import torch.nn as nn
import torch
import numpy as np


class GroupedIdxConv1D(nn.Module):
    def __init__(self, nr_groups, out_dim, kernel_size):
        """
        nr_groups (int): Number of groups to group the input channels in.
        out_dim (int): Number of output channels
        **kwargs: All other arguments are passed to the Conv1D layers.

        Performs 1D convolutions on the input data of shape [bs x ch x T], where the data is grouped over the channel axis based on the provided group_idxs.
        A 1D convolution is performed per group_idx, instead of per channel in the input. This effectively weight-ties the convolutional kernels such that all inputs that belong to the same group_idx are filtered with the same kernels.

        This is useful to ensure that all stacked mini-windows that originate from the same ExG channel are filtered with the same kernels.
        """

        super().__init__()

        self.nr_groups = nr_groups
        self.out_dim = out_dim

        for group_idx in range(self.nr_groups):
            if group_idx == 0: #Only add the bias once to mimic a standard Conv1D layer
                bias = True 
            else:
                bias = False
            self.add_module(f'conv1D_group{group_idx}',  nn.Conv1d(1, self.out_dim, bias=bias, kernel_size=kernel_size))
                            
    def forward(self, x, group_idxs):
        """
        x (torch.tensor) of shape [bs x ch x T']
        group_idxs (torch.tensor) Integers between 0 and self.nr_groups-1 that indicate for each element in x to which group it belongs. Shape [bs x ch]
        """
        bs, channels = x.shape[0], x.shape[1]

        assert len(x.shape) == 3, f'Input shape is {x.shape}, but should be [bs x ch x T]'
        assert len(group_idxs.shape) == 2, f'Input shape is {group_idxs.shape}, but should be [bs x ch]'
        assert group_idxs.max() < self.nr_groups, f'Group index {group_idxs.max()} is larger than the number of groups {self.nr_groups}'

        x_r = x.reshape(bs*channels,-1) #[bs*ch x T]
        x_conv_outputs = torch.zeros(bs*channels, self.out_dim, 1, device=x_r.device, dtype=x_r.dtype) #[bs*ch x out_dim x 1]
        
        for group_idx in range(self.nr_groups):
            channel_mask = group_idxs.flatten() == group_idx #[bs*ch]

            if channel_mask.sum() == 0: #No data belonging to this group_idx in this batch
                continue
            
            x_partial = x_r[channel_mask] #[bs*nr_selected_entries x T']
            x_conv = self.__getattr__(f'conv1D_group{group_idx}')(x_partial.unsqueeze(1))  # [bs*selected_entries x out_channels x T']

            if x_conv_outputs.shape[-1] != x_conv.shape[-1]: # Make a placeholder for the time dimension in the output tensor as well
                x_conv_outputs = x_conv_outputs.repeat(1,1,x_conv.shape[-1]) #[bs*ch x out_dim x T'']

            x_conv_outputs[channel_mask] = x_conv

        x_conv_outputs = x_conv_outputs.reshape((bs, channels, self.out_dim, -1)) #[bs x ch x channels_out x T'']

        # Sum over the input_channels like in a normal convolutional layer
        return torch.sum(x_conv_outputs, 1) #[bs x out_channels x T'']
    
class Model(nn.Module):
    def __init__(self, hypernet_settings, encoder_settings, sampling_type, nr_classes):
        super().__init__()

        self.mini_window_temporal_reduction_factor = 6 # 6 (multi-channel) mini-windows fit in 30 seconds
        self.nr_data_channels, self.window_L = hypernet_settings['input_dim']
        self.nr_classes = nr_classes

        # Initialize everything related to the hyper-network and mini-window selection
        self.sampling_type = sampling_type
        assert self.sampling_type in ['active', 'random']
        self.k = hypernet_settings.get('k', 5)
        assert self.k > 0
        
        self.N = self.nr_data_channels*self.mini_window_temporal_reduction_factor
        
        if self.sampling_type == 'active':
            self.build_hypernet(self.nr_data_channels, hypernet_settings['output_channels'], hypernet_settings['kernel_sizes'], hypernet_settings['poolings'])
        self.gumbel_softmax_temperature = hypernet_settings['gumbel_softmax_temperature']
        self.init_GS_temperature()
        self.gumbel = torch.distributions.gumbel.Gumbel(0, 1)

        # Initialize encoder and classifier
        self.build_encoder(encoder_settings['output_channels'], encoder_settings['kernel_sizes'], encoder_settings['poolings'])
        self.build_classifier(encoder_settings['output_channels'][-1])


    def init_GS_temperature(self):
        self.decay_factor_GS = torch.as_tensor((np.log(self.gumbel_softmax_temperature['start'])-np.log(self.gumbel_softmax_temperature['end']))/(self.gumbel_softmax_temperature['nr_epochs']))
        self.register_buffer('GS_temp', torch.as_tensor(self.gumbel_softmax_temperature.get('start', 1.0))) 

    def update_GS_temperature(self, epoch):
        self.GS_temp.data = torch.maximum(torch.as_tensor(self.gumbel_softmax_temperature['end']),self.gumbel_softmax_temperature['start'] * np.exp(-self.decay_factor_GS*epoch))

    def build_hypernet(self, input_channels, output_channels, kernel_sizes, poolings):
        self.hypernet = nn.Sequential()

        for block_idx, (output_ch, kernel_size) in enumerate(zip(output_channels, kernel_sizes)):
            self.hypernet.add_module(f'block_{block_idx}_conv1d',nn.Conv1d(input_channels, output_ch, kernel_size=kernel_size, padding='same', padding_mode='reflect'))      

            if block_idx < len(output_channels)-1:  
                self.hypernet.add_module(f'block_{block_idx}_activation', nn.LeakyReLU())
                self.hypernet.add_module(f'block_{block_idx}_pooling', nn.AvgPool1d(poolings[block_idx]))
                self.hypernet.add_module(f'block_{block_idx}_dropout', nn.Dropout(0.1))
            else: # Last block
                self.hypernet.add_module(f'block_{block_idx}_activation', nn.Tanh())
                self.hypernet.add_module('flatten', nn.Flatten(start_dim=1, end_dim=-1))

                # Compute the latent space dimensionality
                latent_dim = self.window_L
                for pool_factor in poolings:
                    latent_dim = latent_dim // pool_factor
                latent_dim *= output_channels[-1]
        
                self.hypernet.add_module('linear', nn.Linear(in_features=latent_dim, out_features=self.N, bias=False))
                self.hypernet.add_module('activationMLP', nn.LeakyReLU())
                self.hypernet.add_module('linear2', nn.Linear(in_features=self.N, out_features=self.N, bias=False))
            
            input_channels = output_ch


    def build_encoder(self, output_channels, kernel_sizes, poolings):
        # Create the first convolutional layer of the encoder separately, as it takes two inputs
        self.encoder_block_0_conv1dGroupedKernels = GroupedIdxConv1D(nr_groups=self.nr_data_channels, out_dim=output_channels[0], kernel_size=kernel_sizes[0])

        self.encoder = nn.Sequential() 
        for block_idx, (output_ch, kernel_size) in enumerate(zip(output_channels, kernel_sizes)):
            if block_idx > 0:
                self.encoder.add_module(f'block_{block_idx}_conv1d',nn.Conv1d(input_channels, output_ch, kernel_size=kernel_size))      

            if block_idx < len(output_channels)-1:  
                self.encoder.add_module(f'block_{block_idx}_activation', nn.LeakyReLU())
                self.encoder.add_module(f'block_{block_idx}_pooling', nn.MaxPool1d(poolings[block_idx]))
                self.encoder.add_module(f'block_{block_idx}_dropout', nn.Dropout(0.1))
            else: # Last block
                self.encoder.add_module(f'block_{block_idx}_pooling', nn.AdaptiveAvgPool1d(1))

            input_channels = output_ch

    def build_classifier(self, input_channels):
        self.classifier = nn.Sequential()
        self.classifier.add_module('linear', nn.Linear(in_features=input_channels, out_features=self.nr_classes))
        self.classifier.add_module('activation', nn.LogSoftmax(dim=-1))


    def compute_sampling_matrix(self, pred_logits):
        # pred_logits are of size: [bs, N=chxnr_mini_windows]
        # Returns a sampling matrix of size: [bs, k, N]

        # In case of training or using random sampling (both during training and validation), add Gumbel noise to make sure that the sampling is stochastic.
        if (self.training == True and self.sampling_type == 'active') or self.sampling_type == 'random':
            gumbel_noise = self.gumbel.sample(list(pred_logits.shape)).to(pred_logits.device)
            perturbed_logits = pred_logits + gumbel_noise
        else: # During evaluation we want to simple select the top-k logits without noise
            perturbed_logits = pred_logits
            
        ### Draw the hard samples
        _, topk_indices = torch.topk(perturbed_logits, self.k, dim=-1) #[bs, k]
        hard_sampling_matrix = torch.nn.functional.one_hot(topk_indices, num_classes=self.N) #[bs, k, N]. 

        ### Create the corresponding soft sample matrix used during training
        if self.training:

            # Copy the N predicted probabilities to have k times this vector, to create the soft sample matrix
            pred_logits = pred_logits.unsqueeze(1).repeat(1,self.k,1) #[bs, k, N]
            
            # Create mask with exclusive cumsum to mask the logits (=replacing with -inf) at the place where samples were drawn.
            # This enforces sampling without replacement in exactly the same was as what happened during hard sampling.
            cum_mask = torch.cumsum(hard_sampling_matrix, dim=1) - hard_sampling_matrix #[bs,k,N]
            pred_logits[cum_mask == 1] = -np.inf            
            gumbel_noise = gumbel_noise.unsqueeze(1)
            soft_sampling_matrix = torch.softmax((pred_logits + gumbel_noise) / self.GS_temp, -1) #[bs, k, N]
            
            # GS estimator
            sample_matrix = soft_sampling_matrix

        else: # We don't need gradients, and we want to evaluate with hard samples
            sample_matrix = hard_sampling_matrix.type(torch.cuda.FloatTensor)

        return sample_matrix, topk_indices
    

    def forward(self, x, epoch):
        """
        x (torch.Tensor): Input data containing a batch of windows of shape [bs, ch, T]
        epoch (int): The current epoch, used to update the Gumbel Softmax temperature

        Returns:
            log_softmax_output (torch.Tensor): The log softmax prediction for each of the classes for every element in the batch. Shape [bs, nr_classes]
        """
        if self.training: self.update_GS_temperature(epoch)

        bs = x.shape[0] 

        if self.sampling_type == 'random':
            pred_logits = torch.ones((bs, self.N), device=x.device) 
        else: #learned sampling: predict logits by the hypernet
            pred_logits = self.hypernet(x) # [bs, N]

        sampling_matrix, topk_indices = self.compute_sampling_matrix(pred_logits) #[bs, k, N], [bs, k]

        x_r = x.reshape((bs, self.N, -1)) #Reshape to: [bs, N, T']
        selected_mini_windows = torch.bmm(sampling_matrix, x_r) # [bs, k, T']

        # Find the original channels of each selected mini-windows, which will be used by the first layer of the encoder
        channel_idxs = topk_indices//self.mini_window_temporal_reduction_factor
        z_intermediate = self.encoder_block_0_conv1dGroupedKernels(selected_mini_windows, channel_idxs)
        z = self.encoder(z_intermediate) #[bs, out_channels, 1] 

        log_softmax_output = self.classifier(torch.swapaxes(z, -1, -2)).squeeze() #[bs, nr_classes]
        return log_softmax_output

