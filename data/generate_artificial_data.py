
import numpy as np
from datetime import date
import h5py
import mne
from mne.time_frequency import morlet
from scipy import signal
from pathlib import Path
import random

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def sample_series_from_markov_matrix(markov_prob_matrix, nr_samples):
    """
    markov_prob_matrix (np.ndarray): Matrix of size nr_classes x nr_classes, where each row denotes the previous class, and the columns denote the next class. 
    """
    markov_prob_matrix = markov_prob_matrix / np.sum(markov_prob_matrix,-1) #Make sure the matrix is row-normalized
    nr_classes = markov_prob_matrix.shape[0]

    sampled_classes = np.zeros(nr_samples, dtype=np.int32)-1
    sampled_classes[0] = np.random.randint(0, nr_classes)
    
    for idx in range(1,nr_samples):
        next_class = np.random.choice(nr_classes, p=markov_prob_matrix[sampled_classes[idx-1]])
        sampled_classes[idx] = int(next_class)
        
    return sampled_classes   

def generate_spindle(min_amplt=0.8, max_amplt=1, sf=100):
    """
    min_amplt (float) Minimum amplitude of the spindle
    max_amplt (float) Maximum amplitude of the spindle.
    sf (int): Sampling frequency in Hz
    """

    # Sampling frequency in Hz
    cf = np.random.choice(np.arange(11,16,0.01))             # Central spindles frequency in Hz
    splindle_dur = np.random.choice(np.arange(0.5,2,0.1))    # spindle duration in sec.


    # Compute a Morlet wavelet
    # Multiply the duration by factor 2 to compensate for the flat ends of the Morlet wavelet
    wlt = morlet(sf, [cf], n_cycles=(splindle_dur*2*sf)/cf)[0] 

    # Make the amplitude independent of the duration, but a random value below max_amplt, and randomize the sign
    sign = np.random.choice([1,-1])
    amplt = ur(min_amplt,max_amplt)
    wlt = sign*amplt*(wlt / wlt.max())

    return wlt.real

def generate_triangular_eye_movement(min_amplt=0.8, max_amplt=1, sf=100):
    """
    max_amplt (float) Maximum amplitude of the spindle.
    sf (int): Sampling frequency in Hz
    """

    # Sampling frequency in Hz
    cf = ur(0.5,2)         # Central trinagular shape with frequency between 0.5 and 2 Hz. This results in an eye movement that takes 0.5 to 2 sec.
    duration_in_sec = 1/cf
    t = np.arange(0, duration_in_sec,1/sf)

    #The last argument determines the skewness of the triangle. When it's 0.5, it's perfectly symmetrical
    # +1 to make sure the triangle starts at zero.
    triangle = signal.sawtooth(2 * np.pi * cf * t, ur(0.4,0.6)) +1

    # Make the amplitude independent of the duration, but a random value below max_amplt, and randomize the sign
    amplt = ur(min_amplt,max_amplt) /2

    return amplt * triangle


def generate_wave_in_freq_range(duration_in_sec, low_freq, high_freq, min_amplt=0.5, max_amplt=1, sf=100):
    """
    duration_in_sec (int): Duration of the generated signal
    low_freq (float): Min. freq in Hz of the range in which the signal will be generated
    high_freq (float): Max. freq in Hz of the range in which the signal will be generated
    max_amplt (float) Maximum amplitude of the spindle.
    sf (int): Sampling frequency in Hz
    """

    cf = ur(low_freq, high_freq)
    time = np.arange(0,duration_in_sec,1/sf)
    angle = ur(0,2*np.pi)
    amplt = ur(min_amplt,max_amplt)
    
    return amplt*np.sin(2*np.pi*cf*time+angle)

def return_start_dur_of_class_one_bouts_in_binary_arr(arr):
    # arr is a 1D array that contains 0 and 1. The function returns the indices where class1 bouts start and its duration.
    
    start_class1_bouts = np.where(np.diff(arr)==1)[0]+1
    end_class1_bouts = np.where(np.diff(arr)==-1)[0]+1

    if arr[0] == 1:
        start_class1_bouts = np.concatenate([[0], start_class1_bouts])
        # end_first_class1_bouts = np.argmin(arr) #Find the first non-1 class.
        # end_class1_bouts = np.concatenate([[end_first_class1_bouts], end_class1_bouts])
    if arr[-1] == 1:
        end_class1_bouts = np.concatenate([end_class1_bouts, [len(arr)-1]])
        
    durations = end_class1_bouts-start_class1_bouts
    return start_class1_bouts, durations


def ur(min_val, max_val, nr_samples=1):
    # Samples <nr_samples> iid samples from a uniform distribution between min_val and max_val.
    return (max_val-min_val)*np.random.rand(nr_samples)+min_val

def generate_artificial_EEG_data(save_dir, nr_channels, nr_patients, fs, duration_in_sec, sec_per_annotation, noise_floor_var, class_markov_prob_matrix):
    """ Generates artifical EEG data of 2 channels.

    Args:
        save_dir (str): Save directory of the generated data.
        nr_channels (int): Number of channels to generate data for.
        nr_patients (int): The number of artifical patients to generate data for.
        fs (int): The sampling frequency.
        duration_in_sec (int): Time duration of the generated signals in secs. 
        sec_per_annotation (int): Number of seconds that encapsulate one annotation. 
        noise_floor_var (float): Variance of the zero-mean AWGN that is used to create a noise floor.
        class_markov_prob_matrix (np.ndarray): Array of size [nr_classes x nr_classes] indicating the Markov transition probabilities.

    """

    nr_classes = 3

    time = np.arange(0,duration_in_sec,1/fs)
    x, y, y_highres = {}, {}, np.zeros((nr_classes,duration_in_sec*fs))
    y_spindle, y_eye_mvt, y_delta_waves = {},{},{}
    #smoothing_kernel_size = sec_per_annotation*fs//4 #Stretch the labels such that the fade-in and fade-out of the convolutional kearnel smears out the label to the neighbour epoch as well, facilitating class boundaries across epoch boundaries.

    for patient_idx in range(nr_patients):
        y[f's{patient_idx}'] = sample_series_from_markov_matrix(class_markov_prob_matrix, nr_samples=duration_in_sec//sec_per_annotation)  

        # Start with filling each channel with background white noise floor
        x[f's{patient_idx}'] = np.random.normal(loc=0.0, scale=np.sqrt(noise_floor_var), size=(nr_channels, len(time)))

        high_res_y = np.repeat(y[f's{patient_idx}'], repeats=fs*sec_per_annotation, axis=-1)
        y_spindle[f's{patient_idx}'], y_eye_mvt[f's{patient_idx}'], y_delta_waves[f's{patient_idx}'] = np.zeros_like(high_res_y), np.zeros_like(high_res_y), np.zeros_like(high_res_y)


        for class_idx in range(nr_classes):

            # Create labels that cross epoch boundaries, and creat moments (at the transitions) where multiple classes can be present at the same time by smoothing out each class.
            y_highres[class_idx] = (high_res_y==class_idx).astype(np.int)
        

            # Mimic light sleep with spindles in the first channel.
            if class_idx == 0:
                # Draw uniform probabilities between 0 and 1 for every high_res label. Compare the drawn probabilities to the class mask, where we set the chance of creating a spindle to 0 when the mask equals zero,
                # and to 3/(fs*sec_per_annotation) when the mask equals 1 (i.e. in this class). This results in, on average, 3 spindles per epoch.
                spindle_mask = ur(0, 1, y_highres[class_idx].size) <  3/(fs*sec_per_annotation)*y_highres[class_idx] 

                for sample_idx  in np.where(spindle_mask)[0]:
                    spindle = generate_spindle()
                    # Place middle of spindle at the index where it is positioned in the spindle_mask, in channel 0.
                    start_sample = np.maximum(0,sample_idx-(len(spindle)//2))
                    end_sample = np.minimum(sample_idx+(int(np.ceil(len(spindle)/2))), len(time))

                    x[f's{patient_idx}'][0, start_sample:end_sample] += spindle[:(end_sample-start_sample)]
                    y_spindle[f's{patient_idx}'][start_sample:end_sample] = 1

            # Mimic deep sleep with delta activity in the first channel.
            elif class_idx == 1:
                start_class1_bouts, durations_class1_bouts = return_start_dur_of_class_one_bouts_in_binary_arr(y_highres[class_idx])

                for bout_start, bout_dur in zip(start_class1_bouts, durations_class1_bouts):

                    # Create between 20% and the full class1 bout of delta waves
                    delta_wave_duration_in_sec = ur((bout_dur/fs)//2, bout_dur/fs)
                    delta_wave = generate_wave_in_freq_range(delta_wave_duration_in_sec, 0.5, 4)

                    # Add delta wave around the middle of the class1 bout.
                    bout_middle = bout_start+bout_dur//2
                    start_sample = np.maximum(0,bout_middle-(len(delta_wave)//2))
                    end_sample = np.minimum(bout_middle+(int(np.ceil(len(delta_wave)/2))), len(time))

                    x[f's{patient_idx}'][0, start_sample:end_sample] += delta_wave[:(end_sample-start_sample)]
                    y_delta_waves[f's{patient_idx}'][start_sample:end_sample] = 1

            # Mimic REM sleep with eye movements in the second channel.
            elif class_idx == 2:
                # Create on average 3 eye movements per epoch
                eye_movement_mask = ur(0, 1, y_highres[class_idx].size) < 3/(fs*sec_per_annotation)*y_highres[class_idx] 
            
                for sample_idx in np.where(eye_movement_mask)[0]:
                    eye_movement = generate_triangular_eye_movement()

                    # Place middle of the eye movement at the index where it is positioned in the eye_movement_mask, in channel 1.
                    start_sample = np.maximum(0,sample_idx-(len(eye_movement)//2))
                    end_sample = np.minimum(sample_idx+(int(np.ceil(len(eye_movement)/2))), len(time))

                    x[f's{patient_idx}'][1, start_sample:end_sample] += eye_movement[:(end_sample-start_sample)]
                    y_eye_mvt[f's{patient_idx}'][start_sample:end_sample] = 1
            else:
                raise NotImplementedError

        # Check for epochs without any of the characteristic features, relabel them to the label of the last epoch that actually had at least one characteristic feature.
        highres_idxs_with_features_mask = np.stack([y_spindle[f's{patient_idx}'], y_delta_waves[f's{patient_idx}'], y_eye_mvt[f's{patient_idx}']],0).sum(axis=0)
        epochs_idxs_wo_features_mask = (np.stack(np.split(highres_idxs_with_features_mask, duration_in_sec//sec_per_annotation)).sum(-1)==0).astype(np.int)
        for epoch_idx, feat_not_present in enumerate(epochs_idxs_wo_features_mask):
            if feat_not_present:
                prev_epoch_idx = epoch_idx-1
                while epochs_idxs_wo_features_mask[prev_epoch_idx] == 1:
                    prev_epoch_idx -= 1
                    if prev_epoch_idx == 0:
                        break
                
                y[f's{patient_idx}'][epoch_idx] = y[f's{patient_idx}'][prev_epoch_idx]


        
        # Save the data
        patient_str = f's{patient_idx}'
        hf = h5py.File(save_dir / f'{patient_str}.h5', 'a')

        for ch_idx, ch_data in enumerate(x[patient_str]):
            hf.create_dataset(f'ch{ch_idx}', data=ch_data)
            
        hf.create_dataset('nr_samples', data=[len(ch_data)])
        hf.create_dataset('fs', data=[fs])
        hf.create_dataset('labels', data=y[patient_str])

        #Store the high-resolution masks per characteristic feature as well.
        hf.create_dataset('y_spindle', data=y_spindle[patient_str]) 
        hf.create_dataset('y_eye_mvt', data=y_eye_mvt[patient_str])
        hf.create_dataset('y_delta_waves', data=y_delta_waves[patient_str])
        hf.close()

        print(f's{patient_idx} finished')
    

if __name__ == '__main__':
    save_dir = Path('data/synthetic_data')
    save_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(1)
    generate_artificial_EEG_data(save_dir = save_dir,
                                 nr_channels = 2,
                                 nr_patients=200, 
                                 fs=100, 
                                 duration_in_sec=5400,  #duration of one generated data stream in seconds
                                 sec_per_annotation=30,  
                                 noise_floor_var=0.1, 
                                 class_markov_prob_matrix=np.array([[0.5, 0.3, 0.2], 
                                                                    [0.2, 0.7, 0.1], 
                                                                    [0.25,0.0,0.75]])
                                )