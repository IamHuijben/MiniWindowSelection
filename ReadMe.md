This repository contains the model implementation belonging to the paper "_Attention on Sleep Stage Specific Characteristics_", published at the 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC).

The synthetic data can also be created from this repository, in order to train a model yourself. 

#### Dependencies:

Download the anaconda package (https://www.anaconda.com/).

In the anaconda prompt run:

conda create -n my_env python==3.7.10
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy, mne, h5py, scipy, matplotlib, sklearn

Then activate the environment with: conda activate my_env

#### Running an experiment

##### Data 
Run generate_artificial_data.py to acquire the same artificial data as we used in the paper.
In the paper we used the first 100 outputs for training, the next 50 for validation, and the last 50 for testing, which is automatically handled in the dataloader.

The data from the Healthbed dataset as used in this study are available from the Sleep Medicine Centre Kempenhaeghe upon reasonable request via M. van Gilst (m.m.v.gilst@tue.nl). The data can be requested by presenting a scientific research question and by fulfilling all the regulations concerning thesharing of the human data (e.g. privacy regulations). The details ofthe agreement will depend on the purpose of the data request andthe entity that is requesting the data (e.g. research institute or corpo-rate). Each request will be evaluated by the Kempenhaeghe Research Board and, depending on the request, approval from an independent medical ethical committee might be required.

##### Running a model
To run the model with active mini-window selection on the generated synthetic data, run training.ipynb.
It will use the first 100 generated data streams for training, the next 50 for validation, and the last 50 for testing (as done in the paper).

To run the model with random, instead of active, mini-window selection, change the sampling_type in the settings dictionary from 'active' to 'random'. 

#### Citation
Please cite the following paper if you find this code useful in your work:

``` -->
@inproceedings{huijben2024,
  title={Attention on Sleep Stage Specific Characteristics},
  author={Huijben, Iris AM and Overeem, Sebastiaan and van Gilst, Merel M and van Sloun, Ruud JG}, 
  booktitle={46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  pages={},
  year={2024},
}
```
