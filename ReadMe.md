<<<<<<< Updated upstream
This repository contains the model implementation belonging to the paper "Attention on Sleep Stage Specific Characteristics", published at the 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC).
The synthetic data can also be created from this repository, in order to train a model yourself. 
=======
This repository contains code to reproduce the experiments on synthetic data, published here . %TODO add link.
>>>>>>> Stashed changes

#### Dependencies:

Download the anaconda package (https://www.anaconda.com/).

In the anaconda prompt run:

conda create -n my_env python==3.7.10
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy, mne, h5py, scipy, matplotlib, sklearn

Then activate the environment with: conda activate my_env

<<<<<<< Updated upstream
#### Running an experiment

##### Data 
Run generate_artificial_data.py to acquire the same artificial data as we used in the paper.
In the paper we used the first 100 outputs for training, the next 50 for validation, and the last 50 for testing.

The data from the Healthbed dataset as used in this study are available from the Sleep Medicine Centre Kempenhaeghe upon reasonable request via M. van Gilst (m.m.v.gilst@tue.nl). The data can be requested by presenting a scientific research question and by fulfilling all the regulations concerning thesharing of the human data (e.g. privacy regulations). The details ofthe agreement will depend on the purpose of the data request andthe entity that is requesting the data (e.g. research institute or corpo-rate). Each request will be evaluated by the Kempenhaeghe Research Board and, depending on the request, approval from an independent medical ethical committee might be required.

##### Running a model
To train a model, use the trainning.ipynb notebook.
=======
### Data
Run generate_artificial_data.py to acquire the same artificial data as we used in the paper. 

The HealthBed data used in this study are available from the Sleep Medicine Centre Kempenhaeghe upon reasonable request. The data can be requested by presenting a scientific research question and by fulfilling all the regulations concerning the sharing of the human data. The details of the agreement will depend on the purpose of the data request and the entity that is requesting the data (e.g. research institute or corporate). Each request will be evaluated by the Kempenhaeghe Research Board and, depending on the request, approval from independent medical ethical committee might be required. Access to data from outside the European Union will further depend on the expected duration of the activity; due to the work required from a regulatory point of view, the data is less suitable for activities that are time critical, or require access in short notice. For inquiries regarding availability, please contact Merel van Gilst (m.m.v.gilst@tue.nl).
 
#### Running an experiment
To run the model with active mini-window selection on the generated synthetic data, run training.ipynb.
It will use the first 100 generated data streams for training, the next 50 for validation, and the last 50 for testing (as done in the paper).

To run the model with random, instead of active, mini-window selection, change the sampling_type in the settings dictionary from 'active' to 'random'. 
>>>>>>> Stashed changes

#### Citation
Please cite the following paper if you find this code useful in your work:

@inproceedings{huijben2023,
<<<<<<< Updated upstream
  title={Attention on Sleep Stage Specific Characteristics},
  author={Huijben, Iris AM and Overeem, Sebastiaan and Van Gilst, Merel M and Van Sloun, Ruud}, 
  booktitle={46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  pages={},
  year={2024},
}
``` -->
=======
  title={Detecting Sleep Stage Specific Data Regions},
  author={Huijben, Iris AM, Overeem, Sebastiaan and Van Gilst, Merel M and Van Sloun, Ruud JG},
  booktitle={International Conference of the IEEE Engineering in Medicine and Biology Society},
  year={2024},
}
>>>>>>> Stashed changes
