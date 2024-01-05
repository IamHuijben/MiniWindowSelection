This repository contains all code to reproduce the paper about mini-window selection in multi-channel time series.

#### Dependencies:

Download the anaconda package (https://www.anaconda.com/).

In the anaconda prompt run:

conda create -n my_env python==3.7.10
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy, mne, h5py, scipy, matplotlib, sklearn

Then activate the environment with: conda activate my_env

#### Running an experiment
First run generate_artificial_data.py to acquire the same artificial data as we used in the paper.
In the paper we used the first 100 outputs for training, the next 50 for validation, and the last 50 for testing.


##### Data processing
The physiological and audio experiments start with running preprocessing/physiological_data_main.py or preprocessing/audio_main.py scripts to acquire the data in the right format. 
Both datasets first need to be downloaded locally to run these scripts.
The toy experiments do not need preprocessing anymore, as we published the generated dataset in the data/random_walk_freq_sinusoids_processed folder. However, to rerun or change preprocessing, you can use preprocessing/synthetic_data_main.py

##### Running a model
After having the data in place, you can train one of the different models from the experiments/<model_type>/<experiment_type>/main.py files for the different models and experiments. 
The trained model will automatically be saved in your working directory.
Note that training the SOM, supervised classifier, or running the fit_pca_kmeans.py script require an already trained CPC model to run on top of.

All model folders contain experiment_specific.yml files that specify all the settings as used in the paper.

<!-- #### Citation
Please cite the following paper if you find this code useful in your work:

```
@inproceedings{huijben2023,
  title={Som-cpc: Unsupervised Contrastive Learning with Self-Organizing Maps for Structured Representations of High-Rate Time Series},
  author={Huijben, Iris AM and Nijdam, Arthur Andreas and Overeem, Sebastiaan and Van Gilst, Merel M and Van Sloun, Ruud},
  booktitle={International Conference on Machine Learning},
  pages={14132--14152},
  year={2023},
  organization={PMLR}
}
``` -->