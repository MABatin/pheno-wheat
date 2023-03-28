# Installation

---

After cloning the repo:\
`cd pheno-wheat`\
`pip install virtualenv` (if you don't already have virtualenv installed)\
`python3 -m virtualenv envwheat` to create the virtual environment for the project\
`source envwheat/bin/activate` to activate virtual environment\
Use `pip install -r requirements.txt` to install all the requirements

Install mmdetection and mmsegmentation by following:
1. mmdetection
````
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
````
2. mmsegmentation
````
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
````
# Instructions

---
### 1. Spike Segmentation

Download the [Spike segmentation dataset](https://drive.google.com/file/d/1O5Iauv3vrC3NFLrJDZwnUPTdb72uZxSY/view?usp=share_link) \
Extract the zip file in the **data** directory in the following structure:
```
pheno-wheat
├── data
│   └── SPIKE_main
│       ├── annotations
│       ├── test
│       ├── train
│       └── val
└── ...
```
Run `./train_spike.sh` to train a model specified in the `train_spike.json` file under the key
*config*.\
(create from `train_spike.json.template`)

(if permission denied error shows up use the command `chmod +x ./train_spike.sh`)

Run `./test_spike.sh` to test the model specified in the `test_spike.json` file under the key 
*config* loaded from the checkpoint file specified under the key *checkpoint*. \
(create from `test_spike.json.template`)

To run inference on a single image in `demo/` folder, run `./inference.sh`. 

### 2. Spikelet Segmentation

_Ongoing project_


