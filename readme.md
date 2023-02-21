# Installation

---

After cloning the repo:\
`cd pheno-wheat`\
`pip install virtualenv` (if you don't already have virtualenv installed)\
`virtualenv envwheat` to create the virtualenv for the project\
Use `pip install -r requirements.txt` to install all the requirements

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
Run `train_spike.sh` to train a model specified in the `train_spike.json` file under the key
*config*.\
Run `test_spike.sh` to test the model specified in the `test_spike.json` file under the key 
*config* loaded from the checkpoint file 
specified under the key *checkpoint*

### 2. Spikelet Segmentation

Download the [Spikelet segmentation dataset]()\

- [ ] Complete the spikelet annotation 
- [ ] Write scripts for training and testing spikelet segmentation models
- [ ] Explore different semantic segmentation architectures (including U-net)
- [ ] Compare results
- [ ] Publish another paper 😂


