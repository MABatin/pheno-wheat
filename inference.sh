#!/usr/bin/env bash

SOURCE=./envwheat/bin/activate

if [ -d "$SOURCE" ]
then
`source $SOURCE`
else
SOURCE=~/envonpremise/bin/activate
`source $SOURCE`
fi
python -m image_demo \
/pheno-wheat/demo/spike_yellow.png \
/pheno-wheat/configs/models/WheatSpikeNet.py \
/pheno-wheat/checkpoints/WheatSpikeNet.pth

