#!/usr/bin/env bash

SOURCE=./envwheat/bin/activate

if [ -d "$SOURCE" ]
then
`source $SOURCE`
else
SOURCE=~/envonpremise/bin/activate
`source $SOURCE`
fi
python -m test_spike
