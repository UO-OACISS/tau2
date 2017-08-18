#!/bin/bash

currdir=${PWD}
echo ${PWD}
export PYCOOLR_NODE=cerberus.nic.uoregon.edu

./src/pycoolr/pycoolr-plot/coolr.py configs/beaconcerberus.cfg
