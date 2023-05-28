#!/bin/bash

# If operating system name contains Darwnin: MacOS. Else Linux
if uname -s | grep -iqF Darwin; then
    MYPATH="$HOME/miniconda3"
    curl -LO "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    sh Miniconda3-latest-MacOSX-x86_64.sh -b -p $MYPATH
else
    MYPATH="/goinfre/$USER/miniconda3"
    curl -LO "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    sh Miniconda3-latest-Linux-x86_64.sh -b -p $MYPATH
fi