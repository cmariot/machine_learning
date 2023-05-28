#!/bin/bash

# If operating system name contains Darwnin: MacOS. Else Linux
if uname -s | grep -iqF Darwin; then
    MYPATH="$HOME/miniconda3"
    $MYPATH/bin/conda init zsh
    $MYPATH/bin/conda config --set auto_activate_base false
else
    MYPATH="/goinfre/$USER/miniconda3"
    $MYPATH/bin/conda init zsh
    $MYPATH/bin/conda config --set auto_activate_base false
fi 

source ~/.zshrc