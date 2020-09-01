# SeisSrcInv

A full waveform seismic source mechanism inversion package. This package takes Green's functions and real, observed seismograms and performs a Bayesian inversion. The package also allows for the inversion results to be plotted.

## To cite:

The method is detailed completely here:
T Hudson, AM Brisbourne, F Walter, D Graff, R White, A Smith “Icequake source mechanisms for studying glacial sliding” JGR Earth Surface (in review) (preprint available here: https://doi.org/10.1002/essoar.10502610.1)

Zenodo citation:
[![DOI](https://zenodo.org/badge/177765984.svg)](https://zenodo.org/badge/latestdoi/177765984)

## To install:
### Using Pip:
pip install SeisSrcInv

### Manually:
Download SeisSrcInv from https://github.com/TomSHudson/SeisSrcInv
Install by:
python setup.py install

## Usage:
For example of usage, see example.ipynb jupyter-notebook or example.pdf. The jupyter-notebook should run once SeisSrcInv is installed.
For other optional arguments try:
help(SeisSrcInv.inversion.run)
help(SeisSrcInv.plot.run)
