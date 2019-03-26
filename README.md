# SeisSrcInv

A full waveform seismic source mechanism inversion package. This package takes Green's functions and real, observed seismograms and performs a Bayesian inversion. The package also allows for the inversion results to be plotted.

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
