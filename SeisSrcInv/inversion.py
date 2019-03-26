#!python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to undertake full waveform inversion for a source. Can invert for DC, full unconstrained MT, DC-crack or single force sources.

# Input variables:
# See run() function description.

# Output variables:
# See run() function description.

# Created by Tom Hudson, 28th March 2018

# Notes:
# Currently performs variance reduction on normalised data (as there is an issue with amplitude scaling when using DC events)
# Units of greens functions produced by fk are:
#   10^-20 cm/(dyne cm) - for DC and explosive sources
#   10^-15 cm/dyne - for single force
#   Therefore DC/explosive sources have displacements that are 10^3 times smaller - this correction factor is applied when importing greens functions
#   Also, convert to SI units, so system isn't in cm and dyne but in metres and newtons. Conversion factor is 10^5 (for 1/dyne) x 100 (for cm) = 10^7
#   (See fk readme)

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh # For calculating eigenvalues and eigenvectors of symetric (Hermitian) matrices
import scipy.signal as signal # For cross-correlation calculations
import os,sys
from obspy import UTCDateTime
import pickle
import random # For entirely random number generation
import math
import multiprocessing


# Specify parameters:
datadir = '/Users/tomhudson/Python/obspy_scripts/fk/test_data/output_data_for_inversion_MT_and_single_force_Rhone_gl_event_20180214185538'
outdir = "./python_FW_outputs"
real_data_fnames = ['real_data_RA51_l.txt', 'real_data_RA52_l.txt', 'real_data_RA53_l.txt', 'real_data_RA54_l.txt', 'real_data_RA55_l.txt', 'real_data_RA56_l.txt', 'real_data_RA57_l.txt', 'real_data_RA51_q.txt', 'real_data_RA52_q.txt', 'real_data_RA53_q.txt', 'real_data_RA54_q.txt', 'real_data_RA55_q.txt', 'real_data_RA56_q.txt', 'real_data_RA57_q.txt', 'real_data_RA51_t.txt', 'real_data_RA52_t.txt', 'real_data_RA53_t.txt', 'real_data_RA54_t.txt', 'real_data_RA55_t.txt', 'real_data_RA56_t.txt', 'real_data_RA57_t.txt'] ##['real_data_SKR01_z.txt', 'real_data_SKR02_z.txt', 'real_data_SKR03_z.txt', 'real_data_SKR04_z.txt', 'real_data_SKR05_z.txt', 'real_data_SKR06_z.txt', 'real_data_SKR07_z.txt', 'real_data_SKG08_z.txt', 'real_data_SKG13_z.txt'] ##['real_data_ST01_z.txt', 'real_data_ST02_z.txt', 'real_data_ST03_z.txt', 'real_data_ST04_z.txt', 'real_data_ST05_z.txt', 'real_data_ST08_z.txt'] #['real_data_RA51_l.txt', 'real_data_RA52_l.txt', 'real_data_RA53_l.txt']#, 'real_data_RA51_r.txt', 'real_data_RA52_r.txt', 'real_data_RA53_r.txt', 'real_data_RA51_t.txt', 'real_data_RA52_t.txt', 'real_data_RA53_t.txt'] #['real_data_ST01_z.txt', 'real_data_ST02_z.txt', 'real_data_ST03_z.txt', 'real_data_ST04_z.txt', 'real_data_ST05_z.txt', 'real_data_ST06_z.txt', 'real_data_ST07_z.txt', 'real_data_ST08_z.txt', 'real_data_ST09_z.txt', 'real_data_ST10_z.txt'] #['real_data_RA51_z.txt', 'real_data_RA52_z.txt', 'real_data_RA53_z.txt', 'real_data_RA51_r.txt', 'real_data_RA52_r.txt', 'real_data_RA53_r.txt', 'real_data_RA51_t.txt', 'real_data_RA52_t.txt', 'real_data_RA53_t.txt'] # List of real waveform data files within datadir corresponding to each station (i.e. length is number of stations to invert for)
MT_green_func_fnames = ['green_func_array_MT_RA51_l.txt', 'green_func_array_MT_RA52_l.txt', 'green_func_array_MT_RA53_l.txt', 'green_func_array_MT_RA54_l.txt', 'green_func_array_MT_RA55_l.txt', 'green_func_array_MT_RA56_l.txt', 'green_func_array_MT_RA57_l.txt', 'green_func_array_MT_RA51_q.txt', 'green_func_array_MT_RA52_q.txt', 'green_func_array_MT_RA53_q.txt', 'green_func_array_MT_RA54_q.txt', 'green_func_array_MT_RA55_q.txt', 'green_func_array_MT_RA56_q.txt', 'green_func_array_MT_RA57_q.txt', 'green_func_array_MT_RA51_t.txt', 'green_func_array_MT_RA52_t.txt', 'green_func_array_MT_RA53_t.txt', 'green_func_array_MT_RA54_t.txt', 'green_func_array_MT_RA55_t.txt', 'green_func_array_MT_RA56_t.txt', 'green_func_array_MT_RA57_t.txt'] ##['green_func_array_MT_SKR01_z.txt', 'green_func_array_MT_SKR02_z.txt', 'green_func_array_MT_SKR03_z.txt', 'green_func_array_MT_SKR04_z.txt', 'green_func_array_MT_SKR05_z.txt', 'green_func_array_MT_SKR06_z.txt', 'green_func_array_MT_SKR07_z.txt', 'green_func_array_MT_SKG08_z.txt', 'green_func_array_MT_SKG13_z.txt'] ##['green_func_array_MT_ST01_z.txt', 'green_func_array_MT_ST02_z.txt', 'green_func_array_MT_ST03_z.txt', 'green_func_array_MT_ST04_z.txt', 'green_func_array_MT_ST05_z.txt', 'green_func_array_MT_ST08_z.txt'] #['green_func_array_MT_RA51_l.txt', 'green_func_array_MT_RA52_l.txt', 'green_func_array_MT_RA53_l.txt']#, 'green_func_array_MT_RA51_r.txt', 'green_func_array_MT_RA52_r.txt', 'green_func_array_MT_RA53_r.txt', 'green_func_array_MT_RA51_t.txt', 'green_func_array_MT_RA52_t.txt', 'green_func_array_MT_RA53_t.txt'] #['green_func_array_MT_ST01_z.txt', 'green_func_array_MT_ST02_z.txt', 'green_func_array_MT_ST03_z.txt', 'green_func_array_MT_ST04_z.txt', 'green_func_array_MT_ST05_z.txt', 'green_func_array_MT_ST06_z.txt', 'green_func_array_MT_ST07_z.txt', 'green_func_array_MT_ST08_z.txt', 'green_func_array_MT_ST09_z.txt', 'green_func_array_MT_ST10_z.txt'] #['green_func_array_MT_RA51_z.txt', 'green_func_array_MT_RA52_z.txt', 'green_func_array_MT_RA53_z.txt', 'green_func_array_MT_RA51_r.txt', 'green_func_array_MT_RA52_r.txt', 'green_func_array_MT_RA53_r.txt', 'green_func_array_MT_RA51_t.txt', 'green_func_array_MT_RA52_t.txt', 'green_func_array_MT_RA53_t.txt'] # List of Green's functions data files (generated using fk code) within datadir corresponding to each station (i.e. length is number of stations to invert for)
single_force_green_func_fnames = ['green_func_array_single_force_RA51_l.txt', 'green_func_array_single_force_RA52_l.txt', 'green_func_array_single_force_RA53_l.txt', 'green_func_array_single_force_RA54_l.txt', 'green_func_array_single_force_RA55_l.txt', 'green_func_array_single_force_RA56_l.txt', 'green_func_array_single_force_RA57_l.txt', 'green_func_array_single_force_RA51_q.txt', 'green_func_array_single_force_RA52_q.txt', 'green_func_array_single_force_RA53_q.txt', 'green_func_array_single_force_RA54_q.txt', 'green_func_array_single_force_RA55_q.txt', 'green_func_array_single_force_RA56_q.txt', 'green_func_array_single_force_RA57_q.txt', 'green_func_array_single_force_RA51_t.txt', 'green_func_array_single_force_RA52_t.txt', 'green_func_array_single_force_RA53_t.txt', 'green_func_array_single_force_RA54_t.txt', 'green_func_array_single_force_RA55_t.txt', 'green_func_array_single_force_RA56_t.txt', 'green_func_array_single_force_RA57_t.txt'] ##['green_func_array_single_force_SKR01_z.txt', 'green_func_array_single_force_SKR02_z.txt', 'green_func_array_single_force_SKR03_z.txt', 'green_func_array_single_force_SKR04_z.txt', 'green_func_array_single_force_SKR05_z.txt', 'green_func_array_single_force_SKR06_z.txt', 'green_func_array_single_force_SKR07_z.txt', 'green_func_array_single_force_SKG08_z.txt', 'green_func_array_single_force_SKG13_z.txt'] ##['green_func_array_single_force_ST01_z.txt', 'green_func_array_single_force_ST02_z.txt', 'green_func_array_single_force_ST03_z.txt', 'green_func_array_single_force_ST04_z.txt', 'green_func_array_single_force_ST05_z.txt', 'green_func_array_single_force_ST08_z.txt'] #['green_func_array_single_force_RA51_l.txt', 'green_func_array_single_force_RA52_l.txt', 'green_func_array_single_force_RA53_l.txt']#, 'green_func_array_single_force_RA51_r.txt', 'green_func_array_single_force_RA52_r.txt', 'green_func_array_single_force_RA53_r.txt', 'green_func_array_single_force_RA51_t.txt', 'green_func_array_single_force_RA52_t.txt', 'green_func_array_single_force_RA53_t.txt'] #['green_func_array_single_force_ST01_z.txt', 'green_func_array_single_force_ST02_z.txt', 'green_func_array_single_force_ST03_z.txt', 'green_func_array_single_force_ST04_z.txt', 'green_func_array_single_force_ST05_z.txt', 'green_func_array_single_force_ST06_z.txt', 'green_func_array_single_force_ST07_z.txt', 'green_func_array_single_force_ST08_z.txt', 'green_func_array_single_force_ST09_z.txt', 'green_func_array_single_force_ST10_z.txt'] #['green_func_array_single_force_RA51_z.txt', 'green_func_array_single_force_RA52_z.txt', 'green_func_array_single_force_RA53_z.txt', 'green_func_array_single_force_RA51_r.txt', 'green_func_array_single_force_RA52_r.txt', 'green_func_array_single_force_RA53_r.txt', 'green_func_array_single_force_RA51_t.txt', 'green_func_array_single_force_RA52_t.txt', 'green_func_array_single_force_RA53_t.txt'] # List of Green's functions data files (generated using fk code) within datadir corresponding to each station (i.e. length is number of stations to invert for)
data_labels = ["RA51, L", "RA52, L", "RA53, L", "RA54, L", "RA55, L", "RA56, L", "RA57, L", "RA51, Q", "RA52, Q", "RA53, Q", "RA54, Q", "RA55, Q", "RA56, Q", "RA57, Q", "RA51, T", "RA52, T", "RA53, T", "RA54, T", "RA55, T", "RA56, T", "RA57, T"] ##["SKR01, Z", "SKR02, Z", "SKR03, Z", "SKR04, Z", "SKR05, Z", "SKR06, Z", "SKR07, Z", "SKG08, Z", "SKG13, Z"] ##["ST01, Z", "ST02, Z", "ST03, Z", "ST04, Z", "ST05, Z", "ST08, Z"] #["RA51, L", "RA52, L", "RA53, L"]#, "RA51, R", "RA52, R", "RA53, R", "RA51, T", "RA52, T", "RA53, T"] #["ST01, Z", "ST02, Z", "ST03, Z", "ST04, Z", "ST05, Z", "ST06, Z", "ST07, Z", "ST08, Z", "ST09, Z", "ST10, Z"] #["RA51, Z", "RA52, Z", "RA53, Z", "RA51, R", "RA52, R", "RA53, R", "RA51, T", "RA52, T", "RA53, T"] # Format of these labels must be of the form "station_name, comp" with the comma
inversion_type = "single_force_crack_no_coupling" # Inversion type can be: full_mt, full_mt_Lune_samp, DC, single_force, DC_single_force_couple, DC_single_force_no_coupling, DC_crack_couple, or single_force_crack_no_coupling. (if single force, greens functions must be 3 components rather than 6)
perform_normallised_waveform_inversion = False ###False # Boolean - If True, performs normallised waveform inversion, whereby each synthetic and real waveform is normallised before comparision. Effectively removes overall amplitude from inversion if True. Should use True if using VR comparison method.
compare_all_waveforms_simultaneously = False # Bolean - If True, compares all waveform observations together to give one similarity value. If False, compares waveforms from individual recievers separately then combines using equally weighted average. Default = True.
num_samples = 10000 #1000000 # Number of samples to perform Monte Carlo over
comparison_metric = "VR" # Options are VR (variation reduction), CC (cross-correlation of static signal), CC-shift (cross-correlation of signal with shift allowed), or PCC (Pearson correlation coeficient), gau (Gaussian based method for estimating the true statistical probability) (Note: CC is the most stable, as range is naturally from 0-1, rather than -1 to 1)
manual_indices_time_shift_MT = [23, 22, 21, 23, 25, 23, 23, 22, 22, 21, 23, 25, 23, 24, 24, 24, 21, 23, 28, 28, 25] # Values by which to shift greens functions (must be integers here)
manual_indices_time_shift_SF = [22, 22, 21, 22, 24, 23, 23, 22, 22, 21, 22, 24, 23, 24, 23, 23, 20, 22, 27, 27, 24] # Values by which to shift greens functions (must be integers here)
cut_phase_start_vals = [] # Indices by which to begin cut phase (must be integers, and specified for every trace, if specified). (Default is not to cut the P and S phases) (must specify cut_phase_end_vals too)
cut_phase_length = 100 # Length to cut phases by. Integer. Currently this number must be constant, as code cannot deal with different data lengths.
nlloc_hyp_filename = "NLLoc_data/loc.20180214.185538.grid0.loc.hyp" ##"NLLoc_data/loc.Tom__RunNLLoc000.20090121.042009.grid0.loc.hyp" #"NLLoc_data/loc.run1.20171222.022435.grid0.loc.hyp" #"NLLoc_data/loc.Tom__RunNLLoc000.20090121.042009.grid0.loc.hyp" # Nonlinloc filename for saving event data to file in MTFIT format (for plotting, further analysis etc)
plot_switch = True # If True, will plot outputs to screen
num_processors = 1 #1 # Number of processors to run for (default is 1)
set_pre_time_shift_values_to_zero_switch = True # If true, sets values before time shift to zero, to account for rolling the data on itself (default is True)
only_save_non_zero_solns_switch = False # If True, will only save results with a non-zero probability.
return_absolute_similarity_values_switch = True # If True, will also save absolute similarity values, as well as the normallised values. (will be saved to the output dict as )
invert_for_ratio_of_multiple_media_greens_func_switch = False # If True, allows for invertsing for the ratio of two sets of greens functions, for different media, relative to one another (with the split in greens function fnames sepcified by green_func_fnames_split_index).
green_func_fnames_split_index = 6 # Index of first greens function fname for second medium
green_func_phase_labels = ['P','P','P','P','P','P','P','S','S','S','S','S','S','S','S','S','S','S','S','S','S'] # List of same length as data_labels, to specify the phase associated with each greens function. Can be "P", "S", or "surface". If this parameter is specified then will use multiple greens function ratios.
invert_for_relative_magnitudes_switch = True # If True, inverts for relative magnitude. Notes: Must have perform_normallised_waveform_inversion=False; Will then vary magnitude by 10^lower range to upper range, specified by rel_exp_mag_range. (Default is False)
rel_exp_mag_range = [-3.0, 3.0] # Values of lower and upper exponent for 10^x , e.g. [-3.0,3.0] would be relative magnitude range from 10^-3 to 10^3 (Default is [0.0,0.0])



# ------------------- Define various functions used in script -------------------
def load_input_data(datadir, real_data_fnames, green_func_fnames, manual_indices_time_shift=[], cut_phase_start_vals=[], cut_phase_length=0, set_pre_time_shift_values_to_zero_switch=True):
    """Function to load input data and output as arrays of real data and greens functions.
    Inputs: arrays containing filenames of files with real data (columns for P (L component) only at the moment) and greens functions data (For M_xx, M_yy, M_zz, M_xy, M_xz, M_yz), respectively. Optional input is manual_indices_time_shift (an array/list of manual integer index time shifts for each station).
    Outputs: Real data array of shape (t, n) where t is number of time data points and n is number of stations; greens functions array of shape (t, g_n) where g_n is the number of greens functions components."""
    # Set up data storage arrays:
    tmp_real_data = np.loadtxt(datadir+"/"+real_data_fnames[0],dtype=float)
    tmp_green_func_data = np.loadtxt(datadir+"/"+green_func_fnames[0],dtype=float)
    num_time_pts = len(tmp_real_data) # Number of time points
    num_green_func_comp = len(tmp_green_func_data[0,:]) # Number of greens functions components
    real_data_array = np.zeros((len(real_data_fnames), num_time_pts), dtype=float)
    green_func_array_raw = np.zeros((len(real_data_fnames), num_green_func_comp, num_time_pts), dtype=float)
    
    # Loop over files, saving real and greens functions data to arrays:
    for i in range(len(real_data_fnames)):
        real_data_array[i, :] = np.loadtxt(datadir+"/"+real_data_fnames[i],dtype=float)
        green_func_array_raw[i, :, :] = np.transpose(np.loadtxt(datadir+"/"+green_func_fnames[i],dtype=float))
    
    # Shift greens functions by manually specified amount in time (if specified):
    # (for allignment so that real data and greens functions are alligned)
    if not len(manual_indices_time_shift) == 0:
        green_func_array = np.zeros(np.shape(green_func_array_raw), dtype=float)
        for i in range(len(manual_indices_time_shift)):
            green_func_array[i,:,:] = np.roll(green_func_array_raw[i,:,:], manual_indices_time_shift[i], axis=1) # Roll along time axis
            if set_pre_time_shift_values_to_zero_switch == True:
                green_func_array[i,:,0:manual_indices_time_shift[i]] = 0. # and set values before orignal pre-roll start to zero
    else:
        green_func_array = green_func_array_raw
    
    # Cut out phases rather than using whole length of data, if specified:
    if len(cut_phase_start_vals)>0:
        real_data_array_cut_phases = np.zeros((len(real_data_array[:,0]), cut_phase_length), dtype=float)
        green_func_array_cut_phases = np.zeros((len(green_func_array[:,0]), num_green_func_comp, cut_phase_length), dtype=float)
        for i in range(len(real_data_fnames)):
            real_data_array_cut_phases[i, :] = real_data_array[i, int(cut_phase_start_vals[i]):int(cut_phase_start_vals[i]+cut_phase_length)]
            green_func_array_cut_phases[i, :, :] = green_func_array[i, :, int(cut_phase_start_vals[i]):int(cut_phase_start_vals[i]+cut_phase_length)]
        real_data_array = real_data_array_cut_phases
        green_func_array = green_func_array_cut_phases
    
    return real_data_array, green_func_array
    

def load_input_data_multiple_media(datadir, real_data_fnames, green_func_fnames, green_func_fnames_split_index, manual_indices_time_shift=[], cut_phase_start_vals=[], cut_phase_length=0, set_pre_time_shift_values_to_zero_switch=True):
    """Function to load input data and output as arrays of real data and greens functions, with greens functions for multiple media.
    Inputs: arrays containing filenames of files with real data (columns for P (L component) only at the moment) and greens functions data (For M_xx, M_yy, M_zz, M_xy, M_xz, M_yz), respectively. Optional input is manual_indices_time_shift (an array/list of manual integer index time shifts for each station).
    Outputs: Real data array of shape (t, n) where t is number of time data points and n is number of stations; greens functions array of shape (t, g_n) where g_n is the number of greens functions components."""
    # Get new fname arrays for each medium:
    green_func_fnames_media_1 = green_func_fnames[:green_func_fnames_split_index]
    green_func_fnames_media_2 = green_func_fnames[green_func_fnames_split_index:]
    if not len(green_func_fnames_media_1) == len(green_func_fnames_media_2):
        print "Greens functions fname array is not correct. Consider whether green_func_fnames_split_index value is correct for splitting the two mediums."
        sys.exit()
    
    # Set up data storage arrays:
    tmp_real_data = np.loadtxt(datadir+"/"+real_data_fnames[0],dtype=float)
    tmp_green_func_data = np.loadtxt(datadir+"/"+green_func_fnames[0],dtype=float)
    num_time_pts = len(tmp_real_data) # Number of time points
    num_green_func_comp = len(tmp_green_func_data[0,:]) # Number of greens functions components
    real_data_array = np.zeros((len(real_data_fnames), num_time_pts), dtype=float)
    green_func_array_raw = np.zeros((len(green_func_fnames_media_1), num_green_func_comp, num_time_pts, 2), dtype=float)
    
    # Loop over files, saving real and greens functions data to arrays:
    for i in range(len(real_data_fnames)):
        real_data_array[i, :] = np.loadtxt(datadir+"/"+real_data_fnames[i],dtype=float)
        # Load in for medium 1:
        green_func_array_raw[i, :, :, 0] = np.transpose(np.loadtxt(datadir+"/"+green_func_fnames_media_1[i],dtype=float))
        # Load in for medium 2:
        green_func_array_raw[i, :, :, 1] = np.transpose(np.loadtxt(datadir+"/"+green_func_fnames_media_2[i],dtype=float))
    
    # Shift greens functions by manually specified amount in time (if specified):
    # (for allignment so that real data and greens functions are alligned)
    if not len(manual_indices_time_shift) == 0:
        green_func_array = np.zeros(np.shape(green_func_array_raw), dtype=float)
        for i in range(len(manual_indices_time_shift)):
            green_func_array[i,:,:,0] = np.roll(green_func_array_raw[i,:,:,0], manual_indices_time_shift[i], axis=1) # Roll along time axis, for medium 1
            green_func_array[i,:,:,1] = np.roll(green_func_array_raw[i,:,:,1], manual_indices_time_shift[i], axis=1) # Roll along time axis, for medium 2
            if set_pre_time_shift_values_to_zero_switch == True:
                green_func_array[i,:,0:manual_indices_time_shift[i], :] = 0. # and set values before orignal pre-roll start to zero
    else:
        green_func_array = green_func_array_raw
    
    # Cut out phases rather than using whole length of data, if specified:
    if len(cut_phase_start_vals)>0:
        real_data_array_cut_phases = np.zeros((len(real_data_array[:,0]), cut_phase_length), dtype=float)
        green_func_array_cut_phases = np.zeros((len(green_func_array[:,0]), num_green_func_comp, cut_phase_length, 2), dtype=float)
        for i in range(len(real_data_fnames)):
            real_data_array_cut_phases[i, :] = real_data_array[i, int(cut_phase_start_vals[i]):int(cut_phase_start_vals[i]+cut_phase_length)]
            green_func_array_cut_phases[i, :, :, :] = green_func_array[i, :, int(cut_phase_start_vals[i]):int(cut_phase_start_vals[i]+cut_phase_length), :]
        real_data_array = real_data_array_cut_phases
        green_func_array = green_func_array_cut_phases
    
    return real_data_array, green_func_array
    

def get_overall_real_and_green_func_data(datadir, real_data_fnames, MT_green_func_fnames, single_force_green_func_fnames, inversion_type, manual_indices_time_shift_MT=[], manual_indices_time_shift_SF=[], cut_phase_start_vals=[], cut_phase_length=0, set_pre_time_shift_values_to_zero_switch=True, invert_for_ratio_of_multiple_media_greens_func_switch=False, green_func_fnames_split_index=0):
    """Function to load input data, depending upon inversion type. Primarily function to control use of load_input_data() function.
    Note: Has multiple manual_indices_time_shift... inputs, as can specify different offsets for MT greens functions and single force greens functions if desired."""
    # Load input data into arrays:
    if inversion_type=="full_mt" or inversion_type=="full_mt_Lune_samp" or inversion_type=="DC" or inversion_type=="DC_crack_couple":
        if invert_for_ratio_of_multiple_media_greens_func_switch:
            real_data_array, green_func_array = load_input_data_multiple_media(datadir, real_data_fnames, MT_green_func_fnames, green_func_fnames_split_index, manual_indices_time_shift_MT, cut_phase_start_vals=cut_phase_start_vals, cut_phase_length=cut_phase_length, set_pre_time_shift_values_to_zero_switch=set_pre_time_shift_values_to_zero_switch)
        else:
            real_data_array, green_func_array = load_input_data(datadir, real_data_fnames, MT_green_func_fnames, manual_indices_time_shift_MT, cut_phase_start_vals=cut_phase_start_vals, cut_phase_length=cut_phase_length, set_pre_time_shift_values_to_zero_switch=set_pre_time_shift_values_to_zero_switch)
        # correct for different units of single force to DC (see note in script header):
        green_func_array = green_func_array*(10**3)
    elif inversion_type=="single_force":
        if invert_for_ratio_of_multiple_media_greens_func_switch:
            real_data_array, green_func_array = load_input_data_multiple_media(datadir, real_data_fnames, single_force_green_func_fnames, green_func_fnames_split_index, manual_indices_time_shift_SF, cut_phase_start_vals=cut_phase_start_vals, cut_phase_length=cut_phase_length, set_pre_time_shift_values_to_zero_switch=set_pre_time_shift_values_to_zero_switch)
        else:
            real_data_array, green_func_array = load_input_data(datadir, real_data_fnames, single_force_green_func_fnames, manual_indices_time_shift_SF, cut_phase_start_vals=cut_phase_start_vals, cut_phase_length=cut_phase_length, set_pre_time_shift_values_to_zero_switch=set_pre_time_shift_values_to_zero_switch)
    elif inversion_type=="DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling" or inversion_type == "single_force_crack_no_coupling":
        if invert_for_ratio_of_multiple_media_greens_func_switch:
            real_data_array, MT_green_func_array = load_input_data_multiple_media(datadir, real_data_fnames, MT_green_func_fnames, green_func_fnames_split_index, manual_indices_time_shift_MT, cut_phase_start_vals=cut_phase_start_vals, cut_phase_length=cut_phase_length, set_pre_time_shift_values_to_zero_switch=set_pre_time_shift_values_to_zero_switch)
            real_data_array, SF_green_func_array = load_input_data_multiple_media(datadir, real_data_fnames, single_force_green_func_fnames, green_func_fnames_split_index, manual_indices_time_shift_SF, cut_phase_start_vals=cut_phase_start_vals, cut_phase_length=cut_phase_length, set_pre_time_shift_values_to_zero_switch=set_pre_time_shift_values_to_zero_switch)
        else:
            real_data_array, MT_green_func_array = load_input_data(datadir, real_data_fnames, MT_green_func_fnames, manual_indices_time_shift_MT, cut_phase_start_vals=cut_phase_start_vals, cut_phase_length=cut_phase_length, set_pre_time_shift_values_to_zero_switch=set_pre_time_shift_values_to_zero_switch)
            real_data_array, SF_green_func_array = load_input_data(datadir, real_data_fnames, single_force_green_func_fnames, manual_indices_time_shift_SF, cut_phase_start_vals=cut_phase_start_vals, cut_phase_length=cut_phase_length, set_pre_time_shift_values_to_zero_switch=set_pre_time_shift_values_to_zero_switch)
        # correct for different units of single force to DC (see note in script header):
        MT_green_func_array = MT_green_func_array*(10**3)
        # And combine all greens functions:
        green_func_array = np.hstack((MT_green_func_array, SF_green_func_array))
    # And convert to SI units (see note in script header):
    green_func_array = green_func_array*(10**7)
    return real_data_array, green_func_array

def get_full_MT_array(mt):
    full_MT = np.array( ([[mt[0],mt[3]/np.sqrt(2.),mt[4]/np.sqrt(2.)],
                          [mt[3]/np.sqrt(2.),mt[1],mt[5]/np.sqrt(2.)],
                          [mt[4]/np.sqrt(2.),mt[5]/np.sqrt(2.),mt[2]]]) )
    return full_MT
    
    
def get_six_MT_from_full_MT_array(full_MT):
    six_MT = np.array([full_MT[0,0], full_MT[1,1], full_MT[2,2], np.sqrt(2.)*full_MT[0,1], np.sqrt(2.)*full_MT[0,2], np.sqrt(2.)*full_MT[1,2]])
    return six_MT
    
def find_eigenvalues_from_sixMT(sixMT):
    """Function to find ordered eigenvalues given 6 moment tensor."""
    # Get full MT:
    MT_current = sixMT
    # And get full MT matrix:
    full_MT_current = get_full_MT_array(MT_current)
    # Find the eigenvalues for the MT solution and sort into descending order:
    w,v = eigh(full_MT_current) # Find eigenvalues and associated eigenvectors for the symetric (Hermitian) MT matrix (for eigenvalue w[i], eigenvector is v[:,i])
    full_MT_eigvals_sorted = np.sort(w)[::-1] # Sort eigenvalues into descending order
    # Calculate gamma and delta (lat and lon) from the eigenvalues:
    lambda1 = full_MT_eigvals_sorted[0]
    lambda2 = full_MT_eigvals_sorted[1]
    lambda3 = full_MT_eigvals_sorted[2]
    return lambda1, lambda2, lambda3
    

def rot_mt_by_theta_phi(full_MT, theta=np.pi, phi=np.pi):
    """Function to rotate moment tensor by angle theta and phi (rotation about Y and then Z axes). Theta and phi must be in radians."""
    rot_theta_matrix = np.vstack(([np.cos(theta), 0., np.sin(theta)],[0., 1., 0.],[-1.*np.sin(theta), 0., np.cos(theta)]))
    rot_phi_matrix = np.vstack(([np.cos(phi), -1.*np.sin(phi), 0.], [np.sin(phi), np.cos(phi), 0.], [0., 0., 1.]))
    full_MT_first_rot = np.dot(rot_theta_matrix, np.dot(full_MT, np.transpose(rot_theta_matrix)))
    full_MT_second_rot = np.dot(rot_phi_matrix, np.dot(full_MT_first_rot, np.transpose(rot_phi_matrix)))
    return full_MT_second_rot
    
def rot_single_force_by_theta_phi(single_force_vector, theta=np.pi, phi=np.pi):
    """Function to rotate NED single force by angle theta and phi, consistent with same theta, phi for mt rotation (rotate about Y and then Z axes). Theta and phi must be in radians."""
    rot_theta_matrix = np.vstack(([np.cos(theta), 0., np.sin(theta)],[0., 1., 0.],[-1.*np.sin(theta), 0., np.cos(theta)]))
    rot_phi_matrix = np.vstack(([np.cos(phi), -1.*np.sin(phi), 0.], [np.sin(phi), np.cos(phi), 0.], [0., 0., 1.]))
    single_force_vector_first_rot = np.dot(rot_theta_matrix, single_force_vector)
    single_force_vector_second_rot = np.dot(rot_phi_matrix, single_force_vector_first_rot)
    return single_force_vector_second_rot

def perform_inversion(real_data_array, green_func_array):
    """Function to perform inversion using real data and greens functions to obtain the moment tensor. (See Walter 2009 thesis, Appendix C for theoretical details).
    Inputs are: real_data_array - array of size (k,t_samples), containing real data to invert for; green_func_array - array of size (k, n_mt_comp, t_samples), containing the greens function data for the various mt components and each station (where k is the number of stations*station components to invert for, t_samples is the number of time samples, and n_mt_comp is the number of moment tensor components specficied in the greens functions array).
    Outputs are: M - tensor of length n_mt_comp, containing the moment tensor (or single force) inverted for."""
    # Perform the inversion:
    D = np.transpose(np.array([np.hstack(list(real_data_array[:]))])) # equivilent to matlab [real_data_array[0]; real_data_array[1]; real_data_array[2]]
    G =  np.transpose(np.vstack(np.hstack(list(green_func_array[:])))) # equivilent to matlab [green_func_array[0]; green_func_array[1]; green_func_array[2]]
    M, res, rank, sing_values_G = np.linalg.lstsq(G,D) # Equivilent to M = G\D; for G not square. If G is square, use linalg.solve(G,D)
    return M


def forward_model(green_func_array, M):
    """Function to forward model for a given set of greens functions and a specified moment tensor (or single force tensor).
    Inputs are: green_func_array - array of size (k, n_mt_comp, t_samples), containing the greens function data for the various mt components and each station; and M - tensor of length n_mt_comp, containing the moment tensor (or single force) to forward model for (where k is the number of stations*station components to invert for, t_samples is the number of time samples, and n_mt_comp is the number of moment tensor components specficied in the greens functions array)
    Outputs are: synth_forward_model_result_array - array of size (k, t_samples), containing synthetic waveforms for a given moment tensor."""
    # And get forward model synthetic waveform result:
    synth_forward_model_result_array = np.zeros(np.shape(green_func_array[:,0,:]), dtype=float)
    # Loop over signals:
    for i in range(len(green_func_array[:,0,0])):
        # Loop over components of greens function and MT solution (summing):
        for j in range(len(M)):
            synth_forward_model_result_array[i,:] += green_func_array[i,j,:]*M[j] # greens function for specific component over all time * moment tensor component
    return synth_forward_model_result_array


def plot_specific_forward_model_result(real_data_array, synth_forward_model_result_array, data_labels, plot_title="", perform_normallised_waveform_inversion=True):
    """Function to plot real data put in with specific forward model waveform result."""
    fig, axarr = plt.subplots(len(real_data_array[:,0]), sharex=True)
    for i in range(len(axarr)):
        if perform_normallised_waveform_inversion:
            axarr[i].plot(real_data_array[i,:]/np.max(np.absolute(real_data_array[i,:])),c='k', alpha=0.6) # Plot real data
            axarr[i].plot(synth_forward_model_result_array[i,:]/np.max(np.absolute(synth_forward_model_result_array[i,:])),c='r',linestyle="--", alpha=0.6) # Plot synth data
        else:
            axarr[i].plot(real_data_array[i,:],c='k', alpha=0.6) # Plot real data
            axarr[i].plot(synth_forward_model_result_array[i,:],c='r',linestyle="--", alpha=0.6) # Plot synth data
        axarr[i].set_title(data_labels[i])
    plt.suptitle(plot_title)
    plt.show()


def generate_random_MT():
    """Function to generate random moment tensor using normal distribution projected onto a 6-sphere method. (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972).
    Returns a random normalised 6 MT."""
    # Generate 6 indepdendent normal deviates:
    six_MT_unnormalised = np.array([np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0)], dtype=float)
    # Normallise sample onto unit 6-sphere:
    six_MT_normalised = six_MT_unnormalised/(np.sum(six_MT_unnormalised**2)**-0.5) # As in Muller (1959)
    # And normallise so that moment tensor magnitude = 1:
    six_MT_normalised = six_MT_normalised/((np.sum(six_MT_normalised**2))**0.5)
    # And set to correct dimensions (so matrix multiplication in forward model works correctly):
    six_MT_normalised = np.reshape(six_MT_normalised, (6, 1))
    return six_MT_normalised


def generate_random_MT_Lune_samp():
    """Function to generate random moment tensor using normal distribution projected onto a 3-sphere for orientation
     and gamma+delta varied normal distribution projected onto a 2-sphere for Lune space method. 
    (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972, Tape and Tape 2012,2013). Returns a random normalised 6 MT."""
    # 1. Get randomly varied Lune parameters (gamma and delta):
    # Define U rotation matrix (See Tape and Tape 2012/2013):
    U_rot_matrix = (1./np.sqrt(6))*np.vstack(([np.sqrt(3.),0.,-np.sqrt(3.)],[-1.,2.,-1.], [np.sqrt(2.),np.sqrt(2.),np.sqrt(2.)]))
    # Get a random sample 3-vector on a 3-unit sphere to use to calculate random delta and gamma Lune angles:
    delta = np.random.uniform(-np.pi/2., np.pi/2.) # theta, but shifted to range between -pi/2 and pi/2 (See Tape and Tape 2012/2013)
    beta = (np.pi/2.) - delta # Beta is simply phase shift of delta (See Tape and Tape 2012/2013)
    gamma = np.random.uniform(-np.pi/6., np.pi/6.) # phi, but shifted to range between -pi/6 and pi/6 (See Tape and Tape 2012/2013)
    # Get eigenvalues from delta,gamma,beta:
    lune_space_uvw_vec = np.vstack(([np.cos(gamma)*np.sin(beta)], [np.sin(gamma)*np.sin(beta)], [np.cos(beta)]))
    lambda_vec = np.dot(np.transpose(U_rot_matrix), lune_space_uvw_vec) # (See Tape and Tape 2012, eq. 20)
    Lune_space_MT = np.vstack(([lambda_vec[0],0.,0.],[0.,lambda_vec[1],0.], [0.,0.,lambda_vec[2]])) # MT with principle axes in u,v,w Lune space
    # 2. Get theta and phi angles to rotate Lune_space_MT by to randomly rotate into x,y,z space:
    # Get a random sample 3-vector on a 3-unit sphere to use to calculate random theta and phi rotation angles:
    a_unnormalised = np.array([np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0)], dtype=float) # Generate 3 indepdendent normal deviates
    a_normalised = a_unnormalised/(np.sum(a_unnormalised**2)**-0.5) # Normallise sample onto unit 3-sphere - As in Muller (1959)
    # And normallise so that vector magnitude = 1:
    a_normalised = a_normalised/((np.sum(a_normalised**2))**0.5)
    x = a_normalised[0]
    y = a_normalised[1]
    z = a_normalised[2]
    theta = np.arctan2(np.sqrt((x**2)+(y**2)),z)
    phi = np.arctan2(y,x)
    # 3. Rotate Lune_space_MT from u,v,w coords to x,y,z coords:
    random_MT = rot_mt_by_theta_phi(Lune_space_MT, theta, phi)
    random_six_MT = get_six_MT_from_full_MT_array(random_MT)
    # And normallise so that moment tensor magnitude = 1:
    random_six_MT_normalised = random_six_MT/((np.sum(random_six_MT**2))**0.5)
    # And set to correct dimensions (so matrix multiplication in forward model works correctly):
    random_six_MT_normalised = np.reshape(random_six_MT_normalised, (6, 1))
    return random_six_MT_normalised
    
    
def generate_random_DC_MT():
    """Function to generate random DC constrained moment tensor using normal distribution projected onto a 3-sphere method (detailed in algorithm B2,B3, Pugh 2015). (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972).
    Returns a random normalised 6 MT. DC component is derived from eigenvalues using CDC decomposition as in Tape and Tape 2013. DC moment tensor is specifed then rotated by random theta and phi to give random DC mt."""
    # Specify DC moment tensor to rotate:
    DC_MT_to_rot = np.vstack(([0.,0.,1.],[0.,0.,0.], [1.,0.,0.])) # DC moment tensor
    # Get a random sample 3-vector on a 3-unit sphere to use to calculate random theta and phi rotation angles:
    a_unnormalised = np.array([np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0)], dtype=float) # Generate 3 indepdendent normal deviates
    a_normalised = a_unnormalised/(np.sum(a_unnormalised**2)**-0.5) # Normallise sample onto unit 3-sphere - As in Muller (1959)
    # And normallise so that vector magnitude = 1:
    a_normalised = a_normalised/((np.sum(a_normalised**2))**0.5)
    x = a_normalised[0]
    y = a_normalised[1]
    z = a_normalised[2]
    theta = np.arctan2(np.sqrt((x**2)+(y**2)),z) #np.arccos(z)
    phi = np.arctan2(y,x) #np.arccos(x/np.sin(theta))
    # And rotate DC moment tensor by random 3D angle:
    random_DC_MT = rot_mt_by_theta_phi(DC_MT_to_rot, theta, phi)
    random_DC_six_MT = get_six_MT_from_full_MT_array(random_DC_MT)
    # And normallise so that moment tensor magnitude = 1:
    random_DC_six_MT_normalised = random_DC_six_MT/((np.sum(random_DC_six_MT**2))**0.5)
    # And set to correct dimensions (so matrix multiplication in forward model works correctly):
    random_DC_six_MT_normalised = np.reshape(random_DC_six_MT_normalised, (6, 1))
    return random_DC_six_MT_normalised
    
    
def generate_random_single_force_vector():
    """Function to generate random single force vector (F_x,F_y,F_z) using normal distribution projected onto a 3-sphere method. (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972).
    Returns a random normalised single force 3-vector."""
    # Generate 3 indepdendent normal deviates:
    single_force_vector_unnormalised = np.array([np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0)], dtype=float)
    # Normallise sample onto unit 3-sphere:
    single_force_vector_normalised = single_force_vector_unnormalised/(np.sum(single_force_vector_unnormalised**2)**-0.5) # As in Muller (1959)
    # And normallise so that moment tensor magnitude = 1:
    single_force_vector_normalised = single_force_vector_normalised/((np.sum(single_force_vector_normalised**2))**0.5)
    # And set to correct dimensions (so matrix multiplication in forward model works correctly):
    single_force_vector_normalised = np.reshape(single_force_vector_normalised, (3, 1))
    return single_force_vector_normalised
    
def generate_random_DC_single_force_coupled_tensor():
    """Function to generate random DC-single-force coupled tensor (M_11,M_22,M_33,M_12,M_13_M_23,F_x,F_y,F_z) using normal distribution projected onto a 3-sphere method. (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972).
    Returns a random normalised DC-single-force tensor of length 9 and the fraction amplitude of DC (0-1.).
    Note on coupling: Coupled so that single force in direction of slip vector of DC solution."""
    # 1. --- Specify DC moment tensor to rotate ---:
    DC_MT_to_rot = np.vstack(([0.,0.,1.],[0.,0.,0.], [1.,0.,0.])) # DC moment tensor
    # 2. --- Specify single force 3-vector in same direction as slip of DC solution ---:
    NED_single_force_vector_to_rotate = np.array([1.,0.,0.], dtype=float)
    # 3. --- Rotate DC moment tensor and single force by same random rotation on sphere:
    # 3.a. Get a random sample 3-vector on a 3-unit sphere to use to calculate random theta and phi rotation angles:
    a_unnormalised = np.array([np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0)], dtype=float) # Generate 3 indepdendent normal deviates
    a_normalised = a_unnormalised/(np.sum(a_unnormalised**2)**-0.5) # Normallise sample onto unit 3-sphere - As in Muller (1959)
    # And normallise so that vector magnitude = 1:
    a_normalised = a_normalised/((np.sum(a_normalised**2))**0.5)
    x = a_normalised[0]
    y = a_normalised[1]
    z = a_normalised[2]
    theta = np.arctan2(np.sqrt((x**2)+(y**2)),z) #np.arccos(z)
    phi = np.arctan2(y,x) #np.arccos(x/np.sin(theta))
    # 3.b. Rotate DC moment tensor by random 3D angle:
    random_DC_MT = rot_mt_by_theta_phi(DC_MT_to_rot, theta, phi)
    random_DC_six_MT = get_six_MT_from_full_MT_array(random_DC_MT)
    random_DC_six_MT_normalised = random_DC_six_MT/((np.sum(random_DC_six_MT**2))**0.5) # And normallise so that moment tensor magnitude = 1
    random_DC_six_MT_normalised = np.reshape(random_DC_six_MT_normalised, (6, 1)) # And set to correct dimensions (so matrix multiplication in forward model works correctly)
    # 3.c. Rotate Single force 3-vector by the same random 3D angle:
    random_coupled_NED_single_force = rot_single_force_by_theta_phi(NED_single_force_vector_to_rotate, theta, phi)
    random_coupled_single_force = np.array([random_coupled_NED_single_force[1], random_coupled_NED_single_force[0], random_coupled_NED_single_force[2]]) # Convert single force from NED coords to END coords (as that is what greens functions are in)
    random_coupled_single_force = np.reshape(random_coupled_single_force, (3, 1))
    # 4. --- Get random fraction amplitude DC and single force components (sum of amplitudes is to 1) ---:
    random_amp_frac = random.random() # random number between 0. and 1.
    random_DC_six_MT_normalised = random_DC_six_MT_normalised*random_amp_frac
    random_coupled_single_force = random_coupled_single_force*(1.-random_amp_frac)
    # 5. --- Finally combine to tensor of length 9 ---:
    random_DC_single_force_coupled_tensor = np.vstack((random_DC_six_MT_normalised, random_coupled_single_force))
    return random_DC_single_force_coupled_tensor, random_amp_frac

def generate_random_DC_single_force_uncoupled_tensor():
    """Function to generate random DC-single-force uncoupled tensor (M_11,M_22,M_33,M_12,M_13_M_23,F_x,F_y,F_z) using normal distribution projected onto a 3-sphere method. (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972).
    Returns a random normalised DC-single-force tensor of length 9 and the fraction amplitude of DC (0-1.).
    Note on coupling: Uncoupled, i.e. single force and DC slip vector can be oriented in any direction with respect to one another."""
    # Generate random DC MT and single force:
    random_DC_MT_normallised = generate_random_DC_MT() # Generate a random DC sample
    random_single_force_normallised = generate_random_single_force_vector()
    # And split the amplitude of DC to single force randomly:
    random_amp_frac = random.random() # random number between 0. and 1.
    random_DC_MT_normallised = random_DC_MT_normallised*random_amp_frac
    random_single_force_normallised = random_single_force_normallised*(1.-random_amp_frac)
    # Finally combine to tensor of length 9:
    random_DC_single_force_uncoupled_tensor = np.vstack((random_DC_MT_normallised, random_single_force_normallised))
    return random_DC_single_force_uncoupled_tensor, random_amp_frac
    
def generate_random_DC_crack_coupled_tensor():
    """Function to generate random DC-crack coupled tensor (M_11,M_22,M_33,M_12,M_13_M_23) using normal distribution projected onto a 3-sphere method. (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972).
    Returns a random normalised MT tensor of length 6 and the fraction amplitude of DC (0-1.).
    Note on coupling: Coupled, i.e. DC slip vector is perpendicular to crack expansion vector."""
    # 1. Generate random DC MT and crack MT:
    # generate DC MT:
    DC_MT_to_rot = np.vstack(([0.,0.,1.],[0.,0.,0.], [1.,0.,0.])) # DC moment tensor
    # generate crack MT (from Tape 2013, eq. 41 and Fig. 6):
    # To randomly generate lune_perim_angle (from Tape 2013 eq. 41 and Fig. 6):
    # --- To randomly generate lune_perim_angle (from Tape 2013 eq. 41 and Fig. 6) ---:
    # Get between small range:
    theta_lune_sphere = np.random.uniform(-1.,1.)*np.pi/2.
    random_num = random.random()
    if random_num <= 0.5:
        phi_lune_sphere = 0. #np.pi/6. #0. #np.pi/6.
    elif random_num > 0.5:
        phi_lune_sphere = np.pi/3 #-1.*np.pi/6. #np.pi/3 #-1.*np.pi/6.
    # calculate lune_perim_angle, allowing for outside tan(-pi/2->pi/2):
    lune_perim_angle = np.arctan(np.sin(phi_lune_sphere)/np.sin(theta_lune_sphere)) # Generates uniform distribution of lune crack angle in Lune plot space #random.random()*2.*np.pi # Random number in uniform distribution betwen 0 and 2 pi
    # And redistribute evenly everywhere on boundary:
    random_num = random.random()
    if random_num>0.25 and random_num<=0.5:
        lune_perim_angle = lune_perim_angle+np.pi # Allow to use full 2 pi space
    if random_num>0.5 and random_num<=0.75:
        lune_perim_angle = lune_perim_angle+np.pi/2 # Allow to use full 2 pi space
    if random_num>0.75 and random_num<=1.0:
        lune_perim_angle = lune_perim_angle+3*np.pi/2 # Allow to use full 2 pi space
    # --- ---
    # random_num = random.random()
    # if random_num <= 0.5:
    #     theta_lune_sphere = random.random()*np.pi/2.
    # elif random_num > 0.5:
    #     theta_lune_sphere = -1.*random.random()*np.pi/2.
    # random_num = random.random()
    # if random_num <= 0.5:
    #     phi_lune_sphere = np.pi/6.
    # elif random_num > 0.5:
    #     phi_lune_sphere = -1.*np.pi/6.
    # lune_perim_angle = np.arctan(np.sin(phi_lune_sphere)/np.sin(theta_lune_sphere)) # Generates uniform distribution of lune crack angle in Lune plot space #random.random()*2.*np.pi # Random number in uniform distribution betwen 0 and 2 pi
    crack_MT_to_rot = ((((4*(np.sin(lune_perim_angle)**2)) + (np.cos(lune_perim_angle)**2))**-0.5)/np.sqrt(3.)) * np.vstack(([np.cos(lune_perim_angle)-(np.sqrt(2)*np.sin(lune_perim_angle)),0.,0.],[0.,np.cos(lune_perim_angle)-(np.sqrt(2)*np.sin(lune_perim_angle)),0.], [0.,0.,np.cos(lune_perim_angle)+(2.*np.sqrt(2)*np.sin(lune_perim_angle))])) # crack moment tensor
    # 2. Combine DC and crack tensors:
    random_amp_frac = random.random() # random number between 0. and 1., for relative amplitude of DC and crack fractions.
    DC_crack_MT_to_rot =random_amp_frac*DC_MT_to_rot + (1.-random_amp_frac)*crack_MT_to_rot
    # 3. Randomly rotate DC-crack MT to random orientation:
    # Get a random sample 3-vector on a 3-unit sphere to use to calculate random theta and phi rotation angles:
    a_unnormalised = np.array([np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0)], dtype=float) # Generate 3 indepdendent normal deviates
    a_normalised = a_unnormalised/(np.sum(a_unnormalised**2)**-0.5) # Normallise sample onto unit 3-sphere - As in Muller (1959)
    # And normallise so that vector magnitude = 1:
    a_normalised = a_normalised/((np.sum(a_normalised**2))**0.5)
    x = a_normalised[0]
    y = a_normalised[1]
    z = a_normalised[2]
    theta = np.arctan2(np.sqrt((x**2)+(y**2)),z) #np.arccos(z)
    phi = np.arctan2(y,x) #np.arccos(x/np.sin(theta))
    DC_crack_MT_rotated = rot_mt_by_theta_phi(DC_crack_MT_to_rot, theta, phi)
    # 4. Normalise and get 6 MT:
    # Get 6 MT:
    DC_crack_six_MT_rotated = get_six_MT_from_full_MT_array(DC_crack_MT_rotated)
    # And normallise so that moment tensor magnitude = 1:
    DC_crack_six_MT_rotated_normalised = DC_crack_six_MT_rotated/((np.sum(DC_crack_six_MT_rotated**2))**0.5)
    # And set to correct dimensions (so matrix multiplication in forward model works correctly):
    DC_crack_six_MT_rotated_normalised = np.reshape(DC_crack_six_MT_rotated_normalised, (6, 1))
    return DC_crack_six_MT_rotated_normalised, random_amp_frac

def generate_random_single_force_crack_uncoupled_tensor():
    """Function to generate random single-force-crack uncoupled tensor (M_11,M_22,M_33,M_12,M_13_M_23F_x,F_y,F_z) using normal distribution projected onto a 3-sphere method. (Based on Pugh 2015,Appendix B and Muller, 1959; Marsaglia, 1972).
    Returns a random normalised MT tensor of length 6 and the fraction amplitude of single-force (0-1.).
    Note on coupling: Uncoupled, i.e. single-force vector can be at any angle relative to crack plane."""
    # 1. Generate random single force vector and crack MT:
    # generate random single force vector:
    random_SF_vector = generate_random_single_force_vector()
    # generate crack MT (from Tape 2013, eq. 41 and Fig. 6):
    # To randomly generate lune_perim_angle (from Tape 2013 eq. 41 and Fig. 6):
    # --- To randomly generate lune_perim_angle (from Tape 2013 eq. 41 and Fig. 6) ---:
    # Get between small range:
    theta_lune_sphere = np.random.uniform(-1.,1.)*np.pi/2.
    random_num = random.random()
    if random_num <= 0.5:
        phi_lune_sphere = 0. #np.pi/6. #0. #np.pi/6.
    elif random_num > 0.5:
        phi_lune_sphere = np.pi/3 #-1.*np.pi/6. #np.pi/3 #-1.*np.pi/6.
    # calculate lune_perim_angle, allowing for outside tan(-pi/2->pi/2):
    lune_perim_angle = np.arctan(np.sin(phi_lune_sphere)/np.sin(theta_lune_sphere)) # Generates uniform distribution of lune crack angle in Lune plot space #random.random()*2.*np.pi # Random number in uniform distribution betwen 0 and 2 pi
    # And redistribute evenly everywhere on boundary:
    random_num = random.random()
    if random_num>0.25 and random_num<=0.5:
        lune_perim_angle = lune_perim_angle+np.pi # Allow to use full 2 pi space
    if random_num>0.5 and random_num<=0.75:
        lune_perim_angle = lune_perim_angle+np.pi/2 # Allow to use full 2 pi space
    if random_num>0.75 and random_num<=1.0:
        lune_perim_angle = lune_perim_angle+3*np.pi/2 # Allow to use full 2 pi space
    # --- ---
    # random_num = random.random()
    # if random_num <= 0.5:
    #     theta_lune_sphere = random.random()*np.pi/2.
    # elif random_num > 0.5:
    #     theta_lune_sphere = -1.*random.random()*np.pi/2.
    # random_num = random.random()
    # if random_num <= 0.5:
    #     phi_lune_sphere = np.pi/6.
    # elif random_num > 0.5:
    #     phi_lune_sphere = -1.*np.pi/6.
    # lune_perim_angle = np.arctan(np.sin(phi_lune_sphere)/np.sin(theta_lune_sphere)) # Generates uniform distribution of lune crack angle in Lune plot space #random.random()*2.*np.pi # Random number in uniform distribution betwen 0 and 2 pi
    crack_MT_to_rot = ((((4*(np.sin(lune_perim_angle)**2)) + (np.cos(lune_perim_angle)**2))**-0.5)/np.sqrt(3.)) * np.vstack(([np.cos(lune_perim_angle)-(np.sqrt(2)*np.sin(lune_perim_angle)),0.,0.],[0.,np.cos(lune_perim_angle)-(np.sqrt(2)*np.sin(lune_perim_angle)),0.], [0.,0.,np.cos(lune_perim_angle)+(2.*np.sqrt(2)*np.sin(lune_perim_angle))])) # crack moment tensor
    # 2. Randomly rotate crack MT to random orientation:
    # Get a random sample 3-vector on a 3-unit sphere to use to calculate random theta and phi rotation angles:
    a_unnormalised = np.array([np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0), np.random.normal(loc=0.0, scale=1.0)], dtype=float) # Generate 3 indepdendent normal deviates
    a_normalised = a_unnormalised/(np.sum(a_unnormalised**2)**-0.5) # Normallise sample onto unit 3-sphere - As in Muller (1959)
    # And normallise so that vector magnitude = 1:
    a_normalised = a_normalised/((np.sum(a_normalised**2))**0.5)
    x = a_normalised[0]
    y = a_normalised[1]
    z = a_normalised[2]
    theta = np.arccos(z)
    phi = np.arccos(x/np.sin(theta))
    crack_MT_rotated = rot_mt_by_theta_phi(crack_MT_to_rot, theta, phi)
    # 3. Convert crack MT to 6 MT:
    crack_six_MT_rotated = get_six_MT_from_full_MT_array(crack_MT_rotated)
    # And set to correct dimensions (so matrix multiplication in forward model works correctly):
    crack_six_MT_rotated = np.reshape(crack_six_MT_rotated, (6, 1))
    # 4. Split the amplitude of crack to single force randomly:
    random_amp_frac = random.random() # random number between 0. and 1.
    random_SF_vector = random_SF_vector*random_amp_frac
    crack_six_MT_rotated = crack_six_MT_rotated*(1.-random_amp_frac)
    # 5. Finally combine to tensor of length 9:
    random_crack_single_force_uncoupled_tensor = np.vstack((crack_six_MT_rotated, random_SF_vector))
    return random_crack_single_force_uncoupled_tensor, random_amp_frac

def variance_reduction(data, synth):
    """Function to perform variance reduction of data and synthetic. Based on Eq. 2.1 in Walter 2009 thesis. Originally from Templeton and Dreger 2006.
    (When using all data in flattened array into this function, this is approximately equivilent to L2 norm (e.g. Song2011))."""
    VR = 1. - (np.sum(np.square(data-synth))/np.sum(np.square(data))) # Calculate variance reduction (according to Templeton and Dreger 2006, Walter 2009 thesis) # Needs <0. filter (as below)
    # And account for amplitude difference error (between synth and data) giving negative results:
    # Note: This is artificial! Can avoid by setting normalisation of data before input.
    if VR < 0.:
        VR = 0.
    return VR

def variance_reduction_normallised(data, synth):
    """Function to perform variance reduction of data and synthetic. Based on Eq. 2.1 in Walter 2009 thesis. Originally from Templeton and Dreger 2006.
    (When using all data in flattened array into this function, this is approximately equivilent to L2 norm (e.g. Song2011))."""
    ###VR = 1. - (np.sum(np.square(data-synth))/(len(data)*np.max(np.absolute(np.square(data-synth)))))
    VR = 1. - (np.sum(np.square(data-synth))/np.square(np.max(np.absolute(data))+np.max(np.absolute(synth)))*len(data)) # Calculate variation reduction, altered to give value between 0 and 1 (data and synth must be normallised)
    ###VR = 1. - (np.sum(np.square(data-synth))/np.sum(np.square(data))) # Calculate variance reduction (according to Templeton and Dreger 2006, Walter 2009 thesis) # Needs <0. filter (as below)
    # And account for amplitude difference error (between synth and data) giving negative results:
    # Note: This is artificial! Can avoid by setting normalisation of data before input.
    # if VR < 0.:
    #     VR = 0.
    return VR
    
def cross_corr_comparison(data, synth):
    """Function performing cross-correlation between long waveform data (data) and template.
    Performs normalized cross-correlation in fourier domain (since it is faster).
    Returns normallised correlation coefficients. See notes on cross-correlation search for more details on normalisation theory"""
    synth_normalised = (synth - np.mean(synth))/np.std(synth)/len(synth) # normalise and detrend
    std_data = np.std(data)    
    #correlation of unnormalized data and normalized template:
    f_corr = signal.correlate(data, synth_normalised, mode='valid', method='fft') # Note: Needs scipy version 19 or greater (mode = full or valid, use full if want to allow shift)
    ncc = np.true_divide(f_corr, std_data) #normalization of the cross correlation
    ncc_max = np.max(ncc) # get max cross-correlation coefficient (only if use full mode)
    if ncc_max<0.:
        ncc_max = 0.
    return ncc_max
    
def cross_corr_comparison_shift_allowed(data, synth, max_samples_shift_limit=5):
    """Function performing cross-correlation between long waveform data (data) and template, allowing a shift of +/- specified number of samples.
    Performs normalized cross-correlation in fourier domain (since it is faster).
    Returns normallised correlation coefficients. See notes on cross-correlation search for more details on normalisation theory."""
    # Upsample the data, via interpolation, in order to allow for finer shifts:
    upsamp_factor = 4
    data_high_res = np.interp(np.arange(0.,len(data),1./upsamp_factor), np.arange(len(data)), data)
    synth_high_res = np.interp(np.arange(0.,len(synth),1./upsamp_factor), np.arange(len(synth)), synth)
    # Loop over sample shifts, calculating ncc for each component:
    samps_to_shift = np.arange(-1*max_samples_shift_limit*upsamp_factor, max_samples_shift_limit*upsamp_factor, dtype=int)
    ncc_values = np.zeros(len(samps_to_shift),dtype=float)
    for a in range(len(samps_to_shift)):
        samp_to_shift = samps_to_shift[a]
        data_tmp = np.roll(data_high_res,samp_to_shift)
        synth_tmp = np.roll(synth_high_res,samp_to_shift)
        ncc_values[a] = cross_corr_comparison(data_tmp, synth_tmp)
    # And get max. cross-correlation coefficient:
    ncc_max = np.max(ncc_values)
    return ncc_max
    
def pearson_correlation_comparison(data, synth):
    """Function to compare similarity of data and synth using Pearson correlation coefficient. Returns number between 0. and 1. (as negatives are anti-correlation, set PCC<0 to 0)."""
    mean_data = np.average(data)
    mean_synth = np.average(synth)
    cov_data_synth = np.sum((data-mean_data)*(synth-mean_synth))/len(data)
    PCC = cov_data_synth/(np.std(data)*np.std(synth)) # Pearson correlation coefficient (-1 to 1, where 0 is no correlation, -1 is anti-correlation and 1 is correlation.)
    if PCC<0.:
        PCC = 0.
    return PCC

def gaussian_comparison(data, synth):
    """Function to perform gaussian comparison of data and synthetic. See equations below for how implemented. Noise level taken from -60:-10 samples at end of trace."""
    data_uncert = np.average(np.absolute(data[-60:-10])) # Get approximate noise level from data
    gau_prob = np.exp(-1*np.sum(((data-synth)**2)/(2*(data_uncert**2)))) # Find gaussian based probability (between zero and 1)
    return gau_prob

def compare_synth_to_real_waveforms(real_data_array, synth_waveforms_array, comparison_metric, perform_normallised_waveform_inversion=True, compare_all_waveforms_simultaneously=True):
    """Function to compare synthetic to real waveforms via specified comparison metrix, and can do normallised if specified and can compare all waveforms together, or separately then equal weight average combine."""
    # Note: Do for all stations combined!
    # Note: Undertaken currently on normalised real and synthetic data!
    synth_waveform_curr_sample = synth_waveforms_array

    # Compare waveforms for individual recievers/components together, as 1 1D array (if compare_all_waveforms_simultaneously=True)
    if compare_all_waveforms_simultaneously:
        # Compare normallised if specified:
        if perform_normallised_waveform_inversion:
            # Normalise:
            real_data_array_normalised = real_data_array.copy()
            synth_waveform_curr_sample_normalised = synth_waveform_curr_sample.copy()
            for j in range(len(real_data_array[:,0])):
                real_data_array_normalised[j,:] = real_data_array[j,:]/np.max(np.absolute(real_data_array[j,:]))
                synth_waveform_curr_sample_normalised[j,:] = synth_waveform_curr_sample[j,:]/np.max(np.absolute(synth_waveform_curr_sample[j,:]))
            # And do waveform comparison, via specified method:
            if comparison_metric == "VR":
                # get variance reduction:
                similarity_curr_sample = variance_reduction(real_data_array_normalised.flatten(), synth_waveform_curr_sample_normalised.flatten())  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
            elif comparison_metric == "CC":
                # And get cross-correlation value:
                similarity_curr_sample = cross_corr_comparison(real_data_array_normalised.flatten(), synth_waveform_curr_sample_normalised.flatten())  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
            elif comparison_metric == "PCC":
                # And get Pearson correlation coeficient:
                similarity_curr_sample = pearson_correlation_comparison(real_data_array_normalised.flatten(), synth_waveform_curr_sample_normalised.flatten())  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
            elif comparison_metric == "CC-shift":
                # And get cross-correlation value, with shift allowed:
                similarity_curr_sample = cross_corr_comparison_shift_allowed(real_data_array_normalised.flatten(), synth_waveform_curr_sample_normalised.flatten(), max_samples_shift_limit=5)
            elif comparison_metric == "gau":
                # And get gaussian probability estimate:
                similarity_curr_sample = gaussian_comparison(real_data_array_normalised.flatten(), synth_waveform_curr_sample_normalised.flatten())
        else:
            # Do waveform comparison, via specified method:
            if comparison_metric == "VR":
                # get variance reduction:
                similarity_curr_sample = variance_reduction(real_data_array.flatten(), synth_waveform_curr_sample.flatten())  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
            elif comparison_metric == "CC":
                # And get cross-correlation value:
                similarity_curr_sample = cross_corr_comparison(real_data_array.flatten(), synth_waveform_curr_sample.flatten())  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
            elif comparison_metric == "PCC":
                # And get Pearson correlation coeficient:
                similarity_curr_sample = pearson_correlation_comparison(real_data_array.flatten(), synth_waveform_curr_sample.flatten())  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
            elif comparison_metric == "CC-shift":
                # And get cross-correlation value, with shift allowed:
                similarity_curr_sample = cross_corr_comparison_shift_allowed(real_data_array.flatten(), synth_waveform_curr_sample.flatten(), max_samples_shift_limit=5)
            elif comparison_metric == "gau":
                # And get gaussian probability estimate:
                similarity_curr_sample = gaussian_comparison(real_data_array.flatten(), synth_waveform_curr_sample.flatten())

    # Compare waveforms for individual recievers/components separately then equal weight (if compare_all_waveforms_simultaneously=False):
    else:
        # Compare normallised if specified:
        if perform_normallised_waveform_inversion:
            # Normalise:
            real_data_array_normalised = real_data_array.copy()
            synth_waveform_curr_sample_normalised = synth_waveform_curr_sample.copy()
            for j in range(len(real_data_array[:,0])):
                real_data_array_normalised[j,:] = real_data_array[j,:]/np.max(np.absolute(real_data_array[j,:]))
                synth_waveform_curr_sample_normalised[j,:] = synth_waveform_curr_sample[j,:]/np.max(np.absolute(synth_waveform_curr_sample[j,:]))
            # And do waveform comparison, via specified method:
            similarity_ind_stat_comps = np.zeros(len(real_data_array_normalised[:,0]), dtype=float)
            for k in range(len(similarity_ind_stat_comps)):
                if comparison_metric == "VR":
                    # get variance reduction:
                    similarity_ind_stat_comps[k] = variance_reduction(real_data_array_normalised[k,:], synth_waveform_curr_sample_normalised[k,:])  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
                elif comparison_metric == "CC":
                    # And get cross-correlation value:
                    similarity_ind_stat_comps[k] = cross_corr_comparison(real_data_array_normalised[k,:], synth_waveform_curr_sample_normalised[k,:])  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
                elif comparison_metric == "PCC":
                    # And get Pearson correlation coeficient:
                    similarity_ind_stat_comps[k] = pearson_correlation_comparison(real_data_array_normalised[k,:], synth_waveform_curr_sample_normalised[k,:])  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
                elif comparison_metric == "CC-shift":
                    # And get cross-correlation value, with shift allowed:
                    similarity_ind_stat_comps[k] = cross_corr_comparison_shift_allowed(real_data_array_normalised[k,:], synth_waveform_curr_sample_normalised[k,:], max_samples_shift_limit=5)
                elif comparison_metric == "gau":
                    # And get gaussian probability estimate:
                    similarity_curr_sample = gaussian_comparison(real_data_array_normalised[k,:], synth_waveform_curr_sample_normalised[k,:])
        else:
            # Do waveform comparison, via specified method:
            similarity_ind_stat_comps = np.zeros(len(real_data_array[:,0]), dtype=float)
            for k in range(len(similarity_ind_stat_comps)):
                if comparison_metric == "VR":
                    # get variance reduction:
                    similarity_ind_stat_comps[k] = variance_reduction(real_data_array[k,:], synth_waveform_curr_sample[k,:])  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
                elif comparison_metric == "CC":
                    # And get cross-correlation value:
                    similarity_ind_stat_comps[k] = cross_corr_comparison(real_data_array[k,:], synth_waveform_curr_sample[k,:])  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
                elif comparison_metric == "PCC":
                    # And get Pearson correlation coeficient:
                    similarity_ind_stat_comps[k] = pearson_correlation_comparison(real_data_array[k,:], synth_waveform_curr_sample[k,:])  # P(data|model_i) (???) or at least an approximate measure of it! # Flattened arrays as do for all stations
                elif comparison_metric == "CC-shift":
                    # And get cross-correlation value, with shift allowed:
                    similarity_ind_stat_comps[k] = cross_corr_comparison_shift_allowed(real_data_array[k,:], synth_waveform_curr_sample[k,:], max_samples_shift_limit=5)
                elif comparison_metric == "gau":
                    # And get gaussian probability estimate:
                    similarity_curr_sample = gaussian_comparison(real_data_array[k,:], synth_waveform_curr_sample[k,:])
        # And get equal weighted similarity value:
        similarity_curr_sample = np.average(similarity_ind_stat_comps)
    
    return similarity_curr_sample

def PARALLEL_worker_mc_inv(procnum, num_samples_per_processor, inversion_type, M_amplitude, green_func_array, real_data_array, comparison_metric, perform_normallised_waveform_inversion, compare_all_waveforms_simultaneously, return_dict_MTs, return_dict_similarity_values_all_samples, return_dict_MT_single_force_rel_amps, return_dict_medium_1_medium_2_rel_amp_ratios, invert_for_ratio_of_multiple_media_greens_func_switch, green_func_phase_labels, num_phase_types_for_media_ratios, invert_for_relative_magnitudes_switch=False, rel_exp_mag_range=[1.,1.]):
    """Parallel worker function for perform_monte_carlo_sampled_waveform_inversion function."""
    print "Processing for process:", procnum, "for ", num_samples_per_processor, "samples."
    
    # Define temp data stores for current process:
    tmp_MTs = np.zeros((len(green_func_array[0,:,0]), num_samples_per_processor), dtype=float)
    tmp_similarity_values_all_samples = np.zeros(num_samples_per_processor, dtype=float)
    if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling" or inversion_type == "DC_crack_couple" or inversion_type == "single_force_crack_no_coupling":
        tmp_MT_single_force_rel_amps = np.zeros(num_samples_per_processor, dtype=float)
    else:
        tmp_MT_single_force_rel_amps = []
    if invert_for_ratio_of_multiple_media_greens_func_switch:
        tmp_medium_1_medium_2_rel_amp_ratios = np.zeros(num_samples_per_processor, dtype=float)
    else:
        tmp_medium_1_medium_2_rel_amp_ratios = []
    if invert_for_ratio_of_multiple_media_greens_func_switch:
        if num_phase_types_for_media_ratios>0:
            tmp_frac_medium_2_diff_phases_dict = {} # Dictionary for temp storing of phase fractions of medium 1
            tmp_medium_1_medium_2_rel_amp_ratios_multi_phases = np.zeros((num_samples_per_processor, 3), dtype=float)
        else:
            tmp_medium_1_medium_2_rel_amp_ratios_multi_phases = []
        
    # Sort greens function storage if processing for multiple media:
    if invert_for_ratio_of_multiple_media_greens_func_switch:
        green_func_array_total_both_media = green_func_array.copy()
        
    # 3. Loop over samples, checking how well a given MT sample synthetic wavefrom from the forward model compares to the real data:
    for i in range(num_samples_per_processor):
        # Generate random medium amplitude ratio and associated greens functions (if required):
        if invert_for_ratio_of_multiple_media_greens_func_switch:
            # If want to invert for ratio of meduim 1 to medium 2 separately for different phases:
            if num_phase_types_for_media_ratios>0:
                # Generate different phase fractions:
                tmp_frac_medium_2_diff_phases_dict["P"] = np.random.uniform(0.0, 1.0)
                tmp_frac_medium_2_diff_phases_dict["S"] = np.random.uniform(0.0, 1.0)
                tmp_frac_medium_2_diff_phases_dict["surface"] = np.random.uniform(0.0, 1.0)
                # Generate associated greens functions:
                green_func_array = np.zeros(np.shape(green_func_array_total_both_media[:,:,:,0]), dtype=float)
                # Loop over greens function for each station-phase:
                for j in range(len(green_func_phase_labels)):
                    tmp_frac_medium_2 = tmp_frac_medium_2_diff_phases_dict[green_func_phase_labels[j]] # Get fraction for specific phase, for specific greens functions for specific station-phase
                    green_func_array[j, :, :] = (1. - tmp_frac_medium_2)*green_func_array_total_both_media[j,:,:,0] + tmp_frac_medium_2*green_func_array_total_both_media[j,:,:,1]                
            # Otherwise generate single fraction value and associated greens functions:
            else:
                frac_medium_2 = np.random.uniform(0.0, 1.0)
                green_func_array = (1. - frac_medium_2)*green_func_array[:,:,:,0] + frac_medium_2*green_func_array[:,:,:,1]
            
        # 4. Generate synthetic waveform for current sample:
        # Vary moment amplitude randomly if specified:
        if invert_for_relative_magnitudes_switch:
            M_amplitude_exp_factor = np.random.uniform(low=rel_exp_mag_range[0], high=rel_exp_mag_range[1])
            M_amplitude = 10.**M_amplitude_exp_factor
        # And generate waveform from source mechanism tensor:
        if inversion_type=="full_mt":
            MT_curr_sample = generate_random_MT()*M_amplitude # Generate a random MT sample
        elif inversion_type=="full_mt_Lune_samp":
            MT_curr_sample = generate_random_MT_Lune_samp()*M_amplitude # Generate a random MT sample, sampled uniformly in Lune space
        elif inversion_type=="DC":
            MT_curr_sample = generate_random_DC_MT()*M_amplitude # Generate a random DC sample
        elif inversion_type=="single_force":
            MT_curr_sample = generate_random_single_force_vector()*M_amplitude # Generate a random single force sample
        elif inversion_type == "DC_single_force_couple":
            MT_curr_sample, random_DC_to_single_force_amp_frac = generate_random_DC_single_force_coupled_tensor() # Generate a random DC-single-force coupled sample, with associated relative amplitude of DC to single force
            MT_curr_sample = MT_curr_sample*M_amplitude
        elif inversion_type == "DC_single_force_no_coupling":
            MT_curr_sample, random_DC_to_single_force_amp_frac = generate_random_DC_single_force_uncoupled_tensor()
            MT_curr_sample = MT_curr_sample*M_amplitude
        elif inversion_type == "DC_crack_couple":
            MT_curr_sample, random_DC_to_single_force_amp_frac = generate_random_DC_crack_coupled_tensor()
            MT_curr_sample = MT_curr_sample*M_amplitude
        elif inversion_type == "single_force_crack_no_coupling":
            MT_curr_sample, random_DC_to_single_force_amp_frac = generate_random_single_force_crack_uncoupled_tensor()
            MT_curr_sample = MT_curr_sample*M_amplitude
        synth_waveform_curr_sample = forward_model(green_func_array, MT_curr_sample) # Note: Greens functions must be of similar amplitude units going into here...
    
        # 5. Compare real data to synthetic waveform (using variance reduction or other comparison metric), to assign probability that data matches current model:
        similarity_curr_sample = compare_synth_to_real_waveforms(real_data_array, synth_waveform_curr_sample, comparison_metric, perform_normallised_waveform_inversion, compare_all_waveforms_simultaneously)      
            
        # 6. Append results to data store:
        tmp_MTs[:,i] = MT_curr_sample[:,0]
        tmp_similarity_values_all_samples[i] = similarity_curr_sample
        if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling" or inversion_type == "DC_crack_couple" or inversion_type == "single_force_crack_no_coupling":
            tmp_MT_single_force_rel_amps[i] = random_DC_to_single_force_amp_frac
        if invert_for_ratio_of_multiple_media_greens_func_switch:
            if num_phase_types_for_media_ratios>0:
                tmp_medium_1_medium_2_rel_amp_ratios_multi_phases[i,0] = tmp_frac_medium_2_diff_phases_dict["P"]
                tmp_medium_1_medium_2_rel_amp_ratios_multi_phases[i,1] = tmp_frac_medium_2_diff_phases_dict["S"]
                tmp_medium_1_medium_2_rel_amp_ratios_multi_phases[i,2] = tmp_frac_medium_2_diff_phases_dict["surface"]
            else:
                tmp_medium_1_medium_2_rel_amp_ratios[i] = frac_medium_2
            
        if i % 10000 == 0:
            print "Processor number:", procnum, "- Processed for",i,"samples out of",num_samples_per_processor,"samples"
    
    # 7. And convert misfit measure to likelihood function probability:
    tmp_similarity_values_all_samples = np.exp(-(1.-tmp_similarity_values_all_samples)/2.)
    
    # And return values back to script:
    return_dict_MTs[procnum] = tmp_MTs
    return_dict_similarity_values_all_samples[procnum] = tmp_similarity_values_all_samples
    return_dict_MT_single_force_rel_amps[procnum] = tmp_MT_single_force_rel_amps
    if num_phase_types_for_media_ratios>0:
        return_dict_medium_1_medium_2_rel_amp_ratios[procnum] = tmp_medium_1_medium_2_rel_amp_ratios_multi_phases
    else:
        return_dict_medium_1_medium_2_rel_amp_ratios[procnum] = tmp_medium_1_medium_2_rel_amp_ratios
    print "Finished processing process:", procnum, "for ", num_samples_per_processor, "samples."

def perform_monte_carlo_sampled_waveform_inversion(real_data_array, green_func_array, num_samples=1000, M_amplitude=1.,inversion_type="full_mt",comparison_metric="CC",perform_normallised_waveform_inversion=True, compare_all_waveforms_simultaneously=True, num_processors=1, return_absolute_similarity_values_switch=False, invert_for_ratio_of_multiple_media_greens_func_switch=False, green_func_phase_labels=[], num_phase_types_for_media_ratios=0, invert_for_relative_magnitudes_switch=False, rel_exp_mag_range=[1.,1.]):
    """Function to use random Monte Carlo sampling of the moment tensor to derive a best fit for the moment tensor to the data.
    Notes: Currently does this using M_amplitude (as makes comparison of data realistic) (alternative could be to normalise real and synthetic data).
    Inversion type can be: full_mt, DC or single_force. If it is full_mt or DC, must give 6 greens functions in greeen_func_array. If it is a single force, must use single force greens functions (3).
    Comparison metrics can be: VR (variation reduction), CC (cross-correlation of static signal), or PCC (Pearson correlation coeficient).
    RUNS IN PARALLEL, with specified number of processors."""
    
    # 1. Set up data stores to write inversion results to:
    MTs = np.zeros((len(green_func_array[0,:,0]), num_samples), dtype=float)
    similarity_values_all_samples = np.zeros(num_samples, dtype=float)
    MTp = np.zeros(num_samples, dtype=float)
    if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling" or inversion_type == "DC_crack_couple" or inversion_type == "single_force_crack_no_coupling":
        MT_single_force_rel_amps = np.zeros(num_samples, dtype=float)
    else:
        MT_single_force_rel_amps = []
    if invert_for_ratio_of_multiple_media_greens_func_switch:
        if num_phase_types_for_media_ratios>0:
            medium_1_medium_2_rel_amp_ratios = np.zeros((num_samples, 3), dtype=float) # To store multi phase amplitude ratios per sample
        else:
            medium_1_medium_2_rel_amp_ratios = np.zeros(num_samples, dtype=float) # To store single amplitude ratio per sample
    else:
        medium_1_medium_2_rel_amp_ratios = []
    
    # 2. Assign prior probabilities:
    # Note: Don't need to assign p_data as will find via marginalisation
    p_model = 1./num_samples # P(model) - Assume constant P(model)
    
    # 3. to 6. are parallelised here:
    # Submit to multiple processors:
    # Setup multiprocessing:
    manager = multiprocessing.Manager()
    return_dict_MTs = manager.dict() # for returning data
    return_dict_similarity_values_all_samples = manager.dict() # for returning data
    return_dict_MT_single_force_rel_amps = manager.dict() # for returning data
    return_dict_medium_1_medium_2_rel_amp_ratios = manager.dict() # for returning data
    jobs = []
    num_samples_per_processor = int(num_samples/num_processors)
    # Loop over processes doing smapling:
    for procnum in range(num_processors):
        p = multiprocessing.Process(target=PARALLEL_worker_mc_inv, args=(procnum, num_samples_per_processor, inversion_type, M_amplitude, green_func_array, real_data_array, comparison_metric, perform_normallised_waveform_inversion, compare_all_waveforms_simultaneously, return_dict_MTs, return_dict_similarity_values_all_samples, return_dict_MT_single_force_rel_amps, return_dict_medium_1_medium_2_rel_amp_ratios, invert_for_ratio_of_multiple_media_greens_func_switch, green_func_phase_labels, num_phase_types_for_media_ratios, invert_for_relative_magnitudes_switch, rel_exp_mag_range))
        jobs.append(p) # Append process to list so that can join together later
        p.start() # Start process
    # Join processes back together:
    for proc in jobs:
        proc.join()
    # And get overall data:
    for procnum in range(num_processors):
        MTs[:,int(procnum*num_samples_per_processor):int((procnum+1)*num_samples_per_processor)] = return_dict_MTs[procnum]
        similarity_values_all_samples[int(procnum*num_samples_per_processor):int((procnum+1)*num_samples_per_processor)] = return_dict_similarity_values_all_samples[procnum]
        if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling" or inversion_type == "DC_crack_couple" or inversion_type == "single_force_crack_no_coupling":
            MT_single_force_rel_amps[int(procnum*num_samples_per_processor):int((procnum+1)*num_samples_per_processor)] = return_dict_MT_single_force_rel_amps[procnum]
        if invert_for_ratio_of_multiple_media_greens_func_switch:
            # If have multi phase amplitude ratios rather than just one multi media amplitude ratio per sample:
            if len(np.shape(medium_1_medium_2_rel_amp_ratios)) == 2:
                medium_1_medium_2_rel_amp_ratios[int(procnum*num_samples_per_processor):int((procnum+1)*num_samples_per_processor), :] = return_dict_medium_1_medium_2_rel_amp_ratios[procnum]
            # Else if only have one multi media amplitude ratio per sample:
            else:
                medium_1_medium_2_rel_amp_ratios[int(procnum*num_samples_per_processor):int((procnum+1)*num_samples_per_processor)] = return_dict_medium_1_medium_2_rel_amp_ratios[procnum]
    # From PARALLEL_worker_mc_inv function, have obtained: MTs, similarity_values_all_samples (and MT_single_force_rel_amps if required)
    
    # 7. Get P(model|data):
    p_data = np.sum(p_model*similarity_values_all_samples) # From marginalisation, P(data) = sum(P(model_i).P(data|model_i))
    MTp = similarity_values_all_samples*p_model/p_data

    # 8. Any final inversion specific data processing:    
    if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling" or inversion_type == "DC_crack_couple" or inversion_type == "single_force_crack_no_coupling":
        MTs = np.vstack((MTs, MT_single_force_rel_amps)) # For passing relative amplitude DC as well as MT and single force components
    if invert_for_ratio_of_multiple_media_greens_func_switch:
        # If have multi phase amplitude ratios rather than just one multi media amplitude ratio per sample:
        if len(np.shape(medium_1_medium_2_rel_amp_ratios)) == 2:
            # Stack for P, S and surface phase amplitude ratio values:
            MTs = np.vstack((MTs, medium_1_medium_2_rel_amp_ratios[:,0])) # For passing relative amplitude of medium 1 to medium 2 greens functions
            MTs = np.vstack((MTs, medium_1_medium_2_rel_amp_ratios[:,1])) # For passing relative amplitude of medium 1 to medium 2 greens functions
            MTs = np.vstack((MTs, medium_1_medium_2_rel_amp_ratios[:,2])) # For passing relative amplitude of medium 1 to medium 2 greens functions
        # Else if only have one multi media amplitude ratio per sample:
        else:
            MTs = np.vstack((MTs, medium_1_medium_2_rel_amp_ratios)) # For passing relative amplitude of medium 1 to medium 2 greens functions
    
    # Any final additional data returning processing:
    if return_absolute_similarity_values_switch:
        MTp_absolute = similarity_values_all_samples
    else:
        MTp_absolute = []
        
    return MTs, MTp, MTp_absolute
    
def get_event_uid_and_station_data_MTFIT_FORMAT_from_nonlinloc_hyp_file(nlloc_hyp_filename):
    """Function to get event uid and station data (station name, azimuth, takeoff angle, polarity) from nonlinloc hyp file. This data is required for writing to file for plotting like MTFIT data."""
    # Array shape is (num_stations, 4) in the order: station name, azimuth, takeoff angle, polarity
    
    # Get event UID:
    # Get event origin times:
    # Get event time from NLLoc file for basal icequake:
    os.system("grep 'GEOGRAPHIC' "+nlloc_hyp_filename+" > ./tmp_event_GEO_line.txt")
    GEO_line = np.loadtxt("./tmp_event_GEO_line.txt", dtype=str)
    event_origin_time = UTCDateTime(GEO_line[2]+GEO_line[3]+GEO_line[4]+GEO_line[5]+GEO_line[6]+GEO_line[7])
    # And remove temp files:
    os.system("rm ./tmp_*GEO_line.txt")
    uid = event_origin_time.strftime("%Y%m%d%H%M%S%f")
    
    # And get station arrival times and azimuth+takeoff angles for each phase, for event:
    os.system("awk '/PHASE ID/{f=1;next} /END_PHASE/{f=0} f' "+nlloc_hyp_filename+" > ./tmp_event_PHASE_lines.txt") # Get phase info and write to tmp file
    PHASE_lines = np.loadtxt("./tmp_event_PHASE_lines.txt", dtype=str) # And import phase lines as np str array
    arrival_times_dict = {} # Create empty dictionary to store data (with keys: event_origin_time, station_arrivals {station {station_P_arrival, station_S_arrival}}})
    arrival_times_dict['event_origin_time'] = event_origin_time
    arrival_times_dict['station_arrival_times'] = {}
    arrival_times_dict['azi_takeoff_angles'] = {}
    # Loop over stations:
    for i in range(len(PHASE_lines[:,0])):
        station = PHASE_lines[i, 0]
        station_current_phase_arrival = UTCDateTime(PHASE_lines[i,6]+PHASE_lines[i,7]+PHASE_lines[i,8])
        station_current_azimuth_event_to_sta = float(PHASE_lines[i,22])
        station_current_toa_event_to_sta = float(PHASE_lines[i,24])
        station_current_toa_sta_inclination = 180. - station_current_toa_event_to_sta
        # See if station entry exists, and if does, write arrival to array, otherwise, create entry and write data to file:
        # For station arrival times:
        try:
            arrival_times_dict['station_arrival_times'][station]
        except KeyError:
            # If entry didnt exist, create it and fill:
            if PHASE_lines[i, 4] == "P":
                arrival_times_dict['station_arrival_times'][station] = {}
                arrival_times_dict['station_arrival_times'][station]["P"] = station_current_phase_arrival
            elif PHASE_lines[i, 4] == "S":
                arrival_times_dict['station_arrival_times'][station] = {}
                arrival_times_dict['station_arrival_times'][station]["S"] = station_current_phase_arrival
        # And if entry did exist:
        else:
            if PHASE_lines[i, 4] == "P":
                arrival_times_dict['station_arrival_times'][station]["P"] = station_current_phase_arrival
            elif PHASE_lines[i, 4] == "S":
                arrival_times_dict['station_arrival_times'][station]["S"] = station_current_phase_arrival
        # And for azimuth and takeoff angle:
        try:
            arrival_times_dict['azi_takeoff_angles'][station]
        except KeyError:
            # If entry didnt exist, create it and fill:
            if PHASE_lines[i, 4] == "P":
                arrival_times_dict['azi_takeoff_angles'][station] = {}
                arrival_times_dict['azi_takeoff_angles'][station]["P_azimuth_sta_to_event"] = station_current_azimuth_event_to_sta
                arrival_times_dict['azi_takeoff_angles'][station]["P_toa_sta_inclination"] = station_current_toa_sta_inclination
        # And if entry did exist:
        else:
            if PHASE_lines[i, 4] == "P":
                arrival_times_dict['azi_takeoff_angles'][station]["P_azimuth_sta_to_event"] = station_current_azimuth_event_to_sta
                arrival_times_dict['azi_takeoff_angles'][station]["P_toa_sta_inclination"] = station_current_toa_sta_inclination  
    
    # And clean up:
    os.system("rm ./tmp*PHASE_lines.txt")
    
    # And create stations array:
    stations = []
    for i in range(len(arrival_times_dict['azi_takeoff_angles'])):
        station = arrival_times_dict['azi_takeoff_angles'].keys()[i]
        azi = arrival_times_dict['azi_takeoff_angles'][station]["P_azimuth_sta_to_event"]
        toa = arrival_times_dict['azi_takeoff_angles'][station]["P_toa_sta_inclination"]
        pol = 0 # Assign zero polarity, as not needed for full waveform
        stations.append([np.array([station], dtype=str), np.array([[azi]], dtype=float), np.array([[toa]], dtype=float), np.array([[pol]], dtype=int)])
    #stations = np.array(stations) # HERE!!! (need to find out what type of object stations is!)
    
    return uid, stations

def remove_zero_prob_results(MTp, MTs):
    """Function to remove zero probability results from FW outputs. Inputs are: MTp - array containing probability values for each solution; MTs - array containing MT inversion information."""
    non_zero_indices = np.argwhere(MTp>0.)
    MTs_out = np.take(MTs, non_zero_indices, axis=1)[:,:,0] # Get non-zero MT solutions only
    MTp_out = MTp[non_zero_indices][:,0]
    return MTp_out, MTs_out
    
def save_to_MTFIT_style_file(MTs, MTp, nlloc_hyp_filename, inversion_type, outdir, MTp_absolute=[]):
    """Function to save data to MTFIT style file, containing arrays of uid, MTs (array of 6xn for possible MT solutions), MTp (array of length n storing probabilities of each solution) and stations (station name, azimuth, takeoff angle, polarity (set to zero here)).
    Output is a pickled file containing a dictionary of uid, stations, MTs and MTp."""
    # Get uid and stations data:
    uid, stations = get_event_uid_and_station_data_MTFIT_FORMAT_from_nonlinloc_hyp_file(nlloc_hyp_filename)
    # Write all data to output dict:
    out_dict = {}
    out_dict["MTs"] = MTs
    out_dict["MTp"] = MTp
    out_dict["uid"] = uid
    out_dict["stations"] = stations
    if len(MTp_absolute)>0:
        out_dict["MTp_absolute"] = MTp_absolute
    # And save to file:
    out_fname = outdir+"/"+uid+"_FW_"+inversion_type+".pkl"
    print "Saving FW inversion to file:", out_fname
    pickle.dump(out_dict, open(out_fname, "wb"))
    

def get_synth_forward_model_most_likely_result(MTs, MTp, green_func_array, inversion_type, invert_for_ratio_of_multiple_media_greens_func_switch=False, green_func_phase_labels=[], num_phase_types_for_media_ratios=0):
    """Function to get most likely synth forward model result, based on the inversion outputs.
    Used for obtaining most likely inversion result information."""
    if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling" or inversion_type == "DC_crack_couple" or inversion_type == "single_force_crack_no_coupling":
        if invert_for_ratio_of_multiple_media_greens_func_switch:
            if num_phase_types_for_media_ratios>0:
                # Generate different phase fractions:
                tmp_frac_medium_2_diff_phases_dict={}
                tmp_frac_medium_2_diff_phases_dict["P"] = MTs[-3, np.where(MTp==np.max(MTp))[0][0]]
                tmp_frac_medium_2_diff_phases_dict["S"] = MTs[-2, np.where(MTp==np.max(MTp))[0][0]]
                tmp_frac_medium_2_diff_phases_dict["surface"] = MTs[-1, np.where(MTp==np.max(MTp))[0][0]]
                # Create actual greens functions for this solution:
                green_func_array_for_most_likely_amp_ratio = np.zeros(np.shape(green_func_array[:,:,:,0]), dtype=float)
                for j in range(len(green_func_phase_labels)):
                    tmp_frac_medium_2 = tmp_frac_medium_2_diff_phases_dict[green_func_phase_labels[j]] # Get fraction for specific phase, for specific greens functions for specific station-phase
                    green_func_array_for_most_likely_amp_ratio[j, :, :] = (1. - tmp_frac_medium_2)*green_func_array[j,:,:,0] + tmp_frac_medium_2*green_func_array[j,:,:,1]
                # And get result:
                synth_forward_model_most_likely_result_array = forward_model(green_func_array_for_most_likely_amp_ratio, MTs[:-4, np.where(MTp==np.max(MTp))[0][0]])
            else:
                frac_medium_2 = MTs[-1, np.where(MTp==np.max(MTp))[0][0]]
                green_func_array_for_most_likely_amp_ratio = (1. - frac_medium_2)*green_func_array[:,:,:,0] + frac_medium_2*green_func_array[:,:,:,1]
                synth_forward_model_most_likely_result_array = forward_model(green_func_array_for_most_likely_amp_ratio, MTs[:-2, np.where(MTp==np.max(MTp))[0][0]])
        else:
            synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:-1, np.where(MTp==np.max(MTp))[0][0]])
    else:
        if invert_for_ratio_of_multiple_media_greens_func_switch:
            if num_phase_types_for_media_ratios>0:
                # Generate different phase fractions:
                tmp_frac_medium_2_diff_phases_dict={}
                tmp_frac_medium_2_diff_phases_dict["P"] = MTs[-3, np.where(MTp==np.max(MTp))[0][0]]
                tmp_frac_medium_2_diff_phases_dict["S"] = MTs[-2, np.where(MTp==np.max(MTp))[0][0]]
                tmp_frac_medium_2_diff_phases_dict["surface"] = MTs[-1, np.where(MTp==np.max(MTp))[0][0]]
                # Create actual greens functions for this solution:
                green_func_array_for_most_likely_amp_ratio = np.zeros(np.shape(green_func_array[:,:,:,0]), dtype=float)
                for j in range(len(green_func_phase_labels)):
                    tmp_frac_medium_2 = tmp_frac_medium_2_diff_phases_dict[green_func_phase_labels[j]] # Get fraction for specific phase, for specific greens functions for specific station-phase
                    green_func_array_for_most_likely_amp_ratio[j, :, :] = (1. - tmp_frac_medium_2)*green_func_array[j,:,:,0] + tmp_frac_medium_2*green_func_array[j,:,:,1]
                # And get result:
                synth_forward_model_most_likely_result_array = forward_model(green_func_array_for_most_likely_amp_ratio, MTs[:-3, np.where(MTp==np.max(MTp))[0][0]])
            else:
                frac_medium_2 = MTs[-1, np.where(MTp==np.max(MTp))[0][0]]
                green_func_array_for_most_likely_amp_ratio = (1. - frac_medium_2)*green_func_array[:,:,:,0] + frac_medium_2*green_func_array[:,:,:,1]
                synth_forward_model_most_likely_result_array = forward_model(green_func_array_for_most_likely_amp_ratio, MTs[:-1, np.where(MTp==np.max(MTp))[0][0]])
        else:
            synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:, np.where(MTp==np.max(MTp))[0][0]])
    
    return synth_forward_model_most_likely_result_array

def save_specific_waveforms_to_file(real_data_array, synth_data_array, data_labels, nlloc_hyp_filename, inversion_type, outdir):
    """Function to save specific waveforms to dictionary format file."""
    # Put waveform data in dict format:
    out_wf_dict = {}
    for i in range(len(data_labels)):
        out_wf_dict[data_labels[i]] = {}
        out_wf_dict[data_labels[i]]["real_wf"] = real_data_array[i,:]
        out_wf_dict[data_labels[i]]["synth_wf"] = synth_data_array[i,:]
    # Get uid for filename:
    uid, stations = get_event_uid_and_station_data_MTFIT_FORMAT_from_nonlinloc_hyp_file(nlloc_hyp_filename)
    # And write to file:
    out_fname = outdir+"/"+uid+"_FW_"+inversion_type+".wfs"
    print "Saving FW inversion to file:", out_fname
    pickle.dump(out_wf_dict, open(out_fname, "wb"))
    
def run_multi_medium_inversion(datadir, outdir, real_data_fnames, MT_green_func_fnames, single_force_green_func_fnames, data_labels, inversion_type, perform_normallised_waveform_inversion, compare_all_waveforms_simultaneously, num_samples, comparison_metric, manual_indices_time_shift_MT, manual_indices_time_shift_SF, nlloc_hyp_filename, cut_phase_start_vals=[], cut_phase_length=0, plot_switch=False, num_processors=1, set_pre_time_shift_values_to_zero_switch=True, only_save_non_zero_solns_switch=False, return_absolute_similarity_values_switch=False, invert_for_ratio_of_multiple_media_greens_func_switch=False, green_func_fnames_split_index=0, green_func_phase_labels=[], invert_for_relative_magnitudes_switch=False, rel_exp_mag_range=[1.,1.]):
    """Function to run the inversion script."""
    
    # Load input data (completely, for specific inversion type):
    real_data_array, green_func_array = get_overall_real_and_green_func_data(datadir, real_data_fnames, MT_green_func_fnames, single_force_green_func_fnames, inversion_type, manual_indices_time_shift_MT=manual_indices_time_shift_MT, manual_indices_time_shift_SF=manual_indices_time_shift_SF, cut_phase_start_vals=cut_phase_start_vals, cut_phase_length=cut_phase_length, set_pre_time_shift_values_to_zero_switch=set_pre_time_shift_values_to_zero_switch, invert_for_ratio_of_multiple_media_greens_func_switch=invert_for_ratio_of_multiple_media_greens_func_switch, green_func_fnames_split_index=green_func_fnames_split_index)
    
    # Do initial check/s:
    if len(green_func_phase_labels)>0:
        if not len(green_func_array[:,0,0,0]) == len(green_func_phase_labels):
            print "Error: Greens functions filename array (for medium 1), does not match length of green_func_phase_labels array."
            sys.exit()
    
    # Get number of different phases, if specified:
    num_phase_types_for_media_ratios = 0
    if green_func_phase_labels.count("P")>0:
        num_phase_types_for_media_ratios += 1
    if green_func_phase_labels.count("S")>0:
        num_phase_types_for_media_ratios += 1
    if green_func_phase_labels.count("surface")>0:
        num_phase_types_for_media_ratios += 1
    
    # Define a fraction of the second medium to use for the simple least squares inversion:
    frac_medium_2 = 0.5
    green_func_array_for_lsq_inv = (1. - frac_medium_2)*green_func_array[:,:,:,0] + frac_medium_2*green_func_array[:,:,:,1]
    
    # Perform the inversion:
    M = perform_inversion(real_data_array, green_func_array_for_lsq_inv)
    M_amplitude = ((np.sum(M**2))**0.5)

    # And get forward model synthetic waveform result:
    synth_forward_model_result_array = forward_model(green_func_array_for_lsq_inv, M)

    # And plot the results:
    if plot_switch:
        plot_specific_forward_model_result(real_data_array, synth_forward_model_result_array, data_labels, plot_title="Initial theoretical inversion solution", perform_normallised_waveform_inversion=perform_normallised_waveform_inversion)

    # And save least squares output:
    # Set output arrays to equal least squares output:    
    MTs = M
    similarity_curr_sample = compare_synth_to_real_waveforms(real_data_array, synth_forward_model_result_array, comparison_metric, perform_normallised_waveform_inversion, compare_all_waveforms_simultaneously)      
    MTp = np.array([similarity_curr_sample])
    # And save data to MTFIT style file:
    outdir_least_squares = outdir+"/least_squares_result"
    os.system("mkdir -p "+outdir_least_squares)
    save_to_MTFIT_style_file(MTs, MTp, nlloc_hyp_filename, inversion_type, outdir_least_squares) # Saves pickled dictionary containing data from inversion
    # And save most likely solution and real data waveforms to file:
    if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling" or inversion_type == "DC_crack_couple" or inversion_type == "single_force_crack_no_coupling":
        synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:-1, np.where(MTp==np.max(MTp))[0][0]])
    else:
        synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:, np.where(MTp==np.max(MTp))[0][0]])
    save_specific_waveforms_to_file(real_data_array, synth_forward_model_most_likely_result_array, data_labels, nlloc_hyp_filename, inversion_type, outdir_least_squares)


    # And do Monte Carlo random sampling to obtain PDF of moment tensor:
    MTs, MTp, MTp_absolute = perform_monte_carlo_sampled_waveform_inversion(real_data_array, green_func_array, num_samples, M_amplitude=M_amplitude,inversion_type=inversion_type, comparison_metric=comparison_metric, perform_normallised_waveform_inversion=perform_normallised_waveform_inversion, compare_all_waveforms_simultaneously=compare_all_waveforms_simultaneously, num_processors=num_processors, return_absolute_similarity_values_switch=return_absolute_similarity_values_switch, invert_for_ratio_of_multiple_media_greens_func_switch=invert_for_ratio_of_multiple_media_greens_func_switch, green_func_phase_labels=green_func_phase_labels, num_phase_types_for_media_ratios=num_phase_types_for_media_ratios, invert_for_relative_magnitudes_switch=invert_for_relative_magnitudes_switch, rel_exp_mag_range=rel_exp_mag_range)

    # Check that probability of output is non-zero:
    if math.isnan(MTp[0]):
        print "Error: Sum of probabilities is equal to zero - therefore no adiquate solution could be found and inversion is terminating."
        sys.exit()
    
    # Remove zero probability values if specified:
    if only_save_non_zero_solns_switch:
        MTp, MTs = remove_zero_prob_results(MTp, MTs)

    # And plot most likely solution:
    if plot_switch:
        if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling" or inversion_type == "DC_crack_couple" or inversion_type == "single_force_crack_no_coupling":
            if invert_for_ratio_of_multiple_media_greens_func_switch:
                if num_phase_types_for_media_ratios>0:
                    # Generate different phase fractions:
                    tmp_frac_medium_1_diff_phases_dict={}
                    tmp_frac_medium_1_diff_phases_dict["P"] = MTs[-3, np.where(MTp==np.max(MTp))[0][0]]
                    tmp_frac_medium_1_diff_phases_dict["S"] = MTs[-2, np.where(MTp==np.max(MTp))[0][0]]
                    tmp_frac_medium_1_diff_phases_dict["surface"] = MTs[-1, np.where(MTp==np.max(MTp))[0][0]]
                    # Create actual greens functions for this solution:
                    green_func_array_for_most_likely_amp_ratio = np.zeros(np.shape(green_func_array[:,:,:,0]), dtype=float)
                    for j in range(len(green_func_phase_labels)):
                        tmp_frac_medium_1 = tmp_frac_medium_1_diff_phases_dict[green_func_phase_labels[j]] # Get fraction for specific phase, for specific greens functions for specific station-phase
                        green_func_array_for_most_likely_amp_ratio[j, :, :] = (1. - tmp_frac_medium_1)*green_func_array[j,:,:,0] + tmp_frac_medium_1*green_func_array[j,:,:,1]
                    # And get result:
                    synth_forward_model_most_likely_result_array = forward_model(green_func_array_for_most_likely_amp_ratio, MTs[:-4, np.where(MTp==np.max(MTp))[0][0]])
                else:
                    frac_medium_1 = MTs[-1, np.where(MTp==np.max(MTp))[0][0]]
                    green_func_array_for_most_likely_amp_ratio = (1. - frac_medium_1)*green_func_array[:,:,:,0] + frac_medium_1*green_func_array[:,:,:,1]
                    synth_forward_model_most_likely_result_array = forward_model(green_func_array_for_most_likely_amp_ratio, MTs[:-2, np.where(MTp==np.max(MTp))[0][0]])
            else:
                synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:-1, np.where(MTp==np.max(MTp))[0][0]])
        else:
            if invert_for_ratio_of_multiple_media_greens_func_switch:
                if num_phase_types_for_media_ratios>0:
                    # Generate different phase fractions:
                    tmp_frac_medium_1_diff_phases_dict={}
                    tmp_frac_medium_1_diff_phases_dict["P"] = MTs[-3, np.where(MTp==np.max(MTp))[0][0]]
                    tmp_frac_medium_1_diff_phases_dict["S"] = MTs[-2, np.where(MTp==np.max(MTp))[0][0]]
                    tmp_frac_medium_1_diff_phases_dict["surface"] = MTs[-1, np.where(MTp==np.max(MTp))[0][0]]
                    # Create actual greens functions for this solution:
                    green_func_array_for_most_likely_amp_ratio = np.zeros(np.shape(green_func_array[:,:,:,0]), dtype=float)
                    for j in range(len(green_func_phase_labels)):
                        tmp_frac_medium_1 = tmp_frac_medium_1_diff_phases_dict[green_func_phase_labels[j]] # Get fraction for specific phase, for specific greens functions for specific station-phase
                        green_func_array_for_most_likely_amp_ratio[j, :, :] = (1. - tmp_frac_medium_1)*green_func_array[j,:,:,0] + tmp_frac_medium_1*green_func_array[j,:,:,1]
                    # And get result:
                    synth_forward_model_most_likely_result_array = forward_model(green_func_array_for_most_likely_amp_ratio, MTs[:-3, np.where(MTp==np.max(MTp))[0][0]])
                else:
                    frac_medium_1 = MTs[-1, np.where(MTp==np.max(MTp))[0][0]]
                    green_func_array_for_most_likely_amp_ratio = (1. - frac_medium_1)*green_func_array[:,:,:,0] + frac_medium_1*green_func_array[:,:,:,1]
                    synth_forward_model_most_likely_result_array = forward_model(green_func_array_for_most_likely_amp_ratio, MTs[:-1, np.where(MTp==np.max(MTp))[0][0]])
            else:
                synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:, np.where(MTp==np.max(MTp))[0][0]])
        plot_specific_forward_model_result(real_data_array, synth_forward_model_most_likely_result_array, data_labels, plot_title="Most likely Monte Carlo sampled solution", perform_normallised_waveform_inversion=perform_normallised_waveform_inversion)
        print "Most likely solution:", MTs[:,np.where(MTp==np.max(MTp))[0][0]]

    # And save data to MTFIT style file:
    save_to_MTFIT_style_file(MTs, MTp, nlloc_hyp_filename, inversion_type, outdir, MTp_absolute=MTp_absolute) # Saves pickled dictionary containing data from inversion
    # And save most likely solution and real data waveforms to file:
    
    
    
    synth_forward_model_most_likely_result_array = get_synth_forward_model_most_likely_result(MTs, MTp, green_func_array, inversion_type, invert_for_ratio_of_multiple_media_greens_func_switch=invert_for_ratio_of_multiple_media_greens_func_switch, green_func_phase_labels=green_func_phase_labels, num_phase_types_for_media_ratios=num_phase_types_for_media_ratios)
    save_specific_waveforms_to_file(real_data_array, synth_forward_model_most_likely_result_array, data_labels, nlloc_hyp_filename, inversion_type, outdir)

    print "Finished"
        
    
def run(datadir, outdir, real_data_fnames, MT_green_func_fnames, single_force_green_func_fnames, data_labels, inversion_type, perform_normallised_waveform_inversion, compare_all_waveforms_simultaneously, num_samples, comparison_metric, manual_indices_time_shift_MT, manual_indices_time_shift_SF, nlloc_hyp_filename, cut_phase_start_vals=[], cut_phase_length=0, plot_switch=False, num_processors=1, set_pre_time_shift_values_to_zero_switch=True, only_save_non_zero_solns_switch=False, return_absolute_similarity_values_switch=False, invert_for_ratio_of_multiple_media_greens_func_switch=False, green_func_fnames_split_index=0, green_func_phase_labels=[], invert_for_relative_magnitudes_switch=False, rel_exp_mag_range=[1.0,1.0]):
    """Function to run the inversion script.
    ------------------ Inputs ------------------
    Required arguments:
    datadir - Path to directory containing real and Greens function waveforms (type: str)
    outdir - Path to output directory for inversion outputs (type: str)
    real_data_fnames - List of strings of real data filenames within <datadir> (type: list)
    MT_green_func_fnames - List of strings of Greens function data filenames, associated with the six fundamental moment tensor components, within <datadir> (type: list)
    single_force_green_func_fnames - List of strings of Greens function data filenames, associated with the three single force components, within <datadir> (type: list)
    data_labels - List of string labels associated with the list <real_data_fnames>. These are used for plotting functions and are therefore passed to the output (type: list)
    inversion_type - The inversion type to undertake. Options are: full_mt, full_mt_Lune_samp, DC, single_force, DC_single_force_couple, DC_single_force_no_coupling, DC_crack_couple, or single_force_crack_no_coupling. (if single force, greens functions must be 3 components rather than 6) (type: str)
    num_samples - Number of samples to perform Monte Carlo over (type: int)
    comparison_metric - Method for comparing data and finding the misfit. Options are VR (variation reduction), CC (cross-correlation of static signal), CC-shift (cross-correlation of signal with shift allowed), or PCC (Pearson correlation coeficient), gau (Gaussian based method for estimating the true statistical probability) (Note: CC is the most stable, as range is naturally from 0-1, rather than -1 to 1) (type: str)
    manual_indices_time_shift_MT -  Values by which to shift greens functions (must be integers here) (type: list of integers)
    manual_indices_time_shift_SF - Values by which to shift greens functions (must be integers here) (type: list of integers)
    nlloc_hyp_filename - Nonlinloc .grid0.loc.hyp filename, for saving event data to file in MTFIT format (for plotting, further analysis etc) (type: str)

    Optional arguments:
    perform_normallised_waveform_inversion - Boolean switch. If True, performs normallised waveform inversion, whereby each synthetic and real waveform is normallised before comparision. Effectively removes overall amplitude from inversion if True. Should use True if using VR comparison method. (type: Bool)
    compare_all_waveforms_simultaneously - Bolean. If True, compares all waveform observations together to give one similarity value. If False, compares waveforms from individual recievers separately then combines using equally weighted average. Default = True. (type: Bool)
    cut_phase_start_vals - Indices by which to begin cut phase (must be integers, and specified for every trace, if specified). (Default is not to cut the P and S phases) (must specify cut_phase_end_vals too) (type: list of integers)
    cut_phase_length - Length to cut phases by. Integer. Currently this number must be constant, as code cannot deal with different data lengths. (type: int)
    plot_switch - If True, will plot some outputs to screen (Default is False) (type: bool)
    num_processors - Number of processors to run for (default is 1) (type: int)
    set_pre_time_shift_values_to_zero_switch - If true, sets values before time shift to zero, to account for rolling the data on itself (default is True) (type: bool)
    only_save_non_zero_solns_switch - If True, will only save results with a non-zero probability. (type: bool)
    return_absolute_similarity_values_switch - If True, will also save absolute similarity values, as well as the normallised values. (will be saved to the output dict as MTp_absolute). (type: bool)
    invert_for_ratio_of_multiple_media_greens_func_switch -If True, allows for invertsing for the ratio of two sets of greens functions, for different media, relative to one another (with the split in greens function fnames sepcified by green_func_fnames_split_index). (type: bool)
    green_func_fnames_split_index - Index of first greens function fname for second medium (if performing multi-media inversion) (type: int)
    green_func_phase_labels - List of same length as data_labels, to specify the phase associated with each greens function. Can be "P", "S", or "surface". If this parameter is specified then will use multiple greens function ratios.
    invert_for_relative_magnitudes_switch - If True, inverts for relative magnitude. Notes: Must have perform_normallised_waveform_inversion=False; Will then vary magnitude by 10^lower range to upper range, specified by rel_exp_mag_range. (Default is False)
    rel_exp_mag_range - Values of lower and upper exponent for 10^x , e.g. [-3.0,3.0] would be relative magnitude range from 10^-3 to 10^3 (Default is [0.0,0.0]) (used if invert_for_relative_magnitudes_switch is set to True) (type: list of 2 x int)
    
    ------------------ Outputs ------------------
    Returns:
    inversion result in form of .pkl (python pickled binary file) to <outdir> (path specified above)
    waveforms associated with the most likely inversion result in the form of a .wfs file (for plotting using plotting scripts)
    """

    # Run specific multi medium inversion, if specified:
    if invert_for_ratio_of_multiple_media_greens_func_switch:
        # For muliple media greens functions:
        run_multi_medium_inversion(datadir, outdir, real_data_fnames, MT_green_func_fnames, single_force_green_func_fnames, data_labels, inversion_type, perform_normallised_waveform_inversion, compare_all_waveforms_simultaneously, num_samples, comparison_metric, manual_indices_time_shift_MT, manual_indices_time_shift_SF, nlloc_hyp_filename, cut_phase_start_vals=cut_phase_start_vals, cut_phase_length=cut_phase_length, plot_switch=plot_switch, num_processors=num_processors, set_pre_time_shift_values_to_zero_switch=set_pre_time_shift_values_to_zero_switch, only_save_non_zero_solns_switch=only_save_non_zero_solns_switch, return_absolute_similarity_values_switch=return_absolute_similarity_values_switch, invert_for_ratio_of_multiple_media_greens_func_switch=invert_for_ratio_of_multiple_media_greens_func_switch, green_func_fnames_split_index=green_func_fnames_split_index, green_func_phase_labels=green_func_phase_labels)
    else:
        # Run for normal, single set of greens functions:
        
        # Load input data (completely, for specific inversion type):
        real_data_array, green_func_array = get_overall_real_and_green_func_data(datadir, real_data_fnames, MT_green_func_fnames, single_force_green_func_fnames, inversion_type, manual_indices_time_shift_MT=manual_indices_time_shift_MT, manual_indices_time_shift_SF=manual_indices_time_shift_SF, cut_phase_start_vals=cut_phase_start_vals, cut_phase_length=cut_phase_length, set_pre_time_shift_values_to_zero_switch=set_pre_time_shift_values_to_zero_switch)
        
        # Perform the inversion:
        M = perform_inversion(real_data_array, green_func_array)
        M_amplitude = ((np.sum(M**2))**0.5)

        # And get forward model synthetic waveform result:
        synth_forward_model_result_array = forward_model(green_func_array, M)
    
        # And plot the results:
        if plot_switch:
            plot_specific_forward_model_result(real_data_array, synth_forward_model_result_array, data_labels, plot_title="Initial theoretical inversion solution", perform_normallised_waveform_inversion=perform_normallised_waveform_inversion)
    
        # And save least squares output:
        # Set output arrays to equal least squares output:    
        MTs = M
        similarity_curr_sample = compare_synth_to_real_waveforms(real_data_array, synth_forward_model_result_array, comparison_metric, perform_normallised_waveform_inversion, compare_all_waveforms_simultaneously)      
        MTp = np.array([similarity_curr_sample])
        # And save data to MTFIT style file:
        outdir_least_squares = outdir+"/least_squares_result"
        os.system("mkdir -p "+outdir_least_squares)
        save_to_MTFIT_style_file(MTs, MTp, nlloc_hyp_filename, inversion_type, outdir_least_squares) # Saves pickled dictionary containing data from inversion
        # And save most likely solution and real data waveforms to file:
        if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling" or inversion_type == "DC_crack_couple" or inversion_type == "single_force_crack_no_coupling":
            synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:-1, np.where(MTp==np.max(MTp))[0][0]])
        else:
            synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:, np.where(MTp==np.max(MTp))[0][0]])
        save_specific_waveforms_to_file(real_data_array, synth_forward_model_most_likely_result_array, data_labels, nlloc_hyp_filename, inversion_type, outdir_least_squares)
    
    
        # And do Monte Carlo random sampling to obtain PDF of moment tensor:
        MTs, MTp, MTp_absolute = perform_monte_carlo_sampled_waveform_inversion(real_data_array, green_func_array, num_samples, M_amplitude=M_amplitude,inversion_type=inversion_type, comparison_metric=comparison_metric, perform_normallised_waveform_inversion=perform_normallised_waveform_inversion, compare_all_waveforms_simultaneously=compare_all_waveforms_simultaneously, num_processors=num_processors, return_absolute_similarity_values_switch=return_absolute_similarity_values_switch, invert_for_relative_magnitudes_switch=invert_for_relative_magnitudes_switch, rel_exp_mag_range=rel_exp_mag_range)
    
        # Check that probability of output is non-zero:
        if math.isnan(MTp[0]):
            print "Error: Sum of probabilities is equal to zero - therefore no adiquate solution could be found and inversion is terminating."
            sys.exit()
        
        # Remove zero probability values if specified:
        if only_save_non_zero_solns_switch:
            MTp, MTs = remove_zero_prob_results(MTp, MTs)
    
        # And plot most likely solution:
        if plot_switch:
            if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling" or inversion_type == "DC_crack_couple" or inversion_type == "single_force_crack_no_coupling":
                synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:-1, np.where(MTp==np.max(MTp))[0][0]])
            else:
                synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:, np.where(MTp==np.max(MTp))[0][0]])
            plot_specific_forward_model_result(real_data_array, synth_forward_model_most_likely_result_array, data_labels, plot_title="Most likely Monte Carlo sampled solution", perform_normallised_waveform_inversion=perform_normallised_waveform_inversion)
            print "Most likely solution:", MTs[:,np.where(MTp==np.max(MTp))[0][0]]
    
        # And save data to MTFIT style file:
        save_to_MTFIT_style_file(MTs, MTp, nlloc_hyp_filename, inversion_type, outdir, MTp_absolute=MTp_absolute) # Saves pickled dictionary containing data from inversion
        # And save most likely solution and real data waveforms to file:
        if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling" or inversion_type == "DC_crack_couple" or inversion_type == "single_force_crack_no_coupling":
            synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:-1, np.where(MTp==np.max(MTp))[0][0]])
        else:
            synth_forward_model_most_likely_result_array = forward_model(green_func_array, MTs[:, np.where(MTp==np.max(MTp))[0][0]])
        save_specific_waveforms_to_file(real_data_array, synth_forward_model_most_likely_result_array, data_labels, nlloc_hyp_filename, inversion_type, outdir)

        print "Finished"

# ------------------- End of defining various functions used in script -------------------


# ------------------- Main script for running -------------------
if __name__ == "__main__":
    # Run functions via main run function:
    run(datadir, outdir, real_data_fnames, MT_green_func_fnames, single_force_green_func_fnames, data_labels, inversion_type, perform_normallised_waveform_inversion, compare_all_waveforms_simultaneously, num_samples, comparison_metric, manual_indices_time_shift_MT, manual_indices_time_shift_SF, nlloc_hyp_filename, cut_phase_start_vals, cut_phase_length, plot_switch, num_processors, set_pre_time_shift_values_to_zero_switch, only_save_non_zero_solns_switch, return_absolute_similarity_values_switch, invert_for_relative_magnitudes_switch, rel_exp_mag_range)

