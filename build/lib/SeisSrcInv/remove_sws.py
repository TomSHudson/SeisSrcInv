#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Functions to correct a trace for shear wave splitting back to original signal, pre anisotropic effects.


# Input variables:
# tr_N - The north trace (format: obspy trace)
# tr_E - The east trace (format: obspy trace)
# dt - lag time of fast to slow arrivals
# phi - Fast direction angle to correct by, in degrees

# Output variables:

# Created by Tom Hudson, 14th August 2018

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
import matplotlib.pyplot as plt
import obspy


# -------------------------------Define Main Functions In Script-------------------------------
    
def twoD_rotation(data_N, data_E, rot_angle_deg):
    """To rotate N and E traces horizontally by an angle anticlockwise, rot_angle_deg (in degrees). Returns a 2,n array of N and E tr values.
    (see https://service.iris.edu/irisws/rotation/docs/1/help/ for theory)"""
    rot_angle_rad = np.pi * rot_angle_deg/180.
    # Define rotation matrix:
    rot_matrix = np.zeros((2,2), dtype=float)
    rot_matrix[0,0] = np.cos(rot_angle_rad)
    rot_matrix[0,1] = np.sin(rot_angle_rad)
    rot_matrix[1,0] = -1.*np.sin(rot_angle_rad)
    rot_matrix[1,1] = np.cos(rot_angle_rad)
    # Perform rotation:
    data_vect = np.vstack([data_E,data_N]) # Note that E and N are way round for x,y
    data_vect_rot = np.matmul(rot_matrix, data_vect)
    # Return data (north first):
    data_N_rot = data_vect_rot[1,:]
    data_E_rot = data_vect_rot[0,:]
    return data_N_rot, data_E_rot

def remove_shear_wave_spliting(tr_N, tr_E, dt, phi):
    """Overall main function to remove shear wave splitting.
    Input variables:
    tr_N - The north trace (format: obspy trace)
    tr_E - The east trace (format: obspy trace)
    dt - lag time of fast to slow arrivals
    phi - Fast direction angle to correct by, in degrees
    Outputs:
    tr_N_corr - North trace with shear wave splitting effect removed
    tr_E_corr - North trace with shear wave splitting effect removed
    """
    
    # Get data from traces:
    data_N = tr_N.data
    data_E = tr_E.data
    
    # 1. Get angle to rotate by:
    rot_angle = -phi # Negative since want to rotate clockwise, but 2D rotation is counterclockwise. Use N trace as first data array in rotation.
    
    # 2. Rotate traces (to S1-S2 orientation):
    data_1_rot, data_2_rot = twoD_rotation(data_N, data_E, rot_angle)
        
    # 3. Time shift the slow trace:
    num_samples_to_shift = int(dt*tr_N.stats.sampling_rate)
    data_2_rot = np.roll(data_2_rot, -1*num_samples_to_shift)
    
    # 4. Rotate data back to original orientation:
    data_N_corr, data_E_corr = twoD_rotation(data_1_rot, data_2_rot, -1.*rot_angle)
    
    # And return data:
    tr_N_corr = tr_N.copy()
    tr_E_corr = tr_E.copy()
    tr_N_corr.data = data_N_corr
    tr_E_corr.data = data_E_corr
    return tr_N_corr, tr_E_corr


def remove_shear_wave_spliting_for_st(st_zne_in, stations, dts, phis, plot_switch=False):
    """Function to remove shear wave splitting for all specified stations in an entire stream.
    Works by linearising S wave arrivals on horizontals, using derived splitting parameters for
    each station (dt and phi).
    Arguments:
    Required:
    st_zne_in - Obspy stream of data to linearise S wave arrival for. (obspy stream)
    stations - Station names to proess for (list of strs)
    dts - The fast to slow S delay times corresponding to each station in stations, in seconds (list of floats)
    phis - The fast angles, relative to north, corresponding to each station in stations, in degrees (list of float)
    Optional:
    plot_switch - If true, plots data before and after correction (bool)

    Returns:
    st_zne_sws_removed - Obspy stream with shear-wave-spliting removed. (obspy stream)"""
    # Setup output stream:
    st_zne_sws_removed = st_zne_in.copy()
    # Loop over stations:
    for i in range(len(stations)):
        # Get N and E comps and remove SWS:
        tr_N = st_zne_in.select(station=stations[i], channel='??N')[0]
        tr_E = st_zne_in.select(station=stations[i], channel='??E')[0]
        tr_N_corr, tr_E_corr = remove_shear_wave_spliting(tr_N, tr_E, dts[i], phis[i])
        st_zne_sws_removed.select(station=stations[i], channel='??N')[0].data = tr_N_corr.data
        st_zne_sws_removed.select(station=stations[i], channel='??E')[0].data = tr_E_corr.data
        # And plot, if specified:
        if plot_switch:
            plt.figure()
            plt.plot(st_zne_in.select(station=stations[i], channel='??E')[0].data, 
                        st_zne_in.select(station=stations[i], channel='??N')[0].data, label="before")
            plt.plot(st_zne_sws_removed.select(station=stations[i], channel='??E')[0].data, 
                        st_zne_sws_removed.select(station=stations[i], channel='??N')[0].data, label="after sws removed")
            plt.xlabel('E')
            plt.ylabel('N')
            plt.legend()
            plt.show()
    
    return st_zne_sws_removed

    
