#!/usr/bin/python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to get data from the greens functions and real data and save to inversion input data format.

# Input variables:
# station_labels - Array of station labels to process
# dist_labels - Array containing distance labels associated with stations (for getting fk data)
# green_func_dir - directory containing output greens functions created by fk
# outdir - Output directory where outputs data for use in inversion
# high_pass_freq - High pass frequency to apply to all data
# low_pass_freq - Low pass frequency to apply to all data
# comp_list_MT - Array of components for MT greens functions
# comp_list_single_force - Array of components for single force greens functions
# Real data dir containing nonlinloc hyp file and trimmed mseed files containing event

# Output variables:
# Outputs all data to: outdir
# Data output are:
# 1.  Greens functions for MT inversion
# 2.  Greens functions for single force inversion
# 3.  Real data for inversion

# Created by Tom Hudson, 14th March 2018

# Notes:

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
# Import modules:
import numpy as np
import obspy
import glob
import matplotlib.pyplot as plt
import os
from obspy import UTCDateTime as UTCDateTime
from scipy.signal import decimate
import NonLinLocPy.read_nonlinloc as read_nonlinloc

#----------------------------------- Define parameters -----------------------------------
# # Define parameters:
# src_depth_labels = ["src_depth_2.017"]
# station_labels = ["ST01", "ST02", "ST03", "ST04", "ST05", "ST08"] ##["SKR01", "SKR02", "SKR03", "SKR04", "SKR05", "SKR06", "SKR07", "SKG08", "SKG13"] ##["ST01", "ST02", "ST03", "ST04", "ST05", "ST08"] ##["ST01", "ST02", "ST03", "ST04", "ST05", "ST06", "ST07", "ST08", "ST09", "ST10"] #["RA51", "RA52", "RA53", "RA54", "RA55", "RA56", "RA57"] ##["RA51", "RA52", "RA53", "RA54", "RA55", "RA56", "RA57"] ##["SKR01", "SKR02", "SKR03", "SKR04", "SKR05", "SKR06", "SKR07", "SKG08", "SKG13"] ##["ST01", "ST02", "ST03", "ST04", "ST05", "ST06", "ST07", "ST08", "ST09", "ST10"] #["SKG08", "SKR02", "SKR03", "SKG13", "SKR01", "SKR07", "SKR04", "SKG09", "SKR06", "SKR05", "SKG12", "SKG11"]
# dist_labels = ["0.8817", "0.7000", "1.4975", "0.5588", "1.6484", "3.9617"] ##["0.2121", "0.3048", "0.5187", "0.6238", "0.5816", "0.4270", "0.3923", "0.9799", "0.6943"] ##["0.8817", "0.7000", "1.4975", "0.5588", "1.6484", "3.9617"] ##["0.8817", "0.7000", "1.4975", "0.5588", "1.6484", "4.9324", "3.9179", "3.9617", "4.1303", "2.9080"] ##["0.1454", "0.0866", "0.1653", "0.0799", "0.1257", "0.1445", "0.1595"] ##["0.2121", "0.3048", "0.5187", "0.6238", "0.5816", "0.4270", "0.3923", "0.9799", "0.6943"] ##["0.9131", "0.7360", "1.5130", "0.5253", "1.6901", "4.8891", "3.8746", "3.9195", "4.0887", "2.8648"] #["0.3041", "0.3960", "0.5012", "0.5926", "0.6787", "0.7712", "0.8841", "1.1177", "1.1183", "1.1457", "1.7361", "1.9808"]
# azi_source_to_stat_labels = ["288.74", "209.53", "174.47", "102.07", "228.57", "51.38"] ##["197.3", "129.7", "166.1", "204.1", "247.7", "306.6", "34.1", "127.0", "69.4"] ##["288.74", "209.53", "174.47", "102.07", "228.57", "51.38"] ##["288.74", "209.53", "174.47", "102.07", "228.57", "65.20", "66.23", "51.38", "80.45", "68.01"] ##["71.3", "276.6", "162.8", "131.0", "23.2", "319.6", "243.5"] ##["197.3", "129.7", "166.1", "204.1", "247.7", "306.6", "34.1", "127.0", "69.4"] ##["253.17", "328.56", "3.99", "75.03", "311.04", "114.79", "113.75", "128.77", "99.38", "111.93"] #["143.0", "292.0", "252.0", "5.0", "282.0", "330.0", "255.0", "214.0", "302.0", "276.0", "336.0", "293.0"]
# instrument_gains = np.array([5., 1., 5., 5., 5., 1.], dtype=float)*31.9*2.04082E+07 ##np.array([31.9*2.04082E+07, 31.9*2.04082E+07, 31.9*2.04082E+07, 31.9*2.04082E+07, 31.9*2.04082E+07, 31.9*2.04082E+07, 31.9*2.04082E+07,2400.*988142.,2390.*1.04275E+06], dtype=float) ##np.array([5., 1., 5., 5., 5., 1.], dtype=float)*31.9*2.04082E+07  #np.array([4.00E+08, 4.00E+08, 6.40E+08, 1.89E+09, 1.89E+09, 1.89E+09, 1.89E+09], dtype=float) #np.ones(7, dtype=float) #np.array([31.9*2.04082E+07, 31.9*2.04082E+07, 31.9*2.04082E+07, 31.9*2.04082E+07, 31.9*2.04082E+07, 31.9*2.04082E+07, 31.9*2.04082E+07,2400.*988142.,2390.*1.04275E+06], dtype=float) ##np.array([5., 1., 5., 5., 5., 5., 1.,1.,5.,5.], dtype=float)*31.9*2.04082E+07  #np.ones(len(station_labels), dtype=float)*31.9*2.04082E+07 # Array of station gains
# green_func_dir = "/Users/tomhudson/Python/obspy_scripts/fk/fk_workdirs/Rutford_event_example_multi_depth_source_vel_stf/bed_till_granite/output_Rutford_event_example_for_inversion_MT_and_single_force" ##"fk_workdirs/Skeidararjokull_20140629184210_event_multi_depth_source/output_Skeidararjokull_gl_general_for_inversion_MT_and_single_force/src_depth_0.540" ##"fk_workdirs/Rutford_event_example_multi_depth_source/output_Rutford_event_example_for_inversion_MT_and_single_force/src_depth_2.040" ##"fk_workdirs/example_Rhone_gl_20180214185538/output_Rhone_gl_with_bed_general_for_inversion_MT_and_single_force" ##"fk_workdirs/Skeidararjokull_20140629184210_event/output_Skeidararjokull_gl_general_for_inversion_MT_and_single_force" ##"fk_workdirs/Rutford_event_example/output_Rutford_event_example_for_inversion_MT_and_single_force" # "fk_workdirs/example_Rhone_gl/output_Rhone_gl_with_bed_general_for_inversion_MT_and_single_force"
# outdir = "/Users/tomhudson/Python/obspy_scripts/fk/test_data/output_data_for_inversion_MT_and_single_force_Rutford_event_example_multi_depth_source_SWS_corrected/bed_till_granite" ##"test_data/output_data_for_inversion_MT_and_single_force_Skeidararjokull_20140629184210_event_multi_depth_source/src_depth_0.540" ##"test_data/output_data_for_inversion_MT_and_single_force_Rutford_event_example_multi_depth_source_SWS_corrected/src_depth_2.040" ##"test_data/output_data_for_inversion_MT_and_single_force_Rhone_gl_event_20180214185538" #"output_data_for_inversion_MT_and_single_force"
# high_pass_freq = 10.0 #20.0 #10.0 #5.0 #20.0 #5.0 #10.0 #5.0
# low_pass_freq = 200.0 #120.0 #200.0 #100.0 #120.0 #100.0 ##200.0 #100.0
# num_greens_func_samples = 1024 #125 #1024 #125 #168 #256 #125 #256 #1024 #512 # Note: This value is after downsampling, and must be <= to the number of samples of the greens functions input into this code (can be less if required!) (168 for Rhone data)
# comp_list_MT = ["xx", "yy", "zz", "xy", "xz", "yz"]
# comp_list_single_force = ["X", "Y", "Z"]
# comp_list_actual_waveforms = ['z','r','t','l','q'] # Waveform components (l and q components are calculated in this script)
# ZNE_switch = True #False ##True ##False ##True # If real data is in ZNE format, will need to be rotated so set to True
# real_event_nonlinloc_hyp_file = "real_Rutford_event_example/loc.Tom__RunNLLoc000.20090121.042009.grid0.loc.hyp" ##"real_Skeidararjokull_events/loc.Tom__RunNLLoc000.20140629.184210.grid0.loc.hyp" ##"real_Rutford_event_example/loc.Tom__RunNLLoc000.20090121.042009.grid0.loc.hyp" ##"real_Rhone_gl_event_20180214185538/loc.20180214.185538.grid0.loc.hyp" ##"real_Skeidararjokull_events/loc.Tom__RunNLLoc000.20140629.184210.grid0.loc.hyp" ##"real_Rutford_event_example/loc.Tom__RunNLLoc000.20090121.042009.grid0.loc.hyp"
# mseed_fname = "real_Rutford_event_example/mseed/20090121042009180_ice_flow_dir_corrected_SWS_corrected_horizontal_polarity_corrected.m" ##"real_Skeidararjokull_events/mseed/20140629184210000.m" ##"real_Rutford_event_example/mseed/20090121042009180_SWS_corrected_ice_flow_dir_corrected.m" ##"real_Rhone_gl_event_20180214185538/mseed/20180214185538_500Hz.m" ##"real_Skeidararjokull_events/mseed/20140629184210000.m" ##"real_Rutford_event_example/mseed/20090121042009180.m"
# convert_displacement_to_velocity = False # If greens function data in is displacement (defined by source time function) then differentiate data to convert to velocity for comparision (done before filtering)
# downsample_greens_func_factor = 10 # Integer factor to downsample greens functions data by (if greater than 10, consider downsampling multiple times - this code does not do thatsc) (Downsampling done after filtering)
# upsample_real_data_factor = 1 # Integer factor to upsample real data by (upsampling done before filtering)


#----------------------------------- End: Define parameters -----------------------------------


#----------------------------------- Define functions -----------------------------------

def get_arrival_time_data_from_NLLoc_hyp_files(nlloc_hyp_filename):
    """Function to get phase data from NLLoc file."""
    # Import nonlinloc data:
    nlloc_hyp_data = read_nonlinloc.read_hyp_file(nlloc_hyp_filename)

    # Setup data store:
    arrival_times_dict = {} # Create empty dictionary to store data (with keys: event_origin_time, station_arrivals {station {station_P_arrival, station_S_arrival}}})

    # Get event origin times:
    arrival_times_dict['event_origin_time'] = nlloc_hyp_data.origin_time
    
    # And get station arrival times and azimuth+takeoff angles for each phase, for event:
    arrival_times_dict['station_arrival_times'] = {}
    arrival_times_dict['azi_takeoff_angles'] = {}

    # Loop over stations:
    stations_list = list(nlloc_hyp_data.phase_data.keys())
    for i in range(len(stations_list)):
        station = stations_list[i]
        arrival_times_dict['station_arrival_times'][station] = {}
        arrival_times_dict['azi_takeoff_angles'][station] = {}

        # Loop over phases:
        phases = list(nlloc_hyp_data.phase_data[station].keys())
        for j in range(len(phases)):
            phase = phases[j]
            # Get arrival time:
            arrival_times_dict['station_arrival_times'][station][phase] = nlloc_hyp_data.phase_data[station][phase]['arrival_time']
            # Get station to event azimuth and takeoff angle:
            if phase == "P":
                station_current_azimuth_event_to_sta = nlloc_hyp_data.phase_data[station][phase]['SAzim']
                station_current_toa_event_to_sta = nlloc_hyp_data.phase_data[station][phase]['RDip']
                if station_current_azimuth_event_to_sta > 180.:
                    station_current_azimuth_sta_to_event = 180. - (360. - station_current_azimuth_event_to_sta)
                elif station_current_azimuth_event_to_sta <= 180.:
                    station_current_azimuth_sta_to_event = 360. - (180. - station_current_azimuth_event_to_sta)
                station_current_toa_event_to_sta = float(PHASE_lines[i,24])
                if station_current_toa_event_to_sta > 90.:
                    station_current_toa_sta_inclination = station_current_toa_event_to_sta - 90.
                elif station_current_toa_event_to_sta <= 90.:
                    station_current_toa_sta_inclination = station_current_toa_event_to_sta + 90.
                arrival_times_dict['azi_takeoff_angles'][station]["P_azimuth_sta_to_event"] = station_current_azimuth_sta_to_event
                arrival_times_dict['azi_takeoff_angles'][station]["P_toa_sta_inclination"] = station_current_toa_sta_inclination

    return arrival_times_dict


def rotate_ZRT_to_LQT(tr_z,tr_r,tr_t,back_azi,event_inclin_angle_at_station):
    """Function to rotate ZRT traces into LQT and save to outdir.
    Requires: tr_z,tr_r,tr_t - traces for z,r,t components; back_azi - back azimuth angle from reciever to event in degrees from north; 
    event_inclin_angle_at_station - inclination angle of arrival at receiver, in degrees from vertical down."""
    # Get ZRT stream
    st_ZRT = obspy.core.Stream()
    tr_z.stats.channel = "HHZ"
    tr_z.stats.component = "Z"
    st_ZRT.append(tr_z)
    tr_r.stats.channel = "HHR"
    tr_r.stats.component = "R"
    st_ZRT.append(tr_r)
    tr_t.stats.channel = "HHT"
    tr_t.stats.component = "T"
    st_ZRT.append(tr_t)
    # Rotate to ZNE:
    st_ZNE = st_ZRT.copy()
    st_ZNE.rotate(method='RT->NE', back_azimuth=back_azi)
    # Rotate to LQT:
    st_LQT = st_ZNE.copy()
    st_LQT.rotate(method='ZNE->LQT', back_azimuth=back_azi, inclination=event_inclin_angle_at_station)
    return st_LQT


def get_greens_functions_from_file(green_func_dir, dist_label, actual_waveform_comp, num_greens_func_samples, greens_func_comp_list, real_arrival_times_dict, high_pass_freq, low_pass_freq, convert_displacement_to_velocity, downsample_greens_func_factor):
    """Function to get Greens functions from file."""
    # Create data store:
    green_func_array = np.zeros((num_greens_func_samples, len(greens_func_comp_list)), dtype=float)
    # Loop over Greens function components:
    for j in range(len(greens_func_comp_list)):
        comp = greens_func_comp_list[j]
        # Get LQT greens functions (currently replicates over loop, but ignore inefficiency for now):
        tr_z = obspy.read(glob.glob(green_func_dir+"/*"+dist_label+"*"+comp+".z")[0])[0]
        tr_r = obspy.read(glob.glob(green_func_dir+"/*"+dist_label+"*"+comp+".r")[0])[0]
        tr_t = obspy.read(glob.glob(green_func_dir+"/*"+dist_label+"*"+comp+".t")[0])[0]
        back_azi = real_arrival_times_dict['azi_takeoff_angles'][stat]["P_azimuth_sta_to_event"]
        event_inclin_angle_at_station = real_arrival_times_dict['azi_takeoff_angles'][stat]["P_toa_sta_inclination"]
        st_LQT = rotate_ZRT_to_LQT(tr_z,tr_r,tr_t,back_azi,event_inclin_angle_at_station)
        # And save:
        st_LQT.select(channel="HHL").write(green_func_dir+"/auto_created_"+dist_label+"___"+comp+"."+"l", format="SAC")
        st_LQT.select(channel="HHQ").write(green_func_dir+"/auto_created_"+dist_label+"___"+comp+"."+"q", format="SAC")
        # Note: T does not need to be saved as already exists.

        # Get green func:                
        green_curr_comp_st = obspy.read(glob.glob(green_func_dir+"/*"+dist_label+"*"+comp+"."+actual_waveform_comp)[0])
        # And differentaite if greens functions are for displacement:
        if convert_displacement_to_velocity==True:
            green_curr_comp_st[0].data = np.gradient(green_curr_comp_st[0].data)
        # And filter:
        green_tmp = green_curr_comp_st.filter('bandpass', freqmin=high_pass_freq, freqmax=low_pass_freq, corners=4, zerophase=False)[0].data
        # And downsample (if required):
        if downsample_greens_func_factor>1:
            green_tmp = decimate(green_tmp, downsample_greens_func_factor, ftype='iir', zero_phase=False) # downsample data. Uses scipy.signal.decimate which has an inbuilt anti-aliasing filter. Note: If downsample_factor > 13, downsample multiple times.
        # And save data to array:
        green_func_array[:,j] = green_tmp[0:num_greens_func_samples]

    return green_func_array


def run(station_labels, dist_labels, azi_source_to_stat_labels, green_func_dir, outdir, high_pass_freq, low_pass_freq, num_greens_func_samples, comp_list_MT, comp_list_single_force, comp_list_actual_waveforms, ZNE_switch, real_event_nonlinloc_hyp_file, mseed_fname, instrument_gains, convert_displacement_to_velocity, downsample_greens_func_factor, upsample_real_data_factor):
    """Main function to run script."""
    # Get real event arrival times:
    real_arrival_times_dict = get_arrival_time_data_from_NLLoc_hyp_files(real_event_nonlinloc_hyp_file)
    
    # Make outdir if not already made:
    os.mkdir(outdir)
    
    # Initialise data stores:
    # Greens functions and mseed:
    green_func_array_MT = np.zeros((num_greens_func_samples, len(comp_list_MT)), dtype=float)
    green_func_array_single_force = np.zeros((num_greens_func_samples, len(comp_list_single_force)), dtype=float)
    synth_MT_random_check_green_func_array = np.zeros((num_greens_func_samples, 1), dtype=float) # Only 1 comp as simulate whole source in 1. Note: Not a greens function anymore, as has the full MT information.
    synth_single_force_random_check_green_func_array = np.zeros((num_greens_func_samples, 1), dtype=float) # Only 1 comp as simulate whole source in 1. Note: Not a greens function anymore, as has the full MT information.

    # Process data:
    # Loop over stations:
    for i in range(len(station_labels)):
        stat = station_labels[i]
        dist_label = dist_labels[i] # The distance label associated with the current station
        # Loop over actual waveform components (e.g. z,r,t):
        for k in range(len(comp_list_actual_waveforms)):
            actual_waveform_comp = comp_list_actual_waveforms[k]
            print("Processing for:", stat, actual_waveform_comp)

            # 1. Process modelled data:
            # 1.a. Process for MT components:
            green_func_array_MT = get_greens_functions_from_file(green_func_dir, dist_label, actual_waveform_comp, num_greens_func_samples, comp_list_MT, real_arrival_times_dict, high_pass_freq, low_pass_freq, convert_displacement_to_velocity, downsample_greens_func_factor)
            # And save Greens functions to file:
            np.savetxt(os.path.join(outdir, "green_func_array_MT_"+stat+"_"+actual_waveform_comp+".txt"), green_func_array_MT)
            print("Output file:", os.path.join(outdir, "green_func_array_MT_"+stat+"_"+actual_waveform_comp+".txt"))
    
            # 1.b. And process for single force components, if they exist:  
            green_func_array_single_force = get_greens_functions_from_file(green_func_dir, dist_label, actual_waveform_comp, num_greens_func_samples, comp_list_single_force, real_arrival_times_dict, high_pass_freq, low_pass_freq, convert_displacement_to_velocity, downsample_greens_func_factor)
            # And save Greens functions to file:
            np.savetxt(os.path.join(outdir, "green_func_array_single_force_"+stat+"_"+actual_waveform_comp+".txt"), green_func_array_single_force)
            print("Output file:", os.path.join(outdir, "green_func_array_single_force_"+stat+"_"+actual_waveform_comp+".txt"))

            # 2. Process real data:
            # Import data from file:
            st_real = obspy.read(mseed_fname)
            # Upsample data if required:
            if upsample_real_data_factor>1:
                for i_st in range(len(st_real)):
                    curr_st_data_tmp = st_real[i_st].data
                    orig_xvals = np.arange(len(curr_st_data_tmp))
                    new_xvals = np.arange(0,len(curr_st_data_tmp),1./upsample_real_data_factor)
                    upsampled_st_data_tmp = np.interp(new_xvals, orig_xvals, curr_st_data_tmp)
                    st_real[i_st].data = upsampled_st_data_tmp[:-int(upsample_real_data_factor-1)] # Make have last sample same as orig data
                    st_real[i_st].stats.sampling_rate = int(st_real[i_st].stats.sampling_rate*upsample_real_data_factor) # To make traces match start and end time (effectively updates start and end time)
            # Rotate data from ZNE to ZRT if neccessary:
            if ZNE_switch:
                st_real_unrotated = st_real.copy()
                st_real.clear()
                #if actual_waveform_comp.upper() == "Z":
                st_real.append(st_real_unrotated.select(station=stat,component="Z")[0])
                #elif actual_waveform_comp.upper() == "R" or actual_waveform_comp.upper() == "T":
                current_azi_source_to_stat = float(azi_source_to_stat_labels[i])
                if current_azi_source_to_stat>180.:
                    current_azi_stat_to_source = current_azi_source_to_stat-180.
                else:
                    current_azi_stat_to_source = current_azi_source_to_stat+180.
                st_real_unrotated.select(station=stat).rotate('NE->RT', back_azimuth=current_azi_stat_to_source)
                st_real.append(st_real_unrotated.select(station=stat,component="R")[0])
                st_real.append(st_real_unrotated.select(station=stat,component="T")[0])
            # And get LQT components too:
            tr_z = st_real.select(station=stat,component="Z")[0]
            tr_r = st_real.select(station=stat,component="R")[0]
            tr_t = st_real.select(station=stat,component="T")[0]
            back_azi = real_arrival_times_dict['azi_takeoff_angles'][stat]["P_azimuth_sta_to_event"]
            event_inclin_angle_at_station = real_arrival_times_dict['azi_takeoff_angles'][stat]["P_toa_sta_inclination"]
            st_LQT = rotate_ZRT_to_LQT(tr_z,tr_r,tr_t,back_azi,event_inclin_angle_at_station)
            st_real.append(st_LQT.select(station=stat,component="L")[0])
            st_real.append(st_LQT.select(station=stat,component="Q")[0])
            st_real_filt = st_real.copy()
            st_real_filt.filter('bandpass', freqmin=high_pass_freq, freqmax=low_pass_freq, corners=4, zerophase=False)
            stat_arrival_time = real_arrival_times_dict["station_arrival_times"][stat]["P"]
            #st_real_filt.trim(starttime=stat_arrival_time-0.1, endtime=stat_arrival_time+1.0)
            sampling_rate = st_real_filt[0].stats.sampling_rate
            st_real_filt.trim(starttime=stat_arrival_time-(50.*upsample_real_data_factor)/sampling_rate, endtime=stat_arrival_time+5.0)
            real_data_out = np.transpose(st_real_filt.select(station=stat, component=actual_waveform_comp.upper())[0].data[0:num_greens_func_samples]/instrument_gains[i])
            np.savetxt(os.path.join(outdir, "real_data_"+stat+"_"+actual_waveform_comp+".txt"), real_data_out)
            print("Output file:", os.path.join(outdir, "real_data_"+stat+"_"+actual_waveform_comp+".txt"))


#----------------------------------- End: Define functions -----------------------------------

    
#----------------------------------- Run script -----------------------------------
if __name__ == "__main__":
    
    # Run functions via main run function:
    for src_depth_label in src_depth_labels:
        green_func_dir_tmp = os.path.join(green_func_dir, src_depth_label)
        outdir_tmp = os.path.join(outdir, src_depth_label)
        run(station_labels, dist_labels, azi_source_to_stat_labels, green_func_dir_tmp, outdir_tmp, high_pass_freq, low_pass_freq, num_greens_func_samples, comp_list_MT, comp_list_single_force, comp_list_actual_waveforms, ZNE_switch, real_event_nonlinloc_hyp_file, mseed_fname, instrument_gains, convert_displacement_to_velocity, downsample_greens_func_factor, upsample_real_data_factor)
    
    print("Finished")
        
        


