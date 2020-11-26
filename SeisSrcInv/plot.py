#!python

#-----------------------------------------------------------------------------------------------------------------------------------------

# Script Description:
# Script to take unconstrained moment tensor or DC or single force inversion result (from full waveform inversion) and plot results.

# Input variables:
# See run() function description.

# Output variables:
# See run() function description.

# Created by Tom Hudson, 9th April 2018

#-----------------------------------------------------------------------------------------------------------------------------------------

# Import neccessary modules:
import numpy as np
from numpy.linalg import eigh # For calculating eigenvalues and eigenvectors of symetric (Hermitian) matrices
import matplotlib
import matplotlib.pyplot as plt
import obspy
import scipy.io as sio # For importing .mat MT solution data
import scipy.optimize as opt # For curve fitting
import math # For plotting contours as line
import os,sys
import random
from matplotlib import path # For getting circle bounding path for MT plotting
from obspy.imaging.scripts.mopad import MomentTensor, BeachBall, NED2USE # For getting nodal planes for unconstrained moment tensors
from obspy.core.event.source import farfield # For calculating MT radiation patterns
from matplotlib.patches import Polygon, Circle # For plotting MT radiation patterns
import matplotlib.patches as mpatches # For adding patches for creating legends etc
from matplotlib.collections import PatchCollection # For plotting MT radiation patterns
import glob
import pickle
import matplotlib.gridspec as gridspec

# ------------------------------------------------ Specify module functions ------------------------------------------------

def load_MT_dict_from_file(matlab_data_filename):
    # If output from MTFIT:
    if matlab_data_filename[-3:] == "mat":
        data=sio.loadmat(matlab_data_filename)
        i=0
        while True:
            try:
                # Load data UID from matlab file:
                if data['Events'][0].dtype.descr[i][0] == 'UID':
                    uid=data['Events'][0][0][i][0]
                if data['Events'][0].dtype.descr[i][0] == 'Probability':
                    MTp=data['Events'][0][0][i][0] # stored as a n length vector, the probability
                if data['Events'][0].dtype.descr[i][0] == 'MTSpace':
                    MTs=data['Events'][0][0][i] # stored as a 6 by n array (6 as 6 moment tensor components)
                MTp_absolute = []
                i+=1
            except IndexError:
                break
    
        try:
            stations = data['Stations']
        except KeyError:
            stations = []
    # Else if output from full waveform inversion:
    elif matlab_data_filename[-3:] == "pkl":
        FW_dict = pickle.load(open(matlab_data_filename,"rb"))
        uid = FW_dict["uid"]
        MTp = FW_dict["MTp"]
        MTs = FW_dict["MTs"]
        stations = np.array(FW_dict["stations"], dtype=object)
        # And try to get absolute similarity/probability values:
        try:
            MTp_absolute = FW_dict["MTp_absolute"]
        except KeyError:
            MTp_absolute = []
        
    else:
        print("Cannot recognise input filename.")
        
    return uid, MTp, MTp_absolute, MTs, stations
    
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

def load_MT_waveforms_dict_from_file(waveforms_data_filename):
    """Function to read waveforms dict output from full_waveform_inversion."""
    wfs_dict = pickle.load(open(waveforms_data_filename, "rb"))
    return wfs_dict

def get_full_MT_array(mt):
    full_MT = np.array( ([[mt[0],mt[3]/np.sqrt(2.),mt[4]/np.sqrt(2.)],
                          [mt[3]/np.sqrt(2.),mt[1],mt[5]/np.sqrt(2.)],
                          [mt[4]/np.sqrt(2.),mt[5]/np.sqrt(2.),mt[2]]]) )
    return full_MT
    
def TP_FP(T,P):
    """Converts T,P vectors to the fault normal and slip
    Converts the 3x3 Moment Tensor to the fault normal and slip vectors.
    Args
        T: numpy matrix of T vectors.
        P: numpy matrix of P vectors.
    Returns
        (numpy.matrix, numpy.matrix): tuple of Normal and slip vectors
    (Note: Function from MTFIT.MTconvert)
    """
    if T.ndim==1:
        T=np.matrix(T)
    if P.ndim==1:
        P=np.matrix(P)
    if T.shape[0]!=3:
        T=T.T
    if P.shape[0]!=3:
        P=P.T
    N1=(T+P)/np.sqrt(np.diag(np.matrix(T+P).T*np.matrix(T+P)))
    N2=(T-P)/np.sqrt(np.diag(np.matrix(T-P).T*np.matrix(T-P)))
    return N1,N2

def MT33_TNPE(MT33):
    """Converts 3x3 matrix to T,N,P vectors and the eigenvalues
    Converts the 3x3 Moment Tensor to the T,N,P vectors and the eigenvalues.
    Args
        MT33: 3x3 numpy matrix
    Returns
        (numpy.matrix, numpy.matrix, numpy.matrix, numpy.array): tuple of T, N, P vectors and Eigenvalue array
    (Note: Function from MTFIT.MTconvert)
    """
    E,L=np.linalg.eig(MT33)
    idx = E.argsort()[::-1]
    E=E[idx]
    L=L[:,idx]
    T=L[:,0]
    P=L[:,2]
    N=L[:,1]
    return T,N,P,E

def FP_SDR(normal,slip):   
    """Converts fault normal and slip to strike dip rake
    Coordinate system is North East Down.
    Args
        normal: numpy matrix - Normal vector
        slip: numpy matrix - Slip vector
    Returns
        (float, float, float): tuple of strike, dip and rake angles in radians
    (Note: Function from MTFIT.MTconvert)
    """ 
    if type(slip)==np.ndarray:
        slip=slip/np.sqrt(np.sum(slip*slip))
    else:
        slip=slip/np.sqrt(np.diag(slip.T*slip))
    if type(normal)==np.ndarray:
        normal=normal/np.sqrt(np.sum(normal*normal))
    else:
        normal=normal/np.sqrt(np.diag(normal.T*normal))
    slip[:,np.array(normal[2,:]>0).flatten()]*=-1
    normal[:,np.array(normal[2,:]>0).flatten()]*=-1
    normal=np.array(normal)
    slip=np.array(slip)
    strike,dip=normal_SD(normal)
    rake=np.arctan2(-slip[2],slip[0]*normal[1]-slip[1]*normal[0])
    strike[dip>np.pi/2]+=np.pi
    rake[dip>np.pi/2]=2*np.pi-rake[dip>np.pi/2]
    dip[dip>np.pi/2]=np.pi-dip[dip>np.pi/2]
    strike=np.mod(strike,2*np.pi)
    rake[rake>np.pi]-=2*np.pi
    rake[rake<-np.pi]+=2*np.pi
    return np.array(strike).flatten(),np.array(dip).flatten(),np.array(rake).flatten()

def normal_SD(normal):
    """
    Convert a plane normal to strike and dip
    Coordinate system is North East Down.
    Args
        normal: numpy matrix - Normal vector
    Returns
        (float, float): tuple of strike and dip angles in radians
    """
    if not isinstance(normal, np.matrixlib.defmatrix.matrix):
        normal = np.array(normal)/np.sqrt(np.sum(normal*normal, axis=0))
    else:
        normal = normal/np.sqrt(np.diag(normal.T*normal))
    normal[:, np.array(normal[2, :] > 0).flatten()] *= -1
    normal = np.array(normal)
    strike = np.arctan2(-normal[0], normal[1])
    dip = np.arctan2((normal[1]**2+normal[0]**2),
                     np.sqrt((normal[0]*normal[2])**2+(normal[1]*normal[2])**2))
    strike = np.mod(strike, 2*np.pi)
    return strike, dip

def MT33_SDR(MT33):
    """Converts 3x3 matrix to strike dip and rake values (in radians)
    Converts the 3x3 Moment Tensor to the strike, dip and rake. 
    Args
        MT33: 3x3 numpy matrix
    Returns
        (float, float, float): tuple of strike, dip, rake angles in radians
    (Note: Function from MTFIT.MTconvert)
    """
    T,N,P,E=MT33_TNPE(MT33)
    N1,N2=TP_FP(T,P)
    return FP_SDR(N1,N2)

def rotation_matrix(axis, theta):
    """
    Function to get rotation matrix given an axis of rotation and a rotation angle. Based on Euler-Rodrigues formula.
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    (From: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector)
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def get_slip_vector_from_full_MT(full_MT):
    """Function to get slip vector from full MT."""
    # Get sorted eigenvectors:
    # Find the eigenvalues for the MT solution and sort into descending order:
    w,v = eigh(full_MT) # Find eigenvalues and associated eigenvectors for the symetric (Hermitian) MT matrix (for eigenvalue w[i], eigenvector is v[:,i])
    full_MT_eigvals_sorted = np.sort(w)[::-1] # Sort eigenvalues into descending order
        
    # Get T-axis (eigenvector corresponding to highest eigenvalue):
    T_axis_eigen_value = full_MT_eigvals_sorted[0] # Highest eigenvalue value
    T_axis_eigenvector_idx = np.where(w==T_axis_eigen_value)[0][0]
    T_axis_vector = v[:,T_axis_eigenvector_idx]
    
    # Get null-axis (eigenvector corresponding to intermediate eigenvalue):
    null_axis_eigen_value = full_MT_eigvals_sorted[1] # Intermediate eigenvalue value
    null_axis_eigenvector_idx = np.where(w==null_axis_eigen_value)[0][0]
    null_axis_vector = v[:,null_axis_eigenvector_idx]
    
    # Get P-axis (eigenvector corresponding to lowest eigenvalue) (for completeness only):
    P_axis_eigen_value = full_MT_eigvals_sorted[2] # Lowest eigenvalue value
    P_axis_eigenvector_idx = np.where(w==P_axis_eigen_value)[0][0]
    P_axis_vector = v[:,P_axis_eigenvector_idx]
    
    # Get slip vector:
    # (For fault plane 1)
    slip_vector = (1./np.sqrt(2))*(T_axis_vector - P_axis_vector)
    normal_axis_vector = (1./np.sqrt(2))*(T_axis_vector + P_axis_vector)
    # s,d,r = MT33_SDR(full_MT)
    # sdr = [s[0], d[0], r[0]]
    # normal_axis_vector = np.array([- np.sin(sdr[1]) * np.sin(sdr[0]),
    #                              - np.sin(sdr[1]) * np.sin(sdr[0]), 
    #                                 np.cos(sdr[1])])
    # slip_vector = [np.cos(sdr[2]) * np.cos(sdr[0]) + np.sin(sdr[2]) * np.cos(sdr[1]) * np.sin(sdr[0]), 
    #                 - np.cos(sdr[2]) * np.sin(sdr[0]) + np.sin(sdr[2]) * np.cos(sdr[1]) * np.cos(sdr[0]),
    #                 np.sin(sdr[2]) * np.sin(sdr[1])]

    return slip_vector, normal_axis_vector, T_axis_vector, null_axis_vector, P_axis_vector

def convert_spherical_coords_to_cartesian_coords(r,theta,phi):
    """Function to take spherical coords and convert to cartesian coords. (theta between 0 and pi, phi between 0 and 2pi)"""
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z

def Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z):
    """Function to take 3D grid coords for a cartesian coord system and convert to 2D equal area projection."""
    # if z.all()>-1.0:
    #     X = x * np.sqrt(1/(1+z))
    #     Y = y * np.sqrt(1/(1+z))
    # else:
    X = x * np.sqrt(np.absolute(1/(1+z)))
    Y = y * np.sqrt(np.absolute(1/(1+z)))
    return X,Y


def stereographic_equal_angle_projection_conv_XY_plane_for_MTs(x,y,z):
    """Function to take 3D grid coords for a cartesian coord system and convert to 2D stereographic equal angle projection."""
    X = x / (1-z)
    Y = y / (1-z)
    return X,Y


def rotate_threeD_coords_about_spec_axis(x, y, z, rot_angle, axis_for_rotation="x"):
    """Function to rotate about x-axis (right-hand rule). Rotation angle must be in radians."""
    if axis_for_rotation == "x":
        x_rot = x
        y_rot = (y*np.cos(rot_angle)) - (z*np.sin(rot_angle))
        z_rot = (y*np.sin(rot_angle)) + (z*np.cos(rot_angle))
    elif axis_for_rotation == "y":
        x_rot = (x*np.cos(rot_angle)) + (z*np.sin(rot_angle))
        y_rot = y
        z_rot = (z*np.cos(rot_angle)) - (x*np.sin(rot_angle))
    elif axis_for_rotation == "z":
        x_rot = (x*np.cos(rot_angle)) - (y*np.sin(rot_angle))
        y_rot = (x*np.sin(rot_angle)) + (y*np.cos(rot_angle))
        z_rot = z
    return x_rot, y_rot, z_rot
    
# def strike_dip_rake_to_slip_vector(strike, dip, rake):
#     """Function to convert strike, dip, rake to slip vector.
#     Returns normalised slip vector, in NED format.
#     Input angles must be in radians."""
#     # Define initial vector to rotate:
#     vec = np.array([0.,1.,0.]) # In ENU format
#     # Rotate by strike about z-axis:
#     vec[0],vec[1],vec[2] = rotate_threeD_coords_about_spec_axis(vec[0],vec[1],vec[2], -strike, axis_for_rotation="z")
#     # Rotate by dip about x-axis:
#     vec[0],vec[1],vec[2] = rotate_threeD_coords_about_spec_axis(vec[0],vec[1],vec[2], dip, axis_for_rotation="x")
#     # Rotate by rake anticlockwise about z-axis:
#     vec[0],vec[1],vec[2] = rotate_threeD_coords_about_spec_axis(vec[0],vec[1],vec[2], rake, axis_for_rotation="y")
#     # Convert vec to NED foramt:
#     vec_NED = np.array([vec[1],vec[0],-1.*vec[2]])
#     return vec_NED


def create_and_plot_bounding_circle_and_path(ax):
    """Function to create and plot bounding circle for plotting MT solution. 
    Inputs are ax to plot on. Outputs are ax and bounding_circle_path (defining area contained by bounding circle)."""
    # Setup bounding circle:
    theta = np.ones(200)*np.pi/2
    phi = np.linspace(0.,2*np.pi,len(theta))
    r = np.ones(len(theta))
    x,y,z = convert_spherical_coords_to_cartesian_coords(r,theta,phi)
    X_bounding_circle,Y_bounding_circle = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
    ax.plot(Y_bounding_circle,X_bounding_circle, c="k")

    # And create bounding path from circle:
    path_coords = [] # list to store path coords
    for i in range(len(X_bounding_circle)):
        x_tmp = X_bounding_circle[i]
        y_tmp = Y_bounding_circle[i]
        path_coords.append((x_tmp,y_tmp))
    bounding_circle_path = path.Path(path_coords) # bounding path that can be used to find points
    #bounding_circle_path.contains_points([(.5, .5)])
    
    return ax, bounding_circle_path

def get_nodal_plane_xyz_coords(mt_in):
    """Function to get nodal plane coords given 6 MT in, in NED coords. Returns 2 arrays, describing the two nodal planes in terms of x,y,z coords on a unit sphere."""
    ned_mt = mt_in # 6 MT
    mopad_mt = MomentTensor(ned_mt,system='NED') # In north, east, down notation
    bb = BeachBall(mopad_mt, npoints=200)
    bb._setup_BB(unit_circle=True)
    neg_nodalline = bb._nodalline_negative # extract negative nodal plane coords (in 3D x,y,z)
    pos_nodalline = bb._nodalline_positive # extract positive nodal plane coords (in 3D x,y,z)
    return neg_nodalline, pos_nodalline


def plot_radiation_pattern_for_given_NED_full_MT(ax, radiation_pattern_full_MT, bounding_circle_path, lower_upper_hemi_switch="lower", radiation_MT_phase="P", unconstrained_vs_DC_switch="unconstrained", plot_plane="EN"):
    """Function to plot radiation pattern on axis ax, given 6 MT describing MT to plot radiation pattern for and other args.
    Outputs axis ax with radiation pattern plotted."""
    # Get MT to plot radiation pattern for:
    ned_mt = [radiation_pattern_full_MT[0,0], radiation_pattern_full_MT[1,1], radiation_pattern_full_MT[2,2], radiation_pattern_full_MT[0,1], radiation_pattern_full_MT[0,2], radiation_pattern_full_MT[1,2]]

    # Get spherical points to sample for radiation pattern:
    if unconstrained_vs_DC_switch == "DC":
        theta = np.linspace(0,np.pi,200)
    else:
        theta = np.linspace(0,np.pi,100)
    phi = np.linspace(0.,2*np.pi,len(theta))
    r = np.ones(len(theta))
    THETA,PHI = np.meshgrid(theta, phi)
    theta_flattened = THETA.flatten()
    phi_flattened = PHI.flatten()
    r_flattened = np.ones(len(theta_flattened))
    x,y,z = convert_spherical_coords_to_cartesian_coords(r_flattened,theta_flattened,phi_flattened)
    radiation_field_sample_pts = np.vstack((x,y,z))
    # get radiation pattern using farfield fcn:
    if radiation_MT_phase=="P":
        disp = farfield(ned_mt, radiation_field_sample_pts, type="P") # Gets radiation displacement vector
        disp_magn = np.sum(disp * radiation_field_sample_pts, axis=0) # Magnitude of displacement (alligned with radius)  ???np.sqrt???
    elif radiation_MT_phase=="S":
        disp = farfield(ned_mt, radiation_field_sample_pts, type="S") # Gets radiation displacement vector
        disp_magn = np.sqrt(np.sum(disp * disp, axis=0)) # Magnitude of displacement (perpendicular to radius)
    disp_magn /= np.max(np.abs(disp_magn)) # Normalised magnitude of displacemnet
    
    # If solution is DC, convert radiation pattern to 1/-1:
    if unconstrained_vs_DC_switch == "DC":
        disp_magn[disp_magn>=0.] = 1.
        disp_magn[disp_magn<0.] = -1.

    # Create 2D XY radial mesh coords (for plotting) and plot radiation pattern:
    theta_spacing = theta[1]-theta[0]
    phi_spacing = phi[1]-phi[0]
    patches = []
    # Plot majority of radiation points as polygons:
    for b in range(len(disp_magn)):
        # Convert coords if upper hemisphere plot rather than lower hemisphere:
        if lower_upper_hemi_switch=="upper":
            theta_flattened[b] = np.pi-theta_flattened[b]
            #phi_flattened[b] = phi_flattened[b]-np.pi
        # Get coords at half spacing around point:
        theta_tmp = np.array([theta_flattened[b]-(theta_spacing/2.), theta_flattened[b]+(theta_spacing/2.)])
        phi_tmp = np.array([phi_flattened[b]-(phi_spacing/2.), phi_flattened[b]+(phi_spacing/2.)])
        # And check that doesn't go outside boundaries:
        if theta_flattened[b] == 0. or theta_flattened[b] == np.pi:
            continue # ignore as outside boundaries
        if phi_flattened[b] == 0.:# or phi_flattened[b] == 2*np.pi:
            continue # ignore as outside boundaries
        THETA_tmp, PHI_tmp = np.meshgrid(theta_tmp, phi_tmp)
        R_tmp = np.ones(4,dtype=float)
        x,y,z = convert_spherical_coords_to_cartesian_coords(R_tmp,THETA_tmp.flatten(),PHI_tmp.flatten())
        # Perform rotation of plot plane if required:
        if plot_plane == "EZ":
            x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
        elif plot_plane == "NZ":
            x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
            x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
        X, Y = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
        # And plot (but ONLY if within bounding circle):
        if bounding_circle_path.contains_point((X[0],Y[0]), radius=0):
            poly_corner_coords = [(Y[0],X[0]), (Y[2],X[2]), (Y[3],X[3]), (Y[1],X[1])]
            if unconstrained_vs_DC_switch == "DC":
                polygon_curr = Polygon(poly_corner_coords, closed=True, facecolor=matplotlib.cm.binary(128 + int(disp_magn[b]*128)), alpha=0.6)
            elif unconstrained_vs_DC_switch == "unconstrained":
                polygon_curr = Polygon(poly_corner_coords, closed=True, facecolor=matplotlib.cm.RdBu(128 - int(disp_magn[b]*128)), alpha=0.6)
            ax.add_patch(polygon_curr)
    # Plot final point (theta,phi=0,0) (beginning point):
    if unconstrained_vs_DC_switch == "DC":
        centre_area = Circle([0.,0.], radius=theta_spacing/2., facecolor=matplotlib.cm.binary(128 + int(disp_magn[b]*128)), alpha=0.6)
    elif unconstrained_vs_DC_switch == "unconstrained":
        centre_area = Circle([0.,0.], radius=theta_spacing/2., facecolor=matplotlib.cm.RdBu(128 - int(disp_magn[b]*128)), alpha=0.6)
    ax.add_patch(centre_area)
    
    return ax

def plot_nodal_planes_for_given_NED_full_MT(ax, full_MT_for_nodal_planes, bounding_circle_path, lower_upper_hemi_switch="lower", alpha_nodal_planes=0.3, plot_plane="EN"):
    """Function for plotting nodal planes on axis ax, for given 6MT in NED format for nodal planes."""
    # Get ned 6 mt to plot nodal plaens for
    ned_mt = [full_MT_for_nodal_planes[0,0], full_MT_for_nodal_planes[1,1], full_MT_for_nodal_planes[2,2], full_MT_for_nodal_planes[0,1], full_MT_for_nodal_planes[0,2], full_MT_for_nodal_planes[1,2]]

    # Get 3D nodal planes:
    plane_1_3D, plane_2_3D = get_nodal_plane_xyz_coords(ned_mt)
    # And switch vertical if neccessary:
    if lower_upper_hemi_switch=="upper":
        plane_1_3D[2,:] = -1*plane_1_3D[2,:] # as positive z is down, therefore down gives spherical projection
        plane_2_3D[2,:] = -1*plane_2_3D[2,:] # as positive z is down, therefore down gives spherical projection
    # Specify 3D coords explicitely:
    x1, y1, z1 = plane_1_3D[0],plane_1_3D[1],plane_1_3D[2]
    x2, y2, z2 = plane_2_3D[0],plane_2_3D[1],plane_2_3D[2]
    # Perform rotation of plot plane if required:
    if plot_plane == "EZ":
        x1, y1, z1 = rotate_threeD_coords_about_spec_axis(x1, y1, z1, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
        x2, y2, z2 = rotate_threeD_coords_about_spec_axis(x2, y2, z2, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
    elif plot_plane == "NZ":
        x1,y1,z1 = rotate_threeD_coords_about_spec_axis(x1, y1, z1, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
        x1,y1,z1 = rotate_threeD_coords_about_spec_axis(x1, y1, z1, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
        x2,y2,z2 = rotate_threeD_coords_about_spec_axis(x2, y2, z2, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
        x2,y2,z2 = rotate_threeD_coords_about_spec_axis(x2, y2, z2, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
    # And convert to 2D:
    X1,Y1 = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x1, y1, z1)
    X2,Y2 = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x2, y2, z2)

    # Get only data points within bounding circle:
    path_coords_plane_1 = [] # list to store path coords
    path_coords_plane_2 = [] # list to store path coords
    for j in range(len(X1)):
        path_coords_plane_1.append((X1[j],Y1[j]))
    for j in range(len(X2)):
        path_coords_plane_2.append((X2[j],Y2[j]))
    stop_plotting_switch = False # If true, would stop plotting on current axis (as can't)
    try:
        path_coords_plane_1_within_bounding_circle = np.vstack([p for p in path_coords_plane_1 if bounding_circle_path.contains_point(p, radius=0)])
        path_coords_plane_2_within_bounding_circle = np.vstack([p for p in path_coords_plane_2 if bounding_circle_path.contains_point(p, radius=0)])
        path_coords_plane_1_within_bounding_circle = np.vstack((path_coords_plane_1_within_bounding_circle, path_coords_plane_1_within_bounding_circle[0,:])) # To make no gaps
        path_coords_plane_2_within_bounding_circle = np.vstack((path_coords_plane_2_within_bounding_circle, path_coords_plane_2_within_bounding_circle[0,:])) # To make no gaps
        X1_within_bounding_circle = path_coords_plane_1_within_bounding_circle[:,0]
        Y1_within_bounding_circle = path_coords_plane_1_within_bounding_circle[:,1]
        X2_within_bounding_circle = path_coords_plane_2_within_bounding_circle[:,0]
        Y2_within_bounding_circle = path_coords_plane_2_within_bounding_circle[:,1]
    except ValueError:
        print("(Skipping current nodal plane solution as can't plot.)")
        stop_plotting_switch = True # Stops rest of script plotting on current axis

    # And plot 2D nodal planes:
    if not stop_plotting_switch:
        # Plot plane 1:
        for a in range(len(X1_within_bounding_circle)-1):
            if np.abs(Y1_within_bounding_circle[a]-Y1_within_bounding_circle[a+1])<0.25 and np.abs(X1_within_bounding_circle[a]-X1_within_bounding_circle[a+1])<0.25:
                ax.plot([Y1_within_bounding_circle[a], Y1_within_bounding_circle[a+1]],[X1_within_bounding_circle[a], X1_within_bounding_circle[a+1]], color="k", alpha=alpha_nodal_planes, marker="None")
            else:
                continue # And don't plot line between bounding circle intersections
        # And plot plane 2:
        for a in range(len(X2_within_bounding_circle)-1):
            if np.abs(Y2_within_bounding_circle[a]-Y2_within_bounding_circle[a+1])<0.25 and np.abs(X2_within_bounding_circle[a]-X2_within_bounding_circle[a+1])<0.25:
                ax.plot([Y2_within_bounding_circle[a], Y2_within_bounding_circle[a+1]],[X2_within_bounding_circle[a], X2_within_bounding_circle[a+1]], color="k", alpha=alpha_nodal_planes, marker="None")
            else:
                continue # And don't plot line between bounding circle intersections
    
    return ax

def plot_nodal_planes_for_given_single_force_vector(ax, single_force_vector_to_plot, alpha_single_force_vector=0.8, plot_plane="EN"):
    """Function for plotting single force vector on beachball style plot."""
    
    # normalise:
    single_force_vector_to_plot = single_force_vector_to_plot/np.sqrt(single_force_vector_to_plot[0]**2 + single_force_vector_to_plot[1]**2 + single_force_vector_to_plot[2]**2) #np.max(np.absolute(single_force_vector_to_plot))
    
    # Convert 3D vector to 2D plane coords:
    # Note: Single force vector in is is NED format
    x = np.array([single_force_vector_to_plot[1]])
    y = np.array([single_force_vector_to_plot[0]])
    z = np.array([single_force_vector_to_plot[2]])*-1. # -1 factor as single force z is down (?) whereas 
    # Perform rotation of plot plane if required:
    if plot_plane == "EZ":
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
    elif plot_plane == "NZ":
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
    X, Y = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
    
    # And plot:
    ax.quiver([0.],[0.],Y,X,color="#0B7EB3",alpha=alpha_single_force_vector, angles='xy', scale_units='xy', scale=1)
    
    return ax

def convert_cart_coords_to_spherical_coords(x,y,z):
    """Function to convert cartesian coords to spherical coords."""
    r = np.sqrt(x**2 + y**2 + z**2) # Get radius
    theta = np.arccos(z/r) # Get theta (angle from +ve vertical)
    # Get phi (angle from +ve x):
    if y==0:
        if x>=0:
            phi = 0.
        elif x<0:
            phi = np.pi
    elif y>0:
        phi = np.arccos(x/(r*np.sin(theta)))
    elif y<0:
        phi = (2.*np.pi) - np.arccos(x/(r*np.sin(theta)))
    return r, theta, phi
    
def shift_twoD_data_array_to_have_max_in_centre(twoD_array, axis_0_labels, axis_1_labels):
    """Function to do 2D roll of data to shift maximum value to centre of array, rolling 2D array and array labels."""
    # Find maximum:
    axis_0_max_idx = np.where(twoD_array==np.max(twoD_array))[0][0]
    axis_1_max_idx = np.where(twoD_array==np.max(twoD_array))[1][0]
    # Find centre coords:
    axis_0_centre_idx = int(len(twoD_array[:,0])/2.)
    axis_1_centre_idx = int(len(twoD_array[0,:])/2.)
    # Get roll amounts:
    axis_0_roll_amount = axis_0_centre_idx - axis_0_max_idx
    axis_1_roll_amount = axis_1_centre_idx - axis_1_max_idx
    # Roll axis 0:
    twoD_array = np.roll(twoD_array, axis_0_roll_amount, axis=0)
    axis_0_labels = np.roll(axis_0_labels, axis_0_roll_amount)
    # Roll axis 1:
    twoD_array = np.roll(twoD_array, axis_1_roll_amount, axis=1)
    axis_1_labels = np.roll(axis_1_labels, axis_1_roll_amount)
    return twoD_array, axis_0_labels, axis_1_labels

def get_uncertainty_estimate_bounds_full_soln(MTs, MTp, inversion_type, n_data_frac=0.1, use_gau_fit=False, DC_switch_slip_vector=False):
    """Function to get uncertainty estimate of the direction/orientation of the radiation pattern.
    Currently takes a fraction of the data (e.g. 10%) and fits a Gaussian to the data, taking the full width half maximum as the measure (more representatitive than the standard deviation.)
    DC_switch_slip_vector is only for DC solutions, and switches the slip vector from the primary to secondary nodal plane.
    Returns the upper and lower bounds of the uncertainty in direction in terms of theta and phi as well as x,y,z.
    Note: Currently doesn't centre data for fitting gaussian."""
    
    # Get top n solutions:
    num_data_samples = int(n_data_frac*len(MTp))
    MT_indices = MTp.argsort()[-int(num_data_samples):][::-1] # Get indices of sorted array
    MTs_to_process = MTs[:,MT_indices]
    MTp_to_process = MTp[MT_indices]
    
    # Create 2D array bin for data:
    num_degrees_per_bin = 5
    theta_phi_bins = np.zeros((int(180/num_degrees_per_bin), int(360/num_degrees_per_bin)), dtype=float)
    theta_bin_labels = np.arange(0.+(np.pi/180.)/2., np.pi, num_degrees_per_bin*np.pi/180.)
    phi_bin_labels = np.arange(0.+(np.pi/360.)/2., 2.*np.pi, num_degrees_per_bin*2.*np.pi/360.)
    
    # Get x,y,z coords of slip vector/single force vector (depending upon inversion type):
    # Note: x,y,z in NED coordinate system.
    if inversion_type == "single_force":
        x_array = MTs_to_process[1,:]
        y_array = MTs_to_process[0,:]
        z_array = -1*MTs_to_process[2,:]
    elif inversion_type == "DC":
        x_array = np.zeros(len(MTs_to_process[0,:]), dtype=float)
        y_array = np.zeros(len(MTs_to_process[0,:]), dtype=float)
        z_array = np.zeros(len(MTs_to_process[0,:]), dtype=float)
        for a in range(len(MTs_to_process[0,:])):
            # Get slip direction for current DC MT:
            MT_curr = MTs_to_process[:,a]
            MT_curr_full_MT = get_full_MT_array(MT_curr)
            slip_vector, normal_axis_vector, T_axis_vector, null_axis_vector, P_axis_vector = get_slip_vector_from_full_MT(MT_curr_full_MT) # Get slip vector from DC radiation pattern
            if DC_switch_slip_vector:
                slip_direction = P_axis_vector # Note - should be = slip_vector, but T axis vector actually gives slip direction at the moment. Not sure if this is because of negative eigenvalues in get_slip_vector_from_full_MT() function.
            else:
                slip_direction = T_axis_vector # Note - should be = slip_vector, but T axis vector actually gives slip direction at the moment. Not sure if this is because of negative eigenvalues in get_slip_vector_from_full_MT() function.
            x_array[a] = slip_direction[0]
            y_array[a] = slip_direction[1]
            z_array[a] = slip_direction[2]
    
    # Loop over MT samples:
    for i in range(len(MTs_to_process[0,:])):
        # Calculate theta and phi for a specific MT:
        r, theta, phi = convert_cart_coords_to_spherical_coords(x_array[i],y_array[i],z_array[i])
        theta_bin_val, theta_bin_idx = find_nearest(theta_bin_labels,theta)
        phi_bin_val, phi_bin_idx = find_nearest(phi_bin_labels,phi)
        # Bin the data (in terms of theta, phi):
        theta_phi_bins[theta_bin_idx, phi_bin_idx] += MTp_to_process[i]
    
    # Centre data to maximum bin value, for Gaussian fitting (and shift bin labels too):
    ###theta_phi_bins, theta_bin_labels_new, phi_bin_labels_new = shift_twoD_data_array_to_have_max_in_centre(theta_phi_bins, theta_bin_labels, phi_bin_labels)
    
    if use_gau_fit:
        # # Fit 2D gaussian to theta-phi sphere data:
        # Note: May not work if scatter of solutions lies over the phi=0,phi=2pi boundary, as fits to 2D plane, not spherical surface.
        # # And fit gaussian (to normalised data):
        theta_bin_labels_normallised = theta_bin_labels/np.max(theta_bin_labels)
        twoD_gauss_fitted_data_theta_phi = fit_twoD_Gaussian(theta_bin_labels_normallised, phi_bin_labels, theta_phi_bins, initial_guess_switch=False)
    
        # Get full half width maximum (FWHM) (or could calculatestandard deviation) in theta and phi (from maximum point of gaussian):
        theta_max_idx = np.where(twoD_gauss_fitted_data_theta_phi==np.max(twoD_gauss_fitted_data_theta_phi))[0][0]
        theta_gau_max = theta_bin_labels[theta_max_idx]
        phi_max_idx = np.where(twoD_gauss_fitted_data_theta_phi==np.max(twoD_gauss_fitted_data_theta_phi))[1][0]
        phi_gau_max = phi_bin_labels[phi_max_idx]
        val, idx = find_nearest(twoD_gauss_fitted_data_theta_phi[:,phi_max_idx], np.max(twoD_gauss_fitted_data_theta_phi[:,phi_max_idx])/2.)  # Calculate FWHM of gaussian
        #val, idx = find_nearest(twoD_gauss_fitted_data_theta_phi[:,phi_max_idx], np.std(twoD_gauss_fitted_data_theta_phi[:,phi_max_idx])) # Calculate std of gaussian
        std_theta = np.absolute(theta_bin_labels[theta_max_idx] - theta_bin_labels[idx])
        val, idx = find_nearest(twoD_gauss_fitted_data_theta_phi[theta_max_idx,:], np.max(twoD_gauss_fitted_data_theta_phi[theta_max_idx,:])/2.)  # Calculate FWHM of gaussian
        #val, idx = find_nearest(twoD_gauss_fitted_data_theta_phi[theta_max_idx,:], np.std(twoD_gauss_fitted_data_theta_phi[theta_max_idx,:])) # Calculate std of gaussian
        std_phi = np.absolute(phi_bin_labels[phi_max_idx] - phi_bin_labels[idx])
        theta_uncert_bounds = np.array([theta_gau_max-std_theta, theta_gau_max+std_theta])
        phi_uncert_bounds = np.array([phi_gau_max-std_phi, phi_gau_max+std_phi])
        
    else:
        # Get REAL DATA standard deviation in theta and phi (from maximum point of real data):
        theta_max_idx = np.where(theta_phi_bins==np.max(theta_phi_bins))[0][0]
        theta_gau_max = theta_bin_labels[theta_max_idx]
        phi_max_idx = np.where(theta_phi_bins==np.max(theta_phi_bins))[1][0]
        phi_gau_max = phi_bin_labels[phi_max_idx]
        val, idx = find_nearest(theta_phi_bins[:,phi_max_idx], np.std(theta_phi_bins[:,phi_max_idx]))  # Calculate FWHM of gaussian
        #val, idx = find_nearest(twoD_gauss_fitted_data_theta_phi[:,phi_max_idx], np.std(twoD_gauss_fitted_data_theta_phi[:,phi_max_idx])) # Calculate std of gaussian
        std_theta = np.absolute(theta_bin_labels[theta_max_idx] - theta_bin_labels[idx])
        val, idx = find_nearest(theta_phi_bins[theta_max_idx,:], np.std(theta_phi_bins[theta_max_idx,:]))  # Calculate FWHM of gaussian
        #val, idx = find_nearest(twoD_gauss_fitted_data_theta_phi[theta_max_idx,:], np.std(twoD_gauss_fitted_data_theta_phi[theta_max_idx,:])) # Calculate std of gaussian
        std_phi = np.absolute(phi_bin_labels[phi_max_idx] - phi_bin_labels[idx])
        theta_uncert_bounds = [theta_gau_max-std_theta, theta_gau_max+std_theta]
        phi_uncert_bounds = [phi_gau_max-std_phi, phi_gau_max+std_phi]
    
    # And get x,y,z values for uncertainty bounds:
    r = 1.
    x_uncert_bounds = r*np.sin(theta_uncert_bounds)*np.cos(phi_uncert_bounds)
    y_uncert_bounds = r*np.sin(theta_uncert_bounds)*np.sin(phi_uncert_bounds)
    z_uncert_bounds = r*np.cos(theta_uncert_bounds)
    
    return x_uncert_bounds, y_uncert_bounds, z_uncert_bounds, theta_uncert_bounds, phi_uncert_bounds
    
def plot_uncertainty_vector_area_for_full_soln(ax, max_likelihood_vector, x_uncert_bounds, y_uncert_bounds, z_uncert_bounds, plot_plane="EN"):
    """Function for plotting single force vector on beachball style plot. All vectors should be input in NED coordinates."""
    
    # normalise:
    max_likelihood_vector = max_likelihood_vector/np.sqrt(max_likelihood_vector[0]**2 + max_likelihood_vector[1]**2 + max_likelihood_vector[2]**2) #np.max(np.absolute(max_likelihood_vector))
    
    # Convert 3D vectors to 2D plane coords:
    # 1. For most likely vector direction:
    # Note: Single force vector in is is NED format
    x = np.array([max_likelihood_vector[0]])
    y = np.array([max_likelihood_vector[1]])
    z = np.array([max_likelihood_vector[2]])
    # Perform rotation of plot plane if required:
    if plot_plane == "EZ":
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
    elif plot_plane == "NZ":
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
    X_most_likely, Y_most_likely = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
    # 2. For lower uncertainty vector direction:
    # Note: Single force vector in is is NED format
    x = np.array([x_uncert_bounds[0]])
    y = np.array([y_uncert_bounds[0]])
    z = np.array([z_uncert_bounds[0]])
    # Perform rotation of plot plane if required:
    if plot_plane == "EZ":
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
    elif plot_plane == "NZ":
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
    X_lower_uncert, Y_lower_uncert = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
    # 3. For lower uncertainty vector direction:
    # Note: Single force vector in is is NED format
    x = np.array([x_uncert_bounds[1]])
    y = np.array([y_uncert_bounds[1]])
    z = np.array([z_uncert_bounds[1]])
    # Perform rotation of plot plane if required:
    if plot_plane == "EZ":
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
    elif plot_plane == "NZ":
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
        x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
    X_upper_uncert, Y_upper_uncert = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
    
    # And plot:
    ax.quiver([0.],[0.],Y_most_likely,X_most_likely,color="#DB3E1F",alpha=0.6, angles='xy', scale_units='xy', scale=1) # Plot maximum vector line
    ax.plot([0., Y_lower_uncert],[0., X_lower_uncert], color="#DB3E1F", ls="--", alpha=0.6) # Plot minimum uncertainty vector line
    ax.plot([0., Y_upper_uncert],[0., X_upper_uncert], color="#DB3E1F", ls="--", alpha=0.6) # Plot maximum uncertainty vector line
    
    return ax

def plot_full_waveform_result_beachball(MTs_to_plot, wfs_dict, radiation_pattern_MT=[], MTp_max_prob_value=-1, stations=[], lower_upper_hemi_switch="lower", figure_filename=[], num_MT_solutions_to_plot=20, inversion_type="unconstrained", radiation_MT_phase="P", plot_plane="EN", plot_uncertainty_switch=False, uncertainty_MTs=[], uncertainty_MTp=[], plot_wfs_on_focal_mech_switch=True, DC_switch_slip_vector=False):
    """Function to plot full waveform DC constrained inversion result on sphere, then project into 2D using an equal area projection.
    Input MTs are np array of NED MTs in shape [6,n] where n is number of solutions. Also takes optional radiation_pattern_MT, which it will plot a radiation pattern for.
        Note: x and y coordinates switched for plotting to take from NE to EN
        Note: stations is a dictionary containing station info."""
    
    # Setup figure:
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111) #projection="3d")
    
    # Add some settings for main figure:
    ax.set_xlabel("E")
    ax.set_ylabel("N")
    ax.set_xlim(-3.5,3.5)
    ax.set_ylim(-3.5,3.5)
    ax.quiver([-2.6],[2.5],[0.5],[0.],color="k",alpha=0.8, angles='xy', scale_units='xy', scale=1) # Plot x direction label
    ax.quiver([-2.5],[2.4],[0.],[0.5],color="k",alpha=0.8, angles='xy', scale_units='xy', scale=1) # Plot y direction label
    if plot_plane=="EN":
        plt.text(-2.0,2.5,"E",color="k", fontsize=10, horizontalalignment="center", verticalalignment='center',alpha=1.0, zorder=100) # x label
        plt.text(-2.5,3.0,"N",color="k", fontsize=10, horizontalalignment="center", verticalalignment='center',alpha=1.0, zorder=100) # y label
    elif plot_plane=="EZ":
        plt.text(-2.0,2.5,"E",color="k", fontsize=10, horizontalalignment="center", verticalalignment='center',alpha=1.0, zorder=100) # x label
        plt.text(-2.5,3.0,"Z",color="k", fontsize=10, horizontalalignment="center", verticalalignment='center',alpha=1.0, zorder=100) # y label
    elif plot_plane=="NZ":
        plt.text(-2.0,2.5,"N",color="k", fontsize=10, horizontalalignment="center", verticalalignment='center',alpha=1.0, zorder=100) # x label
        plt.text(-2.5,3.0,"Z",color="k", fontsize=10, horizontalalignment="center", verticalalignment='center',alpha=1.0, zorder=100) # y label
    plt.axis('off')
    
    # Add similarity value to plot, if supplied:
    if plot_uncertainty_switch:
        if MTp_max_prob_value>=0.0:
            plt.title("Similarity: "+np.str(MTp_max_prob_value))
    
    # Setup bounding circle and create bounding path from circle:
    ax, bounding_circle_path = create_and_plot_bounding_circle_and_path(ax)
    
    # Plot radiation pattern and nodal planes if inversion_type is DC or unconstrained:
    if inversion_type == "DC" or inversion_type == "unconstrained":
        unconstrained_vs_DC_switch = inversion_type
        
        # Plot radiation pattern if provided with radiation pattern MT to plot:
        if not len(radiation_pattern_MT)==0:
            # if inversion_type == "unconstrained":
            radiation_pattern_full_MT = get_full_MT_array(radiation_pattern_MT)
            plot_radiation_pattern_for_given_NED_full_MT(ax, radiation_pattern_full_MT, bounding_circle_path, lower_upper_hemi_switch=lower_upper_hemi_switch, radiation_MT_phase=radiation_MT_phase, unconstrained_vs_DC_switch=unconstrained_vs_DC_switch, plot_plane=plot_plane) # Plot radiation pattern

        # Plot MT nodal plane solutions:
        # Get samples to plot:
        # IF single sample, plot most likely (assocaited with radiation pattern):
        if num_MT_solutions_to_plot == 1:
            if not len(radiation_pattern_MT)==0:
                curr_MT_to_plot = get_full_MT_array(radiation_pattern_MT)
                ax = plot_nodal_planes_for_given_NED_full_MT(ax, curr_MT_to_plot, bounding_circle_path, lower_upper_hemi_switch, alpha_nodal_planes=0.3, plot_plane=plot_plane)
        # else if number of samples > 1:
        else:
            if len(MTs_to_plot[0,:]) > num_MT_solutions_to_plot:
                sample_indices = random.sample(range(len(MTs_to_plot[0,:])),num_MT_solutions_to_plot) # Get random sample of MT solutions to plot
            else:
                sample_indices = range(len(MTs_to_plot[0,:]))
            # Loop over MT solutions, plotting nodal planes:
            for i in sample_indices:
                # Get current mt:
                curr_MT_to_plot = get_full_MT_array(MTs_to_plot[:,i])
                # And try to plot current MT nodal planes:
                print("Attempted to plot solution", i)
                ax = plot_nodal_planes_for_given_NED_full_MT(ax, curr_MT_to_plot, bounding_circle_path, lower_upper_hemi_switch, alpha_nodal_planes=0.3, plot_plane=plot_plane)
    
    # Or plot single force vector if inversion_type is single_force:
    elif inversion_type == "single_force":
        single_force_vector_to_plot = radiation_pattern_MT
        ax = plot_nodal_planes_for_given_single_force_vector(ax, single_force_vector_to_plot, alpha_single_force_vector=0.8, plot_plane=plot_plane)
    
    # Plot measure of uncertainty in orientation (if specified):
    if plot_uncertainty_switch:
        if inversion_type=="single_force":
            max_likelihood_vector = np.array([single_force_vector_to_plot[1], single_force_vector_to_plot[0], -1*single_force_vector_to_plot[2]])
            x_uncert_bounds, y_uncert_bounds, z_uncert_bounds, theta_uncert_bounds, phi_uncert_bounds = get_uncertainty_estimate_bounds_full_soln(uncertainty_MTs, uncertainty_MTp, inversion_type, n_data_frac=0.1, use_gau_fit=False)
            ax = plot_uncertainty_vector_area_for_full_soln(ax, max_likelihood_vector, x_uncert_bounds, y_uncert_bounds, z_uncert_bounds, plot_plane=plot_plane)
        elif inversion_type=="DC":
            # Get direction of slip for most likely solution:
            radiation_pattern_full_MT = get_full_MT_array(radiation_pattern_MT)
            slip_vector, normal_axis_vector, T_axis_vector, null_axis_vector, P_axis_vector = get_slip_vector_from_full_MT(radiation_pattern_full_MT) # Get slip vector from DC radiation pattern
            if DC_switch_slip_vector:
                max_likelihood_vector = P_axis_vector # Note - should be = slip_vector, but T axis vector actually gives slip direction at the moment. Not sure if this is because of negative eigenvalues in get_slip_vector_from_full_MT() function.
            else:
                max_likelihood_vector = T_axis_vector # Note - should be = slip_vector, but T axis vector actually gives slip direction at the moment. Not sure if this is because of negative eigenvalues in get_slip_vector_from_full_MT() function.            # Get uncertainty bounds for slip direction:
            x_uncert_bounds, y_uncert_bounds, z_uncert_bounds, theta_uncert_bounds, phi_uncert_bounds = get_uncertainty_estimate_bounds_full_soln(uncertainty_MTs, uncertainty_MTp, inversion_type, n_data_frac=0.1, use_gau_fit=False, DC_switch_slip_vector=DC_switch_slip_vector)
            # And plot slip direction and uncertainty bounds:
            ax = plot_uncertainty_vector_area_for_full_soln(ax, max_likelihood_vector, x_uncert_bounds, y_uncert_bounds, z_uncert_bounds, plot_plane=plot_plane)
                        
    # Plot stations (if provided):
    if not len(stations) == 0:
        # Loop over stations:
        for station in stations:
            station_name = station[0][0]
            # Get params for station:
            # If from MTFIT analysis:
            if isinstance(station[1][0], float):
                azi=(station[1][0]/360.)*2.*np.pi + np.pi
                toa=(station[2][0]/360.)*2.*np.pi
                polarity = station[3][0]
            # Else if from python FW inversion:
            else:
                azi=(float(station[1][0])/360.)*2.*np.pi + np.pi
                toa=(float(station[2][0])/360.)*2.*np.pi
                polarity = int(station[3][0])
            # And get 3D coordinates for station (and find on 2D projection):
            theta = np.pi - toa # as +ve Z = down
            phi = azi
            if lower_upper_hemi_switch=="upper":
                theta = np.pi-theta
                phi = phi-np.pi
            # And correct for points below horizontal plane:
            # Note: Could correct for each plane, but currently only correct to place stations for EN plane, regardless of rotation.
            if theta>np.pi/2.:
                theta = theta - np.pi
                phi = phi + np.pi
            if plot_plane == "EZ":
                if (phi>0. and phi<=np.pi/2.) or (phi>3.*np.pi/2. and phi<=2.*np.pi):
                    theta = theta + np.pi/2.
                    phi = phi + np.pi
            elif plot_plane == "NZ":
                if phi>np.pi and phi<=2.*np.pi:
                    theta = theta + np.pi/2.
                    phi = phi + np.pi
            r = 1.0/np.sqrt(2.) # as on surface of focal sphere (but sqrt(2) as other previous plotting reduces size of sphere.)
            x,y,z = convert_spherical_coords_to_cartesian_coords(r, theta, phi)
            # Perform rotation of plot plane if required:
            if plot_plane == "EZ":
                x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
            elif plot_plane == "NZ":
                x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
                x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
            X, Y = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
            # And plot based on polarity:
            if polarity == 1:
                ax.scatter(Y,X,c="r",marker="^",s=30,alpha=1.0, zorder=100)
            elif polarity == -1:
                ax.scatter(Y,X,c="b",marker="v",s=30,alpha=1.0, zorder=100)
            elif polarity == 0:
                ax.scatter(Y,X,c="#267388",marker='o',s=30,alpha=1.0, zorder=100)
            # And plot station name:
            if plot_wfs_on_focal_mech_switch:
                plt.sca(ax) # Specify axis to work on
                plt.text(Y,X,station_name,color="k", fontsize=10, horizontalalignment="left", verticalalignment='top',alpha=1.0, zorder=100)
            
            # And plot waveforms (real and synthetic):
            if plot_wfs_on_focal_mech_switch:
                # Get current real and synthetic waveforms:
                # Note: Will get all components for current station
                real_wfs_current_station = []
                synth_wfs_current_station = []
                wfs_component_labels_current_station = []
                for wfs_key in list(wfs_dict.keys()):
                    if station_name in wfs_key:
                        real_wfs_current_station.append(wfs_dict[wfs_key]['real_wf']) # Append current real waveforms to wfs for current station
                        synth_wfs_current_station.append(wfs_dict[wfs_key]['synth_wf']) # Append current synth waveforms to wfs for current station
                        wfs_component_labels_current_station.append(wfs_key.split(", ")[1]) # Get current component label
                # and reorder if have Z,R and T components:
                wfs_component_labels_current_station_sorted = list(wfs_component_labels_current_station)
                wfs_component_labels_current_station_sorted.sort()
                if wfs_component_labels_current_station_sorted == ['R','T','Z']:
                    real_wfs_current_station_unsorted = list(real_wfs_current_station)
                    synth_wfs_current_station_unsorted = list(synth_wfs_current_station)
                    idx_tmp = wfs_component_labels_current_station.index("R")
                    real_wfs_current_station[0] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[0] = synth_wfs_current_station_unsorted[idx_tmp]
                    idx_tmp = wfs_component_labels_current_station.index("T")
                    real_wfs_current_station[1] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[1] = synth_wfs_current_station_unsorted[idx_tmp]
                    idx_tmp = wfs_component_labels_current_station.index("Z")
                    real_wfs_current_station[2] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[2] = synth_wfs_current_station_unsorted[idx_tmp]
                    wfs_component_labels_current_station = wfs_component_labels_current_station_sorted
                    radius_factor_wfs_plotting = 3.0
                elif wfs_component_labels_current_station_sorted == ['L','Q','T']:
                    real_wfs_current_station_unsorted = list(real_wfs_current_station)
                    synth_wfs_current_station_unsorted = list(synth_wfs_current_station)
                    idx_tmp = wfs_component_labels_current_station.index("L")
                    real_wfs_current_station[0] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[0] = synth_wfs_current_station_unsorted[idx_tmp]
                    idx_tmp = wfs_component_labels_current_station.index("Q")
                    real_wfs_current_station[1] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[1] = synth_wfs_current_station_unsorted[idx_tmp]
                    idx_tmp = wfs_component_labels_current_station.index("T")
                    real_wfs_current_station[2] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[2] = synth_wfs_current_station_unsorted[idx_tmp]
                    wfs_component_labels_current_station = wfs_component_labels_current_station_sorted
                    radius_factor_wfs_plotting = 3.0
                elif wfs_component_labels_current_station_sorted == ['R-P', 'R-S', 'T-P', 'T-S', 'Z-P', 'Z-S']:
                    real_wfs_current_station_unsorted = list(real_wfs_current_station)
                    synth_wfs_current_station_unsorted = list(synth_wfs_current_station)
                    idx_tmp = wfs_component_labels_current_station.index("R-P")
                    real_wfs_current_station[0] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[0] = synth_wfs_current_station_unsorted[idx_tmp]
                    idx_tmp = wfs_component_labels_current_station.index("R-S")
                    real_wfs_current_station[1] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[1] = synth_wfs_current_station_unsorted[idx_tmp]
                    idx_tmp = wfs_component_labels_current_station.index("T-P")
                    real_wfs_current_station[2] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[2] = synth_wfs_current_station_unsorted[idx_tmp]
                    idx_tmp = wfs_component_labels_current_station.index("T-S")
                    real_wfs_current_station[3] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[3] = synth_wfs_current_station_unsorted[idx_tmp]
                    idx_tmp = wfs_component_labels_current_station.index("Z-P")
                    real_wfs_current_station[4] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[4] = synth_wfs_current_station_unsorted[idx_tmp]
                    idx_tmp = wfs_component_labels_current_station.index("Z-S")
                    real_wfs_current_station[5] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[5] = synth_wfs_current_station_unsorted[idx_tmp]
                    wfs_component_labels_current_station = wfs_component_labels_current_station_sorted
                    radius_factor_wfs_plotting = 3.0
                elif wfs_component_labels_current_station_sorted == ['Z-P', 'R-P']:
                    real_wfs_current_station_unsorted = list(real_wfs_current_station)
                    synth_wfs_current_station_unsorted = list(synth_wfs_current_station)
                    idx_tmp = wfs_component_labels_current_station.index("Z-P")
                    real_wfs_current_station[0] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[0] = synth_wfs_current_station_unsorted[idx_tmp]
                    idx_tmp = wfs_component_labels_current_station.index("R-T")
                    real_wfs_current_station[1] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[1] = synth_wfs_current_station_unsorted[idx_tmp]
                    wfs_component_labels_current_station = wfs_component_labels_current_station_sorted
                    radius_factor_wfs_plotting = 3.0
                elif wfs_component_labels_current_station_sorted == ['R','Z']:
                    real_wfs_current_station_unsorted = list(real_wfs_current_station)
                    synth_wfs_current_station_unsorted = list(synth_wfs_current_station)
                    idx_tmp = wfs_component_labels_current_station.index("Z")
                    real_wfs_current_station[0] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[0] = synth_wfs_current_station_unsorted[idx_tmp]
                    idx_tmp = wfs_component_labels_current_station.index("R")
                    real_wfs_current_station[1] = real_wfs_current_station_unsorted[idx_tmp]
                    synth_wfs_current_station[1] = synth_wfs_current_station_unsorted[idx_tmp]
                    wfs_component_labels_current_station = wfs_component_labels_current_station_sorted
                    radius_factor_wfs_plotting = 3.0
                else:
                    radius_factor_wfs_plotting = 2.0
                    if len(stations)>5:
                        radius_factor_wfs_plotting = 2.0 + 2.*random.random()
                # for wfs_dict_station_idx in range(len(wfs_dict.keys())):
                #     if wfs_dict.keys()[wfs_dict_station_idx].split(",")[0] == station_name:
                #         real_wf_current_stat = wfs_dict[wfs_dict.keys()[wfs_dict_station_idx]]['real_wf']
                #         synth_wf_current_stat = wfs_dict[wfs_dict.keys()[wfs_dict_station_idx]]['synth_wf']
                # Get coords to plot waveform at:
                if plot_plane == "EN":
                    theta = np.pi/2. # Set theta to pi/2 as want to just plot waveforms in horizontal plane (if plot_plane == "EN")
                    r = radius_factor_wfs_plotting #2.0 # as want to plot waveforms beyond extent of focal sphere
                elif plot_plane == "EZ":
                    if theta == np.pi/2. and (phi==np.pi or phi==2.*np.pi):
                        phi = np.pi
                        r = radius_factor_wfs_plotting #2.
                    else:
                        r = np.sqrt(25./((np.cos(theta)**2) + (np.sin(phi)**2))) # as want to plot waveforms beyond extent of focal sphere
                elif plot_plane == "NZ":
                    if theta == np.pi/2. and (phi==np.pi/2. or phi==3.*np.pi/2.):
                        phi = np.pi
                        r = radius_factor_wfs_plotting #2.
                    else:
                        r = radius_factor_wfs_plotting*np.sqrt(25./((np.cos(theta)**2) + (np.cos(phi)**2))) # as want to plot waveforms beyond extent of focal sphere
                x,y,z = convert_spherical_coords_to_cartesian_coords(r, theta, phi)
                # Perform rotation of plot plane if required:
                if plot_plane == "EZ":
                    x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="y") # Rotate axis to get XY -> XZ plane
                elif plot_plane == "NZ":
                    x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="x") # Rotate axis to get XY -> YZ plane
                    x,y,z = rotate_threeD_coords_about_spec_axis(x, y, z, np.pi/2, axis_for_rotation="z") # Flip N and Z axes (so Z is up)
                X_waveform_loc, Y_waveform_loc = Lambert_azimuthal_equal_area_projection_conv_XY_plane_for_MTs(x,y,z)
                data_xy_coords = (Y_waveform_loc, X_waveform_loc)
                disp_coords = ax.transData.transform(data_xy_coords) # And transform data coords into display coords
                fig_inv = fig.transFigure.inverted() # Create inverted figure transformation
                fig_coords = fig_inv.transform((disp_coords[0],disp_coords[1])) # Transform display coords into figure coords (for adding axis)
                # Plot if waveform exists for current station:
                if len(real_wfs_current_station)>0:
                    # Plot line to waveform:
                    ax.plot([Y,Y_waveform_loc],[X,X_waveform_loc],c='k',alpha=0.6)
                    # And plot waveform:
                    left, bottom, width, height = [fig_coords[0]-0.15, fig_coords[1]-0.1, 0.3, 0.2]
                    # for each wf component:
                    if len(real_wfs_current_station)>1:
                        for k in range(len(real_wfs_current_station)):
                            bottom_tmp = bottom + k*height/len(real_wfs_current_station)
                            inset_ax_tmp = fig.add_axes([left, bottom_tmp, width, height/len(real_wfs_current_station)])
                            #inset_ax1 = inset_axes(ax,width="10%",height="5%",bbox_to_anchor=(0.2,0.4))
                            inset_ax_tmp.plot(real_wfs_current_station[k],c='k', alpha=0.6, linewidth=0.75) # Plot real data
                            inset_ax_tmp.plot(synth_wfs_current_station[k],c='#E83313',linestyle="--", alpha=0.6, linewidth=0.5) # Plot synth data
                            # inset_ax_tmp.set_ylabel(wfs_component_labels_current_station[k],loc="left",size=10)
                            plt.title(wfs_component_labels_current_station[k],loc="left",size=8)
                            plt.axis('off')
                    elif len(real_wfs_current_station)==1:
                        inset_ax_tmp = fig.add_axes([left, bottom, width, height])
                        #inset_ax1 = inset_axes(ax,width="10%",height="5%",bbox_to_anchor=(0.2,0.4))
                        inset_ax_tmp.plot(real_wfs_current_station[0],c='k', alpha=0.6, linewidth=0.75) # Plot real data
                        inset_ax_tmp.plot(synth_wfs_current_station[0],c='#E83313',linestyle="--", alpha=0.6, linewidth=0.75) # Plot synth data
                        plt.axis('off')
        
    # And save figure if given figure filename:
    if not len(figure_filename) == 0:
        plt.savefig(figure_filename, dpi=600)
    else:
        plt.show()
        
def plot_prob_distribution_DC_vs_single_force(MTs, MTp, figure_filename=[], inversion_type=""):
    """Function to get the probability distribution based on all samples for % DC vs. single force (which is the final value in MTs). Input is 10xn array of moment tensor samples and a length n array of their associated probability. Output is results plotted and shown to display or saved to file."""
    
    # Setup arrays to store data:
    percentage_DC_all_solns_bins = np.arange(0.,101.,1.)
    probability_percentage_DC_all_solns_bins = np.zeros(len(percentage_DC_all_solns_bins), dtype=float)
    probability_percentage_SF_all_solns_bins = np.zeros(len(percentage_DC_all_solns_bins), dtype=float)
    
    # Loop over MTs:
    for i in range(len(MTs[0,:])):
        MT_prob_current = MTp[i]
        if not MT_prob_current == 0:
            # Get frac DC and frac crack from CDC decomposition:
            frac_DC = MTs[9,i] # Last value from the inversion
            frac_SF = 1. - MTs[9,i]
            # And append probability to bin:
            # For DC:
            val, val_idx = find_nearest(percentage_DC_all_solns_bins,frac_DC*100.)
            probability_percentage_DC_all_solns_bins[val_idx] += MTp[i] # Append probability of % DC to bin
            # And for single force:
            val, val_idx = find_nearest(percentage_DC_all_solns_bins,frac_SF*100.)
            probability_percentage_SF_all_solns_bins[val_idx] += MTp[i] # Append probability of % single force to bin
    
    # Set first and final bins equal to twice the bin value (as they only get values rounded from half the region of other bins):
    probability_percentage_DC_all_solns_bins[0] = probability_percentage_DC_all_solns_bins[0]*2.
    probability_percentage_DC_all_solns_bins[-1] = probability_percentage_DC_all_solns_bins[-1]*2.
    probability_percentage_SF_all_solns_bins[0] = probability_percentage_SF_all_solns_bins[0]*2.
    probability_percentage_SF_all_solns_bins[-1] = probability_percentage_SF_all_solns_bins[-1]*2.
    
    # And plot results:
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    plt.plot(percentage_DC_all_solns_bins[:], probability_percentage_DC_all_solns_bins[:], c='#D94411')
    if inversion_type == "DC_single_force_couple" or inversion_type == "DC_single_force_no_coupling":
        ax1.set_xlabel("Percentage DC")
    elif inversion_type == "single_force_crack_no_coupling":
        ax1.set_xlabel("Percentage crack")
    ax1.set_xlim((0,100))
    ax1.set_ylim((0.,np.max(probability_percentage_DC_all_solns_bins[:])*1.05))
    ###plt.plot(percentage_DC_all_solns_bins, probability_percentage_DC_all_solns_bins, c='k')
    ax2 = ax1.twiny()
    ax2.set_xlim((0,100))
    ax2.set_xlabel("Percentage single force")
    plt.gca().invert_xaxis()
    #plt.plot(percentage_DC_all_solns_bins[:], probability_percentage_crack_all_solns_bins[:], c='#309BD8')
    ax1.set_ylabel("Probability")
    # And do some presentation formatting:
    ax1.tick_params(labelright=True)
    ax1.tick_params(right = 'on')
    ax1.axvline(x=50,ls="--",c="#CDE64E")
    ax1.axvline(x=25,ls="--",c="#A5B16B")
    ax1.axvline(x=75,ls="--",c="#A5B16B")
    #plt.legend()
    # And save figure if given figure filename:
    if not len(figure_filename) == 0:
        plt.savefig(figure_filename, dpi=600)
        print("Saving plot to file:", figure_filename)
    else:
        plt.show()
        
# Lune plotting functions:
def get_frac_of_MTs_using_MT_probs(MTs, MTp, frac_to_sample, return_MTp_samples_switch=False):
    """Function to return fraction of MTs based on highet probabilities."""
    num_events_to_sample = int(len(MTp)*frac_to_sample) # Take top 1 % of samples
    sorted_indices = np.argsort(MTp)[::-1] # reorder into descending order
    # Find indices of solutions in sample:
    sample_indices = sorted_indices[0:num_events_to_sample]
    MTs_sample = MTs[:,sample_indices]
    MTp_sample = MTp[sample_indices]
    print("Sampled",len(MTs_sample[0,:]),"out of",len(MTs[0,:]),"events")
    if return_MTp_samples_switch:
        return MTs_sample, MTp_sample
    else:
        return MTs_sample

def find_delta_gamm_values_from_sixMT(sixMT):
    """Function to find delta and gamma given 6 moment tensor."""
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
    gamma = np.arctan(((-1*lambda1) + (2*lambda2) - lambda3)/((3**0.5)*(lambda1 - lambda3))) # eq. 20a (Tape and Tape 2012)
    beta = np.arccos((lambda1+lambda2+lambda3)/((3**0.5)*((lambda1**2 + lambda2**2 + lambda3**2)**0.5))) # eq. 20b (Tape and Tape 2012)
    delta = (np.pi/2.) - beta # eq. 23 (Tape and Tape 2012)

    return delta, gamma
    
def get_binned_MT_solutions_by_delta_gamma_dict(MTs_sample, MTp_sample):
    """Function to get binned MT solutions by delta and gamma value. Input is array of MTs (in (6,n) shape).
    Output is binned dictionary containing bin values of delta and gamma and all MT solutions that are in the bin."""
    
    # Set up store for binned MT data:
    gamma_delta_binned_MT_store = {} # Will have the entries: gamma_delta_binned_MT_store[delta][gamma][array of MTs (shape(6,n))]

    # Setup delta-gamma bins for data:
    bin_size_delta = np.pi/120. #np.pi/60.
    bin_size_gamma = np.pi/120. #np.pi/60.
    bin_value_labels_delta = np.arange(-np.pi/2,np.pi/2+bin_size_delta, bin_size_delta)
    bin_value_labels_gamma = np.arange(-np.pi/6,np.pi/6+bin_size_gamma, bin_size_gamma)
    bins_delta_gamma = np.zeros((len(bin_value_labels_delta), len(bin_value_labels_gamma)), dtype=float) # array to store bin values (although can also obtain from dictionary sizes)
    max_prob_bins_delta_gamma = np.zeros((len(bin_value_labels_delta), len(bin_value_labels_gamma)), dtype=float)
    num_samples_in_bins_delta_gamma = np.zeros((len(bin_value_labels_delta), len(bin_value_labels_gamma)), dtype=float)
    
    # And setup dict for all binned values:
    for delta in bin_value_labels_delta:
        for gamma in bin_value_labels_gamma:
            try:
                gamma_delta_binned_MT_store["delta="+str(delta)]["gamma="+str(gamma)] = {}
            except KeyError:
                gamma_delta_binned_MT_store["delta="+str(delta)] = {}
                gamma_delta_binned_MT_store["delta="+str(delta)]["gamma="+str(gamma)] = {}
    
    # Loop over events (binning each data point):
    for a in range(len(MTs_sample[0,:])):
        # Get delta and gamma values for sixMT:
        MT_current = MTs_sample[:,a]
        delta, gamma = find_delta_gamm_values_from_sixMT(MT_current)

        # And bin solution into approriate bin:
        idx_delta = (np.abs(bin_value_labels_delta-delta)).argmin()
        idx_gamma = (np.abs(bin_value_labels_gamma-gamma)).argmin()
        bins_delta_gamma[idx_delta,idx_gamma] += MTp_sample[a]
        num_samples_in_bins_delta_gamma[idx_delta,idx_gamma] += 1. # Append 1 to bin
        if MTp_sample[a] > max_prob_bins_delta_gamma[idx_delta,idx_gamma]:
            max_prob_bins_delta_gamma[idx_delta,idx_gamma] = MTp_sample[a]
        
        
        # # And add to dictionary:
        # delta_bin_label_tmp = bin_value_labels_delta[idx_delta]
        # gamma_bin_label_tmp = bin_value_labels_gamma[idx_gamma]
        # try:
        #     tmp_MT_stacked_array = gamma_delta_binned_MT_store["delta="+str(delta_bin_label_tmp)]["gamma="+str(gamma_bin_label_tmp)]["MTs"]
        #     gamma_delta_binned_MT_store["delta="+str(delta_bin_label_tmp)]["gamma="+str(gamma_bin_label_tmp)]["MTs"] = np.hstack((tmp_MT_stacked_array, MT_current.reshape(6,1)))
        # except KeyError:
        #     gamma_delta_binned_MT_store["delta="+str(delta_bin_label_tmp)]["gamma="+str(gamma_bin_label_tmp)]["MTs"] = np.array(MT_current.reshape(6,1)) # If doesnt exist, create new MT store entry
    
    #return gamma_delta_binned_MT_store, bin_value_labels_delta, bin_value_labels_gamma, bins_delta_gamma, num_samples_in_bins_delta_gamma
    return bin_value_labels_delta, bin_value_labels_gamma, bins_delta_gamma, max_prob_bins_delta_gamma, num_samples_in_bins_delta_gamma

def twoD_Gaussian(X, Y, amplitude, xo, yo, sigma_x, sigma_y, theta):
    """Function describing 2D Gaussian. Pass initial guesses for gaussian parameters. Returns 1D ravelled array describing 2D Gaussian function.
    Based on code: https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
    X, Y are 2D np grids (from np.meshgrid)."""
    xo = float(xo)
    yo = float(yo)
    amplitude = float(amplitude)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    gau = amplitude*np.exp( - (a*((X-xo)**2) + 2*b*(X-xo)*(Y-yo) + c*((Y-yo)**2)))
    #gau_out = gau.ravel() # Makes 2D gau array 1D, as otherwise fitting curve function won't work!
    gau_out = np.ravel(gau) # Makes 2D gau array 1D, as otherwise fitting curve function won't work!
    return gau_out

def fit_twoD_Gaussian(x, y, data, initial_guess_switch=False, initial_guess=(1,1,1,1,1)):
    """Function to fit 2D Gaussian to a dataset. x, y are 1D data arrays, data is a 2D array, described by x and y as labels.
    Based on code from:
    https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m"""    
    
    # Mesh grid for 2D Gaussian fit:
    Y, X = np.meshgrid(y, x)
    
    # Fit Gaussian to data:
    data_ravelled = np.ravel(data)
    if initial_guess_switch:
        print("Initial guess parameters for 2D gaussian fit:")
        print(initial_guess)
        popt, pcov = opt.curve_fit(twoD_Gaussian, (X, Y), data_ravelled, p0=initial_guess)
    else:
        popt, pcov = opt.curve_fit(twoD_Gaussian, (X, Y), data_ravelled)
    print("And final parameters derived:")
    print(popt)
    
    # Get fitted data:
    data_fitted = twoD_Gaussian((X, Y), *popt) # Get 2D Gaussian
    data_fitted = np.reshape(data_fitted, np.shape(data)) # and reshape to original data dimensions
    
    return data_fitted
    
def equal_angle_stereographic_projection_conv_YZ_plane(x,y,z):
    """Function to take 3D grid coords for a cartesian coord system and convert to 2D equal area projection."""
    Y = y/(1+x)
    Z = z/(1+x)
    return Y,Z

def plot_Lune(MTs, MTp, six_MT_max_prob=[], frac_to_sample=0.1, figure_filename=[], plot_max_prob_on_Lune=False):
    """Function to plot Lune plot for certain inversions (if Lune plot is relevent, i.e. not DC constrained or single-force constrained).
    Will plot sampled MT solutions on Lune, binned. Will also fit gaussian to this and return the maximum location of the gaussian and the contour coordinates. Also outputs saved figure."""
    
    # Get sample of MT solutions for fitting Gaussian to:
    MTs_sample, MTp_sample = get_frac_of_MTs_using_MT_probs(MTs, MTp, frac_to_sample, return_MTp_samples_switch=True)
    
    # Get bin values for delta-gamma space (for plotting Lune):
    bin_value_labels_delta, bin_value_labels_gamma, bins_delta_gamma, max_prob_bins_delta_gamma, num_samples_in_bins_delta_gamma = get_binned_MT_solutions_by_delta_gamma_dict(MTs_sample, MTp_sample)
    
    # And plot:
    print("Plotting Lune with fitted Gaussian")
    # Set up figure:
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    # Plot major gridlines:
    for phi in [-np.pi/6., np.pi/6.]:
        theta_range = np.linspace(0.0,np.pi,180)
        phi_range = np.ones(len(theta_range))*phi
        r_range = np.ones(len(theta_range))
        # And convert to 2D projection:
        x,y,z = convert_spherical_coords_to_cartesian_coords(r_range,theta_range,phi_range)
        Y_range,Z_range = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
        ax.plot(Y_range,Z_range, color="black")
    # Plot horizontal minor grid lines:
    minor_horiz_interval = np.pi/12.
    for theta in np.arange(0.+minor_horiz_interval, np.pi+minor_horiz_interval, minor_horiz_interval):
        phi_range = np.linspace(-np.pi/6,np.pi/6,90)
        theta_range = np.ones(len(phi_range))*theta
        r_range = np.ones(len(theta_range))
        # And convert to 2D projection:
        x,y,z = convert_spherical_coords_to_cartesian_coords(r_range,theta_range,phi_range)
        Y_range,Z_range = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
        ax.plot(Y_range,Z_range, color="black", linestyle="--", alpha=0.5)
    # Plot vertical minor gridlines:
    minor_vert_interval = np.pi/24.
    for phi in np.arange(-np.pi/6+minor_vert_interval, np.pi/6, minor_vert_interval):
        theta_range = np.linspace(0.0,np.pi,180)
        phi_range = np.ones(len(theta_range))*phi
        r_range = np.ones(len(theta_range))
        # And convert to 2D projection:
        x,y,z = convert_spherical_coords_to_cartesian_coords(r_range,theta_range,phi_range)
        Y_range,Z_range = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
        ax.plot(Y_range,Z_range, color="black", linestyle="--", alpha=0.5)

    # And plot binned data, colored by bin value:
    # Flatten data with respect to biased sampling due to flat distribution in spherical space rather than Lune space:
    # bins_delta_gamma = bins_delta_gamma/num_samples_in_bins_delta_gamma
    # Normallise data:
    if plot_max_prob_on_Lune:
        bins_delta_gamma_normallised = max_prob_bins_delta_gamma/np.max(max_prob_bins_delta_gamma)
        # Remove zero values:
        bins_delta_gamma_normallised[bins_delta_gamma_normallised==0.] = np.nan
        # bins_delta_gamma_normallised = (bins_delta_gamma_normallised-np.min(np.isfinite(bins_delta_gamma_normallised)))/(np.max(np.isfinite(bins_delta_gamma_normallised)) - np.min(np.isfinite(bins_delta_gamma_normallised)))
    else:
        bins_delta_gamma_normallised = bins_delta_gamma/np.max(bins_delta_gamma) # Normalise data
    # Loop over binned data points:
    Y_all = []
    Z_all = []
    c_all = []
    for i in range(len(bin_value_labels_delta)):
        for j in range(len(bin_value_labels_gamma)):
            delta = bin_value_labels_delta[i]
            gamma = bin_value_labels_gamma[j]
            # And plot data coord (if bin greater than 0):
            if bins_delta_gamma_normallised[i,j]>0.:
                x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - delta,gamma)
                Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
                # ax.scatter(Y,Z, color = matplotlib.cm.inferno(int(bins_delta_gamma_normallised[i,j]*256)), alpha=0.6,s=50)
                Y_all.append(Y)
                Z_all.append(Z)
                c_all.append(bins_delta_gamma_normallised[i,j])
    ax.scatter(Y_all,Z_all, c=c_all, cmap="inferno", alpha=0.6,s=50)
    
    # # Plot maximum location and associated contours associated with Guassian fit:
    # # Plot maximum location:
    # delta = max_bin_delta_gamma_values[0]
    # gamma = max_bin_delta_gamma_values[1]
    # x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - delta,gamma)
    # Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
    # ax.scatter(Y,Z, color = "green", alpha=1.0,s=50, marker="X")
    # # And plot 1 stdev contour:
    # contour_bin_delta_values_sorted = []
    # contour_bin_gamma_values_sorted = []
    # for i in range(len(contour_bin_delta_gamma_values_sorted)):
    #     contour_bin_delta_values_sorted.append(contour_bin_delta_gamma_values_sorted[i][0])
    #     contour_bin_gamma_values_sorted.append(contour_bin_delta_gamma_values_sorted[i][1])
    # delta = np.array(contour_bin_delta_values_sorted)
    # gamma = np.array(contour_bin_gamma_values_sorted)
    # x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - delta,gamma)
    # Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
    # ax.plot(Y,Z, color = "green", alpha=0.5)
    
    # Plot location of maximum probability single MT solution (passed as argument):
    if len(six_MT_max_prob)>0:
        delta, gamma = find_delta_gamm_values_from_sixMT(six_MT_max_prob)
        # And plot data coord:
        x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - delta,gamma)
        Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
        ax.scatter(Y,Z, c="gold", alpha=0.8,s=250, marker="*")
    
    # And Finish plot:
    # Plot labels for various defined locations (locations from Tape and Tape 2012, table 1):
    plt.scatter(0.,1.,s=50,color="black")
    plt.text(0.,1.,"Explosion", fontsize=12, horizontalalignment="center", verticalalignment='bottom')
    plt.scatter(0.,-1.,s=50,color="black")
    plt.text(0.,-1.,"Implosion", fontsize=12, horizontalalignment="center", verticalalignment='top')
    x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) - np.arcsin(5/np.sqrt(33)),-np.pi/6.)
    Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
    plt.scatter(Y,Z,s=50,color="red")
    plt.text(Y,Z,"TC$^+$",color="red", fontsize=12, horizontalalignment="right", verticalalignment='bottom')
    x,y,z = convert_spherical_coords_to_cartesian_coords(1.,(np.pi/2.) + np.arcsin(5/np.sqrt(33)),np.pi/6.)
    Y,Z = equal_angle_stereographic_projection_conv_YZ_plane(x,y,z)
    plt.scatter(Y,Z,s=50,color="red")
    plt.text(Y,Z,"TC$^-$",color="red", fontsize=12, horizontalalignment="left", verticalalignment='top')
    plt.scatter(0.,0.,s=50,color="red")
    plt.text(0.,0.,"DC",color="red", fontsize=12, horizontalalignment="center", verticalalignment='top')
    # Various tidying:
    ax.set_xlim(-1.,1.)
    ax.set_ylim(-1.,1.)
    plt.axis('off')
    # And save figure if given figure filename:
    if not len(figure_filename) == 0:
        plt.savefig(figure_filename, dpi=600)
    else:
        plt.show()
    
    # # And return MT data at maximum (and mts within contour?!):
    # # Get all solutions associated with bins inside contour on Lune plot:
    # gamma_delta_binned_MT_store = get_binned_MT_solutions_by_delta_gamma_dict(MTs_sample) # Returns dictionary of all MTs binned by gamma, delta value
    # # And get all values associated with gaussian maximum on Lune plot:
    # max_bin_delta_gamma_indices = np.where(bins_delta_gamma_gau_fitted==np.max(bins_delta_gamma_gau_fitted))
    # max_bin_delta_gamma_values = [bin_value_labels_delta[max_bin_delta_gamma_indices[0][0]], bin_value_labels_gamma[max_bin_delta_gamma_indices[1][0]]]
    # delta = max_bin_delta_gamma_values[0]
    # gamma = max_bin_delta_gamma_values[1]
    # MTs_max_gau_loc = gamma_delta_binned_MT_store["delta="+str(delta)]["gamma="+str(gamma)]["MTs"] # MT solutions associated with gaussian maximum (note: may be different to maximum value due to max value being fit rather than real value)
    #
    # return MTs_max_gau_loc

def sort_wfs_components_current_station(wfs_component_labels_current_station, real_wfs_current_station, synth_wfs_current_station):
    """Function to sort current waveform components."""
    wfs_component_labels_current_station_sorted = list(wfs_component_labels_current_station)
    wfs_component_labels_current_station_sorted.sort()
    if wfs_component_labels_current_station_sorted == ['R','T','Z']:
        real_wfs_current_station_unsorted = list(real_wfs_current_station)
        synth_wfs_current_station_unsorted = list(synth_wfs_current_station)
        idx_tmp = wfs_component_labels_current_station.index("R")
        real_wfs_current_station[0] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[0] = synth_wfs_current_station_unsorted[idx_tmp]
        idx_tmp = wfs_component_labels_current_station.index("T")
        real_wfs_current_station[1] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[1] = synth_wfs_current_station_unsorted[idx_tmp]
        idx_tmp = wfs_component_labels_current_station.index("Z")
        real_wfs_current_station[2] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[2] = synth_wfs_current_station_unsorted[idx_tmp]
        wfs_component_labels_current_station = wfs_component_labels_current_station_sorted
    elif wfs_component_labels_current_station_sorted == ['L','Q','T']:
        real_wfs_current_station_unsorted = list(real_wfs_current_station)
        synth_wfs_current_station_unsorted = list(synth_wfs_current_station)
        idx_tmp = wfs_component_labels_current_station.index("L")
        real_wfs_current_station[0] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[0] = synth_wfs_current_station_unsorted[idx_tmp]
        idx_tmp = wfs_component_labels_current_station.index("Q")
        real_wfs_current_station[1] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[1] = synth_wfs_current_station_unsorted[idx_tmp]
        idx_tmp = wfs_component_labels_current_station.index("T")
        real_wfs_current_station[2] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[2] = synth_wfs_current_station_unsorted[idx_tmp]
        wfs_component_labels_current_station = wfs_component_labels_current_station_sorted
    elif wfs_component_labels_current_station_sorted == ['R-P', 'R-S', 'T-P', 'T-S', 'Z-P', 'Z-S']:
        real_wfs_current_station_unsorted = list(real_wfs_current_station)
        synth_wfs_current_station_unsorted = list(synth_wfs_current_station)
        idx_tmp = wfs_component_labels_current_station.index("R-P")
        real_wfs_current_station[0] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[0] = synth_wfs_current_station_unsorted[idx_tmp]
        idx_tmp = wfs_component_labels_current_station.index("R-S")
        real_wfs_current_station[1] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[1] = synth_wfs_current_station_unsorted[idx_tmp]
        idx_tmp = wfs_component_labels_current_station.index("T-P")
        real_wfs_current_station[2] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[2] = synth_wfs_current_station_unsorted[idx_tmp]
        idx_tmp = wfs_component_labels_current_station.index("T-S")
        real_wfs_current_station[3] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[3] = synth_wfs_current_station_unsorted[idx_tmp]
        idx_tmp = wfs_component_labels_current_station.index("Z-P")
        real_wfs_current_station[4] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[4] = synth_wfs_current_station_unsorted[idx_tmp]
        idx_tmp = wfs_component_labels_current_station.index("Z-S")
        real_wfs_current_station[5] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[5] = synth_wfs_current_station_unsorted[idx_tmp]
        wfs_component_labels_current_station = wfs_component_labels_current_station_sorted
    elif wfs_component_labels_current_station_sorted == ['Z-P', 'R-P']:
        real_wfs_current_station_unsorted = list(real_wfs_current_station)
        synth_wfs_current_station_unsorted = list(synth_wfs_current_station)
        idx_tmp = wfs_component_labels_current_station.index("Z-P")
        real_wfs_current_station[0] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[0] = synth_wfs_current_station_unsorted[idx_tmp]
        idx_tmp = wfs_component_labels_current_station.index("Z-S")
        real_wfs_current_station[1] = real_wfs_current_station_unsorted[idx_tmp]
        synth_wfs_current_station[1] = synth_wfs_current_station_unsorted[idx_tmp]
        wfs_component_labels_current_station = wfs_component_labels_current_station_sorted
    return wfs_component_labels_current_station_sorted, real_wfs_current_station, synth_wfs_current_station

def plot_wfs_of_most_likely_soln_separate_plot(stations, wfs_dict, plot_fname):
    """Function to plot waveforms for the most likely inversion solution and save as separate plot."""
    
    # Setup figure:
    fig = plt.figure(figsize=(8, 3*len(stations)))
    outer_plot_obj = gridspec.GridSpec(len(stations), 1, wspace=0.2, hspace=0.2)
    
    # Loop over each station, plotting waveforms:
    i=0
    for station in stations:
        station_name = station[0][0]
        # Get current real and synthetic waveforms:
        # Note: Will get all components for current station
        real_wfs_current_station = []
        synth_wfs_current_station = []
        wfs_component_labels_current_station = []
        for wfs_key in list(wfs_dict.keys()):
            if station_name in wfs_key:
                real_wfs_current_station.append(wfs_dict[wfs_key]['real_wf']) # Append current real waveforms to wfs for current station
                synth_wfs_current_station.append(wfs_dict[wfs_key]['synth_wf']) # Append current synth waveforms to wfs for current station
                wfs_component_labels_current_station.append(wfs_key.split(", ")[1]) # Get current component label
        # and reorder if have Z,R and T components:
        wfs_component_labels_current_station_sorted, real_wfs_current_station, synth_wfs_current_station = sort_wfs_components_current_station(wfs_component_labels_current_station, real_wfs_current_station, synth_wfs_current_station)
        # And plot:
        if len(real_wfs_current_station) > 0:
            # Setup inner plot for current station:
            inner_plot_obj = gridspec.GridSpecFromSubplotSpec(len(real_wfs_current_station), 1, subplot_spec=outer_plot_obj[i], wspace=0.1, hspace=0.1)
            for j in range(len(real_wfs_current_station)):
                ax_curr = plt.Subplot(fig, inner_plot_obj[j])
                if j==0:
                    ax_curr.set_title(station_name)
                ax_curr.plot(real_wfs_current_station[j],c='k', alpha=0.75, linewidth=2.5) # Plot real data
                ax_curr.plot(synth_wfs_current_station[j],c='#E83313',linestyle="--", alpha=0.75, linewidth=2.0) # Plot synth data
                ax_curr.set_ylabel(wfs_component_labels_current_station_sorted[j])
                ax_curr.spines['top'].set_visible(False)
                ax_curr.spines['right'].set_visible(False)
                ax_curr.spines['bottom'].set_visible(False)
                ax_curr.spines['left'].set_visible(False)
                ax_curr.get_xaxis().set_ticks([])
                ax_curr.get_yaxis().set_ticks([])
                fig.add_subplot(ax_curr)
        i+=1
    
    # And save figure:
    plt.savefig(plot_fname, dpi=300)


def plot_wfs_of_most_likely_soln_separate_plot_das(wfs_dict, plot_fname, fs=1000.):
    """Function to plot waveforms for the most likely inversion solution for DAS data and save as separate plot."""
    # Get real and synth waveform data:
    stations_to_plot = list(wfs_dict.keys())
    print(stations_to_plot)
    real_wfs = np.zeros( (len(wfs_dict[stations_to_plot[0]]['real_wf']), len(stations_to_plot)) )
    for i in range(len(stations_to_plot)):
        real_wfs[:,i] = wfs_dict[stations_to_plot[i]]['real_wf']
    synth_wfs = np.zeros( (len(wfs_dict[stations_to_plot[0]]['synth_wf']), len(stations_to_plot)) )
    for i in range(len(stations_to_plot)):
        synth_wfs[:,i] = wfs_dict[stations_to_plot[i]]['synth_wf']
        
    # Setup figure:
    fig, axes = plt.subplots(ncols=3, figsize=(12,6), sharey=True)

    # Get spatial and time gridded coords:
    X, T = np.meshgrid( 10.0*np.arange(real_wfs.shape[1]), np.arange(real_wfs.shape[0])/fs )

    # Find max. value:
    max_amp = np.max(np.array([np.max(np.abs(real_wfs)), np.max(np.abs(synth_wfs))]))

    # And plot data:
    axes[0].pcolormesh(X, T, real_wfs, cmap='RdBu', vmin=-max_amp, vmax=max_amp)
    axes[1].pcolormesh(X, T, synth_wfs, cmap='RdBu', vmin=-max_amp, vmax=max_amp)
    axes[2].pcolormesh(X, T, real_wfs - synth_wfs, cmap='RdBu', vmin=-max_amp, vmax=max_amp)
    # Do additional labelling:
    for i in range(3):
        axes[i].set_xlabel('Channel no.')
    axes[0].set_ylabel('Time (s)')
    axes[0].set_title('Obs.')
    axes[1].set_title('Model')
    axes[2].set_title('Difference')

    # And save figure:
    plt.savefig(plot_fname, dpi=600)

    
def plot_slip_vector_distribution(MTs, MTp, six_MT_max_prob=[], frac_to_sample=0.1, figure_filename=[]):
    """Function to plot the slip vector distribution in terms of the spherical coordinates theta and phi."""
    
    # Get highest sample of MTs to plot for:
    MTs_sample, MTp_sample = get_frac_of_MTs_using_MT_probs(MTs, MTp, frac_to_sample, return_MTp_samples_switch=True)
    
    # Loop over solutions, finding binned and maximum probabilities:
    theta_bin_vals = np.arange(0.,np.pi,np.pi/100)
    phi_bin_vals = np.arange(0.,2.*np.pi,np.pi/100)
    theta_phi_bins = np.zeros((len(theta_bin_vals), len(phi_bin_vals)), dtype=float)
    theta_phi_bins_num_samples = np.zeros((len(theta_bin_vals), len(phi_bin_vals)), dtype=float)
    theta_phi_bins_max_prob_vals = np.zeros((len(theta_bin_vals), len(phi_bin_vals)), dtype=float)
    for ii in range(len(MTs_sample[:,0])):
        # Get theta and phi rotations, assuming that have been random orientation from Lune space varied diagonallised MT:
        # (Note: based on rotation of theta and phi about y and z axes after diagonal MT from Lune params created)
        MT_curr = MTs_sample[ii,:]
        MTp_curr = MTp_sample[ii]
        full_MT_curr = get_full_MT_array(MT_curr)
        w,v = eigh(full_MT_curr) # Find eigenvalues and associated eigenvectors for the symetric (Hermitian) MT matrix (for eigenvalue w[i], eigenvector is v[:,i])
        lambda_1 = w[0]
        lambda_2 = w[1]
        lambda_3 = w[2]
        mt_22 = full_MT_curr[1,1]
        mt_33 = full_MT_curr[2,2]
        theta = np.arccos((2*mt_33 - (lambda_1 + lambda_3))/(2.*(lambda_3 - lambda_1)))
        phi = np.arccos((2*mt_22 - (lambda_1 + lambda_2))/(2.*(lambda_2 - lambda_1)))
        val, theta_bin_idx = find_nearest(theta_bin_vals,theta)
        val, phi_bin_idx = find_nearest(phi_bin_vals,phi)
        # And update bins:
        theta_phi_bins[theta_bin_idx, phi_bin_idx] += MTp_curr
        theta_phi_bins_num_samples[theta_bin_idx, phi_bin_idx] += 1.
        if MTp_curr>theta_phi_bins_max_prob_vals[theta_bin_idx, phi_bin_idx]:
            theta_phi_bins_max_prob_vals[theta_bin_idx, phi_bin_idx] = MTp_curr
    
    # Mask zero probabilities:
    theta_phi_bins[theta_phi_bins==0] = np.nan
    theta_phi_bins_max_prob_vals[theta_phi_bins==0] = np.nan
    
    # And plot results:
    # Set up figure:
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    # Plot data:
    theta_grid, phi_grid = np.meshgrid(theta_bin_vals, phi_bin_vals)
    ax.pcolormesh(theta_grid, phi_grid, np.transpose(theta_phi_bins_max_prob_vals), cmap="inferno")
    ax.set_xlabel('$\theta$ (rad)')
    ax.set_ylabel('$\phi$ (rad)')
    # Plot max value:
    if len(six_MT_max_prob)>0:
        full_MT_max_prob = get_full_MT_array(six_MT_max_prob)
        w,v = eigh(full_MT_max_prob) # Find eigenvalues and associated eigenvectors for the symetric (Hermitian) MT matrix (for eigenvalue w[i], eigenvector is v[:,i])
        lambda_1 = w[0]
        lambda_2 = w[1]
        lambda_3 = w[2]
        mt_22 = full_MT_max_prob[1,1]
        mt_33 = full_MT_max_prob[2,2]
        theta = np.arccos((2*mt_33 - (lambda_1 + lambda_3))/(2.*(lambda_3 - lambda_1)))
        phi = np.arccos((2*mt_22 - (lambda_1 + lambda_2))/(2.*(lambda_2 - lambda_1)))
        val, theta_bin_idx = find_nearest(theta_bin_vals,theta)
        val, phi_bin_idx = find_nearest(phi_bin_vals,phi)
        # And convert to absolute value to plot within bin space:
        theta = np.abs(theta)
        phi = np.abs(phi)
        ax.scatter(theta,phi, c="gold", alpha=0.8,s=250, marker="*")
    # And show/save figure:
    if len(figure_filename)>0:
        plt.savefig(figure_filename, dpi=600)
    else:
        plt.show()
        

def run(inversion_type, event_uid, datadir, plot_outdir='plots', radiation_MT_phase="P", plot_Lune_switch=True, plot_uncertainty_switch=False, plot_wfs_separately_switch=False, plot_multi_medium_greens_func_inv_switch=False, multi_medium_greens_func_inv_separate_phase_amp_ratios=False, plot_absolute_probability_switch=True, plot_wfs_on_focal_mech_switch=True, plot_max_prob_on_Lune_switch=False, plot_das_wfs_switch=False, fs_das=1000., DC_switch_slip_vector=False, num_MT_solutions_to_plot=1):
    """Function to run main script.
    ------------------ Inputs ------------------
    Required arguments:
    inversion_type - Type of inversion to plot for. Obviously has to match the inversion undertaken. (type: str)
    event_uid - The UID (unique id) of the event inverted for. It is the numbers in the inversion .pkl out filename. (type: str)
    datadir - Path to where the inversion outputs are saved (type: str)

    Optional arguments:
    plot_outdir - Path to where the inversion plot outputs are saved (default: plots) (type: str)
    radiation_MT_phase - Radiation phase to plot (= "P" or "S") (Defalt = "P") (type: str)
    plot_uncertainty_switch - If True, plots uncertainty in direction/orientation of focal mechanism solution (default is False) (type bool)
    plot_Lune_switch - If True, plots Lune (default is True) (type bool)
    plot_max_prob_on_Lune_switch - If True, plots maximum probability solution for each point on Lune, rather than binned probability (default is False) (type bool)
    plot_wfs_separately_switch - If true, plots waveforms separately (default is False) (type bool)
    plot_multi_medium_greens_func_inv_switch - Set as True if plotting multi medium greens function inversion results (default is False) (type bool)
    multi_medium_greens_func_inv_separate_phase_amp_ratios - If inverted for separate phase amplitude ratios for multi medium greens functions, set this as True (default is False) (type bool)
    plot_wfs_on_focal_mech_switch - If True, plots waveforms on focal mechanism plot (default is True) (type bool)
    plot_das_wfs_switch - Switch to plot DAS data, if DAS data used in inversion (default is False) (type bool)
    fs_das - Sampling rate of the DAS data (default is 1000.0) (type float)
    DC_switch_slip_vector - If True, will switch slip vector to the other nodal plane. Default is False. (type bool)
    num_MT_solutions_to_plot - The number of fault plane solutions to plot on the focal sphere. Currently only implemented for DC fault plane plotting. (type int)

    ------------------ Outputs ------------------
    Various outputs as .png files, saved to the directory specified (e.g. "plots/")
    
    """
    
    # Plot for inversion:
    print("Plotting data for inversion")
    
    # Get inversion filenames:
    MT_data_filename = os.path.join(datadir, event_uid+"_FW_"+inversion_type+".pkl")
    MT_waveforms_data_filename = os.path.join(datadir, event_uid+"_FW_"+inversion_type+".wfs")
    try:
        os.mkdir(plot_outdir)
    except FileExistsError:
        print("")

    print("Processing data for:", MT_data_filename)

    # Import MT data and associated waveforms:
    uid, MTp, MTp_absolute, MTs, stations = load_MT_dict_from_file(MT_data_filename)
    wfs_dict = load_MT_waveforms_dict_from_file(MT_waveforms_data_filename)
    # And use absolute probabilities, if specified (and if data is available):
    if plot_absolute_probability_switch:
        if len(MTp_absolute)>0:
            MTp_for_Lune = MTp.copy()
            MTp = MTp_absolute
    else:
        MTp_for_Lune = MTp.copy()
    
    # Get most likely solution and plot:
    index_MT_max_prob = np.argmax(MTp) # Index of most likely MT solution
    MTp_max_prob_value = np.max(MTp) # Similarity value for most likely MT solution
    MT_max_prob = MTs[:,index_MT_max_prob]
    
    # If solution is for multiple medium greens function solution, separate MT data from relative amplitude ratio:
    if plot_multi_medium_greens_func_inv_switch:
        MT_max_prob_tmp = MT_max_prob[0:6]
        if multi_medium_greens_func_inv_separate_phase_amp_ratios:
            amp_ratio_direct_vs_indirect_P = MT_max_prob[-3] # Proportion of amplitude that is from direct vs indirect radiation
            amp_ratio_direct_vs_indirect_S = MT_max_prob[-2] # Proportion of amplitude that is from direct vs indirect radiation
            amp_ratio_direct_vs_indirect_surface = MT_max_prob[-1] # Proportion of amplitude that is from direct vs indirect radiation
            MT_max_prob = MT_max_prob_tmp
            print("----------------------")
            print(" ")
            print("Amplitude ratios of direct to indirect radiation (P, S, surface):", amp_ratio_direct_vs_indirect_P, amp_ratio_direct_vs_indirect_S, amp_ratio_direct_vs_indirect_surface)
            print("(Note: Although gives values for all phases, only phases specified in original solution are actually relevent).")
            print(" ")
            print("----------------------")
        else:
            amp_ratio_direct_vs_indirect = MT_max_prob[-1] # Proportion of amplitude that is from direct vs indirect radiation
            MT_max_prob = MT_max_prob_tmp
            print("----------------------")
            print(" ")
            print("Amplitude ratio of direct to indirect radiation:", amp_ratio_direct_vs_indirect)
            print(" ")
            print("----------------------")
    
    if inversion_type == "full_mt" or inversion_type == "full_mt_Lune_samp":
        inversion_type = "unconstrained"
        # And get full MT matrix:
        full_MT_max_prob = get_full_MT_array(MT_max_prob)
        # Plot MT solutions and radiation pattern of most likely on sphere:
        MTs_to_plot = full_MT_max_prob #MTs_max_gau_loc
        radiation_pattern_MT = MT_max_prob # 6 moment tensor to plot radiation pattern for
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+".png")
            plot_full_waveform_result_beachball(MTs_to_plot, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, MTp_max_prob_value=MTp_max_prob_value, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type=inversion_type, radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane, plot_uncertainty_switch=plot_uncertainty_switch, uncertainty_MTs=MTs, uncertainty_MTp=MTp, plot_wfs_on_focal_mech_switch=plot_wfs_on_focal_mech_switch)
        # And plot waveforms separately (if specified):
        if plot_wfs_separately_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_separate_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot(stations, wfs_dict, plot_fname)
        # And plot Lune for solution:
        if plot_Lune_switch:
            plot_Lune(MTs, MTp_for_Lune, six_MT_max_prob=radiation_pattern_MT, frac_to_sample=1.0, figure_filename=os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_Lune.png"), plot_max_prob_on_Lune=plot_max_prob_on_Lune_switch)
        ###plot_slip_vector_distribution(MTs, MTp, six_MT_max_prob=MT_max_prob, frac_to_sample=0.01, figure_filename="Plots/"+MT_data_filename.split("/")[-1].split(".")[0]+"_slip_vector_dist.png")
        # And plot DAS waveforms, if specified:
        if plot_das_wfs_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_DAS_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot_das(wfs_dict, plot_fname, fs=1000.)
        
    elif inversion_type == "DC":
        # And get full MT matrix:
        full_MT_max_prob = get_full_MT_array(MT_max_prob)
        # Get sampled MT solutions, based upon number of MT solutions to plot:
        MTs_sample = get_frac_of_MTs_using_MT_probs(MTs, MTp, num_MT_solutions_to_plot)
        # Plot MT solutions and radiation pattern of most likely on sphere:
        MTs_to_plot = MTs_sample #full_MT_max_prob #MTs_max_gau_loc
        radiation_pattern_MT = MT_max_prob # 6 moment tensor to plot radiation pattern for
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+".png")
            plot_full_waveform_result_beachball(MTs_to_plot, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, MTp_max_prob_value=MTp_max_prob_value, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=num_MT_solutions_to_plot, inversion_type=inversion_type, radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane, plot_uncertainty_switch=plot_uncertainty_switch, uncertainty_MTs=MTs, uncertainty_MTp=MTp, plot_wfs_on_focal_mech_switch=plot_wfs_on_focal_mech_switch, DC_switch_slip_vector=DC_switch_slip_vector)
        # And plot waveforms separately (if specified):
        if plot_wfs_separately_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_separate_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot(stations, wfs_dict, plot_fname)
        # And plot DAS waveforms, if specified:
        if plot_das_wfs_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_DAS_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot_das(wfs_dict, plot_fname, fs=1000.)
    
    elif inversion_type == "single_force":
        full_MT_max_prob = MT_max_prob
        # Plot MT solutions and radiation pattern of most likely on sphere:
        MTs_to_plot = full_MT_max_prob #MTs_max_gau_loc
        radiation_pattern_MT = MT_max_prob # 6 moment tensor to plot radiation pattern for
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+".png")
            plot_full_waveform_result_beachball(MTs_to_plot, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, MTp_max_prob_value=MTp_max_prob_value, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type=inversion_type, radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane, plot_uncertainty_switch=plot_uncertainty_switch, uncertainty_MTs=MTs, uncertainty_MTp=MTp, plot_wfs_on_focal_mech_switch=plot_wfs_on_focal_mech_switch)
        # And plot waveforms separately (if specified):
        if plot_wfs_separately_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_separate_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot(stations, wfs_dict, plot_fname)
        # And plot DAS waveforms, if specified:
        if plot_das_wfs_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_DAS_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot_das(wfs_dict, plot_fname, fs=1000.)
    
    elif inversion_type == "DC_single_force_couple":
        full_MT_max_prob = get_full_MT_array(MT_max_prob[0:6])
        radiation_pattern_MT = MT_max_prob[0:6]
        single_force_vector_max_prob = MT_max_prob[6:9]
        amp_prop_DC = MT_max_prob[9] # Proportion of amplitude that is DC
        # Plot MT solutions and radiation pattern of most likely on sphere:
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_DC_component.png")
            plot_full_waveform_result_beachball(full_MT_max_prob, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, MTp_max_prob_value=MTp_max_prob_value, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="DC", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane, plot_uncertainty_switch=plot_uncertainty_switch, uncertainty_MTs=MTs, uncertainty_MTp=MTp, plot_wfs_on_focal_mech_switch=plot_wfs_on_focal_mech_switch)
            figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_SF_component.png")
            plot_full_waveform_result_beachball(single_force_vector_max_prob, wfs_dict, radiation_pattern_MT=single_force_vector_max_prob, MTp_max_prob_value=MTp_max_prob_value, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="single_force", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane, plot_uncertainty_switch=plot_uncertainty_switch, uncertainty_MTs=MTs, uncertainty_MTp=MTp, plot_wfs_on_focal_mech_switch=plot_wfs_on_focal_mech_switch)
        # And plot probability distribution for DC vs. single force:
        figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+"DC_vs_SF_prob_dist.png")
        plot_prob_distribution_DC_vs_single_force(MTs, MTp, figure_filename=figure_filename)
        # And plot waveforms separately (if specified):
        if plot_wfs_separately_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_separate_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot(stations, wfs_dict, plot_fname)
        # And plot DAS waveforms, if specified:
        if plot_das_wfs_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_DAS_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot_das(wfs_dict, plot_fname, fs=1000.)

    elif inversion_type == "DC_single_force_no_coupling":
        full_MT_max_prob = get_full_MT_array(MT_max_prob[0:6])
        radiation_pattern_MT = MT_max_prob[0:6]
        single_force_vector_max_prob = MT_max_prob[6:9]
        amp_prop_DC = MT_max_prob[9] # Proportion of amplitude that is DC
        # Plot MT solutions and radiation pattern of most likely on sphere:
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_DC_component.png")
            plot_full_waveform_result_beachball(full_MT_max_prob, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, MTp_max_prob_value=MTp_max_prob_value, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="DC", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane, plot_uncertainty_switch=plot_uncertainty_switch, uncertainty_MTs=MTs, uncertainty_MTp=MTp, plot_wfs_on_focal_mech_switch=plot_wfs_on_focal_mech_switch)
            figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_SF_component.png")
            plot_full_waveform_result_beachball(single_force_vector_max_prob, wfs_dict, radiation_pattern_MT=single_force_vector_max_prob, MTp_max_prob_value=MTp_max_prob_value, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="single_force", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane, plot_uncertainty_switch=plot_uncertainty_switch, uncertainty_MTs=MTs, uncertainty_MTp=MTp, plot_wfs_on_focal_mech_switch=plot_wfs_on_focal_mech_switch)
        # And plot probability distribution for DC vs. single force:
        figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+"DC_vs_SF_prob_dist.png")
        plot_prob_distribution_DC_vs_single_force(MTs, MTp, figure_filename=figure_filename)
        # And plot waveforms separately (if specified):
        if plot_wfs_separately_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_separate_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot(stations, wfs_dict, plot_fname)
        # And plot DAS waveforms, if specified:
        if plot_das_wfs_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_DAS_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot_das(wfs_dict, plot_fname, fs=1000.)

    elif inversion_type == "DC_crack_couple":
        full_MT_max_prob = get_full_MT_array(MT_max_prob[0:6])
        radiation_pattern_MT = MT_max_prob[0:6]
        amp_prop_DC = MT_max_prob[-1] # Proportion of amplitude that is DC
        # Plot MT solutions and radiation pattern of most likely on sphere:
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+".png")
            plot_full_waveform_result_beachball(full_MT_max_prob, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, MTp_max_prob_value=MTp_max_prob_value, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="unconstrained", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane, plot_uncertainty_switch=plot_uncertainty_switch, uncertainty_MTs=MTs, uncertainty_MTp=MTp, plot_wfs_on_focal_mech_switch=plot_wfs_on_focal_mech_switch)
        # And plot waveforms separately (if specified):
        if plot_wfs_separately_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_separate_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot(stations, wfs_dict, plot_fname)
        # And plot Lune for solution:
        if plot_Lune_switch:
            plot_Lune(MTs[0:6,:], MTp_for_Lune, six_MT_max_prob=radiation_pattern_MT, frac_to_sample=0.1, figure_filename=os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_Lune.png")   ) 
        # And plot DAS waveforms, if specified:
        if plot_das_wfs_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_DAS_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot_das(wfs_dict, plot_fname, fs=1000.)

    elif inversion_type == "single_force_crack_no_coupling":
        full_MT_max_prob = get_full_MT_array(MT_max_prob[0:6])
        radiation_pattern_MT = MT_max_prob[0:6]
        single_force_vector_max_prob = MT_max_prob[6:9]
        amp_prop_SF = MT_max_prob[9] # Proportion of amplitude that is DC
        # Plot MT solutions and radiation pattern of most likely on sphere:
        for plot_plane in ["EN","EZ","NZ"]:
            figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_crack_component.png")
            plot_full_waveform_result_beachball(full_MT_max_prob, wfs_dict, radiation_pattern_MT=radiation_pattern_MT, MTp_max_prob_value=MTp_max_prob_value, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="unconstrained", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane, plot_uncertainty_switch=plot_uncertainty_switch, uncertainty_MTs=MTs, uncertainty_MTp=MTp, plot_wfs_on_focal_mech_switch=plot_wfs_on_focal_mech_switch)
            figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_SF_component.png")
            plot_full_waveform_result_beachball(single_force_vector_max_prob, wfs_dict, radiation_pattern_MT=single_force_vector_max_prob, MTp_max_prob_value=MTp_max_prob_value, stations=stations, lower_upper_hemi_switch="upper", figure_filename=figure_filename, num_MT_solutions_to_plot=1, inversion_type="single_force", radiation_MT_phase=radiation_MT_phase, plot_plane=plot_plane, plot_uncertainty_switch=plot_uncertainty_switch, uncertainty_MTs=MTs, uncertainty_MTp=MTp, plot_wfs_on_focal_mech_switch=plot_wfs_on_focal_mech_switch)
        # And plot probability distribution for DC vs. single force:
        figure_filename = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+"crack_vs_SF_prob_dist.png")
        plot_prob_distribution_DC_vs_single_force(MTs, MTp, figure_filename=figure_filename, inversion_type=inversion_type)
        # And plot waveforms separately (if specified):
        if plot_wfs_separately_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_separate_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot(stations, wfs_dict, plot_fname)
        # And plot Lune for solution:
        if plot_Lune_switch:
            plot_Lune(MTs[0:6,:], MTp_for_Lune, six_MT_max_prob=radiation_pattern_MT, frac_to_sample=0.1, figure_filename=os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_"+plot_plane+"_Lune.png"))
        # And plot DAS waveforms, if specified:
        if plot_das_wfs_switch:
            plot_fname = os.path.join(plot_outdir, MT_data_filename.split("/")[-1].split(".")[0]+"_DAS_wfs"+".png")
            plot_wfs_of_most_likely_soln_separate_plot_das(wfs_dict, plot_fname, fs=1000.)


    print("Full MT (max prob.):")
    print(full_MT_max_prob)
    print("(For plotting radiation pattern)")
    
    print("Finished processing unconstrained inversion data for:", MT_data_filename)
    
    print("Finished")
    

# ------------------------------------------------ End of specifing module functions ------------------------------------------------
 

# ------------------- Main script for running -------------------
if __name__ == "__main__":
    
    # Specify event and inversion type:
    for inversion_type in ["full_mt"]:#["DC", "single_force", "full_mt", "full_mt_Lune_samp", "DC_single_force_couple", "DC_single_force_no_coupling", "DC_crack_couple", "single_force_crack_no_coupling"]: #["full_mt", "DC", "single_force", "DC_single_force_couple", "DC_single_force_no_coupling", "DC_crack_couple", "single_force_crack_no_coupling"]:
        ###inversion_type = "single_force_crack_no_coupling" # can be: full_mt, DC, single_force, DC_single_force_couple, DC_single_force_no_coupling, DC_crack_couple, or single_force_crack_no_coupling.
        event_uid = "20140628143352714000" #"20140609182647874000" #"20000101000000050122" #"20000101000000050122" #"20110802061034622300" #"20180731140249174136" #"20180214185538374893" #"20090121042009185230" "20140629184210365600" #"20090121042009165190" #"20180214185538374893" #"20090121042009165190" #"20140629184210365600" #"20090121042009165190" #"20090103051541679700" #"20171222022435216400" # Event uid (numbers in FW inversion filename)
        datadir = "/raid2/tsh37/fk/python_FW_outputs/new_python_FW_outputs/Skeidararjokull_crevassing_icequakes/Skeidararjokull_crevassing_other_events_P_SV_prop/20140628143352737/FW_data_out/src_depth_0.09_200_Hz_7_stat_Z_only" #/7_stat_invert_P_S_dir_indir_amp_rat_sep" #/least_squares_result" #"/raid1/tsh37/fk/python_FW_outputs/orig_VR_method_with_individual_station_comparison_data_unnormallised_ALL_waveform_comps"
        radiation_MT_phase="P" # Radiation phase to plot (= "P" or "S")
        plot_uncertainty_switch = True #False # If True, plots uncertainty in direction/orientation of focal mechanism solution
        plot_Lune_switch = False #True # If True, plots Lune
        plot_max_prob_on_Lune_switch = False # If True, plots maximum probability solution for each point on Lune, rather than binned probability
        plot_wfs_separately_switch = True # If true, plots waveforms separately
        plot_multi_medium_greens_func_inv_switch = False #False # Set as True if plotting multi medium greens function inversion results (default = False)
        multi_medium_greens_func_inv_separate_phase_amp_ratios = False #False # If inverted for separate phase amplitude ratios for multi medium greens functions, set this as True (Default = False)
        plot_wfs_on_focal_mech_switch= False # If True, plots waveforms on focal mechanism plot (default is True)
    
        run(inversion_type, event_uid, datadir, radiation_MT_phase="P", plot_Lune_switch=plot_Lune_switch, plot_uncertainty_switch=plot_uncertainty_switch, plot_wfs_separately_switch=plot_wfs_separately_switch, plot_multi_medium_greens_func_inv_switch=plot_multi_medium_greens_func_inv_switch, multi_medium_greens_func_inv_separate_phase_amp_ratios=multi_medium_greens_func_inv_separate_phase_amp_ratios, plot_wfs_on_focal_mech_switch=plot_wfs_on_focal_mech_switch, plot_max_prob_on_Lune_switch=plot_max_prob_on_Lune_switch)

