#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
&author: P.Cizek
"""

import sys
import numpy as np
import pylab as pl
from scipy import interpolate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math
from pathlib import Path

DTYPE = np.float64 # default dtype for float-types

D_SMOOTHING_FACTOR = 101  #default smoothing factor for smoothing dense gait, velocity and power measurements
D_DIFF_V_EST = 5  #default time difference to compute velocity in [s]

#Hexapod parameters
D_ROBOT_WEIGHT = 2.6 #weight of the robot - default approx 2.6kg
D_G = 9.87 #gravitational acceleration

class Trajectory:
    def __init__(self, filename, robot_id=1, read_pose=True, read_imu=True, read_pow=True, read_gait=True):
        #collected data
        self.TUM = np.empty((0,8))      #ground truth 6-DOF position
        self.imu = np.empty((0,13))     #imu readings
        self.pow = np.empty((0,3))      #voltage and current readings
        self.gait = np.empty((0,38))    #gait data
        
        #calculated variables
        self.v = np.empty((0,1))        #velocity
        self.P =  np.empty((0,1))       #power consumption
        self.CoT = np.empty((0,1))      #cost of transport
        
        
        assert filename is not None, "filename for reading dataset not provided"
        
        #read the localization if exists
        gt_file = Path(filename + ".april")
        gt_file_tum = Path(filename + ".gt")
        if gt_file_tum.is_file() and read_pose is True:
            print("GT data in TUM format found - processing")
            self.TUM = readfile_gt(gt_file_tum, robot_id, form="tum")
            if self.TUM.size == 0: 
                print("  failed to load GT data")
            else:
                print("  loaded " + str(self.TUM[:,1].size) + " positions from tum log")
        elif gt_file.is_file() and read_pose is True:
            print("GT data in april format found - processing")
            self.TUM = readfile_gt(gt_file, robot_id)
            if self.TUM.size == 0: 
                print("  failed to load GT data")
            else:
                print("  loaded " + str(self.TUM[:,1].size) + " positions")
        else:
            print("GT data not found")
        
        #read the acceleration data if exists
        imu_file = Path(filename + ".imu")
        if imu_file.is_file() and read_imu is True:
            print("IMU data found - processing")
            self.imu = readfile_imu(imu_file)
            if self.imu.size == 0: 
                print("  failed to load GT data")
            else:
                print("  loaded " + str(self.imu[:,1].size) + " IMU readings")
        else:
            print("IMU data not found")
        
        #read the power log if exists
        pow_file = Path(filename + ".pow")
        if pow_file.is_file() and read_pow is True:
            print("Power data found - processing")
            self.pow = readfile_pow(pow_file)
            if self.pow.size == 0: 
                print("  failed to load power data")
            else:
                print("  loaded " + str(self.pow[:,1].size) + " voltage/current readings")
        else:
            print("Power data not found")
        
        #read the gait data
        gait_file = Path(filename + ".gait")
        if gait_file.is_file() and read_gait is True:
            print("Gait data found - processing")
            self.gait = readfile_gait(gait_file)
            if self.gait.size == 0: 
                print("  failed to load gait data")
            else:
                print("  loaded " + str(self.gait[:,1].size) + " gait readings")
        else:
            print("Gait data not found")
        
        #align the time span
        if self.TUM.size != 0 and self.pow.size != 0:
            print("aligning pose and power data")
            #align for same time span
            self.TUM, self.pow = self.align_times(self.TUM, self.pow)
            
            #calculate variables
            self.calc_power()
            self.calc_velocity()
            self.calc_cot()
    
    def align_times(self,d0,d1):
        """
        method to align two data logs in time domain
        returns arrays within the same time span 
        @input d0 .. first data log
        @input d1 .. second data log
        """
        t_min = np.max([np.min(d0[:,0]),np.min(d1[:,0])])
        t_max = np.min([np.max(d0[:,0]),np.max(d1[:,0])])
        #filter
        d0 = d0.compress(d0[:,0] > t_min, axis=0)
        d0 = d0.compress(d0[:,0] < t_max, axis=0)
        d1 = d1.compress(d1[:,0] > t_min, axis=0)
        d1 = d1.compress(d1[:,0] < t_max, axis=0)
        return d0, d1
    
    def calc_power(self, sf=D_SMOOTHING_FACTOR):
        """
        method to extract power consumption from power readings
        @input: smooth .. smoothing factor
        """
        self.P = self.pow[:,1]*self.pow[:,2]
        self.P = smooth(self.P, window_len=sf, window='flat')
    
    
    def calc_velocity(self, sf=D_SMOOTHING_FACTOR):
        """
        method to extract power consumption from power readings
        @input: smooth .. smoothing factor
        """
        for i in range(0, self.TUM.shape[0]-1):
            j = i
            while j < self.TUM.shape[0]-1 and (self.TUM[j,0] - self.TUM[i,0]) < D_DIFF_V_EST:
                j+=1
            dt = self.TUM[j,0] - self.TUM[i,0]
            ds = np.linalg.norm([self.TUM[j,1:4] - self.TUM[i,1:4]])
            self.v = np.vstack([self.v, ds/dt]) 
        self.v = np.vstack([self.v, self.v[-1]])
        self.v = self.v.reshape((self.v.shape[0],))
        self.v = smooth(self.v, window_len=sf, window='flat')
    
    def calc_cot(self):
        """
        method to calculate cost of transport
        CoT = P/mgv
        """
        P_i = interpolate.interp1d(self.pow[:,0], self.P)(self.TUM[:,0])
        
        self.CoT = np.divide(P_i,(D_G*D_ROBOT_WEIGHT*self.v))
    
    def __str__(self):
        return str(self.TUM) 
    
    #############################################################################
    ## Plotting functions
    #############################################################################
    
    def plot_path(self, clr='yellow', skipstep=1, fignum=1):
        """
        method to plot the path in 3D 
        @input: clr .. plot color
                skipstep .. plot each n-th sample
        @output: ax .. axes object
        TODO: correct aspect ratio of the graph
        TODO: pass handle of figure to the method to enable overlay drawing 
        """
        xx = [x[1] for x in self.TUM]
        yy = [x[2] for x in self.TUM]
        zz = [x[3] for x in self.TUM]

        fig = plt.figure(num=fignum)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xx[0::skipstep], yy[0::skipstep], zz[0::skipstep], color=clr)
        ax.axis('equal')
        ax.autoscale(tight=True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        
        self.set_axes_equal(ax)
        #plt.draw()
        return ax
   
    def plot_poses(self, skipstep=30, fignum=2):
        """
        method to draw 6D poses
        TODO: correct aspect ratio 
        """
        fig = plt.figure(num=fignum)
        ax = fig.gca(projection='3d')
        
        d = self.TUM[1::skipstep,:]
        v_x = np.tile(np.array([1.0,0,0]),(d.shape[0],1))
        v_y = np.tile(np.array([0,1.0,0]),(d.shape[0],1))
        v_z = np.tile(np.array([0,0,1.0]),(d.shape[0],1))
        
        for i in range(0,d.shape[0]):
            R = q2r(d[i,4:8])
            v_x[i,:] = v_x[i,:].dot(R)
            v_y[i,:] = v_y[i,:].dot(R)
            v_z[i,:] = v_z[i,:].dot(R)
        
        #plot poses
        ax.quiver(d[:,1], d[:,2], d[:,3], v_x[:,0], v_x[:,1], v_x[:,2], length=0.1, arrow_length_ratio=0.1, color='r')
        ax.quiver(d[:,1], d[:,2], d[:,3], v_y[:,0], v_y[:,1], v_y[:,2], length=0.1, arrow_length_ratio=0.1, color='g')
        ax.quiver(d[:,1], d[:,2], d[:,3], v_z[:,0], v_z[:,1], v_z[:,2], length=0.1, arrow_length_ratio=0.1, color='b')
        
        #ax.axis('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        self.set_axes_equal(ax)
        
        plt.draw()
        return ax
    
    
    def plot_voltage_current(self, skipstep=1, fignum=3):
        """
        method to plot measured voltage and current
        """
        tt = [x[0] for x in self.pow]
        u  = [x[1] for x in self.pow]
        i  = [x[2] for x in self.pow]
        
        fig, ax1 = plt.subplots(num=fignum)
        ax1.plot(tt[0::skipstep],u[0::skipstep],'r-')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Voltage [V]',color='r')
        ax1.tick_params('y',colors='r')
        
        ax2 = ax1.twinx()
        ax2.plot(tt[0::skipstep],i[0::skipstep],'b-')
        ax2.set_ylabel('Current [A]',color='b')
        ax2.tick_params('y',colors='b')
        plt.draw()
    
    def plot_power(self, skipstep=1, fignum=4):
        """
        method to plot calculated power consumption
        """
        fig, ax1 = plt.subplots(num=fignum)
        ax1.plot(self.pow[0::skipstep,0],self.P[0::skipstep],'r')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Power [W]')
        plt.draw()
    
    def plot_velocity(self, skipstep=1, fignum=5):
        """
        method to plot calculated velocity
        """
        fig, ax1 = plt.subplots(num=fignum)
        ax1.plot(self.TUM[0::skipstep,0],self.v[0::skipstep],'r')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Velocity [m/s]')
        plt.draw()
        
    def plot_cot(self, skipstep=1, fignum=6):
        """
        method to plot calculated cost of transport
        """
        fig, ax1 = plt.subplots(num=fignum)
        ax1.plot(self.TUM[0::skipstep,0],self.CoT[0::skipstep],'r')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Cost of Transport [-]')
        plt.draw()
        
    
    def set_axes_equal(self, ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
        @input: ax .. matplotlib axis, e.g., as output from plt.gca().
        """
        
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        
        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])
        
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    def save_plot(self, filename):
        """
        Saves the current figure to file
        @input: filename
        """
        plt.gcf().savefig(filename)
    
    
#############################################################################
## Reading data from files
#############################################################################

def readfile_gt(file_ref, robot_id=1, form="april"):
    """
    method to read localization gt data
    @input: filename .. full filename of the gt file 
            robot_id .. robot id 
            form  .. format of the gt data (april - default log from apriltag process tool, tum - tum formated log)
    @output: Q .. trajectory in TUM format
    """
    Q = np.empty((0,8), dtype=DTYPE)
    #open the file and parse the values
    if form == "april":
        with file_ref.open() as fp:
            for line in fp:
                q = line.split()
                
                #if robot_id to sample
                if int(q[1]) == robot_id:
                    #timestamp
                    tt_i = (float(q[2]))
                    
                    #reconstruct pose matrix
                    Q_i = np.zeros((4,4), dtype=DTYPE)
                    for i in range(0, 4):
                        for j in range(0, 4):
                            Q_i[i,j] = float(q[4*i + j + 3])
                    
                    #extract rotation
                    R_i = Q_i[0:3,0:3]
                    
                    #extract translation
                    T_i = Q_i[0:3,3].T
                    
                    #append data
                    a = np.concatenate([[tt_i], T_i, r2q(R_i)])
                    Q = np.vstack([Q, a])
    elif form == "tum":
        with file_ref.open() as fp:
            for line in fp:
                q = line.split()
                #append data
                a = np.array([float(q[0]),float(q[1]),float(q[2]),float(q[3]),float(q[4]),float(q[5]),float(q[6]),float(q[7])])
                Q = np.vstack([Q, a])

    return Q
    
def readfile_pow(file_ref):
    """
    method to read power log  data
    @input: filename .. full filename of the pow file 
    @output: pow .. power log in format: time, voltage[V], current[A]
    """
    pow = np.empty((0,3), dtype=DTYPE)
    #open the file and parse the values
    with file_ref.open() as fp:
        for line in fp:
            q = line.split()
            
            #timestamp
            tt_i = (float(q[0]) + float(q[1])/1000000000)
            #voltage
            V_i = float(q[2])*0.1
            #current
            I_i = (float(q[3])*10.0/1023.0) - 5

            #append data
            a = np.array([tt_i, V_i, I_i], dtype=DTYPE)
            pow = np.vstack([pow, a])
    return pow

def readfile_imu(file_ref):
    """
    method to read imu  data
    @input: filename .. full filename of the imu file 
    @output: pow .. imu log in format: time, orientation, rawdata
    TODO: implement this method
    """
    imu = np.empty((0,13), dtype=DTYPE)
    #open the file and parse the values
    with file_ref.open() as fp:
        for line in fp:
            q = line.split()
            
            #timestamp
            tt_i = (float(q[0]) + float(q[1])/1000000000)
            
            #append data
            a = np.array([tt_i], dtype=DTYPE)
            imu = np.vstack([imu, a])
    
    return imu

def readfile_gait(file_ref):
    """
    method to read imu  data
    @input: filename .. full filename of the imu file 
    @output: pow .. imu log in format: time, orientation, rawdata
    TODO: implement this method
    """
    gait = np.empty((0,38), dtype=DTYPE)
    #open the file and parse the values
    with file_ref.open() as fp:
        for line in fp:
            q = line.split()
            
            #timestamp
            tt_i = (float(q[0]) + float(q[1])/1000000000)
        
            #append data
            a = np.array([tt_i], dtype=DTYPE)
            gait = np.vstack([gait, a])
    
    return gait

#############################################################################
## Helper functions
#############################################################################
def q2r(q):
    """
    method to convert quaternion into rotation matrix
    @input: q .. quaternion (qx,qy,qz,qw)
    @output: R .. rotation matrix
    """
    q = q[[3, 0, 1, 2]]
    R = [[q[0]**2+q[1]**2-q[2]**2-q[3]**2,     2*(q[1]*q[2]-q[0]*q[3]),      2*(q[1]*q[3]+q[0]*q[2])],
         [2*(q[1]*q[2]+q[0]*q[3]),     q[0]**2-q[1]**2+q[2]**2-q[3]**2,      2*(q[2]*q[3]-q[0]*q[1])],
         [2*(q[1]*q[3]-q[0]*q[2]),         2*(q[2]*q[3]+q[0]*q[1]),   q[0]**2-q[1]**2-q[2]**2+q[3]**2]]
    return R

def r2q(R):
    """
    method to convert rotation matrix into quaternion
    @input: R .. rotation matrix
    @output: q .. quaternion (qx,qy,qz,qw)
    """
    q = np.array([R.trace() + 1, R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    q = 1.0/(2.0*math.sqrt(R.trace() + 1.0000001))*q
    q = q[[1, 2, 3, 0]]
    return q

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    @input: x .. the input signal 
            window_len .. the dimension of the smoothing window; should be an odd integer
            window .. the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman' flat window will produce a moving average smoothing.
    @output: y .. the smoothed signal
    """
    assert x.ndim == 1, "smooth only accepts 1 dimension arrays."
    assert x.size >= window_len, "Input vector needs to be bigger than window size."
    assert window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman'], "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    
    s = np.r_[x[int(window_len/2):0:-1],x,x[-1:-int(window_len/2)-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')
    
    y = np.convolve(s, w/w.sum(), mode='valid')
    return y

