#!/usr/bin/env python3
import dataset_utils.dataset_utils as du
import matplotlib.pyplot as plt

"""
import glob
files = glob.glob("./*/*.gt")
for f in files:
    print(f[:-3])
    #load dataset - given path
"""    
P = du.Trajectory("./cubes/t_c_2")

#plot 3D path of the robot
P.plot_path(skipstep=10)
plt.show()

#plot 6D path of the robot 
P.plot_poses()
plt.show()

#plot power readings
P.plot_voltage_current()
plt.show()

#plot power consumption
P.plot_power()
plt.show()

#plot estimated forward velocity 
P.plot_velocity()
plt.show()

#plot estimated forward velocity 
P.plot_cot()
plt.show()