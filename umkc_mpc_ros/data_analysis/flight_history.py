#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:10:00 2023

@author: justin
"""


#load pickle file
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def get_state_control_info(solution):
    """get actual position and control info"""
    control_info = [np.asarray(control[0]) for control in solution]

    state_info = [np.asarray(state[1]) for state in solution]

    return control_info, state_info


def get_info_history(info: list, n_info: int) -> list:
    """get actual position history"""
    info_history = []

    for i in range(n_info):
        states = []
        for state in info:
            states.append(np.float64(state[i][0]))
        info_history.append(states)

    return info_history


def get_info_horizon(info: list, n_info: int) -> list:
    """get actual position history"""
    info_horizon = []

    for i in range(n_info):
        states = []
        for state in info:

            states.append(state[i, 1:])
        info_horizon.append(states)

    return info_horizon


#load pickle file from data 
curr_dir = os.getcwd()
print("Current directory: ", curr_dir)
folder_dir = curr_dir+'/data/'
with open(folder_dir+'history.pkl', 'rb') as handle:
    data = pickle.load(handle)

#load data
#remove first 10 seconds of data
index_parse = 5
control_ref_history = data['control_ref_history'][index_parse:]
obstacle_history = data['obstacle_history'][index_parse:]
state_history = data['state_history'][index_parse:]
trajectory_ref_history = data['trajectory_ref_history'][index_parse:]
time_history = data['time_history'][index_parse:]

idx = 1
phi_rate_cmd = np.rad2deg([np.float64(control[0][idx]) for control in control_ref_history])
theta_rate_cmd = np.rad2deg([np.float64(control[1][idx]) for control in control_ref_history])
psi_rate_cmd = np.rad2deg([np.float64(control[2][idx]) for control in control_ref_history])
airspeed_cmd = np.rad2deg([np.float64(control[3][idx]) for control in control_ref_history])

#plot x y z
# go through list of lists and get the x y z position

state_ref_history = get_info_history(trajectory_ref_history, 6)

x_history = [state[0] for state in state_history]
y_history = [state[1] for state in state_history]
z_history = [state[2] for state in state_history]
phi_history = np.rad2deg([state[3] for state in state_history])
theta_history = np.rad2deg([state[4] for state in state_history])
psi_history = np.rad2deg([state[5] for state in state_history])
airspeed_history = [state[6] for state in state_history]


#plot x y z 3d plot
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_history, y_history, z_history)

#plot start and end points
ax.scatter(x_history[0], y_history[0], z_history[0], c='r', marker='o', label='start')
ax.scatter(x_history[-1], y_history[-1], z_history[-1], c='g', marker='o', label='end')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#set legend
ax.legend()

#%% Plot 3 subplot of roll pitch yaw

fig, axs = plt.subplots(4, 1)
#share x axis
fig.subplots_adjust(hspace=0)
axs[0].plot(time_history, phi_history, label='phi')
axs[0].plot(time_history, phi_rate_cmd, label='phi rate cmd')
#set y label
axs[0].set_ylabel('phi (deg)')

axs[1].plot(time_history, theta_history, label='theta')
axs[1].plot(time_history, theta_rate_cmd, label='theta rate cmd')
axs[1].set_ylabel('theta (deg)')


axs[2].plot(time_history, psi_history, label='psi')
#axs[2].plot(psi_rate_cmd, label='psi rate cmd')
axs[2].set_ylabel('psi (deg)')

axs[3].plot(time_history, airspeed_history, label='airspeed')
axs[3].set_ylabel('airspeed (m/s)')

#set legend
axs[0].legend()
axs[1].legend()
axs[2].legend()

