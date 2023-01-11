import pickle
import os
import matplotlib.pyplot as plt
import numpy as np 

#load pickle file from data 
curr_dir = os.getcwd()
print("Current directory: ", curr_dir)
folder_dir = curr_dir+'/data/'
pkl_name = 'uav_location.pkl'
with open(folder_dir+pkl_name, 'rb') as handle:
    data = pickle.load(handle)


plt.close('all')
idx_parse = 1
uav_location = data['uav_location'][0:10]
effector_position = data['effector_location'][0:10]
uav_wing = data['uav_wing_location'][0:10]

#combine uav_location into one array
uav_array = np.asarray(uav_location)
uav_wing = np.asarray(uav_wing)

#3d plot of effector position 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#set equal axis

if idx_parse == 0 :
    ax.plot(effector_position[:,0], effector_position[:,1], 
        effector_position[:,2], c='r', marker='o')
    
    ax.scatter(uav_array[0], uav_array[1], 
            uav_array[2], label='uav')
    
    ax.scatter(uav_wing[0], uav_wing[1],
            uav_wing[2], label='uav_wing')  

else:
    for effect in effector_position:
        ax.plot(effect[:,0], effect[:,1], 
            effect[:,2], c='r', marker='o')

    ax.plot(uav_array[:,0], uav_array[:,1], 
        uav_array[:,2], c='b', marker='o', label='uav')

    ax.plot(uav_wing[:,0], uav_wing[:,1],
        uav_wing[:,2], c='g', marker='o', label='uav_wing')


ax.axis('equal')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

