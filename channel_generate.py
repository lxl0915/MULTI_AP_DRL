import numpy as np                         # import numpy
import pandas as pd
import matplotlib.pyplot as plt

import math
import os
import csv
import random
import json

import cal_CR as cr

# store devices and APs' coordinate to json        
def save_data_to_json(data,file_name):
    with open(file_name, 'a+', newline= '') as f:
        json.dump(data, f)
        f.write('\n')
        f.close()

def dist_status_generate(devices_num, ap_num, length, ap_distribution_rule, device_distribution_rule, save_flags=0):
    # if set same random speed, every loop will generate same random values.
    #np.random.seed(50)
    
    #print(f"ap_distribution_rule is {ap_distribution_rule}")
    #print(f"device_distribution_rule is {device_distribution_rule}")
    
    if device_distribution_rule:
        # randomly generate devices' coordinate.
        array_size = (devices_num, 2)
        
        #devices_array = np.random.normal(loc = 0.5, scale = 0.4, size = array_size)    # 正态分布的5个随机数组
        devices_array = np.random.random(size=array_size)
        devices_array = np.multiply(devices_array, length)
    else:
        # self-definition devices' coordinate.
        devices_array = []
        
        # uniform generate devices' coordinate.
        for i in range(round(math.sqrt(devices_num))): 
            for j in range(round(math.sqrt(devices_num))):
                device_coor = [length/round(math.sqrt(devices_num))*(i),length/round(math.sqrt(devices_num))*(j)]
                devices_array.append(device_coor)
        if len(devices_array) < devices_num:
            for i in range(devices_num-len(devices_array)):
                device_coor = [round(random.uniform(0, length)),round(random.uniform(0, length))]
                devices_array.append(device_coor)
                
        #device_coor = [(length/(devices_num+2))*(i+1),(length/(devices_num+2))*(i+1)]
        #devices_array.append(device_coor)
    
    
    
    if ap_distribution_rule:
        # randomly generate APs' coordinate.
        array_size = (ap_num, 2)
        APs_array = np.random.random(array_size)
        APs_array = np.multiply(APs_array, length)
    
    else:
        # self-definition APs' coordinate.
        APs_array = []
        for i in range(ap_num):
            '''
            ap_coor = [(length/(ap_num+2))*(i+1),(length/(ap_num+2))*(i+1)]
            APs_array.append(ap_coor)
            '''
            ap_coor = [round(length/2,2),round((length/(ap_num+20))*(i+9),2)]
            APs_array.append(ap_coor)
    
    """
    print("devices coordinate array is:")
    print(devices_array)
    
    print("APs coordinate array is:")
    print(APs_array)
    """
    
    distance = []   # distance:[devices_num][ap_num]
    
    for devices in devices_array:    
        distance_to_ap = []
        for AP in APs_array:
            x_idx = math.fabs(devices[0]-AP[0])
            y_idx = math.fabs(devices[1]-AP[1])
            # calculate the distance and store it to small list
            distance_to_ap.append(math.sqrt(x_idx**2 + y_idx**2))
            
        distance.append(distance_to_ap)
    
        # The return list represents the distance between different devices and APs, it is a two-dimensional list
 
    return devices_array,APs_array,distance
    
def channel_gain_get(distance, A_d, d_e, f_c):
    #print(f"distance are {distance}")
    # according to the reference's formulation, get the channel gains for all devices
    math.pow(100, 2)    
    gain_list = []  # gain_list:[devices_num][ap_num]
    for device_distance in distance:
        dist_list = []
        idx = 0 
        for dist_i in device_distance:
            # generate rayleigh distribution
            ray = np.random.rayleigh(1, len(device_distance))
            # get to the channel gain
            dist_list.append( ray[idx]*A_d*math.pow((3*10**8)/(4*3.1415926*f_c*dist_i),d_e))
            
            idx += 1
        
        # store channel gain for all devices
        gain_list.append(dist_list)

    # return channel gains list, it is a two-dimensional list, dimension is:[devices_num][ap_num]
    return gain_list      

def set_AP_and_Device(ap_num,devices_num):
        ap_type = np.dtype([('compu_capa','f4'), ('power', 'f4')])
        device_type = np.dtype([('compu_capa','f4')])   # devices' local compute capability are not necessary, we just set a iterable object for iterration.
        
        AP = np.zeros(ap_num, dtype= ap_type)  
        devices = np.zeros(devices_num, dtype= device_type)
        
        # set APs' attribution.
        # Batch set ap attributes within reasonable limits
        # compu_capa:1~15GHZ power:2~10W
        ap_compu_capa_random = np.random.randint(5, 20, ap_num) 
        ap_power_random = np.random.randint(2, 10, ap_num)

        for i in range(ap_num):
            AP[i]['compu_capa'] = int(ap_compu_capa_random[i])*10**9
            AP[i]['power'] = ap_power_random[i]
            
        return AP,devices
        

if __name__ == "__main__":
    length = 15                        # the length of the side of the square area in which the device and the AP are located
    devices_num = 10                   # number of users
    ap_num = 4                          # numbers of AP   
    devices_array,APs_array,distance = dist_status_generate(devices_num, ap_num, length, 1, 1)
    
    A_d = 4.11
    d_e = 2.8
    f_c = 915*10**6 
    channel_gain = channel_gain_get(distance, A_d, d_e, f_c)
    #print(f"channe gain is:\n {channel_gain}.")
    
    # save channel and ap, devices' attribution.
    channel = []
    ap_power = []
    
    
    #generate 10000 time slots channel, each distribution for 500
    for i in range(300):
        print(f"this is {i} iterration.")
        # generate distribution and set ap, devices' attribution.
        AP,devices = set_AP_and_Device(ap_num,devices_num)
        #print(f"AP is {AP}")
        
       
        devices_array,APs_array,distance = dist_status_generate(devices_num, ap_num, length, 1, 1)
        for j in range(100):
            channel_gain = channel_gain_get(distance, A_d, d_e, f_c)
            channel.append(channel_gain)
            
            # convert ndarray to normal python data type.
            tmp = list(AP[:]['power'])
            for i in range(len(tmp)):
                tmp[i] = int(tmp[i])
            
            ap_power.append(tmp)
            
    # calculate different offloading methods' wacr for given aps, devices and channel.
    # test cal_CR.py  
    dataset_wacr = [] 
    for device_channel in channel:
        
        cr_model = cr.CR( AP = AP,
        devices = devices,
        ap_num =ap_num,
        devices_num = devices_num,
        channel = device_channel,
        prop_sensi_task = 0.5)
        
        wacr = cr_model.cr_for_all_devices(time_slot_size = 1)
        dataset_wacr.append(wacr)
        #print(f"wacr is {wacr}")

    
    # dump channel to json
    dic_dataset = {"channel":channel,"wacr":dataset_wacr,"ap_power":ap_power}
    with open('data.json', 'w+') as f:
        json.dump(dic_dataset, f)
        
        f.close()
     
    