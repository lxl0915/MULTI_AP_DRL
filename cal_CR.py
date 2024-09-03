import numpy as np
import random
import math

import ap_select
import optimization as op

global ap_type

# Task model for devices
class CR:
    def __init__(
        self,
        AP,
        devices,
        ap_num,
        devices_num,
        channel,
        prop_sensi_task
    ):

        self.ap_num = ap_num
        self.devices_num = devices_num
        self.channel = channel
        self.prop_sensi_task = prop_sensi_task
    
        # init devices and aps
        #self.set_AP_and_Device()
        self.AP = AP
        self.devices = devices
        
    # this function can return wecr of all devices in given 3 different offloading methods.(here is:cd, all local and all offloading.)
    def cr_for_all_devices(self, time_slot_size:1):
        # select ap with different methods
        select = []
        
        miu =0.5
        select.append(ap_select.ap_select(miu, self.devices, self.AP, self.channel, self.prop_sensi_task))
        
        # store results     
        ap_select_result_idx = 0
        # this loop will exec only once, because we just choice one ap selection method.(WADT algorithm)
        for ap_select_result in select:
            #print(f"ap selet result is: {ap_select_result}") 
            
            # storing the select results in somewhere, and now, multiple APs and devices are devided some service subnetwork(this means
            # one AP service for related devices who just selected it). 
            ap_subnet = []  # storing all subnetworks, it is a three-dimensional list:[subnetwork_idx][selected_device_num][device_idx, channel_gain]
            subnet_data = []    #storing subnetworks' data:[selected_device_num][device_idx, channel_gain]
            
            #print(f"sorted ap network is {set(sorted(set(ap_select_result)))}")
            # eg: ap selection result:0 1 3 1 1 0->0 1 3
            for ap_index in set(sorted(set(ap_select_result))):
                subnet_data = []
                device_idx = 0
                
                for result in ap_select_result:
                    select_result = []  # two-dimensional list: [device_idx_in_original_device_list][channel_gain_for_selected_special_ap]
                    
                    # effect_ap_num-iter_num: it denote the ap's index.(0,1,...,effect_ap_num)
                    if result == ap_index :
                        select_result.append(device_idx)
                        select_result.append(self.channel[device_idx][ap_index])
                        
                        subnet_data.append(select_result)

                    device_idx += 1
                    
                    #print(device_idx)
                
                #iter_num -= 1
            
                ap_subnet.append(subnet_data)      
            
            ###########################STEP 4#########################################
            # Loop over each subnetwork as follows
            ap_idx = 0
            # storing subnetwork-weighted effective task computation rate
            total_cr = [] # save different offloading method's wacr in whole wp-mec network.
            sub_effec_task_com_rate = []
            sub_total_task_com_rate = []
            # if number of offloading decision methods changed, we should update it in follow statement.
            for i in range(5):
                sub_effec_task_com_rate.append([])
                sub_total_task_com_rate.append([])
            
            total_cr_diff_sub = []
            for subnet in ap_subnet:
                #ap_subnet:[subnetwork_idx][selected_device_num][device_idx, channel_gain]
                ###########################STEP 5#########################################
                # Each subnetwork loops for offloading decision generation and time resource allocation(we can easy implement these issues by call function offer by Suzhi Bi).
                # Offloading decision generation
                h= []   # h denote channel gain that all devices related with this special ap.
                sub_device = [] # storing special devices' index in this subnet.
                
                for device in subnet:
                    sub_device.append(device[0])
                    h.append(device[1])
                    
                
                # generate different offloading decision
                off_action_list = []
                all_loc = []
                all_off = []
                for i in range(len(h)):
                    all_loc.append(0)
                    all_off.append(1)
                
                # off action generation: all local or all offloading.
                off_action_list.append(all_loc)
                off_action_list.append(all_off)

                # off action generation use coordinate descend method.
                gain0, off_action_cd = op.cd_method(np.array(h), self.AP[ap_idx]['power'])
                off_action_list.append(list(off_action_cd))
                
                # this index is used to allocate time while off_action all value is 1.
                iter_idx = 0
                total_cr_diff_off = []
                for off_action in off_action_list:
                    
                    
                    # Time resource allocation
                    # revert list to ndarray, otherwise bisection will accur error.
                    gain,a,Tj = op.bisection(np.array(h), np.array(off_action),self.AP[ap_idx]['power'], weights=[])
                    #print(f"bisection result is: a = {a} Tj = {Tj}")
                    # if offloading decsion is all off, manual set Tj.(bisection will error)
                    """
                    
                    if 0 not in set(off_action):
                        Tj = []
                        a = 0.15
                        for i in range(len(off_action)):
                            Tj.append(0.85/len(off_action))
                    """
        
                    ###########################STEP 6#########################################
                    # Get the task index of the devices served by each subnetwork (the index value of the starting task in the task list for different devices).
                    #device_num = len(subnet)
                    #task_idx_for_device = task_model.device_task_generate( device_num)
                    
                    ###########################STEP 7#########################################
                    # According to the results of STEP 5 and STEP 6, calculate effective computation rate for every subnetwork.
                    res = self.wcr_for_sub_network(h, off_action, a, Tj, self.AP[ap_idx], time_slot_size)
                    # weighted average computaion rate for different offloading methods in this sub-networks.
                    total_cr_diff_off.append(sum(res)/len(res))
                
                    iter_idx += 1

                total_cr_diff_sub.append(total_cr_diff_off)
                # save wecr to json file
                
                # ap index increase
                ap_idx += 1

            # average cr 
            total_cr_diff_sub = np.array(total_cr_diff_sub)
            for i in range(len(total_cr_diff_sub[0])):
                total_cr.append(round(sum(total_cr_diff_sub[:,i])/len(total_cr_diff_sub), 3))
            
            #print(f"--------------------wecr of different offloading is {total_cr}------------------------------")
            """
            
            write to here
            
            store total_cr for every offloading method to json file.
            
            
            """
                
            ap_select_result_idx +=1
        
        return total_cr
    
    def wcr_for_sub_network(self, h, off_action, a, Tj, ap, time_slot_size:1):
        # parameters and equations
        phi = 100   # denote the number of computation cycles needed to process one bit of raw data.
        p=ap['power']      # ap's power
        u=0.7           # energy harvesting efficiency [6]
        eta1=((u*p)**(1.0/3))/phi # η1:fixed parameter
        ki=10**-26      # denotes the computation energy efficiency coefficient of the processor’s chip [15].
        eta2=u*p/10**-10    # η2:fixed parameter, N0 = 10**-10:denotes the receiver noise power
        
        B=2*10**6       # denotes the communication bandwidth
        Vu=1.1          # vu > 1 indicates the communication overhead in task offloading.

        # total computaion rate, ingnore task execute failed and computation rate equals to offloading rate when offloading. 
        total_cr = []
        
        idx = 0
        off_device_idx = 0
        # iteration for each devices
        for i in off_action:
            # local compute
            if i == 0:
                E_i = u*p*h[idx]*a*time_slot_size
                f_i = math.pow(E_i/ki, 1.0/3) # it is equal to (E_i/ki)**(1.0/3)
                #print(f"local compute f_i is:{f_i}")
                t_i = time_slot_size
                
                local_exec_maxsize = f_i*t_i/phi
                total_cr.append(local_exec_maxsize/time_slot_size)
                
            # offload compute   
            elif i == 1:
                E_i = u*p*h[idx]*a*Tj[off_device_idx]
                P_i = E_i/(Tj[off_device_idx]*time_slot_size)
                N0 = 10**-10
                C = B*np.log((1+P_i*h[idx]/N0))
                #print(f'upload speed rate is : {C}')
                B_i = (Tj[off_device_idx]*time_slot_size*C)/Vu
                
                off_up_maxsize = B_i*Tj[off_device_idx]*time_slot_size
                total_cr.append(off_up_maxsize/time_slot_size)
                
                off_device_idx += 1
            
            # device index increase 
            idx += 1
                
        
        return total_cr
    
    def set_AP_and_Device(self):
        ap_type = np.dtype([('compu_capa','f4'), ('power', 'f4')])
        device_type = np.dtype([('compu_capa','f4')])   # devices' local compute capability are not necessary, we just set a iterable object for iterration.
        
        self.AP = np.zeros(self.ap_num, dtype= ap_type)  
        self.devices = np.zeros(self.devices_num, dtype= device_type)
        
        # set APs' attribution.
        if self.ap_num == 2:
            self.AP[0]['compu_capa'] = 2.5*10**9
            self.AP[0]['power'] = 3
        
            self.AP[1]['compu_capa'] = 1*10**9
            self.AP[1]['power'] = 7
            
        elif self.ap_num == 3:    
            self.AP[0]['compu_capa'] = 10*10**9
            self.AP[0]['power'] = 3
            
            self.AP[1]['compu_capa'] = 7.5*10**9
            self.AP[1]['power'] = 4
            
            self.AP[2]['compu_capa'] = 5*10**9
            self.AP[2]['power'] = 5
        else:
            self.batch_ap_attribution_set()
            
    # Batch set ap attributes within reasonable limits
    # compu_capa:1~15GHZ power:2~10W
    def batch_ap_attribution_set(self):
        ap_compu_capa_random = np.random.randint(5, 20, self.ap_num) 
        ap_power_random = np.random.randint(2, 6, self.ap_num)
    
        for i in range(self.ap_num):
            self.AP[i]['compu_capa'] = int(ap_compu_capa_random[i])*10**9
            self.AP[i]['power'] = ap_power_random[i]
            
        return self.AP