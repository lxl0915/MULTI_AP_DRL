
import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy

# Implementated based on the PyTorch 
from memoryPyTorch import MemoryDNN
from optimization import bisection

import time
import json
import random

def plot_rate(rate_his, color = 'b', rolling_intv=50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))
    rolling_intv = 20

    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), color)
    #plt.fill_between(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = color, alpha = 0.2)
    plt.ylabel('Normalized Computation Rate')
    plt.xlabel('Time Frames')
    plt.show()

def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == "__main__":
    '''
        This algorithm generates K modes from DNN, and chooses with largest
        reward. The mode with largest reward is stored in the memory, which is
        further used to train the DNN.
        Adaptive K is implemented. K = max(K, K_his[-memory_size])
    '''

    N = 10                       # number of users
    n = 15000                    # number of time frames
    K = N                        # initialize K = N
    decoder_mode = 'OP'          # the quantization mode could be 'OP' (Order-preserving) or 'KNN'
    Memory = 1024                # capacity of memory structure
    Delta = 32                   # Update interval for adaptive K

    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))
    # Load data
    if 0:
        channel = sio.loadmat('./data/data_%d' %N)['input_h']
        print(f"channel is {channel}")
        print(f"channel[0] is {channel[0]}")
    if 1:
        with open("data.json") as json_file:
            data = json.load(json_file)
            json_file.close()
            
            #print(data["channel"])
            channel = data["channel"]
            channel = np.array(channel)
            print(f"type of list is {type(channel)}")
            print(f"len of list is {len(channel)}")
            # load data interested
            wacr = data["wacr"]
            ap_power = data["ap_power"]
            # shuffle all the data
            random.shuffle(channel)
            random.shuffle(wacr)
            random.shuffle(ap_power)
            
            ap_power = np.array(ap_power)
            
            #print(f"the nummber of channel is {len(data["channel"])}")
            
            print(channel[1])
            json_file.close()
        
        #print(f"channel is:{channel}")
        #print(f"channel[0] is {channel[0]}")
        #print(f"len of channel[0] is {len(channel[0])}")
    #rate = sio.loadmat('./data/data_%d' %N)['output_obj'] # this rate is only used to plot figures; never used to train DROO.

    # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel * 1000000

    # generate the train and test data sample index
    # data are splitted as 80:20
    # training data are randomly sampled with duplication if n > total data size
    channel_test = channel[0:3000] #only set here, the iteration and dataset is correct.
    wacr_test = wacr[0:3000]
    ap_power_test = ap_power[0:3000]

    channel = channel[-15000:] #only set here, the iteration and dataset is correct.
    wacr = wacr[-15000:]
    ap_power = ap_power[-15000:]
    
   
    
    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n)) # training data size


    mem = MemoryDNN(net = [8, 80, 140, 80,2],      # mem = MemoryDNN(net = [N, 120, 80, N],update output level has 2 neural(2 array),it denote HAP select result and offloading decision for all devices.
                    learning_rate = 0.0001,
                    training_interval=10,
                    batch_size=64,
                    memory_size=Memory
                    )

    start_time = time.time()

    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_idx_his = []
    K_his = []
    for i in range(n):
        print(f"iter is {i}")
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        if i> 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(k_idx_his[-Delta:-1]) +1;
            else:
                max_k = k_idx_his[-1] +1;
            K = min(max_k +1, N)

        if i < n - num_test:
            # training
            i_idx = i % split_idx
        else:
            # test
            i_idx = i - n + num_test + split_idx

        #h = channel[i_idx,:]
        #print(f"channel[{i_idx},:] is {channel[i_idx,:]}")
        # now channel is 3-dimension 
        h = channel[i_idx,:,:]
        ap_power_idx = ap_power[i_idx,:]
        #print(f"len of h is{len(h)}*{len(h[0])}")
        #revert ndarray 10*N list to N*10, then this can times 10*X 
        #h = h.T	#转置
        #print(f"len of revert h is{len(h)}*{len(h[0])}")

        # the action selection must be either 'OP' or 'KNN'
        m_list, m_detail = mem.decode(h, ap_power_idx, K, decoder_mode)

        r_list = []
        h = h.T	#转置
        """
        For all the offloading decisions, the sub-HAP calculates their computation rates, 
        selects the largest as the offloading decision of the sub-network, and finally updates
        the offloading decisions in the replay buffer
        """
        for m in range(len(m_list)):
            tmp_total_r = 0
            for i in range(len(h)):
                for j in range(len(m_detail)):
                    sub_h = []
                    sub_m = []
                    sub_r_list = []
                    
                    
                    if m_detail[j][0] == i:
                        sub_m.append(m_detail[j][1])
                        sub_h.append(h[i][j])
                    tmp_r = 0
                    if len(sub_m) > 0:
                        #sub_r_list.append(bisection(sub_h/1000000, sub_m))
                        ret = bisection(np.array(sub_h)/1000000, np.array(sub_m), ap_power_idx[i])[0]
                        if ret > tmp_r:
                            # If the subnetwork has a larger computation rate with this offloading decision, update the offloading decision (replay buffer).
                            tmp_r = ret
                            m_detail[j][1] = m_list[m][j]
                    tmp_total_r += tmp_r    
            r_list.append(tmp_total_r)             
          
        mem.encode(h, ap_power_idx, m_detail)
        # the main code for DROO training ends here




        # the following codes store some interested metrics for illustrations
        # memorize the largest reward
        rate_his.append(np.max(r_list))
        #rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        tmp =np.array(wacr[i_idx])
        rate_his_ratio.append((rate_his[-1]/N) / np.max(tmp))
        # record the index of largest reward
        k_idx_his.append(np.argmax(r_list))
        # record K in case of adaptive K
        K_his.append(K)
        mode_his.append(m_list[np.argmax(r_list)])


    total_time=time.time()-start_time
    mem.plot_cost()
    plot_rate(rate_his_ratio)

    print("Averaged normalized computation rate:", sum(rate_his_ratio[-num_test: -1])/num_test)
    print('Total time consumed:%s'%total_time)
    print('Average time per channel:%s'%(total_time/n))

    # save data into txt
    save_to_txt(k_idx_his, "k_idx_his.txt")
    save_to_txt(K_his, "K_his.txt")
    save_to_txt(mem.cost_his, "cost_his.txt")
    save_to_txt(rate_his_ratio, "rate_his_ratio.txt")
    save_to_txt(mode_his, "mode_his.txt")
    
    
    
    #call model
    test_rate = []
    test_rate_ratio = []
    for i in range(len(ap_power_test)):
        h = channel_test[i]
        ap_power_idx = ap_power_test[i]

        # the action selection must be either 'OP' or 'KNN'
        m_detail = mem.decode(h, ap_power_idx, K, decoder_mode)[1]
        h = h.T	#转置
        """
        For all the offloading decisions, the sub-HAP calculates their computation rates, 
        selects the largest as the offloading decision of the sub-network, and finally updates
        the offloading decisions in the replay buffer
        """
       
        tmp_total_r = 0
        for i in range(len(h)):
            for j in range(len(m_detail)):
                sub_h = []
                sub_m = []
                sub_r_list = []
                if m_detail[j][0] == i:
                    sub_m.append(m_detail[j][1])
                    sub_h.append(h[i][j])
    
                if len(sub_m) > 0:
                    #sub_r_list.append(bisection(sub_h/1000000, sub_m))
                    ret = bisection(np.array(sub_h)/1000000, np.array(sub_m), ap_power_idx[i])[0]
                    tmp_total_r += ret    
        test_rate.append(tmp_total_r) 
        print(test_rate)
        tmp =np.array(wacr_test[i])
        test_rate_ratio.append((tmp_total_r/N) / np.max(tmp))
        
    print("Test averaged normalized computation rate:", sum(test_rate_ratio)/len(test_rate_ratio))
    plot_rate(test_rate_ratio,color = 'r')