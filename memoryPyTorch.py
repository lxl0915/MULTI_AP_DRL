#  #################################################################
#  This file contains the main DROO operations, including building DNN, 
#  Storing data sample, Training DNN, and generating quantized binary offloading decisions.

#  version 1.0 -- February 2020. Written based on Tensorflow 2 by Weijian Pan and 
#  Liang Huang (lianghuang AT zjut.edu.cn)
#  ###################################################################

from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence

print(torch.__version__)


# DNN network for memory
class MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):

        self.net = net                              #wyc: net =[4,_,_,1]
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        devicese_num = 10
        self.memory = np.zeros((self.memory_size, devicese_num, 10))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3]),
                nn.ReLU(),
                nn.Linear(self.net[3], self.net[4]),
                nn.Sigmoid()
        )

    def remember(self, h, ap_power, m_detail):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        # do not update decision, only test code
        #self.memory[idx, :] = np.hstack((h, m))
        # expand dimension for ap_power
        ap_power = np.expand_dims(ap_power, 0)
        ap_power = ap_power.repeat(h.shape[1], axis=0)
        
        
        tmp = np.hstack((h.T, ap_power))
        self.memory[idx, :] = np.hstack((tmp, m_detail))
        #print("222222222222222222222222222222")

        self.memory_counter += 1

    def encode(self, h, ap_power, m_detail):
        # encoding the entry
        self.remember(h, ap_power, m_detail)
        # train the DNN every 10 step
#        if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = torch.Tensor(batch_memory[:,:,0:8])              #wyc: shape = (128,10,4)
        m_train = torch.Tensor(batch_memory[:,:,8:])             #wyc: shape = (128,10,1)
        
        #print(f"shape is {h_train.shape} {m_train.shape}")
        

        # train the DNN
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001) 
        criterion_1 = nn.CrossEntropyLoss()
        criterion_2 = nn.BCELoss()
        
        self.model.train()
        # optimizer.zero_grad()
        # 2024.8.6:write here,now predict valuse is 128*10*2 sigmoid output, we should transfer it to 128*10*2 formal HAP select result and offloading decision, like:[[2,0],[3,1]...]
        predict = self.model(h_train)
        
        #wyc: predict.shape = (128,10,1)
        loss_1 = criterion_1(predict[:,:,0], m_train[:,:,0])    # HAP select loss(multiclassification problem)
        loss_2 = criterion_2(predict[:,:,1], m_train[:,:,1])    # Offloading loss(2-category problem)
        loss = loss_1 + loss_2
        # torch.set_grad_enabled(True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #print("11111111111111111")

        self.cost = loss.item()
        assert(self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h, ap_power, k = 1, mode = 'OP'):
        # to have batch dimension when feed into Tensor
        #h = torch.Tensor(h[np.newaxis, :])
        h = torch.Tensor(h)
        ap_power = torch.Tensor(ap_power)
        
        # padding tensor
        sentences = [h, ap_power]
        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
        
        h = padded_sentences[0]
        ap_power = padded_sentences[1]
        
        # cat two 10*4 to 10*8
        input = torch.cat((h, ap_power), dim=1)
        
        self.model.eval()
        m_pred = self.model(input)
        m_pred = m_pred.detach().numpy()
        
        # According to different HAPs, the model predictions were classified into multiple categories to be processed separately
        # store results of categorize offloading 
        m_detail = []
        m_list = []
        
        # according to the output of sigmoid, categorize offloading results into 2*HAP categorical values
        for m_one_device in m_pred:
            size = 0.2501
            detail = []
            detail.append(m_one_device[0]//size)
            detail.append(1*(m_one_device[1])>0.5)
            
            m_detail.append(detail)
        
        m_detail = np.array(m_detail)
        # Divided into multiple sub-networks
        flag_1 = 0
        for i in range(self.net[0]):
            flag_2 = 0
            # Some HAPs may have no MD selection, clear data from last cycle
            tmp_data = []
            for data in m_detail:
                if data[0]==i:
                    if flag_2 == 0:
                        flag_2 = 1
                        tmp_data = data
                    else:
                        tmp_data = np.vstack((tmp_data,data))
                        
            if len(tmp_data)>0:  
                #data = np.vstack(data for data in tmp)
                # call generate decision function for all sub-networks
                if mode is 'OP':
                    if flag_1 == 0:
                        flag_1 = 1
                        tmp_m = self.knm(tmp_data, len(tmp_data))
                    else:
                        """
                        Both the number of MDs and the number of offloading decisions were different and could not
                        be spliced, and the decision with the lower number of MDs was duplicated and used to splice
                        the overall decision (this is equivalent to continuing the decision exploration for each HAP
                        network, rather than exploring it as a whole)
                        """
                        tmp_value = self.knm(tmp_data, len(tmp_data))
                        if len(tmp_m)!= len(tmp_value):
                            if len(tmp_m) < len(tmp_value):
                                # Duplicate ndarray data
                                if isinstance(tmp_m, np.ndarray):
                                    tmp_m = tmp_m.tolist()
                                tmp_m = tmp_m*(len(tmp_value)//len(tmp_m) + 1)
                                tmp_m = np.array(tmp_m)
                                tmp_m = tmp_m[0:len(tmp_value),:]
                            elif len(tmp_m) > len(tmp_value):
                                if isinstance(tmp_value, np.ndarray):
                                    tmp_value = tmp_value.tolist()
                                tmp_value = tmp_value*(len(tmp_m)//len(tmp_value) + 1)
                                tmp_value =np.array(tmp_value)
                                tmp_value = tmp_value[0:len(tmp_m),:]
                        #print(f"tmp_value is {tmp_value} tmp_m is {tmp_m} m_detail is {m_detail}")  
                        tmp_m = np.hstack((tmp_m,tmp_value))
                        #print(f"after hstack, tmp_m is {tmp_m}")
                elif mode is 'KNN':
                    if flag_1 == 0:
                        flag_1 = 1
                        tmp_m = self.knn(tmp_data, len(tmp_data))
                    else:
                        """
                        Both the number of MDs and the number of offloading decisions were different and could not
                        be spliced, and the decision with the lower number of MDs was duplicated and used to splice
                        the overall decision (this is equivalent to continuing the decision exploration for each HAP
                        network, rather than exploring it as a whole)
                        """
                        tmp_value = self.knn(tmp_data, len(tmp_data))
                        if len(tmp_m)!= len(tmp_value):
                            if len(tmp_m) < len(tmp_value):
                                tmp_m = tmp_m*(len(tmp_value)/len(tmp_m) + 1)
                                tmp_m = tmp_m[0:len(tmp_value),:]
                            elif len(tmp_m) > len(tmp_value):
                                tmp_value = tmp_value*(len(tmp_m)/len(tmp_value) + 1)
                                tmp_value = tmp_value[0:len(tmp_m),:]
                              
                        tmp_m = np.hstack((tmp_m,tmp_value))
                else:
                    print("The action selection must be 'OP' or 'KNN'")
        
        m_list = tmp_m
        
        return m_list, m_detail


        if 0:
            if mode is 'OP':
                return self.knm(m_pred, k)
            elif mode is 'KNN':
                return self.knn(m_pred, k)
            else:
                print("The action selection must be 'OP' or 'KNN'")

    def knm(self, m, k = 1):
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        if m.ndim == 1: # If there is only one MD, then just flip the decision
            m_list = [[0],[1]]
        else:
            m_list.append(1*(m[:,1]>0.5))
            
    
            m = m[:,1]  # use offloading decision to generate other option.
            if k > 1:
                # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
                m_abs = abs(m-0.5)
                idx_list = np.argsort(m_abs)[:k-1]
                for i in range(k-1):
                    if m[idx_list[i]] >0.5:
                        # set the \hat{x}_{t,(k-1)} to 0
                        m_list.append(1*(m - m[idx_list[i]] > 0))
                    else:
                        # set the \hat{x}_{t,(k-1)} to 1
                        m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list

    def knn(self, m, k = 1):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) is 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()

