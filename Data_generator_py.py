#!/usr/bin/env python
# coding: utf-8

# In[5]:


import example_pendulum_cart_pendulum as example_pendulum
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
#import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import csv

from torch.autograd import Variable


# In[6]:


device = 'cuda:0'


# In[ ]:


environment = "server"
if environment == 'laptop':
    root_dir =R'C:\Users\87106\OneDrive\sindy\progress'
elif environment == 'desktop':
    root_dir = R'E:\OneDrive\sindy\progress'
elif environment == 'server':
    root_dir = R'/mnt/ssd1/stilrmy/Angle_detector/progress'
#the angle_extractor
AE_save_date = '2023-04-25'
AE_save_ver = '1'
#the angle_t_extractor
AtE_save_date = '4-21'
AtE_save_ver = '1'
#genrate path
AE_path = os.path.join(root_dir,AE_save_date,AE_save_ver,'model.pth')
AtE_path = os.path.join(root_dir,'Angle_t_extractor',AtE_save_date,AtE_save_ver,'model.pth')


# In[8]:


#initialize the Angle_extractor and load the parameters
class angle_predict(nn.Module):
    def __init__(self):
        super(angle_predict, self).__init__()
        self.fc1 = nn.Linear(2601, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 2)
    def forward(self, x):
        m = nn.ReLU()
        x = self.fc1(x)
        x = m(x)
        x = self.fc2(x)
        x = m(x)
        x = self.fc3(x)
        x = m(x)
        x = self.fc4(x)
        x = m(x)
        x = self.fc5(x) 
        return x
# AE = angle_predict()
# AE.load_state_dict(torch.load(AE_path))
# AE =AE.to(device)


# In[10]:


#initialize the Angle_t_extractor and load the parameters
class angle_t_predict(nn.Module):
    def __init__(self):
        super(angle_t_predict, self).__init__()
        self.fc1 = nn.Linear(7803, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(64, 1)
    def forward(self, x):
        m = nn.ReLU()
        x = self.fc1(x)
        x = m(x)
        x = self.fc2(x)
        x = m(x)
        x = self.fc3(x)
        x = m(x)
        x = self.fc4(x)
        x = m(x)
        x = self.fc5(x)
        x = m(x)
        x = self.fc6(x) 
        return x
# AtE = angle_t_predict()
# AtE.load_state_dict(torch.load(AtE_path))
# AtE = AtE.to(device)


# In[32]:


def image_process(sample_size,params):
    data = example_pendulum.get_pendulum_data(sample_size,params)
    image = data['x']
    angle = np.zeros(image.shape[0]-2)
    angle_t = np.zeros(image.shape[0]-2)
    angle_tt = np.zeros(image.shape[0]-2)
    '''
    for i in range(image.shape[0]-2):
        input = Variable(torch.tensor(image[i,:],dtype=torch.float32).to(device))
        temp = AE.forward(input)
        temp = temp.cpu().detach().numpy()
        angle[i] = temp[0]
        angle_tt[i] = temp[1]
    
    for i in range(image.shape[0]-2):
        input = torch.tensor(image[i:i+3,:],dtype=torch.float32).to(device)
        input = input.view(-1)
        input = Variable(input)
        temp = AtE.forward(input)
        temp = temp.cpu().detach().numpy()
        angle_t[i] = temp
    '''
    return data['z'],data['dz'],data['ddz']


# In[33]:





# In[ ]:




