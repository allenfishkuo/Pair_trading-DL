# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:48:00 2020

@author: allen
"""
import numpy as np
import time
import sys
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
ext_of_check = "_check.csv"
ext_of_groundtruth  = "_ground truth.csv"

path_to_python ="C:/Users/Allen/pair_trading DL"
path=os.path.dirname(os.path.abspath(__file__))+'/results/'
path_to_check ='C:/Users/Allen/pair_trading DL/check_kmean/'
path_to_20action ='C:/Users/Allen/pair_trading DL/20action Kmean2018/'

def check_equal():
    datelist = [f.split('_')[0] for f in os.listdir(path_to_check)]

    dif_reward = 0
    for date in sorted(datelist):
        count = 0
        gt = pd.read_csv(path_to_20action+date+ext_of_groundtruth,usecols=["reward"])
        check = pd.read_csv(path_to_check+date+ext_of_check,usecols=["reward"])
        gt = gt.values
        check = check.values
        reward_index =[0]*len(gt)
        num = np.arange(1,len(gt)+1,1)
        if date == "20180102":
            for i in range(gt.shape[0]): 
                reward_index[count] = abs(check[i,0] - gt[i,0])/gt[i,0]*100
                if reward_index[count] != 0 :
                  dif_reward += 1  
                count+=1
                
            reward_index = np.array(reward_index)
            plt.bar(num,reward_index)
            plt.xlabel("20180102 stock pairs")
            plt.ylabel("reward error")
            plt.xlim(1,len(gt)+1)
            #plt.ylim(0, 4) 
            print("不同reward 數",dif_reward)
    
check_equal()