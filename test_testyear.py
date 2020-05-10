# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:59:37 2020

@author: allen
"""
import torch
import torch.nn as nn
import numpy as np
import dataloader
import trading_period
import trading_period_by_test
import os 
import pandas as pd

path_to_average = "C:/Users/Allen/pair_trading DL/2016/averageprice/"
ext_of_average = "_averagePrice_min.csv"
path_to_minprice = "C:/Users/Allen/pair_trading DL/2016/minprice/"
ext_of_minprice = "_min_stock.csv"
path_to_compare = "C:/Users/Allen/pair_trading DL/compare2017/"
ext_of_compare = "_table.csv"
path_to_python ="C:/Users/Allen/pair_trading DL"
path_to_groundtruth = "C:/Users/Allen/pair_trading DL/ground truth trading period/"
ext_of_groundtruth = "_ground truth.csv"
path_to_choose = "C:/Users/Allen/pair_trading DL/6action Kmeans/"
max_posion = 5

def test_testyear():
    total_reward = 0
    total_num = 0
    
    #count_test =[0,0,0,0,0,0]
    actions =[[0.5000000000001013, 1.6058835588357105], [1.1231567674441643, 3.009226460170205], [1.6774656461992412, 8.482170812315225], [2.434225143491954, 5.0301795963708305], [3.838786213786223, 7.405844155844149], [50, 100]]
    #counts = [0,0,0,0,0]
    datelist = [f.split('_')[0] for f in os.listdir(path_to_compare)]
    #print(datelist)
    #count = 0
    for date in sorted(datelist):
        
        table = pd.read_csv(path_to_compare+date+ext_of_compare)
        mindata = pd.read_csv(path_to_average+date+ext_of_average)
        tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice)
        gt = pd.read_csv(path_to_choose+date+ext_of_groundtruth,usecols=["action choose"])
        gt = gt.values
        #print(tickdata.shape)
        tickdata = tickdata.iloc[165:]
        tickdata.index = np.arange(0,len(tickdata),1)
        os.chdir(path_to_python)    
        num = np.arange(0,len(table),1)
        for pair in num: #看table有幾列配對 依序讀入

            #action_choose = 0
            #spread =  table.w1[pair] * np.log(mindata[ str(table.stock1[pair]) ]) + table.w2[pair] * np.log(mindata[ str(table.stock2[pair]) ])
            #spread = spread.T.to_numpy()
            #print(spread)
            Bth1 = np.ones((5,1))
            if gt[pair] == 0 :
                open, loss = actions[0][0], actions[0][1]
                
            elif gt[pair] == 1 :
                open, loss = actions[1][0], actions[1][1]
            elif gt[pair] == 2 :
                open, loss = actions[2][0], actions[2][1]
            elif gt[pair] == 3 :
                open, loss = actions[3][0], actions[3][1]
            elif gt[pair] == 4 :
                open, loss = actions[4][0], actions[4][1]
            elif gt[pair] == 5 :
                open, loss = actions[5][0], actions[5][1]    
            
            
            Bth1[2][0] = table.mu[pair]
            Bth1[0][0] = table.mu[pair] +table.stdev[pair]*loss
            Bth1[1][0] = table.mu[pair] +table.stdev[pair]*open
            Bth1[3][0] = table.mu[pair] -table.stdev[pair]*open
            Bth1[4][0] = table.mu[pair] -table.stdev[pair]*loss
            #print(Bth1)
            #plotB1 = np.ones((5,len(s)))*Bth1
            open,loss = 0.75,2.5
            profit,num  = trading_period_by_test.pairs( pair ,165,  table , mindata , tickdata , open ,open, loss ,mindata, max_posion , 0.003, 0.005 , 30000000 )
            #print(profit, open_num)
            total_reward += profit
            #print("profit :",profit)
            total_num += num
            #count_test +=1
            #print(count_test)
    print("利潤  and 開倉次數 :",total_reward ,total_num)
            #print("開倉次數 :",open_num)
test_testyear()