# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:23:13 2020

@author: allen
"""
import torch
import torch.nn as nn
import numpy as np
import dataloader
import trading_period
#import trading_period_by_test
import trading_period_by_test
import trading_period_by_gate
import os 
import pandas as pd
import torch
import torch.utils.data as Data
#from DL_trading import CNN_classsification1
import matplotlib.pyplot as plt
path_to_image = "C:/Users/Allen/pair_trading DL/trading_image_2016/"
path_to_average = "C:/Users/Allen/pair_trading DL/2018/averageprice/"
ext_of_average = "_averagePrice_min.csv"

path_to_minprice = "C:/Users/Allen/pair_trading DL/2018/minprice/"
ext_of_minprice = "_min_stock.csv"

path_to_2016compare = "C:/Users/Allen/pair_trading DL/compare/"
path_to_2017compare = "C:/Users/Allen/pair_trading DL/compare2017/"
path_to_2018compare = "C:/Users/Allen/pair_trading DL/newcompare2018/"
ext_of_compare = "_table.csv"

path_to_python ="C:/Users/Allen/pair_trading DL"

path_to_half = "C:/Users/Allen/pair_trading DL/2016/2016_half/"
path_to_2017half = "C:/Users/Allen/pair_trading DL/2017_halfmin/"
path_to_2018half = "C:/Users/Allen/pair_trading DL/2018_halfmin/"
ext_of_half = "_half_min.csv"

max_posion = 5

def reward(action_list):
    total_reward = 0
    total_num = 0
    profit_count = 0
    all_profit = 0
    #actions =[[0.5000000000001013, 1.6058835588357105], [1.1231567674441643, 3.009226460170205], [1.6774656461992412, 8.482170812315225], [2.434225143491954, 5.0301795963708305], [3.838786213786223, 7.405844155844149], [50, 100]]
    #actions =[[0.5000000000001013, 1.6058835588357105], [0.8422529644270367, 2.7302766798420457], [1.42874957000333, 3.312693498451958], [1.681668686169194, 8.472736714557769], [2.054041204437417, 4.680031695721116], [3.1352641629535314, 5.810311903246376], [4.378200155159055, 8.429014740108636], [5.632843137254913, 16.43431372549023], [6.8013888888889005, 13.081481481481516], [50,100]]
    actions = [[0.5000000000002669, 2.500000000000112], [0.7288428324698772, 4.0090056748083995], [1.1218344155846804, 3.0000000000002496], [1.2162849872773496, 7.4631043256997405], [1.4751902346226717, 3.9999999999997113], [1.749999999999973, 3.4999999999998117], [2.086678832116794, 6.2883211678832325], [2.193017888055368, 4.018753606462444], [2.2499999999999822, 7.500000000000021], [2.6328389830508536, 8.9762711864407], [2.980046948356806, 13.515845070422579], [3.2499999999999982, 5.500000000000034], [3.453852327447829, 11.505617977528125], [3.693027210884357, 6.0739795918367605], [4.000000000000004, 12.500000000000034], [4.151949541284411, 10.021788990825703], [4.752819548872187, 15.016917293233117], [4.8633603238866225, 7.977058029689605], [5.7367647058823605, 13.470588235294136], [6.071428571428564, 16.47435897435901], [6.408839779005503, 10.95488029465933], [7.837962962962951, 12.745370370370392], [8.772727272727282, 18.23295454545456], [9.242088607594926, 14.901898734177237], [100,200]]

    count_test = 0
    datelist = [f.split('_')[0] for f in os.listdir(path_to_2018compare)]
    #print(datelist[167:])
    profit_count = 0
    for date in sorted(datelist[129:168]): #決定交易要從何時開始
        
        table = pd.read_csv(path_to_2018compare+date+ext_of_compare)
        mindata = pd.read_csv(path_to_average+date+ext_of_average)
        tickdata = pd.read_csv(path_to_minprice+date+ext_of_minprice)
        #halfmin = pd.read_csv(path_to_half+date+ext_of_half)
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
            for i in range(25):
                if action_list[count_test] == i :
                    open, loss = actions[i][0], actions[i][1] 

            
            Bth1[2][0] = table.mu[pair]
            Bth1[0][0] = table.mu[pair] +table.stdev[pair]*loss
            Bth1[1][0] = table.mu[pair] +table.stdev[pair]*open
            Bth1[3][0] = table.mu[pair] -table.stdev[pair]*open
            Bth1[4][0] = table.mu[pair] -table.stdev[pair]*loss
            #spread = table.w1[pair] * np.log(tickdata[ str(table.stock1[pair]) ]) + table.w2[pair] * np.log(tickdata[ str(table.stock2[pair]) ])
            #print(Bth1)
            profit, opennum  = trading_period_by_gate.pairs( pair ,165,  table , mindata , tickdata , open ,open, loss ,mindata, max_posion , 0, 0 , 300000000 )
            #print(profit,opennum)
            all_profit += profit
            if profit >= 0 :
                profit_count +=1
            count_test +=1
        total_num += len(table)    
            #print(count_test)
    print("有賺錢的",profit_count)
    print("總共幾個:",total_num)
    
    #print("利潤  and 開倉次數 and 開倉有賺錢的次數/開倉次數:",total_reward ,total_num, profit_count/total_num)
    #print("開倉有賺錢次數 :",profit_count)
    return (profit_count/total_num)*100
#reward()