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
import trading_period_by_test

import trading_period_by_gate
import os 
import pandas as pd
import torch
import torch.utils.data as Data
from DL_trading import CNN_classsification1
import matplotlib.pyplot as plt
path_to_image = "C:/Users/Allen/pair_trading DL/negative profit of 2018/"
path_to_average = "C:/Users/Allen/pair_trading DL/2018/averageprice/"
ext_of_average = "_averagePrice_min.csv"

path_to_minprice = "C:/Users/Allen/pair_trading DL/2018/minprice/"
ext_of_minprice = "_min_stock.csv"

path_to_2016compare = "C:/Users/Allen/pair_trading DL/compare/"
path_to_2017compare = "C:/Users/Allen/pair_trading DL/compare2017/"
path_to_2018compare = "C:/Users/Allen/pair_trading DL/compare2018/"
ext_of_compare = "_table.csv"

path_to_python ="C:/Users/Allen/pair_trading DL"

path_to_half = "C:/Users/Allen/pair_trading DL/2016/2016_half/"
path_to_2017half = "C:/Users/Allen/pair_trading DL/2017_halfmin/"
path_to_2018half = "C:/Users/Allen/pair_trading DL/2018_halfmin/"
ext_of_half = "_half_min.csv"

max_posion = 5

def test_reward():
    total_reward = 0
    total_num = 0
    action_list=[]
    #actions =[[0.5000000000001013, 1.6058835588357105], [1.1231567674441643, 3.009226460170205], [1.6774656461992412, 8.482170812315225], [2.434225143491954, 5.0301795963708305], [3.838786213786223, 7.405844155844149], [50, 100]]
    #actions =[[0.5000000000001013, 1.6058835588357105], [0.8422529644270367, 2.7302766798420457], [1.42874957000333, 3.312693498451958], [1.681668686169194, 8.472736714557769], [2.054041204437417, 4.680031695721116], [3.1352641629535314, 5.810311903246376], [4.378200155159055, 8.429014740108636], [5.632843137254913, 16.43431372549023], [6.8013888888889005, 13.081481481481516], [50,100]]
    actions = [[0.5000000000001013, 1.6058835588357105], [0.6489266547405999, 5.516100178890889], [0.7499999999999054, 2.9999999999997486], [0.9210140679952115, 2.499999999999602], [1.2193892045453651, 3.516477272726988], [1.539372976155364, 7.9955843391227885], [1.7500000000000067, 2.9999999999999147], [1.8311055731762798, 9.058223833257852], [2.356606317411384, 4.500000000000039], [2.5559701492537314, 11.481343283582085],
           [2.9682769367764976, 5.522039180765841], [3.5163502109704545, 6.585970464135011], [4.110335195530733, 8.15502793296091], [4.496926229508203, 17.016393442622984], [4.985197368421058, 13.006578947368432], [5.822303921568625, 9.791666666666659], [5.836683417085425, 15.46733668341711], [7.2098070739549875, 12.65916398713829], [9.019366197183082, 16.031690140845093], [50, 100]]
    #print(actions[0][0])
    #Net = CNN_classsification1()
    #print(Net)
    Net = torch.load('3ResNet20170101_20180931.pkl')
    Net.eval()
    #print(Net)
    whole_year = dataloader.test_data()
    whole_year = torch.FloatTensor(whole_year).cuda()
    #print(whole_year)
    torch_dataset_train = Data.TensorDataset(whole_year)
    whole_test = Data.DataLoader(
            dataset=torch_dataset_train,      # torch TensorDataset format
            batch_size = 1024,      # mini batch size
            shuffle = False,               
            )
    for step, (batch_x,) in enumerate(whole_test):
        #print(batch_x)
        output = Net(batch_x)               # cnn output
        _, predicted = torch.max(output, 1)
        action_choose = predicted.cpu().numpy()
        action_choose = action_choose.tolist()
        action_list.append(action_choose)
   # action_choose = predicted.cpu().numpy()
    action_list =sum(action_list, [])

    
    count_test = 0
    datelist = [f.split('_')[0] for f in os.listdir(path_to_2018compare)]
    #print(datelist[167:])
    profit_count = 0
    for date in sorted(datelist[168:]): #決定交易要從何時開始
        
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
            for i in range(20):
                if action_list[count_test] == i :
                    open, loss = actions[i][0], actions[i][1] 

            
            Bth1[2][0] = table.mu[pair]
            Bth1[0][0] = table.mu[pair] +table.stdev[pair]*loss
            Bth1[1][0] = table.mu[pair] +table.stdev[pair]*open
            Bth1[3][0] = table.mu[pair] -table.stdev[pair]*open
            Bth1[4][0] = table.mu[pair] -table.stdev[pair]*loss
            spread = table.w1[pair] * np.log(tickdata[ str(table.stock1[pair]) ]) + table.w2[pair] * np.log(tickdata[ str(table.stock2[pair]) ])
            #print(Bth1)
            profit, opennum  = trading_period_by_gate.pairs( pair ,165,  table , mindata , tickdata , open ,open, loss ,mindata, max_posion , 0, 0.01 , 300000000 )
            if profit > 0 and opennum == 1 :
                profit_count +=1
                print("有賺錢的pair",profit)
                
                
            elif opennum ==1 and profit < 0 :
                
                print("賠錢的pair :",profit)
                
                plt.figure()
                plt.plot(spread)
                plotB1 = np.ones((5,len(spread)))*Bth1
                plt.plot(range(len(spread)),plotB1.T,"--")
                plt.title("Trade action choose open / loss :"+str(open)+"/"+str(loss)+" and Profit :"+str(profit))
            
                plt.savefig(path_to_image+" number {} Pair Trading DL.png".format(count_test))
                plt.show()
                plt.close()
                
            total_reward += profit            
            total_num += opennum
            count_test +=1
            
            #print(count_test)
    print("利潤  and 開倉次數 and 開倉有賺錢的次數/開倉次數:",total_reward ,total_num, profit_count/total_num)
    print("開倉有賺錢次數 :",profit_count)
    
          
#test()