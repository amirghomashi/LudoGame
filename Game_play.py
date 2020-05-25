#!/usr/bin/env python

#This program provide for course Tools of Artificial Intelligence from SDU University 2020
#The main purpose of this class to run game with paramter such as dicount factor, learning rate , etc.

import ludopy
import numpy as np
from tqdm import tqdm
import os 
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
from train import train_data
from captur import capture_image
from Qplay import Q_play
from os import system
from visualize import visualize_data
from datetime import datetime

class Game_run():
    def __init__(self):
        self.st=Q_play()
        self.perecnt_total=[]
        self.v=visualize_data()
        self.set_number=0
        self.avarage_t=[]

        pass

    def run_call(self,game_count,play_selct,gamma,alfa,epsilon_n,n_file,set_number):
        """
            this method is used to run the game with parameter is define    """
        start=datetime.now()
        self.set_number=set_number
        self.start(game_count,play_selct,0.7,0.7,0.0,1)
        
        ## we define Game play with range with diffrent save file
        # for i in range(4):
        #     self.start_train(game_count,play_selct,0.9,0.9,0.0,i+18)
            
        print("avarage for training is",self.avarage_t)
        end=datetime.now()-start
        print("time over :",end)

    

    def start_train(self,game_count,play_selct,gamma,alfa,epsilon_n,n_file):
        """
            this method is used to crate the Q-table and other file needs to run the game     """

        file_path=f"output/New_data/train_new10.npy"
        file_path_1=f"output/New_data/test_1.npy"
        file_path_2=f"output/New_data/testData_"+str(n_file)+".npy"
        file_path_3=f"output/New_data/test_"+str(n_file)+".txt"
        #file_path_2=f"output/Data/test_train"+n_file+".txt"

        data_2=[]
        data=np.zeros((60, 7))
        if os.path.exists(file_path):
            print("file exsit.....")
        else:
            np.save(file_path,data)
           
        np.save(file_path_2,data_2)
        self.st.file_name=file_path

        file_data_1=np.load(file_path_2,allow_pickle=True)

        inty=int(game_count/self.set_number)
        for i in range(self.set_number):
            self.st.run_game_with_qtable(inty,1,play_selct,gamma,alfa,epsilon_n)

        self.perecnt_total=self.st.percentage
        #file_name=f"output/New_data/test_1.npy"
        np.save(file_path_1,self.perecnt_total)
        total=sum(self.perecnt_total)/self.set_number
        self.avarage_t.append(total)
        print("Avarage of total is",total)
        self.count_train_data(game_count)
        self.v.import_data(file_path_1,file_path_2,file_path_3)  
        self.perecnt_total=[]
        self.st.percentage=[]



    def start(self,game_count,play_selct,gamma,alfa,epsilon_n,n_file):
        """
            this method is used to crate the Q-table and other file needs to run the game     """
        file_path=f"output/New_data/train_new"+str(n_file)+".npy"
        file_path_1=f"output/New_data/test_1.npy"
        file_path_2=f"output/New_data/testData_"+str(n_file)+".npy"
        file_path_3=f"output/New_data/test_"+str(n_file)+".txt"
        #file_path_2=f"output/Data/test_train"+n_file+".txt"

        data_2=[]
        data=np.zeros((60, 7))
        np.save(file_path,data)
        np.save(file_path_2,data_2)
        self.st.file_name=file_path

        file_data_1=np.load(file_path_2,allow_pickle=True)

        inty=int(game_count/self.set_number)
        for i in range(self.set_number):
            self.st.run_game_with_qtable(inty,1,play_selct,gamma,alfa,epsilon_n)

        self.perecnt_total=self.st.percentage
        #file_name=f"output/New_data/test_1.npy"
        np.save(file_path_1,self.perecnt_total)
        total=sum(self.perecnt_total)/self.set_number
        self.avarage_t.append(total)
        print("Avarage of total is",total)
        self.count_train_data(game_count)
        self.v.import_data(file_path_1,file_path_2,file_path_3)  
        self.perecnt_total=[]
        self.st.percentage=[]
        
    
    def count_train_data(self,count):
        """
            this method is used to count the episode of the game      """

        file_name=f"output/New_data/count_data.npy"
        count_data=[]
        if os.path.exists(file_name):
             count_data=np.load(file_name,allow_pickle=True)
        data=np.append(count_data,count)
        np.save(file_name,data)
        count_data=np.load(file_name,allow_pickle=True)
        print(count_data)
        suM_t=sum(count_data)
        print("Train in total is",suM_t)

    def count_game(self):
        """
            this method is used to sum how many episode train the Q-table     """
        file_name=f"output/New_data/count_data.npy"
        count_data=np.load(file_name,allow_pickle=True)
        suM_t=sum(count_data)
        print("train total is ",suM_t)





if __name__ == '__main__':
    print("Start...............")
    type_player=1
    game_count=5000
    play_selct=0
    gamma=0.5
    epsilon=0.0
    alfa=0.1
    #print("Gamma:",gamma,"Alfa",alfa,"espsilon",epsilon)
    list_player=[0,1,2,3]
    run=Game_run()
    run.run_call(game_count,play_selct,gamma,alfa,epsilon,1,50)
   
    

    






    
    
    
    




