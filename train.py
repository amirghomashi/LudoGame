#!/usr/bin/env python

#This program provide for course Tools of Artificial Intelligence from SDU University 2020
#The main purpose of this class to train Q-table base on state and action.

import ludopy
import numpy as np
from tqdm import tqdm
import os 
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt

class train_data():
    def __init__(self):
        self.pre=[]
        self.pre_all=[]
        self.stat=False
        self.file_name=f""
        self.state_s=0
        self.state_old=0
        self.state_s_all=0
        self.state_old_all=0
        self.pre_hist=[]
        self.pre_hit_other=[]
        self.state_s_h=0
        self.state_old_h=0
        self.gamma=0
        self.alfa=0
        

        ## check player get 6 to start game or not
        self.player_start=False
        self.state_l=[]

        # define to check if player knock other player 
        self.knock_status=False
        self.hit_with_other=False
        self.winner_game=False 
        self.safe_home=False 
        self.is_global_home=False
        

        # Define value for reward
        self.reward_l=[0.25,0.2,0.15,0.1,0.05,1,-0.25,-1]

    def traina_data(self,player_i, dice, round_i,player_pieces,pieces):
        """
            this method is used to train Q-table      """
    
        self.Q=np.load(self.file_name,allow_pickle=True)
        
        self.state_action_find(player_pieces)
        reward=self.count_reward(player_i, dice, round_i,player_pieces,pieces)
        Lr=self.alfa
        Gamma=self.gamma
        action=dice
        old_value=self.Q[self.state_old, action]
        value=Lr * ((reward + Gamma * np.max(self.Q[self.state_s, :]) - self.Q[self.state_old, action]))
        current_value=old_value+value
        self.Q[self.state_s, action] = current_value
        np.save(self.file_name, self.Q) 
    
    def count_reward(self, player_i,dice,round_i,player_pieces,piece):
        """
            this method is used to count reward for game history    """
        self.state_l=[]
        reward=0.0
        if round_i==1:
            if dice==6:
                reward=reward+float(self.reward_l[0])
                self.player_start=True
                
        elif round_i>1 and self.player_start:
            self.knock_status=self.knock_stat(player_i,piece,round_i)
            self.hit_with_other=self.player_hit_with_other(round_i,player_pieces)
            self.is_safe_home()

            if dice==6:
                reward=reward+float(self.reward_l[0])
            if self.safe_home:
                reward=reward+float(self.reward_l[1])    
            if self.knock_status:
                reward=reward+float(self.reward_l[2])
            if self.is_star_home:
                 reward=reward+float(self.reward_l[3])
            if self.winner_game:
                reward=reward+float(self.reward_l[5])     
            if self.hit_with_other:
                reward=reward+float(self.reward_l[6]) 
            if self.winner_game==False:
                reward=reward+float(self.reward_l[7])     
        return reward  
    
    def is_player_winner(self,hist,n_player):
        """
            this method is used to find which player is winner of the game     """
        winner=False 
        numm=len(hist)
        player_last_hist=hist[-1][0][n_player]
        winner=all(i ==59 for i in player_last_hist)

    
    def is_star_home(self):
        """
            this method is used to chech if the piece in star home     """
        self.is_global_home=False
        safe_global_number=[5,12,18,25,31,38,44,51]
        if self.state_s in safe_global_number and self.state_s!=self.state_old:
            self.is_global_home=True
            print("Star field:",self.state_s)
        
    def is_safe_home(self):
        """
            this method is used to check is the pice in the safe home      """
        self.safe_home=False
        safe_home_number=[53,54,55,56,57,58,59]
        if self.state_s in safe_home_number and self.state_s!=self.state_old:
            self.safe_home=True    



    def knock_stat(self,player_i,pieces,round_i):
        """
            this method is used to check if player knock other player piece     """
        li=[]
        knock_status=False
    
        for play_i, player_pieces in enumerate(pieces):
            li.append(player_pieces)
        num=len(self.pre_hist)
        if len(self.pre_hist)>0:
            for i in range(num):
                d=self.pre_hist[i]
                s=li[i]
                num=d-s
                
                ch_knock=all(i <=0 for i in num)
                if ch_knock==False:
                    nu=len(d)
                    for j in range(nu):
                        d_d=d[j]
                        s_s=s[j]
                        if d_d>0 and s_s==0:
                            knock_status=True

        self.pre_hist=[]
        self.pre_hist=list(li) 
        return knock_status

    def player_hit_with_other(self,round_i,player_pieces):
        """
            this method is used to check if myplayer is hit with other player     """
        hit_other=False
        li=[]
        for i, piece in enumerate(player_pieces):
            li.append(piece)
        num=len(self.pre_hit_other)
        if len(self.pre_hit_other)>0:
            for i in range(num):
                s=li[i]
                d=self.pre_hit_other[i]
                if s==0 and d>0:
                    hit_other=True  
                       
        self.pre_hit_other=[]
        self.pre_hit_other=list(li)  
        return hit_other       

    def state_action_find(self,player_pieces):
        """
            this method is used to find the action and state from game history      """
        li=[]
        for i, piece in enumerate(player_pieces):
            li.append(piece)
        num=len(self.pre)
        if len(self.pre)>0:
            for i in range(num):
                s=li[i]
                d=self.pre[i]
                #print(d)
                diff=s-d
                if diff>0:
                   self.state_s=s
                   self.state_old=d
                       
        self.pre=[]
        self.pre=list(li)     
                

    
    

    
