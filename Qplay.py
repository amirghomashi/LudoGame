#!/usr/bin/env python

#This program provide for course Tools of Artificial Intelligence from SDU University 2020
#The main purpose of this class to play Ludo game and take history for train the Q-table.

import ludopy
import numpy as np
from tqdm import tqdm
import os 
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
from train import train_data
from captur import capture_image
from os import system


class Q_play():
    def __init__(self):
        self.there_is_a_winner=False
        self.g=ludopy.Game()
        self.player=None
        self.Q=[]
        self.ca=capture_image()
        self.list_winner=[]
        self.number_winner_my_player=0
        self.tr=train_data()
        self.player_last_piece=[]
        self.second_player=0
        self.file_name=""
        self.file_plyer_hist=""
        self.type_play=True
        self.winner=0
        self.my_player_winner=False
        self.gamma_m=0
        self.alfa_m=0
        self.percentage=[]
        self.epsilon=0
        

    
 

    def run_game_with_qtable(self,number,num,n_player,gamma,alfa,esplion_n):
        """
            this method is used to run game by using Q-tablle     """
        self.gamma_m=gamma
        self.alfa_m=alfa
        self.epsilon=esplion_n
        

        for i in range(number):
            system('clear')
            print("Train Nr.",i)
            print("Train left is :",number-i)
            print("Percentage winner:",self.percentage )
            self.game_running(i,num,n_player) 
            

        self.analyises_data(number) 
            

    def game_running(self,i,num,n_player):
        """
            this method is used to play game with semi-smart player or random player:     """
        self.Q=np.load(self.file_name,allow_pickle=True) 
        player_last_game=5 
        palyer_piece_game=[]
        player_picee_my=[]  
        numm_to=0 
        epsilon = self.epsilon
        
        while not self.there_is_a_winner:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner),player_i =self.g.get_observation()
            player_last_game=player_i
            palyer_piece_game=[player_pieces]

            if len(move_pieces):
                if player_i==n_player:   

                    ## define to save data from my player pieces
                    player_picee_my=list([player_pieces])
                    if self.type_play:

                        ## read based on Q-table to choose which piece should be take
                        if np.random.uniform(0, 1) < epsilon:
                            """
                            Explore: select a random action    """
                            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                            #print("player random")
                        else:
                            """
                            Exploit: select the action with max value (future reward)    """
                            piece_to_move=int(self.detect_piece(player_pieces,move_pieces,dice))
                            #print("play Qtabel")
                        
                    else:
                        ## define as random player
                        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))] 
                    
                else:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]    
            else:
                piece_to_move = -1
            _1, _2, _3, _4, _winner, self.there_is_a_winner =self.g.answer_observation(piece_to_move) 
            if self.there_is_a_winner:
                self.winner=player_last_game
                
        
        

        if player_last_game==n_player:   
            self.number_winner_my_player=self.number_winner_my_player+1
            self.my_player_winner=True 
            # print("You win a game.....")
        else:
            self.my_player_winner=False
            # print("You are not win of the game....")

        self.list_winner.append(self.winner)
        self.player_last_piece.append([player_picee_my,n_player,self.winner])
        self.pre_data_after_game(n_player,self.g.hist)
        #self.Save_video_game(1)

        self.g.reset()
        self.there_is_a_winner=False 

    def detect_piece(self,player_pieces,move_pieces,dice):
        """
            this method is used to getting Q-table value     """
        
        num_piece=len(move_pieces)
        _isZero=any(i ==0 for i in player_pieces)
        list_home=[53,54,55,56,57,58,59]
        list_d=[]
        last_data=-20
        move_detect=0
        if dice==6 and _isZero:
            piece_s = [i for i in range(len(player_pieces)) if player_pieces[i] == 0]
            piecF=[i for i in range(len(move_pieces)) if move_pieces[i] == piece_s[0]]
            st=move_pieces[piecF]
            move_detect=st
        else:
            if num_piece>1:
                for i in range(num_piece):
                    f=move_pieces[i]
                    pice_f=player_pieces[f]
                    b=self.Q[pice_f,dice]
                    list_d.extend([pice_f,b])    
                    if b>last_data:
                        last_data=b
                        move_detect=f 
                        
            else:
                move_detect=move_pieces  
        return move_detect  



    def pre_data_after_game(self,n_player,hist):
        """
            this method is used to pre data to store in Q-table with state and action     """
    
        for i in range(1):
            self.tr.winner_game=self.my_player_winner
            for i, moment in (enumerate(hist)):
                pieces, dice, players_i_m, round_i = moment
                if players_i_m==n_player :
                    for player_i, player_pieces in enumerate(pieces):
                        if player_i==n_player :
                            self.tr.file_name=self.file_name
                            self.tr.gamma=self.gamma_m
                            self.tr.alfa=self.alfa_m
                            self.tr.traina_data(player_i,dice,round_i,player_pieces,pieces)
        
    def analyises_data(self,number):
        """
            this method is used to analyise the result from Game     """

        percent=float((self.number_winner_my_player/number)*100)
        self.percentage.append(percent)
        self.number_winner_my_player=0

        # if num_safe_home==3:
        #     second=second+1
        # self.second_player=self.second_player+second 


    def run_game(self,number,num_f,file_path):
        """
            this method is used to run the game     """
        
        for i in range(number):
            self.game_running(i,num_f,file_path)     

    def Save_video_game(self,num):
        """
            this method is used to save game history     """

        file_name_video=f"output/Data/game_video_"+str(num)+".mp4"
        print(file_name_video)
        print("Saving game video")
        self.g.save_hist_video(file_name_video)      


    



    


    

