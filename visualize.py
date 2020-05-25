#!/usr/bin/env python

#This program provide for course Tools of Artificial Intelligence from SDU University 2020
#The main purpose of this class to convert the data after play episode game

import ludopy
import numpy as np
import os 
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt


class visualize_data():
    def __init__(self):
        pass

    def import_data(self,file_1,file_2,file_write):
        """
            this method is used to convert data from .npy file to format txt file    """
        file_name=file_1
        file_name_2=file_2
        file_name_3=f"output/New_data/test_3.npy"
        file_data_first=np.load(file_name,allow_pickle=True)
        file_data_second=np.load(file_name_2,allow_pickle=True)
        file_data_third=[]
         
        ## save file 2
        file_o=np.append(file_data_second,file_data_first)
        np.save(file_name_2,file_o,allow_pickle=False)
        file_data_2=np.load(file_name_2,allow_pickle=True)
        #print(file_data_2)
        self.print_data(file_name,file_write)

    def read_table(self):
        """
            this method is used to read information from Q_table     """

        file_name=f"output/history_new/Senario_4/2/testData_10.npy"
        #file_name_2=f"output/history_new/Senario_4/2/test_10.txt"
        
        self.Q=np.load(file_name,allow_pickle=True) 
        for i in range (len(self.Q)):
            value=self.Q[i]
            print("Row number :",i,value)


    def print_data(self, path_file,file_save):
        """
            this method is used to convert data to txt format     """
        file_data_1=np.load(path_file,allow_pickle=True)
        file_name=file_save
        file1 = open(file_name,"w")
        for i in range(len(file_data_1)):
            val=str(file_data_1[i])+","
            file1.write(val)
        file1.close()


    

    def plot_data_train(self,file_data):
        """
            this method is used to plot and visulize the data     """
        list_data=[]
        num=len(file_data)
        for i in range(num):
            data=float(file_data[i])
            list_data.append(data)
        x=list_data
        plt.plot(x)
        plt.show()
                 


if __name__ == '__main__':
    run=visualize_data()
    run.read_table()   
     
    
