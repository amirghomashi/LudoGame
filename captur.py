#!/usr/bin/env python

#This program provide for course Tools of Artificial Intelligence from SDU University 2020
#The main purpose of this class to capture the image from video recoeded in the game.

import ludopy
import numpy as np
from tqdm import tqdm
import os 
import cv2
from collections import OrderedDict
import matplotlib.pyplot as plt
from train import train_data

class capture_image():
    def __init__(self):
        pass
    def capture(self,path_file,video_path):
        """
            this method is used to capture video to image frame     """
        
        file_path_add=path_file+"/image"
        file_path_read_video=video_path

        if not os.path.exists(file_path_add):
            os.mkdir(file_path_add)
        cap = cv2.VideoCapture(file_path_read_video)
        print("start......")
        counter = 0
        while(1):
            ret, frame = cap.read()
            counter += 1
            file_name=file_path_add+"/imge_new_"+str(counter)+".png"
            if frame is not None and frame.size!=0:
                cv2.imwrite(file_name,frame)  
                print(file_name)
            else:
                break 


if __name__ == '__main__':
    run=capture_image()
    run.capture("System","screencast-2.ogv")               