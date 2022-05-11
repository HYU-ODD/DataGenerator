# -- coding: utf-8 --
from pickle import NONE
import cv2
import os
import argparse

kitti_dir = "/home/adriv/Carla/CARLA_0.9.13/PythonAPI/DataGenerator/data10/training/custom/"
img_dir = "/home/adriv/Carla/CARLA_0.9.13/PythonAPI/DataGenerator/data10/training/image/"
mot_dir = "/home/adriv/Carla/CARLA_0.9.13/PythonAPI/DataGenerator/converted/"
result_dir = "/home/adriv/Carla/CARLA_0.9.13/PythonAPI/DataGenerator/vis/"
mode = 'MOT' # 'KITTI' if you want to convert the origin gt files

def kitti_converter():
    kitti_lst = []
    frame_idx = 0

    while True:
        frame_no = str(frame_idx).zfill(6)
        frame_idx += 1

        kitti_filename = kitti_dir + frame_no + '.txt'
        img_filename = img_dir + frame_no + '.png'
        result_filename = result_dir + frame_no + '.png'

        # 종료조건
        if not os.path.exists(kitti_filename):
            break
        if not os.path.exists(img_filename):
            break
    
        # 결과 폴더 생성
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        img = cv2.imread(img_filename, cv2.IMREAD_COLOR)
        with open(kitti_filename, 'r') as kitti_label:
            while True:
                line = kitti_label.readline()
                if line is None or line == '':
                    break
                
                line = line.split()
                if line[1] != 'Pedestrian':
                    continue

                x1, y1, x2, y2 = int(line[5]), int(line[6]), int(line[7]), int(line[8])

                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)
        
        cv2.imwrite(result_filename, img)


def mot_converter():
    mot_lst = []
    frame_idx = -1
    result_filename = None
    img = None

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(mot_dir + 'gt.txt', 'r') as gt:
        while True:
            line = gt.readline()
            if line == '\n' or line is None: break
            splited = line.split(',')
            if len(splited) < 9: break
            
            frame = int(splited[0])
            
            actor_id = int(splited[1])
            x1, y1, x2, y2 = int(splited[2]), int(splited[3]), int(splited[4]), int(splited[5])
        
            if frame_idx != frame:
                if img is not None:
                    cv2.imwrite(result_filename, img)
                img = cv2.imread(mot_dir + '{0:06d}.png'.format(frame), cv2.IMREAD_COLOR)
                frame_idx = frame
                result_filename = result_dir + '{0:06d}.png'.format(frame)
            
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)
        
        if img is not None:
            cv2.imwrite(result_filename, img)


if __name__ == '__main__':
    if mode == 'KITTI':
        kitti_converter()
    else:
        mot_converter()