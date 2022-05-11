# -- coding: utf-8 --
import cv2
import os

kitti_dir = "/home/adriv/Carla/CARLA_0.9.13/PythonAPI/DataGenerator/data8custom/"
img_dir = "/home/adriv/Carla/CARLA_0.9.13/PythonAPI/DataGenerator/data8/training/image/"
result_dir = "/home/adriv/Carla/CARLA_0.9.13/PythonAPI/DataGenerator/test/result/"

if __name__ == '__main__':
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