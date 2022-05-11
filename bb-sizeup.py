# -- coding: utf-8 --
import os
import argparse


# 수정 1 : 대상 폴더
dir = "/home/adriv/Carla/CARLA_0.9.13/PythonAPI/DataGenerator/data8custom/"

def sizeup(x1, y1, x2, y2, padding, mx):
    return max(0,x1-padding), max(0,y1-padding), min(mx[0],x2+padding), min(mx[1],y2+padding)

def makeline(lst):
    result = str(lst[0])
    for i in range(1, len(lst)):
        result += " "+str(lst[i])
    return result+"\n"

if __name__ == '__main__':
    # 수정 2
    # padding : 얼마나 늘릴지 (상하좌우로 각 padding만큼 증가)
    # max_width,height : 이미지 크기 
    argparser = argparse.ArgumentParser(
        description='bounding box cleanser')
    argparser.add_argument(
        '--padding',
        default=0, type=int)
    argparser.add_argument(
        '--max_width',
        default=1024, type=int)
    argparser.add_argument(
        '--max_height',
        default=720, type=int)
    args = argparser.parse_args()

    kitti_lst = []
    frame_idx = 0
    padding = args.padding
    mx = [args.max_width-1, args.max_height-1] # 0-based 최대 x,y

    while True:
        frame_no = str(frame_idx).zfill(6)
        frame_idx += 1
        file = dir + frame_no + '.txt'

        print(file)

        # 종료조건
        if not os.path.exists(file):
            break
        
        edited_line = []

        with open(file, 'r') as f:
            while True:
                line = f.readline()
                if line is None or line == '':
                    break
                
                line = line.split()

                # custom 일 경우 5,6,7,8
                # kitti 에 적용하려면 4,5,6,7로 하면 됨
                x1, y1, x2, y2 = int(line[5]), int(line[6]), int(line[7]), int(line[8])
                line[5], line[6], line[7], line[8] = sizeup(x1,y1,x2,y2,padding,mx)

                edited_line.append(makeline(line))
        with open(file, 'w') as f:
            f.writelines(edited_line)