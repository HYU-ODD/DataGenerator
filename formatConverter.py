# -- coding: utf-8 --
import os
import sys
import shutil

# config
rootdir = '/home/adriv/Carla/CARLA_0.9.13/PythonAPI/DataGenerator/'
data_dirs = ['data7/', 'data8/']
save_dir = rootdir + 'converted/'


if __name__ == '__main__':
    file_list = []

    for dir in data_dirs:
        path = rootdir + dir + 'training/custom/'
        f = os.listdir(path)

        for file in f:
            file_list.append(path + file)

    file_list.sort()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gt = open(save_dir + 'gt.txt', 'w')

    frame = 0
    for label_path in file_list:
        label = open(label_path, 'r')

        while True:
            line = label.readline()
            if line is None or line == '\n': break
            splited = line.split()
            if len(splited) < 16 or splited[0] is None: break

            actor_id = int(splited[0])
            if actor_id == -1: continue

            left, top, right, bot = splited[5:9]

            obj = str(frame) + ',' + str(actor_id) + ',' + left + ',' + top + ',' + right + ',' + bot + ',1,1,1' 
            
            gt.write(obj + '\n')
        
        img_path = label_path.split('custom/')
        img_path = img_path[0] + 'image/' + img_path[1].split('.')[0] + '.png'
        shutil.copyfile(img_path, save_dir + '{0:06d}.png'.format(frame))
        
        label.close()
        frame += 1
    gt.close()