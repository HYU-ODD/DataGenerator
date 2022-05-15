# -- coding: utf-8 --
import cv2
import os
import argparse

gt_dir = "/home/adriv/Carla/CARLA_0.9.13/PythonAPI/DataGenerator/converted/"

'''
SpaceBar: select a bounding box
BackSpace: delete the selected bounding box
Left: previous frame
Right: next frame
Ctrl+Z: get back
Enter: save new ground truth file (new_gt.txt)
A: move the selected bounding box to the left
D: move the selected bounding box to the right
W: move the selected bounding box up
S: move the selected bounding box down
N: create a new bounding box with the input ID
shift+W: increase the size upward
shift+D: increase the size down
shift+A: increase the size left
shift+D: increase the size right
I: decrease the size down
K: decrease the size upward
J: decrease the size right
L: decrease the size left
'''
''' Option
--start: select starting frame number
'''

def draw_bb(img, gt, frame):
    cv2.putText(img, str(frame), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    for label in gt:
        x, y = int(label[2]), int(label[3])
        w, h = int(label[4]), int(label[5])
        id = label[1]
        cv2.putText(img, id, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description='bounding box cleanser')
    argparser.add_argument(
        '--start',
        default=0, type=int)
    argparser.add_argument(
        '--gt_file',
        default='gt.txt'
    )
    args = argparser.parse_args()
    
    new_gt = []
    start_frame_idx = 0
    with open(gt_dir + args.gt_file, 'r') as label:
        frame = -1
        while True:
            line = label.readline()
            if line is None or line == '':
                break
            
            obj = line.split(',')
            if int(obj[0]) != frame:
                frame = int(obj[0])
                new_gt.append([])
                if frame == args.start:
                    start_frame_idx = len(new_gt) - 1
            new_gt[len(new_gt)-1].append(obj)
    
    finish = 0
    cnt = start_frame_idx
    cache_new_gt = [[] for _ in range(len(new_gt))]
    while 0 <= cnt < len(new_gt):
        gt = new_gt[cnt]
        cache_gt = cache_new_gt[cnt]
        
        if len(gt): frame = int(gt[0][0])
        else: frame = int(cache_gt[0][0])
        
        org = cv2.imread(gt_dir+"image/{0:06d}.png".format(frame), cv2.IMREAD_COLOR)
        if org is None:
            print("Can't open" + gt_dir + "image/{0:06d}.jpg".format(frame))
        src = org.copy()
        draw_bb(src, gt, frame)
        
        idx = -1
        while True:
            cv2.imshow("result", src)
            key = cv2.waitKeyEx() & 0xFF
            print(key)

            if key == 83: # -> 방향키
                cnt += 1
                break
            
            elif key == 27: # esc
                finish = 1
                break
            
            elif key == 81: # <- 방향키
                cnt -= 1
                break
            
            elif key == 32 and len(gt) > 0: # -> spacebar
                idx = idx % len(gt)
                if idx != -1:
                    x, y = int(gt[idx][2]), int(gt[idx][3])
                    w, h = int(gt[idx][4]), int(gt[idx][5])
                    cv2.rectangle(src, (x,y), (x+w, y+h), (0,255,0), 2)
                idx = (idx + 1) % len(gt)
                x, y = int(gt[idx][2]), int(gt[idx][3])
                w, h = int(gt[idx][4]), int(gt[idx][5])
                cv2.rectangle(src, (x,y), (x+w, y+h), (0,0,255), 2)
                
            elif key == 8 and idx != -1 and len(gt) > 0: # backspace
                cache_gt.append(gt[idx])
                gt.remove(gt[idx])
                src = org.copy()
                draw_bb(src, gt, frame)
            
            elif key == ord('d') and idx != -1:
                gt[idx][2] = str(int(gt[idx][2]) + 2)
                src = org.copy()
                draw_bb(src, gt, frame)
                
            elif key == ord('a') and idx != -1:
                gt[idx][2] = str(int(gt[idx][2]) - 2)
                src = org.copy()
                draw_bb(src, gt, frame)
                
            elif key == ord('w') and idx != -1:
                gt[idx][3] = str(int(gt[idx][3]) - 2)
                src = org.copy()
                draw_bb(src, gt, frame)
            
            elif key == ord('s') and idx != -1:
                gt[idx][3] = str(int(gt[idx][3]) + 2)
                src = org.copy()
                draw_bb(src, gt, frame)
                    
            elif (key == 26 or key == 122) and len(cache_gt) > 0: # ctrl+z or z
                label = cache_gt.pop()
                gt.append(label)
                x, y = int(label[2]), int(label[3])
                w, h = int(label[4]), int(label[5])
                cv2.putText(src, label[1], (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)
                cv2.rectangle(src, (x,y), (x+w, y+h), (0,255,0), 2)

            elif key == ord('n'):
                id = int(input('new object id: '))
                gt.append([str(frame), str(id), '100', '100', '50', '50', '1','1','1\n'])
                x, y = int(gt[-1][2]), int(gt[-1][3])
                w, h = int(gt[-1][4]), int(gt[-1][5])
                cv2.putText(src, str(id), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)
                cv2.rectangle(src, (x,y), (x+w, y+h), (0,255,0), 2)

            elif key == 13: # enter
                with open(gt_dir+"new_gt.txt", 'w') as f:
                    for new_label in new_gt:
                        for obj in new_label:
                            for i, component in enumerate(obj):
                                f.write(component)
                                if i+1 < len(obj):
                                    f.write(',')
                print("Saved")

            elif key == 87: # shift + w
                gt[idx][5] = str(int(gt[idx][5]) + 2) # height
                gt[idx][3] = str(int(gt[idx][3]) - 2) # top
                src = org.copy()
                draw_bb(src, gt, frame)
            elif key == 83: # shift + s
                gt[idx][5] = str(int(gt[idx][5]) + 2) # height
                src = org.copy()
                draw_bb(src, gt, frame)
            elif key == 65: # shift + a
                gt[idx][4] = str(int(gt[idx][4]) + 2) # width
                gt[idx][2] = str(int(gt[idx][2]) - 2) # left
                src = org.copy()
                draw_bb(src, gt, frame)
            elif key == 68: # shift + d
                gt[idx][4] = str(int(gt[idx][4]) + 2) # width
                src = org.copy()
                draw_bb(src, gt, frame)

            elif key == ord('i'):
                gt[idx][5] = str(int(gt[idx][5]) - 2) # height
                gt[idx][3] = str(int(gt[idx][3]) + 2) # top
                src = org.copy()
                draw_bb(src, gt, frame)
            elif key == ord('k'):
                gt[idx][5] = str(int(gt[idx][5]) - 2) # height
                src = org.copy()
                draw_bb(src, gt, frame)
            elif key == ord('j'):
                gt[idx][4] = str(int(gt[idx][4]) - 2) # width
                gt[idx][2] = str(int(gt[idx][2]) + 2) # left
                src = org.copy()
                draw_bb(src, gt, frame)
            elif key == ord('l'):
                gt[idx][4] = str(int(gt[idx][4]) - 2) # width
                src = org.copy()
                draw_bb(src, gt, frame)
        if finish: break
        
    if not finish:
        with open(gt_dir+"new_gt.txt", 'w') as f:
                    for new_label in new_gt:
                        for obj in new_label:
                            for i, component in enumerate(obj):
                                f.write(component)
                                if i+1 < len(obj):
                                    f.write(',')
    cv2.destroyAllWindows()
