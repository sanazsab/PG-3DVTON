#obtain image gradient

import os
import numpy as np
from PIL import Image 
import cv2
import json
import pycocotools.mask as maskUtils
from tqdm import tqdm
import math
import argparse
from matplotlib import pyplot as plt
import sys
sys.path.append('.')
from util import tensor2im


def get_mask_from_kps(kps,img_h=256,img_w=192):
    rles = maskUtils.frPyObjects(kps, img_h, img_w)
    rle = maskUtils.merge(rles)
    mask = maskUtils.decode(rle)[...,np.newaxis].astype(np.float32)
    mask = mask * 255.0

    return mask


def get_rectangle_mask(a,b,c,d):
    x1 = a + (b-d)/4
    y1 = b + (c-a)/4
    x2 = a - (b-d)/4
    y2 = b - (c-a)/4

    x3 = c + (b-d)/4
    y3 = d + (c-a)/4
    x4 = c - (b-d)/4
    y4 = d - (c-a)/4
    kps = [x1,y1,x2,y2]

    v0_x = c-a
    v0_y = d-b
    v1_x = x3-x1
    v1_y = y3-y1
    v2_x = x4-x1
    v2_y = y4-y1

    cos1 = (v0_x*v1_x+v0_y*v1_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
    cos2 = (v0_x*v2_x+v0_y*v2_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

    if cos1<cos2:
        kps.extend([x3,y3,x4,y4])
    else:
        kps.extend([x4,y4,x3,y3])

    kps = np.array(kps).reshape(1,-1).tolist()
    mask = get_mask_from_kps(kps)

    return mask


def get_hand_mask(hand_keypoints):
    # shoulder, elbow, wrist
    s_x,s_y,s_c = hand_keypoints[0]
    e_x,e_y,e_c = hand_keypoints[1]
    w_x,w_y,w_c = hand_keypoints[2]

    # up_mask = np.ones((256,192,1))
    # bottom_mask = np.ones((256,192,1))
    # up_mask = np.ones((512,512,1))
    # bottom_mask = np.ones((512,512,1))
    up_mask = np.ones((256,192,1))
    bottom_mask = np.ones((256,192,1))
    if s_c > 0.1 and e_c > 0.1:
        up_mask = get_rectangle_mask(s_x,s_y,e_x,e_y)
        kernel = np.ones((20,20),np.uint8)
        up_mask = cv2.dilate(up_mask,kernel,iterations = 1)
        up_mask = (up_mask > 0).astype(np.float32)[...,np.newaxis]
    if e_c > 0.1 and w_c > 0.1:
        bottom_mask = get_rectangle_mask(e_x,e_y,w_x,w_y)
        bottom_mask = (bottom_mask > 0).astype(np.float32)

    return up_mask, bottom_mask

def get_palm_mask(hand_mask, hand_up_mask, hand_bottom_mask):
    inter_up_mask = (hand_mask + hand_up_mask == 2.0).astype(np.float32)
    inter_bottom_mask = (hand_mask + hand_bottom_mask == 2.0).astype(np.float32)
    palm_mask = hand_mask - inter_up_mask - inter_bottom_mask

    return palm_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--MPV3D_root', type=str, default='/local_storage/users/sanazsab/dataset/train1',help='path to the MPV3D dataset')  # ../
    opt, _ = parser.parse_known_args()

    MPV3D_root = opt.MPV3D_root

    # source dirs
    person_root = os.path.join(MPV3D_root, 'ImgP1')

# -------------------- Segment Palms ------------------------ #

    parse_root = os.path.join(MPV3D_root, 'image-parse')
    pose_root = os.path.join(MPV3D_root, 'pose')
    palmrgb_dst = os.path.join(MPV3D_root, 'palm-rgb')
    os.makedirs(palmrgb_dst, exist_ok=True)
    palmmask_dst = os.path.join(MPV3D_root, 'palm-mask')
    os.makedirs(palmmask_dst, exist_ok=True)

    person_list = sorted(os.listdir(person_root))
    for person_name in tqdm(person_list):
        person_id = person_name.split('_')[0]
        person_path = os.path.join(person_root, person_name)
        parsing_path = os.path.join(parse_root, person_name)
        keypoints_path = os.path.join(pose_root, person_name.replace('.png', '_keypoints.json'))
        palmrgb_outfn = os.path.join(palmrgb_dst, person_name.replace('whole_front.png','palm.png'))
        palmmask_outfn = os.path.join(palmmask_dst, person_name.replace('.png','_palm_mask.png'))

        parsing = np.array(Image.open(parsing_path))
        person = cv2.imread(person_path).astype(np.float32)

        left_arm_mask = (parsing==14).astype(np.float32)
        right_arm_mask = (parsing==15).astype(np.float32)

        left_arm_mask = np.expand_dims(left_arm_mask, 2)
        right_arm_mask = np.expand_dims(right_arm_mask, 2)

        with open(keypoints_path) as f:
            datas = json.load(f)

        keypoints = np.array(datas['people'][0]['pose_keypoints_2d']).reshape((-1,3))
        left_hand_keypoints = keypoints[[5,6,7],:]
        right_hand_keypoints = keypoints[[2,3,4],:]

        left_hand_up_mask, left_hand_botton_mask = get_hand_mask(left_hand_keypoints)
        right_hand_up_mask, right_hand_botton_mask = get_hand_mask(right_hand_keypoints)

        left_palm_mask = get_palm_mask(left_arm_mask, left_hand_up_mask, left_hand_botton_mask)
        right_palm_mask = get_palm_mask(right_arm_mask, right_hand_up_mask, right_hand_botton_mask)

        print(person.shape)
        palm_rgb = person * (left_palm_mask + right_palm_mask)
        palm_mask = (left_palm_mask + right_palm_mask) * 255

        cv2.imwrite(palmrgb_outfn, palm_rgb)
        cv2.imwrite(palmmask_outfn, palm_mask)


# -------Graident------
#target dirs
    gradient_dst = os.path.join(MPV3D_root, 'images-sobel')
    os.makedirs(gradient_dst, exist_ok=True)
    person_list = sorted(os.listdir(person_root))
    for person_name in tqdm(person_list):
        person_path = os.path.join(person_root, person_name)
        person = cv2.imread(person_path)
        print(person.shape)
#       # person=tensor2im(person, imtype=np.uint8)
        person_gray=cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(person_gray,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(person_gray,cv2.CV_64F,0,1,ksize=5)
        gradientx_outfn = os.path.join(gradient_dst, person_name.replace('.png', '_sobelx.png'))
        gradienty_outfn = os.path.join(gradient_dst, person_name.replace('.png', '_sobely.png'))
        plt.imsave(gradientx_outfn, sobelx, cmap='gray')
        plt.imsave(gradienty_outfn, sobely, cmap='gray')
        print(f'Getting image sobel done and saving to {gradient_dst}!')

