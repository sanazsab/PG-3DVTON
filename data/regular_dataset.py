import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os.path as osp
from PIL import Image
import numpy as np
#from torchvision import transforms
import torchvision.transforms as transforms
from torchvision import utils
from utils import pose_utils
from PIL import ImageDraw
from utils.transforms import create_part
import time
import json
import random
import cv2
from data.base_dataset import BaseDataset, get_transform
from utils.util import tensor2im, save_image, save_depth, decode_labels, depth2normal_ortho
np.seterr(divide='ignore', invalid='ignore')
class RegularDataset(BaseDataset):
    def __init__(self, config, augment):    
        self.opt = config
#        if self.opt.train_mode != 'depth':
        self.transforms = augment
        self.isval = self.opt.isval
        self.isdemo = self.opt.isdemo
        self.train_mode = self.opt.train_mode            
        self.fine_width = 192
        self.fine_height = 256
        self.size = (256, 192)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
 
        if self.isdemo:
            self.img_list =[i.strip() for i in open('/local_storage/users/sanazsab/dataset/data_pair.txt', 'r').readlines()]
          #  self.img_list =[i.strip() for i in open('/local_storage/users/sanazsab/dataset/Demos.txt', 'r').readlines()]
            self.mode = 'val'
        else:
            pair_list =[i.strip() for i in open('/local_storage/users/sanazsab/dataset/data_pair.txt', 'r').readlines()]
            train_list = list(filter(lambda p: p.split('\t')[3] == 'train', pair_list))
            train_list = [i for i in train_list]
            test_list = list(filter(lambda p: p.split('\t')[3] == 'test', pair_list))
            test_list = [i for i in test_list]

            if not self.isval:
                self.mode = 'train'
                self.img_list = train_list
            else:
                self.mode = 'val'
                self.img_list = test_list
    
    def __getitem__(self, index):
        t0 = time.time()
        try:
            img_source = self.img_list[index].split('\t')[0]
            img_target = self.img_list[index].split('\t')[1]
            cloth_img = self.img_list[index].split('\t')[2]
            data_suffix = self.img_list[index].split('\t')[3]

        except:
            img_source = self.img_list[index].split(' ')[0]
            img_target = self.img_list[index].split(' ')[1]
            cloth_img = self.img_list[index].split(' ')[2]
            data_suffix = self.img_list[index].split(' ')[3]

        data_suffix = data_suffix if data_suffix == 'train' else 'val'
        source_splitext = os.path.splitext(img_source)[0]
        target_splitext = os.path.splitext(img_target)[0]
        cloth_splitext = os.path.splitext(cloth_img)[0]

        # png or jpg
 
        if self.isdemo:
            source_img_path = os.path.join('/local_storage/users/sanazsab/dataset/images', source_splitext.split('/')[1] + '.jpg')
            target_img_path = os.path.join('/local_storage/users/sanazsab/dataset/images', target_splitext.split('/')[1] + '.jpg')
            cloth_img_path = os.path.join('/local_storage/users/sanazsab/dataset/cloth_image', 'train', cloth_splitext.split('/')[1] + '.jpg')
            cloth_parse_path = os.path.join('/local_storage/users/sanazsab/dataset/cloth_mask', 'train', cloth_splitext + '_mask.png')
        else:
            source_img_path = os.path.join('/local_storage/users/sanazsab/dataset/images', self.mode, source_splitext + '.jpg')
            target_img_path = os.path.join('/local_storage/users/sanazsab/dataset/images', self.mode, target_splitext + '.jpg')
            cloth_img_path = os.path.join('dataset/images', self.mode, cloth_img)

            cloth_parse_path = os.path.join('/local_storage/users/sanazsab/dataset/cloth_mask', self.mode, cloth_splitext + '_mask.png')
        warped_cloth_name = source_splitext.split('/')[0] + '/' + \
                            source_splitext.split('/')[1] + '_' + \
                            target_splitext.split('/')[1] + '_' + \
                            cloth_splitext.split('/')[1] + '_warped_cloth.jpg'
        ### image
        warped_cloth_path = os.path.join('/local_storage/users/sanazsab/dataset', 'warped_cloth', self.mode, warped_cloth_name)
        source_img = self.open_transform(source_img_path, False)
        target_img = self.open_transform(target_img_path, False)
        cloth_img = self.open_transform(cloth_img_path, False)
        cloth_parse = self.parse_cloth(cloth_parse_path)

        try:
            warped_cloth_parse_name = source_splitext.split('/')[0] + '/' + \
                source_splitext.split('/')[1] + '_' + \
                target_splitext.split('/')[1] + '_' + \
                cloth_splitext.split('/')[1] + '_warped_cloth_mask.png'
            ### mask
            warped_cloth_parse_path = os.path.join('/local_storage/users/sanazsab/dataset', 'warped_cloth_mask', self.mode, warped_cloth_parse_name)
            warped_cloth_parse = self.parse_cloth(warped_cloth_parse_path)


            warped_cloth = tensor2im(warped_cloth_img)
            warped_cloth_gray = cv2.cvtColor(warped_cloth, cv2.COLOR_RGB2GRAY)
            warped_cloth_sobelx = cv2.Sobel(warped_cloth_gray, cv2.CV_64F, 1, 0, ksize=5)
            warped_cloth_sobely = cv2.Sobel(warped_cloth_gray, cv2.CV_64F, 0, 1, ksize=5)
            warp_cloth_path = os.path.join('/local_storage/users/sanazsab/dataset', 'warp-cloth-sobel')
            os.makedirs(warp_cloth_path, exist_ok=True)
            cloth_sobelx = os.path.join(warp_cloth_path, cloth_splitext.replace('.jpg', '_sobelx.png'))
            cloth_sobely = os.path.join(warp_cloth_path, cloth_splitext.replace('.jpg', '_sobelx.png'))
            plt.imsave(cloth-sobelx, warped_cloth_sobelx, cmap='gray')
            plt.imsave(cloth-sobely, warped_cloth_sobely, cmap='gray')

            warped_cloth_gradient_sobelx = Image.open(cloth_sobelx).convert('L')
            warped_cloth_gradient_sobely = Image.open(cloth_sobely).convert('L')

            warped_cloth_gradient_sobelx, warped_cloth_gradient_sobely = self.transform(warped_cloth_gradient_sobelx), self.transform(warped_cloth_gradient_sobely)
            target_parse_arm = (target_parse_path == 14).astype(np.float32) + (target_parse_path == 15).astype(np.float32)
            target_parse_arm_cloth = target_parse_arm + target_parse_cloth
            pacm = torch.from_numpy(target_parse_arm_cloth).unsqueeze(0)
            warped_cloth_gradient_sobelx = warped_cloth_gradient_sobelx * pacm - (1 - pacm)  # [-1,1], fill -1 for other parts
            warped_cloth_gradient_sobely = warped_cloth_gradient_sobely * pacm - (1 - pacm)  # [-1,1], fill -1 for other parts 
        except:
            warped_cloth_parse = torch.ones(1,256,192)
            warped_cloth_gradient_sobelx = torch.ones(1,256,192)
            warped_cloth_gradient_sobely = torch.ones(1,256,192)

        if os.path.exists(warped_cloth_path):
            warped_cloth_img = self.open_transform(warped_cloth_path, False)
        else:
            warped_cloth_img = cloth_img
        
        # parsing
        if self.isdemo:
            source_parse_vis_path = os.path.join('/local_storage/users/sanazsab/dataset/parse_cihp' , 'val', source_splitext.split('/')[1] + '_vis.png')
            target_parse_vis_path = os.path.join('/local_storage/users/sanazsab/dataset/parse_cihp' , 'val', target_splitext.split('/')[1] + '_vis.png')
#        source_parse_vis_new = (Image.open(source_parse_vis_path))
            source_parse_vis = self.transforms['3'](Image.open(source_parse_vis_path))
            target_parse_vis = self.transforms['3'](Image.open(target_parse_vis_path))
        
            source_parse_path = os.path.join('/local_storage/users/sanazsab/dataset/parse_cihp', 'val', source_splitext.split('/')[1] + '.png')
            target_parse_path = os.path.join('/local_storage/users/sanazsab/dataset/parse_cihp', 'val', target_splitext.split('/')[1] + '.png')

        else:
            source_parse_vis_path = os.path.join('/local_storage/users/sanazsab/dataset/parse_cihp' , self.mode, source_splitext.split('/')[1] + '_vis.png')
            target_parse_vis_path = os.path.join('/local_storage/users/sanazsab/dataset/parse_cihp' , self.mode, target_splitext.split('/')[1] + '_vis.png')
#        source_parse_vis_new = (Image.open(source_parse_vis_path))
            source_parse_vis = self.transforms['3'](Image.open(source_parse_vis_path))
            target_parse_vis = self.transforms['3'](Image.open(target_parse_vis_path))
        
            source_parse_path = os.path.join('/local_storage/users/sanazsab/dataset/parse_cihp', self.mode, source_splitext.split('/')[1] + '.png')
            target_parse_path = os.path.join('/local_storage/users/sanazsab/dataset/parse_cihp', self.mode, target_splitext.split('/')[1] + '.png')

        source_parse = pose_utils.parsing_embedding(source_parse_path)
        target_parse = pose_utils.parsing_embedding(target_parse_path)


        source_parse_shape = np.array(Image.open(source_parse_path))
        source_parse_shape = (source_parse_shape > 0).astype(np.float32)
        source_parse_shape = Image.fromarray((source_parse_shape*255).astype(np.uint8))
        source_parse_shape = source_parse_shape.resize((self.size[1]//16, self.size[0]//16), Image.BILINEAR) # downsample and then upsample
        source_parse_shape = source_parse_shape.resize((self.size[1], self.size[0]), Image.BILINEAR)
        source_parse_shape = self.transforms['1'](source_parse_shape) # [-1,1]

        source_parse_head = (np.array(Image.open(source_parse_path)) == 1).astype(np.float32) + \
                    (np.array(Image.open(source_parse_path)) == 2).astype(np.float32) + \
                    (np.array(Image.open(source_parse_path)) == 4).astype(np.float32) + \
                    (np.array(Image.open(source_parse_path)) == 13).astype(np.float32)

        target_parse_cloth = (np.array(Image.open(target_parse_path)) == 5).astype(np.float32) + \
                (np.array(Image.open(target_parse_path)) == 6).astype(np.float32) + \
                (np.array(Image.open(target_parse_path)) == 7).astype(np.float32)

        im_mask = torch.from_numpy((np.array(Image.open(target_parse_path)) > 0).astype(np.float32)).unsqueeze(0)


        # prepare for warped cloth
        phead = torch.from_numpy(source_parse_head) # [0,1]
        pcm = torch.from_numpy(target_parse_cloth) # [0,1]
        im = target_img # [-1,1]
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts --> white same as GT ...
        im_h = source_img * phead - (1 - phead) # [-1,1], fill -1 for other parts, thus become black visual
        

        #warped cloth sobel (gradient)
        parse_array = np.array((Image.open(source_parse_path)))    # change from target_parse_path to source_parse_path
     #   palm_path = os.path.join('dataset', 'palm-mask')
        hand_mask = Image.open(os.path.join('/local_storage/users/sanazsab/dataset/train1/palm-mask' , source_splitext.split('/')[1] + '_palm_mask.png')) #change from target to source
        hand_mask = (np.array(hand_mask) > 0).astype(np.float32)
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)
        # parse_neck = (parse_array == 10).astype(np.float32)
        parse_lower = (parse_array == 16).astype(np.float32) + \
                (parse_array == 12).astype(np.float32) + \
                (parse_array == 17).astype(np.float32) + \
                (parse_array == 9).astype(np.float32) + \
                (parse_array == 18).astype(np.float32) + \
                (parse_array == 19).astype(np.float32)
        parse_head_hand_lower = parse_head + hand_mask + parse_lower
        phhlm = torch.from_numpy(parse_head_hand_lower).unsqueeze(0)
        im_hhl = source_img * phhlm - (1 - phhlm) # [-1,1], fill -1 for other parts   #change from im to source_img


        # head (include neck) & arm & lower sobel
        person_sobelx = Image.open(os.path.join('/local_storage/users/sanazsab/dataset/image-sobel', source_splitext.split('/')[1] + '_sobelx.png')).convert('L')
        person_sobely = Image.open(os.path.join('/local_storage/users/sanazsab/dataset/image-sobel', source_splitext.split('/')[1] + '_sobely.png')).convert('L')   #change from target to source
        person_sobelx, person_sobely = self.transform(person_sobelx), self.transform(person_sobely)
        parse_arm = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)
        parse_head_arm_lower = parse_head + parse_arm + parse_lower
        phalm = torch.from_numpy(parse_head_arm_lower).unsqueeze(0)
        image_without_cloth_gradient_sobelx = person_sobelx * phalm - (1 - phalm) # [-1,1], fill -1 for other parts
        image_without_cloth_gradient_sobely = person_sobely * phalm - (1 - phalm) # [-1,1], fill -1 for other parts



        # im depth (front)
        gt_fdepth = np.load(os.path.join('/local_storage/users/sanazsab/dataset', 'depth', target_splitext.split('/')[1] + '_depth.npy'))
        gt_fdepth_m = (gt_fdepth > 0).astype(np.float32)
        gt_fdepth = -1 * (2 * gt_fdepth -1) # viewport -> ndc -> world
        gt_fdepth = gt_fdepth * gt_fdepth_m
        gt_fdepth = torch.from_numpy(gt_fdepth).unsqueeze(0)
        
        initial_fdepth_name = target_splitext.split('/')[1].replace('whole_front', 'initial_front_depth.npy') 
        initial_fdepth_path = os.path.join('/local_storage/users/sanazsab/dataset', 'initial-depth', initial_fdepth_name)
        if os.path.exists(initial_fdepth_path):
            fdepth_initial = np.load(initial_fdepth_path)
            fdepth_initial = torch.from_numpy(fdepth_initial).unsqueeze(0)
        else:
            fdepth_initial = gt_fdepth


        # im depth (back)
        gt_bdepth = np.load(os.path.join('/local_storage/users/sanazsab/dataset', 'depth', target_splitext.split('/')[1].replace('front', 'back_depth.npy')))
        gt_bdepth = np.flip(gt_bdepth, axis = 1) # align with imfd
        gt_bdepth_m = (gt_bdepth > 0).astype(np.float32)
        gt_bdepth = 2 * gt_bdepth -1 # viewport -> ndc -> world
        gt_bdepth = gt_bdepth * gt_bdepth_m
        gt_bdepth = torch.from_numpy(gt_bdepth).unsqueeze(0)

       
        initial_bdepth_name = target_splitext.split('/')[1].replace('whole_front', 'initial_back_depth.npy') 
        initial_bdepth_path = os.path.join('/local_storage/users/sanazsab/dataset', 'initial-depth', initial_bdepth_name)
        if os.path.exists(initial_bdepth_path):
            bdepth_initial = np.load(initial_bdepth_path)
            bdepth_initial = torch.from_numpy(bdepth_initial).unsqueeze(0)
        else:
            bdepth_initial = gt_bdepth


        # pose heatmap embedding
        source_pose_path = os.path.join('/local_storage/users/sanazsab/dataset/pose_coco', self.mode, source_splitext.split('/')[1] +'_keypoints.json')
        with open(source_pose_path, 'r') as f:
            a = json.load(f)
            source_pose = a['people'][0]['pose_keypoints_2d']
        source_pose_loc = pose_utils.pose2loc(source_pose)
        source_pose_embedding = pose_utils.heatmap_embedding(self.size, source_pose_loc)
        

        target_pose_path = os.path.join('/local_storage/users/sanazsab/dataset/pose_coco', self.mode, target_splitext.split('/')[1] +'_keypoints.json')
        with open(target_pose_path, 'r') as f:
            a = json.load(f)
            target_pose = a['people'][0]['pose_keypoints_2d']
        target_pose_loc = pose_utils.pose2loc(target_pose)
        target_pose_embedding = pose_utils.heatmap_embedding(self.size, target_pose_loc)
        target_pose_img, _ = pose_utils.draw_pose_from_cords(target_pose_loc, (256, 192))

        result = {
                'source_parse': source_parse,
                'target_parse': target_parse,
                'source_parse_vis': source_parse_vis,
                'target_parse_vis': target_parse_vis,
                'source_pose_embedding': source_pose_embedding,
                'target_pose_embedding': target_pose_embedding,
                'target_pose_loc': target_pose_loc,
                'source_image': source_img, 
                'target_image': target_img,
                'cloth_image' : cloth_img,
                'cloth_parse' : cloth_parse,
                'source_parse_shape': source_parse_shape,
                'im_h': im_h, # source image head and hair
                'im_c': im_c, # target_cloth_image_warped
                'source_image_name': source_splitext,
                'target_image_name': target_splitext,
                'cloth_image_name': cloth_splitext,
                'warped_cloth_image': warped_cloth_img,
                'warped_cloth_name': warped_cloth_name,
                'warped_cloth_path': warped_cloth_path,
                'source_img_path': source_img_path,
                'target_img_path': target_img_path,
                'target_pose_path': target_pose_path,
                'target_parse_path': target_parse_path, 
                'source_parse_vis_path': source_parse_vis_path,
                'target_parse_vis_path': target_parse_vis_path,
                'target_pose_img': target_pose_img,
                'warped_cloth_parse': warped_cloth_parse,
                'target_parse_cloth': target_parse_cloth,
                'warped_cloth_gradient_sobelx': warped_cloth_gradient_sobelx,
                'warped_cloth_gradient_sobely': warped_cloth_gradient_sobely,
                'im_hhl': im_hhl, #source image head hand lower
                'phhlm': phhlm, #parse head hand lower
                'phalm': phalm, #parse head arm lower
                'image_without_cloth_gradient_sobelx': image_without_cloth_gradient_sobelx,
                'image_without_cloth_gradient_sobely': image_without_cloth_gradient_sobely,
                'person_sobelx': person_sobelx,
                'person_sobely': person_sobely,
                'gt_fdepth': gt_fdepth,  #ground truth
                'gt_bdepth': gt_bdepth,   #ground truth
                'fdepth_initial': fdepth_initial,
                'bdepth_initial': bdepth_initial,
                'im_mask': im_mask,
        }

        return result

    def __len__(self):

        return len(self.img_list)
    
    def make_pair(self, pair_list, test_num, pair_num):

        test_list = list(filter(lambda p: p.split('\t')[3] == 'test', pair_list))
        img_source = [i.split('\t')[0] for i in test_list]
        img_target = [i.split('\t')[1] for i in test_list]
        cloth_img = [i.split('\t')[2] for i in test_list]

        selected_img = random.sample(img_source, test_num)
        selected_target = random.sample(img_target, pair_num)
        selected_cloth = random.sample(cloth_img, pair_num)

        pair_list = []

        with open('demo/uniform_test.txt', 'w') as f:
            for i in range(test_num):
                for j in range(pair_num):
                    pair = selected_img[i] + '\t' + selected_target[j] + '\t' + selected_cloth[j] + '\t' + 'test'
                    f.write(pair + '\n')
                    pair_list.append(pair)
        return pair_list
    
    def open_transform(self, path, downsample=False):
        img = Image.open(path)
        if downsample:
            img = img.resize((96,128),Image.BICUBIC)
        img = self.transforms['3'](img)
        return img
    
    def parse_cloth(self, path, downsample=False):            
        cloth_parse = Image.open(path)
        cloth_parse_array = np.array(cloth_parse)
        cloth_parse = (cloth_parse_array == 255).astype(np.float32) # 0 | 1
        cloth_parse = cloth_parse[np.newaxis, :]
        
        if downsample:
            [X, Y] = np.meshgrid(range(0,192,2),range(0,256,2))                 
            cloth_parse = cloth_parse[:,Y,X]

        cloth_parse = torch.from_numpy(cloth_parse)

        return cloth_parse

if __name__ == '__main__':
    pass
