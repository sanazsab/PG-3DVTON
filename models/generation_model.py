import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn 
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision import models
from .base_model import BaseModel
from . import networks
from models.networks import Define_G, Define_D
import sys
sys.path.append('..')
from utils.transforms import create_part
import torch.nn.functional as F
from utils import pose_utils
from lib.geometric_matching_multi_gpu import GMM
from .base_model import BaseModel
from time import time
from utils import pose_utils
import os.path as osp
from torchvision import utils
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
import functools
import random
from utils import util


class GenerationModel(BaseModel):
    def name(self):
        return 'Generation model: pix2pix | pix2pixHD'
    def __init__(self, opt):
        self.t0 = time()
        BaseModel.__init__(self, opt)
        self.train_mode = opt.train_mode
        self.input_gradient = opt.input_gradient
        self.use_gan_loss = opt.add_gan_loss
        self.use_grad_loss = opt.add_grad_loss
        self.use_normal_loss = opt.add_normal_loss
        if self.use_grad_loss:
            self.compute_grad = networks.Sobel().cuda()


        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G', 'fdepth', 'bdepth']
        if self.use_grad_loss:
            self.loss_names.extend(['fgrad', 'bgrad'])
        if self.use_normal_loss:
            self.loss_names.extend(['fnormal', 'bnormal'])
        if self.use_gan_loss:
            self.loss_names.extend(['fgan','bgan', 'FND', 'BND'])


       # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['warped_cloth_image', 'im_hhl', 'fdepth_initial', 'bdepth_initial']
        if self.input_gradient:
            self.visual_names.extend(['image_without_cloth_gradient_sobelx', 'image_without_cloth_gradient_sobely', 'warped_cloth_image_sobelx', 'warped_cloth_image_sobely'])
        self.visual_names.extend(['imfd_pred','imbd_pred', 'imfd_diff', 'imbd_diff'])
        if self.use_grad_loss:
            self.visual_names.extend(['fgrad_pred_x','fgrad_x', 'fgrad_pred_y', 'fgrad_y', 'fgrad_x_diff', 'fgrad_y_diff', 'bgrad_pred_x', 'bgrad_x', 'bgrad_pred_y', 'bgrad_y',  'bgrad_x_diff', 'bgrad_y_diff'])
        if self.use_normal_loss or self.use_gan_loss:
            self.visual_names.extend(['imfn_pred', 'imfn', 'imbn_pred', 'imbn', 'imfn_diff', 'imbn_diff'])


        if  self.use_gan_loss:
            self.model_names = ['DRM', 'FND', 'BND']
        else:
            self.model_names = ['DRM']

        # resume of networks 
        resume_gmm1 = opt.resume_gmm1
        resume_gmm2 = opt.resume_gmm2
        resume_G_parse = opt.resume_G_parse
        resume_D_parse = opt.resume_D_parse
        resume_G_depthi = opt.resume_G_depthi
        resume_D_depthi = opt.resume_D_depthi
        resume_G_appearance = opt.resume_G_app
        resume_D_appearance = opt.resume_D_app
        resume_G_face = opt.resume_G_face
        resume_D_face = opt.resume_D_face
        resume_G_depth = opt.resume_G_depth
        resume_D_depth = opt.resume_D_depth
        resume_DRM = opt.resume_DRM
       # resume_netFND = opt.resume_netFND
       # resume_netBND = opt.resume_netBND

        # define network
        self.gmm1_model = torch.nn.DataParallel(GMM(opt.input1_ng1, opt.input2_ng1, opt)).cuda()
        self.gmm2_model = torch.nn.DataParallel(GMM(opt.input1_ng2, opt.input2_ng2, opt)).cuda()
        self.generator_parsing = Define_G(opt.input_nc_G_parsing, opt.output_nc_parsing, opt.ndf, opt.netG_parsing, opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids_n)
        self.discriminator_parsing = Define_D(opt.input_nc_D_parsing, opt.ndf, opt.netD_parsing, opt.n_layers_D, 
                                        opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids_n)

        if opt.train_mode == 'depthi':

            self.generator_depthi = Define_G(opt.input_nc_G_depthi, opt.output_nc_depthi, opt.ndf, opt.netG_depthi, opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids_n)
            self.discriminator_depthi = Define_D(opt.input_nc_D_depthi, opt.ndf, opt.netD_depthi, opt.n_layers_D, 
                                        opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids_n)

        self.generator_appearance = Define_G(opt.input_nc_G_app, opt.output_nc_app, opt.ndf, opt.netG_app, opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids_n, with_tanh=False)
        self.discriminator_appearance = Define_D(opt.input_nc_D_app, opt.ndf, opt.netD_app, opt.n_layers_D, 
                                        opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids_n)

        self.generator_face = Define_G(opt.input_nc_D_face, opt.output_nc_face, opt.ndf, opt.netG_face, opt.norm, 
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids_n)
        self.discriminator_face = Define_D(opt.input_nc_D_face, opt.ndf, opt.netD_face, opt.n_layers_D, 
                                        opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids_n)

        if self.input_gradient:
            opt.input_nc += 4
        self.netDRM = networks.define_DRM(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids_n).cuda()
        if self.use_gan_loss: # define front & back normal discriminator
            self.netFND = networks.define_D(opt.input_nc_D, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids_n).cuda()
            self.netBND = networks.define_D(opt.input_nc_D, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids_n).cuda()


        if opt.train_mode == 'gmm':
            setattr(self, 'generator1', self.gmm1_model)
            setattr(self, 'generator2', self.gmm2_model)
#        elif opt.train_mode == 'gmm2':
#            setattr(self, 'generator2', self.gmm2_model)

        elif opt.train_mode == 'depth':
            setattr(self, 'generator_depth', self.netDRM)
            if self.use_gan_loss:
                setattr(self, 'discriminator_FND', self.netFND)
                setattr(self, 'discriminator_BND', self.netBND)
        else:
            setattr(self, 'generator', getattr(self, 'generator_' + self.train_mode))
            setattr(self, 'discriminator', getattr(self, 'discriminator_' + self.train_mode))


        # load networks
        if opt.train_mode == 'appearance':
            self.networks_name = ['gmm', 'gmm', 'parsing', 'parsing' , 'appearance', 'appearance', 'face', 'face']
            self.networks_model = [self.gmm1_model, self.gmm2_model, self.generator_parsing, self.discriminator_parsing, self.generator_appearance, self.discriminator_appearance, 
                             self.generator_face, self.discriminator_face]
            self.resume_path = [resume_gmm1, resume_gmm2, resume_G_parse, resume_D_parse,  resume_G_appearance, resume_D_appearance, resume_G_face, resume_D_face]


    #        self.networks_name = ['gmm', 'gmm' , 'appearance', 'appearance', 'face', 'face']
    #        self.networks_model = [self.gmm1_model, self.gmm2_model, self.generator_appearance, self.discriminator_appearance, 
    #                         self.generator_face, self.discriminator_face]
   #         self.resume_path = [resume_gmm1, resume_gmm2,  resume_G_appearance, resume_D_appearance, resume_G_face, resume_D_face]



        else:
            self.networks_name = ['gmm', 'gmm', 'parsing', 'parsing', 'depthi' , 'depthi' , 'appearance', 'appearance', 'face', 'face', 'depth']
            self.networks_model = [self.gmm1_model, self.gmm2_model, self.generator_parsing, self.discriminator_parsing, self.generator_depthi, self.discriminator_depthi, self.generator_appearance, self.discriminator_appearance, 
                            self.generator_face, self.discriminator_face, self.netDRM]
            self.resume_path = [resume_gmm1, resume_gmm2, resume_G_parse, resume_D_parse, resume_G_depthi, resume_D_depthi,  resume_G_appearance, resume_D_appearance, resume_G_face, resume_D_face, resume_DRM]
        
        self.networks = dict(zip(self.networks_name, self.networks_model))

        for network, resume in zip(self.networks_model, self.resume_path):
            if network != [] and resume != '':
                assert(osp.exists(resume), 'the resume not exits')
                print('loading...')
                self.load_network(network, resume, ifprint=False)

        # define optimizer
        self.optimizer_gmm1 = torch.optim.Adam(self.gmm1_model.parameters(), lr=opt.lr, betas=[0.5, 0.999])
        self.optimizer_gmm2 = torch.optim.Adam(self.gmm2_model.parameters(), lr=opt.lr, betas=[0, 0.999])

        self.optimizer_parsing_G = torch.optim.Adam(self.generator_parsing.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        self.optimizer_parsing_D = torch.optim.Adam(self.discriminator_parsing.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        
        if opt.train_mode == 'depthi':        

            self.optimizer_depthi_G = torch.optim.Adam(self.generator_depthi.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
            self.optimizer_depthi_D = torch.optim.Adam(self.discriminator_depthi.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        
        self.optimizer_appearance_G = torch.optim.Adam(self.generator_appearance.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        self.optimizer_appearance_D = torch.optim.Adam(self.discriminator_appearance.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])

        self.optimizer_face_G = torch.optim.Adam(self.generator_face.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        self.optimizer_face_D = torch.optim.Adam(self.discriminator_face.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])


        self.optimizer_depth_G = torch.optim.Adam(self.netDRM.parameters(), lr=opt.lr, betas=[0.5, 0.999])
        self.optimizers = [self.optimizer_depth_G]
        if self.use_gan_loss:
            self.optimizer_FND = torch.optim.Adam(self.netFND.parameters(), lr=opt.lr, betas=[0.5, 0.999])
            self.optimizer_BND = torch.optim.Adam(self.netBND.parameters(), lr=opt.lr, betas=[0.5, 0.999])
            self.optimizers.extend([self.optimizer_FND, self.optimizer_BND])

        if opt.train_mode == 'gmm':
            self.optimizer_G = [self.optimizer_gmm1, self.optimizer_gmm2]
           # self.optimizer_G = self.optimizer_gmm2

#        elif opt.train_mode == 'gmm2':
#            self.optimizer_G = self.optimizer_gmm2  

        elif opt.joint_all:

            self.optimizer_G = [self.optimizer_gmm1, self.optimizer_gmm2,  self.optimizer_appearance_G, self.optimizer_face_G]
            
            self.optimizer_D = [ self.optimizer_appearance_D, self.optimizer_face_D]

            self.optimizer_G = [self.optimizer_gmm1, self.optimizer_gmm2, self.optimizer_parsing_G, self.optimizer_appearance_G, self.optimizer_face_G]
            
            self.optimizer_D = [self.optimizer_parsing_D, self.optimizer_appearance_D, self.optimizer_face_D]

        elif opt.train_mode == 'depth': #and self.use_gan_loss:
              self.optimizer_G = self.optimizer_depth_G
             # self.optimizer_D = [self.optimizer_FND, self.optimizer_BND]    
        else:
            setattr(self, 'optimizer_G', getattr(self, 'optimizer_' + self.train_mode + '_G'))
            setattr(self, 'optimizer_D', getattr(self, 'optimizer_' + self.train_mode + '_D'))
        self.t1 = time()

    def set_input(self, opt, result):

        self.t2 = time()
        
        self.source_pose_embedding = result['source_pose_embedding'].float().cuda()
        self.target_pose_embedding = result['target_pose_embedding'].float().cuda()
        self.source_image = result['source_image'].float().cuda()
        self.target_image = result['target_image'].float().cuda()
        self.source_parse = result['source_parse'].float().cuda()
        self.target_parse = result['target_parse'].float().cuda()
        self.cloth_parse = result['cloth_parse'].float().cuda()
        self.warped_cloth = result['warped_cloth_image'].float().cuda() # preprocess warped image from gmm model
        self.target_parse_cloth = result['target_parse_cloth'].float().cuda()
        self.target_pose_img = result['target_pose_img'].float().cuda()
        self.image_without_cloth = create_part(self.source_image, self.source_parse, 'image_without_cloth', False)
        self.im_c = result['im_c'].float().cuda() # target warped cloth
        self.cloth_image = result['cloth_image'].float().cuda()

        index = [x for x in list(range(20)) if x != 5 and x != 6 and x != 7]
        real_s_ = torch.index_select(self.source_parse, 1, torch.tensor(index).cuda())
        self.input_parsing = torch.cat((real_s_, self.target_pose_embedding, self.cloth_parse), 1).cuda()
        
        
        if opt.train_mode == 'gmm':  # or opt.train_mode == 'gmm2'
            self.im_h = result['im_h'].float().cuda()
            self.source_parse_shape = result['source_parse_shape'].float().cuda()
            self.agnostic = torch.cat((self.source_parse_shape, self.im_h, self.target_pose_embedding), dim=1)

            if opt.joint_all: # opt.joint
                generated_parsing = F.softmax(self.generator_parsing(self.input_parsing), 1)
            else:
                generated_parsing = F.softmax(self.generator_parsing(self.input_parsing), 1)
                                
            self.input_appearance = torch.cat((self.image_without_cloth, self.warped_cloth, generated_parsing), 1).cuda()

            with torch.no_grad():
                self.generated_inter = self.generator_appearance(self.input_appearance)
                p_rendered, self.m_composite = torch.split(self.generated_inter, 3, 1) 
                self.m_composite = F.sigmoid(self.m_composite)
              
        elif opt.train_mode == 'parsing':
            self.real_s = self.input_parsing
            self.source_parse_vis = result['source_parse_vis'].float().cuda()
            self.target_parse_vis = result['target_parse_vis'].float().cuda()

        elif opt.train_mode == 'depthi':
            self.gt_fdepth = result['gt_fdepth'].float().cuda()
            self.gt_bdepth = result['gt_bdepth'].float().cuda()
            self.real_s = torch.cat((self.gt_fdepth, self.gt_bdepth), 1).cuda()
    
        elif opt.train_mode == 'appearance':

            if opt.joint_all:
                self.im_h = result['im_h'].float().cuda()
                self.source_parse_shape = result['source_parse_shape'].float().cuda()
                self.agnostic = torch.cat((self.source_parse_shape, self.im_h, self.target_pose_embedding), dim=1).cuda()
                self.generated_parsing = F.softmax(self.generator_parsing(self.input_parsing), 1)
                self.generated_fdepthi, self.generated_bdepthi = torch.split(self.generator_depthi(self.input_parsing), [1,1],  1)
                self.generated_fdepthi = F.tanh(self.generated_fdepthi)
                self.generated_bdepthi = F.tanh(self.generated_bdepthi)

            else:    
                with torch.no_grad():          
                    self.generated_parsing = F.softmax(self.generator_parsing(self.input_parsing), 1)
#                    self.generated_fdepthi, self.generated_bdepthi = torch.split(self.generator_depthi(self.input_parsing), [1,1], 1)
#                    self.generated_fdepthi = torch.tanh(self.generated_fdepthi)
#                    self.generated_bdepthi = torch.tanh(self.generated_bdepthi)
            self.input_appearance = torch.cat((self.image_without_cloth, self.warped_cloth, self.generated_parsing), 1).cuda()            
#            self.input_appearance = torch.cat((self.image_without_cloth, self.warped_cloth), 1).cuda()            

#add parse            "attention please"
            generated_parsing_ = torch.argmax(self.generated_parsing, 1, keepdim=True)            
            self.generated_parsing_argmax = torch.Tensor()
            
            for _ in range(20):
                self.generated_parsing_argmax = torch.cat([self.generated_parsing_argmax.float().cuda(), (generated_parsing_ == _).float()], dim=1)
            self.warped_cloth_parse = ((generated_parsing_ == 5) + (generated_parsing_ == 6) + (generated_parsing_ == 7)).float().cuda()

            if opt.save_time:
                self.generated_parsing_vis = torch.Tensor([0]).expand_as(self.target_image)
            else:
                # decode labels cost much time
                _generated_parsing = torch.argmax(self.generated_parsing, 1, keepdim=True)
#                _generated_parsing = _generated_parsing.permute(0,2,3,1).contiguous().int()
#                self.generated_parsing_vis = pose_utils.decode_labels(_generated_parsing) #array
            
            self.real_s = self.source_image
        
        elif opt.train_mode == 'face':
            if opt.joint_all: # opt.joint
                generated_parsing = F.softmax(self.generator_parsing(self.input_parsing), 1)
                self.generated_parsing_face = F.softmax(self.generator_parsing(self.input_parsing), 1)
                self.generated_fdepthi, self.generated_bdepthi = torch.split(self.generator_depthi(self.input_parsing), [1,1],  1)
                self.generated_fdepthi = F.tanh(self.generated_fdepthi)
            else:
                generated_parsing = F.softmax(self.generator_parsing(self.input_parsing), 1)
                self.generated_fdepthi, self.generated_bdepthi = torch.split(self.generator_depthi(self.input_parsing), [1,1],  1)
                self.generated_fdepthi = F.tanh(self.generated_fdepthi)

                "attention please"
                generated_parsing_ = torch.argmax(generated_parsing, 1, keepdim=True)            
                self.generated_parsing_argmax = torch.Tensor()

                for _ in range(20):
                    self.generated_parsing_argmax = torch.cat([self.generated_parsing_argmax.float().cuda(), (generated_parsing_ == _).float()], dim=1)
                                
                # self.generated_parsing_face = generated_parsing_c
                self.generated_parsing_face = self.target_parse
            
            self.input_appearance = torch.cat((self.image_without_cloth, self.warped_cloth, generated_parsing), 1).cuda()
    
            with torch.no_grad():
                self.generated_inter = self.generator_appearance(self.input_appearance)
                p_rendered, m_composite = torch.split(self.generated_inter, 3, 1) 
                p_rendered = F.tanh(p_rendered)
                m_composite = F.sigmoid(m_composite)
                self.generated_image = self.warped_cloth * m_composite + p_rendered * (1 - m_composite)
            
            self.source_face = create_part(self.source_image, self.source_parse, 'face', False)
            self.target_face_real = create_part(self.target_image, self.generated_parsing_face, 'face', False)
            self.target_face_fake = create_part(self.generated_image, self.generated_parsing_face, 'face', False)
            self.generated_image_without_face = self.generated_image - self.target_face_fake
        
            self.input_face = torch.cat((self.source_face, self.target_face_fake), 1).cuda()
            self.real_s = self.source_face

        elif opt.train_mode == 'depth':
            self.warped_cloth_image = result['warped_cloth_image'].float().cuda()
            self.im_hhl = result['im_hhl'].float().cuda()
            self.fdepth_initial = result['fdepth_initial'].float().cuda()
            self.bdepth_initial = result['bdepth_initial'].float().cuda()
            if self.input_gradient:
                self.image_without_cloth_gradient_sobelx = result['image_without_cloth_gradient_sobelx'].float().cuda()
                self.image_without_cloth_gradient_sobely = result['image_without_cloth_gradient_sobely'].float().cuda()
                self.warped_cloth_gradient_sobelx = result['warped_cloth_gradient_sobelx'].float().cuda()
                self.warped_cloth_gradient_sobely = result['warped_cloth_gradient_sobely'].float().cuda()
            self.gt_fdepth = result['gt_fdepth'].float().cuda()
            self.gt_bdepth = result['gt_bdepth'].float().cuda()
            if self.use_grad_loss:
                self.fgrad = self.compute_grad(self.gt_fdepth) # for ground truth
                self.bgrad = self.compute_grad(self.gt_bdepth) # for ground truth


            if self.use_normal_loss or self.use_gan_loss:
                self.im_mask = result['im_mask'].float().cuda()
                self.imfn = util.depth2normal_ortho(self.gt_fdepth).float().cuda() # for ground truth
                self.imbn = util.depth2normal_ortho(self.gt_bdepth).float().cuda() # for ground truth


           # self.input_depth_refine = torch.cat((self.generated_fdepthi, self.generated_bdepthi, self.image_without_cloth, self.warped_cloth, self.warped_cloth_gradient_sobelx, self.warped_cloth_gradient_sobely, self.image_without_cloth_gradient_sobelx, self.image_without_cloth_gradient_sobely ), 1).cuda()            


        elif opt.train_mode == 'joint':
            self.input_joint = torch.cat((self.image_without_cloth, self.warped_cloth, self.generated_parsing), 1).cuda()
    
        self.t3 = time()

        # setattr(self, 'input', getattr(self, 'input_' + self.train_mode))

    def forward(self, opt):
        self.t4 = time()

        if self.train_mode == 'gmm' or opt.joint_all:
            self.grid1, self.theta1 = self.gmm1_model(self.agnostic, self.cloth_image)
            self.warped_cloth_predict = F.grid_sample(self.cloth_image, self.grid1)

            self.grid2, self.theta2 = self.gmm2_model(self.cloth_image, self.warped_cloth_predict)
            self.cloth_predict = F.grid_sample(self.warped_cloth_predict, self.grid2) 

        #    self.grid2, self.theta2 = self.gmm2_model(self.cloth_parse, self.m_composite)
        #    self.cloth_predict = F.grid_sample(self.warped_cloth_predict, self.grid2)

#        if self.train_mode == 'gmm2':
#            self.grid2, self.theta2 = self.gmm2_model(self.cloth_image, self.warped_cloth_predict)
#            self.cloth_predict = F.grid_sample(self.warped_cloth_predict, self.grid2)  


        if opt.train_mode == 'parsing':
            self.fake_t = F.softmax(self.generator_parsing(self.input_parsing), dim=1)
            self.real_t = self.target_parse
       

        if opt.train_mode == 'depthi':
            self.generated_fdepthi, self.generated_bdepthi = torch.split(self.generator_depthi(self.input_parsing), 1, 1)
            self.generated_fdepthi = F.tanh(self.generated_fdepthi)
            self.generated_bdepthi = F.tanh(self.generated_bdepthi)
            self.fake_ft =  self.generated_fdepthi
            self.fake_bt =  self.generated_bdepthi
            self.real_ft = self.gt_fdepth
            self.real_bt = self.gt_bdepth
            self.fake_t = self.generated_fdepthi
            self.real_t = self.gt_fdepth
            self.fdepth_diff = self.fake_ft - self.real_ft
            self.bdepth_diff = self.fake_bt - self.real_bt
 
        if opt.train_mode == 'appearance':
            generated_inter = self.generator_appearance(self.input_appearance)
            p_rendered, m_composite = torch.split(generated_inter, 3, 1) 
            p_rendered = F.tanh(p_rendered)
            self.m_composite = F.sigmoid(m_composite)
            p_tryon = self.warped_cloth * self.m_composite + p_rendered * (1 - self.m_composite)
            self.fake_t = p_tryon
            self.real_t = self.target_image

            if opt.joint_all and opt.joint_parse_loss:

                generate_face = create_part(self.fake_t, self.generated_parsing_argmax, 'face', False)
                generate_image_without_face = self.fake_t - generate_face

                real_s_face = create_part(self.source_image, self.source_parse, 'face', False)
                self.real_t_face = create_part(self.target_image, self.generated_parsing_argmax, 'face', False)
                input = torch.cat((real_s_face, generate_face), dim=1)

                self.fake_t_face = self.generator_face(input)
                ###residual learning
                #"""attention"""
                fake_t_face = create_part(fake_t_face, self.generated_parsing, 'face', False)
                fake_t_face = generate_face + fake_t_face
                self.fake_t_face = create_part(self.fake_t_face, self.generated_parsing_argmax, 'face', False)
                ### fake image
                self.fake_t = generate_image_without_face + self.fake_t_face

        if opt.train_mode == 'face':
            self.fake_t = self.generator_face(self.input_face)
            
            if opt.face_residual:
                self.fake_t = create_part(self.fake_t, self.generated_parsing_face, 'face', False)
                self.fake_t = self.target_face_fake + self.fake_t
            
            self.fake_t = create_part(self.fake_t, self.generated_parsing_face, 'face', False)
            self.refined_image = self.generated_image_without_face + self.fake_t
            self.real_t = create_part(self.target_image, self.generated_parsing_face, 'face', False)

        if opt.train_mode == 'depth':

            if self.input_gradient:
                self.input_depth = torch.cat([self.fdepth_initial, self.bdepth_initial, self.warped_cloth_image, self.im_hhl, self.warped_cloth_gradient_sobelx, self.warped_cloth_gradient_sobelx, self.image_without_cloth_gradient_sobelx, self.image_without_cloth_gradient_sobely], 1).cuda()
            else:
                self.input_depth = torch.cat([self.fdepth_initial, self.bdepth_initial, self.warped_cloth_image, self.im_hhl], 1).cuda()
            outputs_depth = self.netDRM(self.input_depth)
            self.imfd_pred, self.imbd_pred = torch.split(outputs_depth, 1, 1)
            self.imfd_pred = torch.tanh(self.imfd_pred)
            self.imbd_pred = torch.tanh(self.imbd_pred)

            if self.use_grad_loss:
                self.fgrad_pred = self.compute_grad(self.imfd_pred)
                self.bgrad_pred = self.compute_grad(self.imbd_pred)

            if self.use_normal_loss or self.use_gan_loss:
                self.imfn_pred = util.depth2normal_ortho(self.imfd_pred)
                self.imbn_pred = util.depth2normal_ortho(self.imbd_pred)



        self.t5 = time()

    def backward_G(self, opt):
        self.t6 = time()
                
        if opt.train_mode == 'gmm':
            self.loss1 = self.criterionL1(self.warped_cloth_predict, self.im_c)
#            self.loss = self.loss1

            self.loss2 = self.criterionL1(self.cloth_predict, self.cloth_image)
            self.loss = self.loss1 * opt.lambda_gmm1 + self.loss2 * opt.lambda_gmm2

            self.loss.backward()
            self.t7 = time()

            
            return

 #       if opt.train_mode == 'gmm2':
 #           self.loss2 = self.criterionL1(self.cloth_predict, self.cloth_image)
 #           self.loss = self.loss2
 #           self.loss.backward(retain_graph=True)
 #           self.t7 = time()
 #           return

        if opt.train_mode != 'depth':
            fake_st = torch.cat((self.real_s, self.fake_t), 1)
            pred_fake = self.discriminator(fake_st)

        
        if opt.train_mode == 'parsing':
            self.loss_G_GAN = self.criterionGAN(pred_fake,True)
            self.loss_G_BCE = self.criterionBCE_re(self.fake_t, self.real_t) * opt.lambda_L1

            self.loss_G = self.loss_G_GAN + self.loss_G_BCE
            self.loss_G.backward()

        if opt.train_mode == 'depthi':
            self.criterionDepth = torch.nn.L1Loss()
            self.loss_fdepthi = opt.lambda_depth * self.criterionDepth(self.fake_ft, self.real_ft)
            self.loss_bdepthi = opt.lambda_depth * self.criterionDepth(self.fake_bt, self.real_bt)
    
            self.loss_G = self.loss_fdepthi + self.loss_bdepthi
            self.loss_G.backward()
    
        if opt.train_mode == 'appearance':
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) * opt.G_GAN
            # vgg_loss
            loss_vgg1, _ = self.criterion_vgg(self.fake_t, self.real_t, self.target_parse, True, True, True)
            loss_vgg2, _ = self.criterion_vgg(self.fake_t, self.real_t, self.target_parse, True, True, True)
            self.loss_G_vgg = (loss_vgg1 + loss_vgg2) * opt.G_VGG
            self.loss_G_mask = self.criterionL1(self.m_composite, self.warped_cloth_parse) * opt.mask
            if opt.mask_tvloss:
                self.loss_G_mask_tv = self.criterion_tv(self.m_composite)
            else:
                self.loss_G_mask_tv = torch.Tensor([0]).cuda()
            self.loss_G_L1 = self.criterion_smooth_L1(self.fake_t, self.real_t) * opt.lambda_L1
    
            if opt.joint_all and opt.joint_parse_loss:
                self.loss1 = self.criterionL1(self.warped_cloth_predict, self.im_c)
#            self.loss = self.loss1

                self.loss2 = self.criterionL1(self.cloth_predict, self.cloth_image)
                self.loss_G_gmm = (self.loss1 + self.loss2) * opt.joint_G_gmm
                self.loss_G_parsing = self.criterionBCE_re(self.generated_parsing, self.target_parse) * opt.joint_G_parsing
                self.loss_G_face = self.criterionL1(self.fake_t_face, self.real_t_face) * opt.face_L1
                self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_vgg + self.loss_G_mask + self.loss_G_parsing + self.loss_G_gmm + self.loss_G_face
            else:
#                self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_vgg + self.loss_G_mask_tv
                self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_vgg + self.loss_G_mask + self.loss_G_mask_tv
            self.loss_G.backward()
    
        if opt.train_mode == 'face':
            _, self.loss_G_vgg = self.criterion_vgg(self.fake_t, self.real_t, self.generated_parsing_face, False, False,
                                                    False)  # part, gram, neareast
            self.loss_G_vgg = self.loss_G_vgg * opt.face_vgg
            self.loss_G_L1 = self.criterionL1(self.fake_t, self.real_t) * opt.face_L1
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) * opt.face_gan
            self.loss_G_refine = self.criterionL1(self.refined_image, self.target_image) * opt.face_img_L1
    
            self.loss_G = self.loss_G_vgg + self.loss_G_L1 + self.loss_G_GAN + self.loss_G_refine
    
        if opt.train_mode == 'depth':
    
            self.criterionDepth = networks.DepthLoss().cuda()
            if self.use_grad_loss:
                self.criterionGrad = networks.DepthGradLoss().cuda()
            if self.use_gan_loss:
                self.criterionGAN = networks.GANLoss(opt.gan_mode).cuda()
            if self.use_normal_loss:
                self.criterionNormal = networks.NormalLoss()
    
            self.loss_fdepth = self.opt.lambda_depth * self.criterionDepth(self.imfd_pred, self.gt_fdepth)
            self.loss_bdepth = self.opt.lambda_depth * self.criterionDepth(self.imbd_pred, self.gt_bdepth)
            self.loss_G = self.loss_fdepth + self.loss_bdepth
    
            if self.use_grad_loss:
                self.loss_fgrad = self.opt.lambda_grad * self.criterionGrad(self.fgrad_pred, self.fgrad)
                self.loss_bgrad = self.opt.lambda_grad * self.criterionGrad(self.bgrad_pred, self.bgrad)
                self.loss_G += self.loss_fgrad + self.loss_bgrad
    
            if self.use_normal_loss:
                self.loss_fnormal = self.opt.lambda_normal * self.criterionNormal(self.imfn_pred, self.imfn)
                self.loss_bnormal = self.opt.lambda_normal * self.criterionNormal(self.imbn_pred, self.imbn)
                self.loss_G += self.loss_fnormal + self.loss_bnormal
    
            if self.use_gan_loss:  # G(fake_input) should fake the discriminator
                pred_fake_fnormal = self.netFND(torch.cat([self.im_mask, self.imfn_pred], 1))
                pred_fake_bnormal = self.netBND(torch.cat([self.im_mask, self.imbn_pred], 1))
                self.loss_fgan = self.opt.lambda_gan * self.criterionGAN(pred_fake_fnormal, True)
                self.loss_bgan = self.opt.lambda_gan * self.criterionGAN(pred_fake_bnormal, True)
                self.loss_G += self.loss_fgan + self.loss_bgan        

            self.loss_G.backward()

        self.t7 = time()

    def backward_D(self, opt):
        self.t8 = time()

        fake_st = torch.cat((self.real_s, self.fake_t), 1)
        real_st = torch.cat((self.real_s, self.real_t), 1)
        pred_fake = self.discriminator(fake_st.detach())
        pred_real = self.discriminator(real_st) # batch_size,1, 30,30
        
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
    
        self.loss_D.backward()

        self.t9 = time()
    

    def backward_FND(self, opt):
        if opt.train_mode == 'depth':
        # Fake; stop backprop to the generator by detaching imbn_pred
            pred_fake = self.netFND(torch.cat([self.im_mask, self.imfn_pred.detach()], 1))
            loss_FND_fake = self.criterionGAN(pred_fake, False)

        # Real
            pred_real = self.netFND(torch.cat([self.im_mask, self.imfn], 1))
            loss_FND_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
            self.loss_FND = (loss_FND_fake + loss_FND_real) * 0.5
            self.loss_FND.backward()

   
    def backward_BND(self, opt):
        if opt.train_mode == 'depth':
        # Fake; stop backprop to the generator by detaching imbn_pred
            pred_fake = self.netBND(torch.cat([self.im_mask, self.imbn_pred.detach()], 1))
            loss_BND_fake = self.criterionGAN(pred_fake, False)

        # Real
            pred_real = self.netBND(torch.cat([self.im_mask, self.imbn], 1))
            loss_BND_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
            self.loss_BND = (loss_BND_fake + loss_BND_real) * 0.5
            self.loss_BND.backward()


    def optimize_parameters(self, opt):
        
        self.t10 = time()
        self.forward(opt)                    # compute fake images: G(A)

        if opt.train_mode == 'gmm':
            for _ in self.optimizer_G:
                _.zero_grad()
            
            self.backward_G(opt)

            for _ in self.optimizer_G:
                _.step()

#            self.optimizer_G.zero_grad()        # set G's gradients to zero
#            self.backward_G(opt)                  # calculate graidents for G
#            self.optimizer_G.step()
           
            self.t11 = time()
            return

#        if opt.train_mode == 'gmm2':
#            self.optimizer_G.zero_grad()        # set G's gradients to zero
#            self.backward_G(opt)                  # calculate graidents for G
#            self.optimizer_G.step()
           
#            self.t11 = time()
#            return

        

        if opt.train_mode == 'depth' and self.use_gan_loss: #  update D
            self.set_requires_grad(self.netFND, True)  # enable backprop for FND
            self.optimizer_FND.zero_grad()             # set FND's gradients to zero
            self.backward_FND(opt)                        # calculate gradients for FND
            self.optimizer_FND.step()                  # update FND's weights
            self.set_requires_grad(self.netFND, False) # FND requires no gradients when optimizing G

            self.set_requires_grad(self.netBND, True)  # enable backprop for BND
            self.optimizer_BND.zero_grad()             # set BND's gradients to zero
            self.backward_BND(opt)                        # calculate gradients for BND
            self.optimizer_BND.step()                  # update BND's weights
            self.set_requires_grad(self.netBND, False) # BND requires no gradients when optimizing G

        elif opt.joint_all:
             for _ in self.optimizer_D:
                 _.zero_grad()

             self.backward_D(opt)

             for _ in self.optimizer_D:
                 _.step()
        
        elif opt.train_mode != 'depth':

        # update D
            self.set_requires_grad(self.discriminator, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D(opt)                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
        # update G
            self.set_requires_grad(self.discriminator, False)  # D requires no gradients when optimizing G
        if opt.joint_all:
            for _ in self.optimizer_G:
                _.zero_grad()
            
            self.backward_G(opt)

            for _ in self.optimizer_G:
                _.step()
        else:
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G(opt)                  # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights

        self.t11 = time()

    def save_result(self, opt, epoch, iteration):        
        if opt.train_mode == 'gmm':
            images = [self.cloth_image, self.warped_cloth_predict.detach(), self.cloth_predict.detach(), self.im_c]

        if opt.train_mode == 'parsing':
            fake_t_vis = pose_utils.decode_labels(torch.argmax(self.fake_t, dim=1, keepdim=True).permute(0,2,3,1).contiguous())
            images = [self.source_parse_vis, self.target_parse_vis, self.target_pose_img, fake_t_vis]

        if opt.train_mode == 'depthi':
            images = [self.gt_bdepth, self.gt_fdepth, self.fake_ft, fake_bt]


        if opt.train_mode == 'appearance':
            images = [self.source_image, self.im_c, self.target_image, self.cloth_image, self.generated_parsing_vis, self.fake_t.detach()]
#            images = [self.source_image, self.im_c, self.target_image, self.cloth_image, self.fake_t.detach()]

        if opt.train_mode == 'face':
            images = [self.generated_image.detach(), self.refined_image.detach(), self.source_image, self.target_image, self.real_t, self.fake_t.detach()]


        if opt.train_mode != 'depth':
#            images = [ self.imfn_pred.detach(), self.imbn_pred.detach()]
            pose_utils.save_img(images, os.path.join(self.vis_path, opt.train_mode + 'weighgmm',  str(epoch) + '_' + str(iteration) + '.jpg'))

    def save_model(self, opt, epoch):
        if opt.train_mode == 'gmm':
            model_G1 = osp.join(self.save_dir, 'generator1', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss1))
            model_G2 = osp.join(self.save_dir, 'generator2', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss2))
           
            if not osp.exists(osp.join(self.save_dir, 'generator1')):
                os.makedirs(osp.join(self.save_dir, 'generator1'))

#        if opt.train_mode == 'gmm2':
#            model_G2 = osp.join(self.save_dir, 'generator2', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss))
            
            elif not osp.exists(osp.join(self.save_dir, 'generator2')):
                os.makedirs(osp.join(self.save_dir, 'generator2'))

            torch.save(self.generator1.state_dict(), model_G1)
            torch.save(self.generator2.state_dict(), model_G2)

        if opt.train_mode == 'depth': #and self.use_gan_loss:
            model_DRM = osp.join(self.save_dir, 'generator_depth', 'checkpoint_DRM_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_G))
           # model_FND = osp.join(self.save_dir, 'discrimintaor_FND', 'checkpoint_FND_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_FND))
           # model_BND = osp.join(self.save_dir, 'discrimintaor_BND', 'checkpoint_BND_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_BND))
           
            if not osp.exists(osp.join(self.save_dir, 'generator_depth')):
                os.makedirs(osp.join(self.save_dir, 'generator_depth'))

            
            #elif not osp.exists(osp.join(self.save_dir, 'discrimintaor_FND')):
            #    os.makedirs(osp.join(self.save_dir, 'discrimintaor_FND'))

            #elif not osp.exists(osp.join(self.save_dir, 'discrimintaor_BND')):
            #    os.makedirs(osp.join(self.save_dir, 'discrimintaor_BND'))

            torch.save(self.generator_depth.state_dict(), model_DRM)
            if self.use_gan_loss:
                torch.save(self.discriminator_FND.state_dict(), model_FND)
                torch.save(self.discriminator_BND.state_dict(), model_BND)

        
        elif not opt.joint_all and opt.train_mode != 'gmm':  
            model_G = osp.join(self.save_dir, 'generator', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_G))
            model_D = osp.join(self.save_dir, 'dicriminator', 'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_D))
            if not osp.exists(osp.join(self.save_dir, 'generator')):
                os.makedirs(osp.join(self.save_dir, 'generator'))
            if not osp.exists(osp.join(self.save_dir, 'dicriminator')):
                os.makedirs(osp.join(self.save_dir, 'dicriminator'))
        
            torch.save(self.generator.state_dict(), model_G)
            torch.save(self.discriminator.state_dict(), model_D)

        if opt.joint_all:
            model_G_gmm1 = osp.join(self.save_dir, 'gmm1_model', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_G))
          
            model_G_parsing = osp.join(self.save_dir, 'generator_parsing', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_G))
            model_D_parsing = osp.join(self.save_dir, 'dicriminator_parsing', 'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_D))

      #      model_G_depthi = osp.join(self.save_dir, 'generator_depthi', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_G))
      #      model_D_depthi = osp.join(self.save_dir, 'dicriminator_depthi', 'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_D))

            model_G_appearance = osp.join(self.save_dir, 'generator_appearance', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_G))
            model_D_appearance = osp.join(self.save_dir, 'dicriminator_appearance', 'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_D))

            model_G_face = osp.join(self.save_dir, 'generator_face', 'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_G))
            model_D_face = osp.join(self.save_dir, 'dicriminator_face', 'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar'%(epoch, self.loss_D))

            joint_save_dirs = [osp.join(self.save_dir, 'gmm1_model'), osp.join(self.save_dir, 'generator_parsing'), osp.join(self.save_dir, 'dicriminator_parsing'),
                                osp.join(self.save_dir, 'generator_depthi'), osp.join(self.save_dir, 'dicriminator_depthi'),
                                osp.join(self.save_dir, 'generator_appearance'), osp.join(self.save_dir, 'dicriminator_appearance'),
                                osp.join(self.save_dir, 'generator_face'), osp.join(self.save_dir, 'dicriminator_face')]
            for _ in joint_save_dirs:
                if not osp.exists(_):
                    os.makedirs(_)            
            torch.save(self.gmm1_model.state_dict(), model_G_gmm1)
            torch.save(self.generator_parsing.state_dict(), model_G_parsing)
            torch.save(self.generator_depthi.state_dict(), model_G_depthi)
            torch.save(self.generator_appearance.state_dict(), model_G_appearance)
            torch.save(self.generator_face.state_dict(), model_G_face)
            torch.save(self.discriminator_appearance.state_dict(), model_D_appearance)
       
    def print_current_errors(self, opt, epoch, i):
        if opt.train_mode == 'gmm':
            errors = {'loss_L1': self.loss1.item()}
        #    errors = {'loss_L1': self.loss.item()}

#        if opt.train_mode == 'gmm2':
#            errors = {'loss_L1': self.loss.item()}

        if opt.train_mode == 'appearance':
            errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_vgg':self.loss_G_vgg.item(), 'loss_G_mask':self.loss_G_mask.item(),
                        'loss_G_L1': self.loss_G_L1.item(), 'loss_D':self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake':self.loss_D_real.item(), 'loss_G_mask_tv': self.loss_G_mask_tv.item()}

          #  errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_vgg':self.loss_G_vgg.item(),
           #             'loss_G_L1': self.loss_G_L1.item(), 'loss_D':self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake':self.loss_D_real.item(), 'loss_G_mask_tv': self.loss_G_mask_tv.item()}
            
            
            if opt.joint_all and opt.joint_parse_loss:
                errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_vgg':self.loss_G_vgg.item(), 'loss_G_mask':self.loss_G_mask.item(), 'loss_G_gmm':self.loss_G_gmm.item(), 'loss_G_face': self.loss_G_face.item(),
                        'loss_G_L1': self.loss_G_L1.item(), 'loss_D':self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake':self.loss_D_real.item(), 'loss_G_parsing': self.loss_G_parsing.item(), 'loss_G_face': self.loss_G_face.item()}


        if opt.train_mode == 'parsing':
            errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_BCE': self.loss_G_BCE.item(), 
                    'loss_D':self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake':self.loss_D_real.item()}

        if opt.train_mode == 'depthi':
            errors = {'loss_G': self.loss_G.item(), 'loss_fdepthi': self.loss_fdepthi.item(), 'loss_bdepthi': self.loss_bdepthi.item(), 
                    'loss_D':self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake':self.loss_D_real.item()}


        if opt.train_mode == 'face':
            errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_vgg':self.loss_G_vgg.item(), 'loss_G_refine':self.loss_G_refine.item(),
                        'loss_G_L1': self.loss_G_L1.item(), 'loss_D':self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake':self.loss_D_real.item()}

        if opt.train_mode == 'depth':
            errors = {'loss_G': self.loss_G.item()}

            if self.use_grad_loss: 
                errors = {'loss_G': self.loss_G.item()} 

            if self.use_normal_loss: 
                errors = {'loss_G': self.loss_G.item()}

            if self.use_gan_loss: 
                errors = {'loss_G': self.loss_G.item(), 'loss_FND':self.loss_FND.item(), 'loss_BND':self.loss_BND.item()} 

 
        t = self.t11 - self.t2
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in sorted(errors.items()):
            if v != 0:
                message += '%s: %.3f ' % (k, v)
        
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def compute_visuals(self):
        """Calculate additional output images for tensorbard visualization"""
        self.imfd_diff = self.imfd_pred - self.gt_fdepth
        self.imbd_diff = self.imbd_pred - self.gt_bdepth
        if self.use_grad_loss:
            self.fgrad_pred_x = self.fgrad_pred[:,0,:,:].unsqueeze(1)
            self.fgrad_pred_y = self.fgrad_pred[:,1,:,:].unsqueeze(1)
            self.bgrad_pred_x = self.bgrad_pred[:,0,:,:].unsqueeze(1)
            self.bgrad_pred_y = self.bgrad_pred[:,1,:,:].unsqueeze(1)
            self.fgrad_x = self.fgrad[:,0,:,:].unsqueeze(1)
            self.fgrad_y = self.fgrad[:,1,:,:].unsqueeze(1)
            self.bgrad_x = self.bgrad[:,0,:,:].unsqueeze(1)
            self.bgrad_y = self.bgrad[:,1,:,:].unsqueeze(1)
            self.fgrad_x_diff = self.fgrad_pred_x - self.fgrad_x
            self.fgrad_y_diff = self.fgrad_pred_y - self.fgrad_y
            self.bgrad_x_diff = self.bgrad_pred_x - self.bgrad_x
            self.bgrad_y_diff = self.bgrad_pred_y - self.bgrad_y
        if self.use_normal_loss:
            self.imfn_diff = -torch.nn.functional.cosine_similarity(self.imfn_pred, self.imfn, dim=1, eps=1e-12).unsqueeze(1)
            self.imbn_diff = -torch.nn.functional.cosine_similarity(self.imbn_pred, self.imbn, dim=1, eps=1e-12).unsqueeze(1)            
 
