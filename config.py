import argparse

class Config():

    def __init__(self):
        pass

    def parse(self):
        parser = argparse.ArgumentParser(description='GAN generation')

        ###GMM
        parser.add_argument('--input1_ng1', type=int, default=22, help='# of input image channels: 3 for RGB and 1 for grayscale')  # [3,3,20]  36 / 23 / 26 parsing/gen/MPV gen/
        parser.add_argument('--input2_ng1', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale') 
        parser.add_argument('--input1_ng2', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')  # [3,3,20]  36 / 23 / 26 parsing/gen/MPV gen/
        parser.add_argument('--input2_ng2', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')  

        ###parsing
        parser.add_argument('--input_nc_G_parsing', type=int, default=36, help='# of input image channels: 3 for RGB and 1 for grayscale')  # [3,3,20]  36 / 23 / 26 parsing/gen/MPV gen/
        parser.add_argument('--input_nc_D_parsing', type=int, default=56, help='# of input image channels: 3 for RGB and 1 for grayscale')  # 40 / 6  / 6
        parser.add_argument('--output_nc_parsing', type=int, default=20, help='# of output image channels: 3 for RGB and 1 for grayscale') # 20 / 3 / 3
        parser.add_argument('--netD_parsing', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG_parsing', type=str, default='unet_256', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')


        ###depthi
        parser.add_argument('--input_nc_G_depthi', type=int, default=36, help='# of input image channels: 3 for RGB and 1 for grayscale')  # [3,3,20]  36 / 23 / 26 parsing/gen/MPV gen/
        parser.add_argument('--input_nc_D_depthi', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')  # 40 / 6  / 6
        parser.add_argument('--output_nc_depthi', type=int, default=2, help='# of output image channels: 3 for RGB and 1 for grayscale') # 20 / 3 / 3
        parser.add_argument('--netD_depthi', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG_depthi', type=str, default='unet_256', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')


        ###appearance
    #    parser.add_argument('--input_nc_G_app', type=int, default=26, help='# of input image channels: 3 for RGB and 1 for grayscale')  # [3,3,20]  36 / 23 / 26 parsing/gen/MPV gen/
        parser.add_argument('--input_nc_G_app', type=int, default=6, help='# of input image channels: 3 for RGB and 1 for grayscale')  # [3,3,20]  36 / 23 / 26 parsing/gen/MPV gen/
        parser.add_argument('--input_nc_D_app', type=int, default=6, help='# of input image channels: 3 for RGB and 1 for grayscale')  # 40 / 6  / 6
        parser.add_argument('--output_nc_app', type=int, default=4, help='# of output image channels: 3 for RGB and 1 for grayscale') # 20 / 3 / 3
        parser.add_argument('--netD_app', type=str, default='resnet_blocks', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG_app', type=str, default='treeresnet', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')

        ###face
        parser.add_argument('--input_nc_G_face', type=int, default=6, help='# of input image channels: 3 for RGB and 1 for grayscale')  # [3,3,20]  36 / 23 / 26 parsing/gen/MPV gen/
        parser.add_argument('--input_nc_D_face', type=int, default=6, help='# of input image channels: 3 for RGB and 1 for grayscale')  # 40 / 6  / 6
        parser.add_argument('--output_nc_face', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale') # 20 / 3 / 3
        parser.add_argument('--netD_face', type=str, default='resnet_blocks', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG_face', type=str, default='treeresnet', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        
       ###depth
        parser.add_argument('--input_gradient', action='store_true', default=True, help='if specified, add image (sobel) gradient to input')
        parser.add_argument('--input_nc', type=int, default=8, help='input nc for DRM generator [base: 8 | input gradient: + 4]')
        parser.add_argument('--output_nc', type=int, default=2, help='output nc for DRM generator')
        parser.set_defaults(display_ncols=2)  # rewrite default values for this model.
        parser.set_defaults(netD='basic')
        parser.add_argument('--input_nc_D', type=int, default=4, help='3-channel normal map and 1-channel mask')
        parser.add_argument('--add_gan_loss', action='store_true', help='if specified, use gan loss')
        parser.add_argument('--add_grad_loss', action='store_true', default=True, help='if specified, use depth gardient loss')
        parser.add_argument('--add_normal_loss', action='store_true', help='if specified, use normal loss')
        parser.add_argument('--lambda_depth', type=float, default=10.0, help='weight for the depth loss')
        parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for the gan loss')
        parser.add_argument('--lambda_grad', type=float, default=5.0, help='weight for the depth gradinet loss')
        parser.add_argument('--lambda_normal', type=float, default=1.5, help='weight for the depth gradinet loss')




        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')  #default False || not dropout
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--decay_iters', type=int, default=10, help='epochs for learning rate decay to zero')
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--gpu_ids', type=list, default=[0, 1, 2 , 3, 4, 5, 6, 7 , 8, 9])
        parser.add_argument('--gpu_ids_n', type=str, default=2)
        parser.add_argument('--beta1', type=float, default=0.5)
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--epoch', type=int, default=200)
        parser.add_argument('--size', type=tuple, default=(256,192))
        parser.add_argument('--num_workers', type=int, default=4)   #16
        parser.add_argument('--gan_mode', type=str, default='lsgan')  # lsgan or vanilla? lsgan is better compared with vanilla  make sense bceloss
        parser.add_argument('--save_epoch_freq', type=int, default=1)
        parser.add_argument('--print_freq', type=int, default=10)
        parser.add_argument('--val_freq', type=int, default=200)
        parser.add_argument('--batch_size_t', type=int, default=64)     #128 64  depth:1 face=8
        parser.add_argument('--batch_size_v', type=int, default=16)   #16

        parser.add_argument('--suffix', default='new', type=str)
        parser.add_argument('--train_mode', default='parsing', type=str)
        parser.add_argument('--dataset', default='MPV', type=str)
        parser.add_argument('--dataset_mode', default='regular', type=str)
        
        parser.add_argument('--lambda_L1', type=float, default=10)
        parser.add_argument('--G_GAN', type=float, default=1)
        parser.add_argument('--G_VGG', type=float, default=10)
        parser.add_argument('--mask', type=float, default=10)
        parser.add_argument('--G_nn', type=float, default=1) # nnloss
        parser.add_argument('--lambda_gmm1', type=float, default=1)
        parser.add_argument('--lambda_gmm2', type=float, default=1)

        parser.add_argument('--face_vgg', type=float, default=1)
        parser.add_argument('--face_L1', type=float, default=10)
        parser.add_argument('--face_img_L1', type=float, default=1)
        parser.add_argument('--face_gan', type=float, default=3)   # gan loss

        parser.add_argument('--use_gmm', default=False, action='store_true')  
        parser.add_argument('--grid_size', type=int, default=5) # the same as cpvton  
        parser.add_argument('--fine_height', type=int, default=256)  
        parser.add_argument('--fine_width', type=int, default=192) 

        parser.add_argument('--joint', default=False, action='store_true')
        parser.add_argument('--joint_all', default=False, action='store_true')

        # forward
        parser.add_argument('--forward', default='normal', type=str)
        parser.add_argument('--isdemo', default=False, action='store_true')
        parser.add_argument('--isval', default=False, action='store_true')
        parser.add_argument('--forward_save_path', default='end2end', type=str)
        parser.add_argument('--save_time', default=False, action='store_true')

        # for edgetoshoe
        parser.add_argument('--dataroot', default=False, action='store_true')
        parser.add_argument('--pool_size', type=int, default=100)

        ### resume dir
       # parser.add_argument('--resume_gmm1', default='/Midgard/home/sanazsab/MPV3DTON/pretrained_checkpoint/step_009000.pth', type=str)
        parser.add_argument('--resume_gmm1', default='/Midgard/home/sanazsab/MPV3DTON/net_model/gmm_checkpoint/new/generator1/checkpoint_G_epoch_197_loss_0.04963_pth.tar', type=str)
        parser.add_argument('--resume_gmm2', default='/Midgard/home/sanazsab/MPV3DTON/net_model/gmm_checkpoint/new/generator2/checkpoint_G_epoch_197_loss_0.06546_pth.tar', type=str)
       # parser.add_argument('--resume_gmm2', default='/Midgard/home/sanazsab/MPV3DTON/net_model/gmm_checkpoint/generator2/checkpoint_G_epoch_40_loss_0.09704_pth.tar', type=str)
        parser.add_argument('--resume_G_parse', default='/Midgard/home/sanazsab/MPV3DTON/pretrained_checkpoint/parsing.tar', type=str)
 #       parser.add_argument('--resume_G_app', default='/Midgard/home/sanazsab/MPV3DTON/net_model/joint_checkpoint/refactornew/generator_appearance/checkpoint_G_epoch_118_loss_0.36910_pth.tar', type=str) #pretrained_checkpoint/app.tar      without parsing mode
        parser.add_argument('--resume_G_app', default='/Midgard/home/sanazsab/MPV3DTON/pretrained_checkpoint/app.tar', type=str) #pretrained_checkpoint/app.tar
      #  parser.add_argument('--resume_G_app', default='/Midgard/home/sanazsab/MPV3DTON/net_model/joint_checkpoint/refactor/generator_appearance/checkpoint_G_epoch_77_loss_2.98864_pth.tar', type=str) #pretrained_chec$
        parser.add_argument('--resume_G_face', default='/Midgard/home/sanazsab/MPV3DTON/pretrained_checkpoint/face.tar', type=str)
       # parser.add_argument('--resume_G_face', default='/Midgard/home/sanazsab/MPV3DTON/net_model/joint_checkpoint/generator_face/checkpoint_G_epoch_16_loss_3.03893_pth.tar', type=str)
        parser.add_argument('--resume_G_depth', default='/Midgard/home/sanazsab/MPV3DTON/net_model/depth_checkpoint/generator_depth/checkpoint_DRM_epoch_199_loss_4.52575_pth.tar', type=str)
        parser.add_argument('--resume_G_depthi', default='/Midgard/home/sanazsab/MPV3DTON/net_model/depthi_checkpoint/generator/checkpoint_G_epoch_197_loss_0.35948_pth.tar', type=str)

        parser.add_argument('--resume_D_parse', default='', type=str)
      #  parser.add_argument('--resume_D_app', default='/Midgard/home/sanazsab/MPV3DTON/net_model/joint_checkpoint/refactor/dicriminator_appearance/checkpoint_D_epoch_165_loss_0.24961_pth.tar', type=str)        
      #  parser.add_argument('--resume_D_face', default='/Midgard/home/sanazsab/MPV3DTON/net_model/face_checkpoint/dicriminator/checkpoint_D_epoch_199_loss_0.12830_pth.tar', type=str)
      #  parser.add_argument('--resume_D_depthi', default='/Midgard/home/sanazsab/MPV3DTON/net_model/depthi_checkpoint/dicriminator/checkpoint_D_epoch_199_loss_0.00000_pth.tar', type=str)    

        parser.add_argument('--resume_D_parse', default='', type=str)
      #  parser.add_argument('--resume_D_app', default='/Midgard/home/sanazsab/MPV3DTON/net_model/joint_checkpoint/refactor/dicriminator_appearance/checkpoint_D_epoch_165_loss_0.24961_pth.tar', type=str)        
      #  parser.add_argument('--resume_D_face', default='/Midgard/home/sanazsab/MPV3DTON/net_model/face_checkpoint/dicriminator/checkpoint_D_epoch_199_loss_0.12830_pth.tar', type=str)
      #  parser.add_argument('--resume_D_depthi', default='/Midgard/home/sanazsab/MPV3DTON/net_model/depthi_checkpoint/dicriminator/checkpoint_D_epoch_199_loss_0.00000_pth.tar', type=str)
        parser.add_argument('--resume_D_app', default='', type=str)
        parser.add_argument('--resume_D_face', default='', type=str)
        parser.add_argument('--resume_D_depthi', default='', type=str)
        parser.add_argument('--resume_D_depth', default='', type=str)
        parser.add_argument('--resume_DRM', default='', type=str)
#        parser.add_argument('--resume_FND', default='', type=str)
#        parser.add_argument('--resume_BND', default='', type=str)

        ### face refinement
        parser.add_argument('--face_residual', default=False, action='store_true')
        ### joint with parsing loss
        parser.add_argument('--joint_parse_loss', default=False, action='store_true')
        parser.add_argument('--joint_G_parsing', type=float, default=1)
        parser.add_argument('--joint_G_gmm', type=float, default=2)
        parser.add_argument('--mask_tvloss', default=False, action='store_true')

        ### train | val | demo
        parser.add_argument('--warp_cloth', default=False, action='store_true')

        args = parser.parse_args()
        print(args)
        return args
        

if __name__ == "__main__":
    pass
