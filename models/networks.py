import torch
import torch.nn as nn
from torch.nn import init
import os
import functools
from torch.optim import lr_scheduler
from .network_g import TreeResnetGenerator, UnetGenerator
from .network_d import PatchDiscriminator, PixelDiscriminator, ResnetDiscriminator
import torch.nn.functional as F
import numpy as np

def init_weights(net, init_type='normal', init_gain=0.02):
    # initial the network
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>    

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[0, 1, 2, 3, 4, 5 ,6, 7, 9]):
#    print(str(gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
   
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net = torch.nn.DataParallel(net).cuda()
    init_weights(net, init_type, init_gain=init_gain)

    return net

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_layer == 'none':
        norm_layer = lambda x:Identity()
    else:
        raise NotImplementedError('norm layer [%s] is not implemented'%norm_type)
    return norm_layer

class Define_G(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=6, with_tanh=True):
        super(Define_G, self).__init__()
        net = None
        norm_layer = get_norm_layer(norm_type=norm)
        self.netG = netG
        if netG == 'treeresnet':
            net = TreeResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, with_tanh=with_tanh)
        elif netG == 'unet_128':
            net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        elif netG == 'unet_256':
            net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
        self.model = init_net(net, init_type, init_gain, gpu_ids)
    
    def forward(self, x):
        return self.model(x)
        

class Define_D(nn.Module):
    def __init__(self, input_nc, ndf, netD, n_layers_D=3, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=6, n_blocks=3):
        super(Define_D, self).__init__()
        net = None
        norm_layer = get_norm_layer(norm_type=norm)
        if netD == 'basic':  # default PatchGAN classifier
            net = PatchDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
        elif netD == 'n_layers':  # more options
            net = PatchDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
        elif netD == 'pixel':     # classify if each pixel is real or fake 
            net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
        elif netD == 'resnet_blocks':
            net = ResnetDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=True, n_blocks=n_blocks)  # use_sigmoid
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
        self.model = init_net(net, init_type, init_gain, gpu_ids)
        
    def forward(self, x):
        return self.model(x)



class DRM(nn.Module):
    def __init__(self, in_channel, out_channel, ngf=32, norm_layer=nn.InstanceNorm2d):
        super(DRM, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ngf = ngf
        
        # size -> size / 2
        self.l0 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf * 2, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 2)
        )

        # size / 2 -> size / 4
        self.l1 = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 4)
        )

        # size / 4 -> size / 8
        self.l2 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 3, padding=1, stride=2),
            nn.ELU(),
            norm_layer(self.ngf * 8)
        )

        # size / 8 -> size / 16
        self.l3 = nn.Sequential(
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            norm_layer(self.ngf * 16)
        )

        self.block1 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        self.block2 = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 16, self.ngf * 16, 3, padding=1, stride=1)
        )

        # size / 16 -> size / 8
        self.l3u = nn.Sequential(
            nn.Conv2d(self.ngf * 24, self.ngf * 8, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 8, self.ngf * 8, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 8)
        )

        # size / 8 -> size / 4
        self.l2u = nn.Sequential(
            nn.Conv2d(self.ngf * 12, self.ngf * 4, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 4, self.ngf * 4, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 4)
        )

        # size / 4 -> size / 2
        self.l1u = nn.Sequential(
            nn.Conv2d(self.ngf * 6, self.ngf * 2, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, padding=1, stride=1),
            nn.ELU(),
            norm_layer(self.ngf * 2)
        )

        # size / 2 -> size
        self.l0u = nn.Sequential(
            nn.Conv2d(self.ngf * 2, self.ngf, 1, padding=0, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.ngf, 3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(self.ngf, self.out_channel, 3, padding=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input_data, inter_mode='bilinear'):
        x0 = self.l0(input_data)
        x1 = self.l1(x0)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x3 = self.block1(x3) + x3
        x3 = self.block2(x3) + x3
        x3u = nn.functional.interpolate(x3, size=x2.shape[2:4], mode=inter_mode)
        x3u = self.l3u(torch.cat((x3u, x2), dim=1))
        x2u = nn.functional.interpolate(x3u, size=x1.shape[2:4], mode=inter_mode)
        x2u = self.l2u(torch.cat((x2u, x1), dim=1))
        x1u = nn.functional.interpolate(x2u, size=x0.shape[2:4], mode=inter_mode)
        x1u = self.l1u(torch.cat((x1u, x0), dim=1))
        x0u = nn.functional.interpolate(x1u, size=input_data.shape[2:4], mode=inter_mode)
        x0u = self.l0u(x0u)
        return x0u


def define_DRM(input_nc, output_nc=2, ngf=32, norm='instanc', init_type='normal', init_gain=0.02, gpu_ids=[]):

    norm_layer = get_norm_layer(norm_type=norm)

    net = DRM(input_nc, output_nc, ngf, norm_layer)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):

    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == 'spectral_norm':
        net = SNDiscriminator(input_nc, ndf)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, depth_pred, depth_gt):
        loss_depth = torch.log(torch.abs(depth_pred - depth_gt) + 1).mean()
        
        return loss_depth

class DepthGradLoss(nn.Module):
    def __init__(self):
        super(DepthGradLoss, self).__init__()

    def forward(self, depth_grad_pred, depth_grad_gt):
        depth_grad_gt_dx = depth_grad_gt[:, 0, :, :].unsqueeze(1)
        depth_grad_gt_dy = depth_grad_gt[:, 1, :, :].unsqueeze(1)
        depth_grad_pred_dx = depth_grad_pred[:, 0, :, :].unsqueeze(1)
        depth_grad_pred_dy = depth_grad_pred[:, 1, :, :].unsqueeze(1)
        
        loss_dx = torch.log(torch.abs(depth_grad_pred_dx - depth_grad_gt_dx) + 1).mean()
        loss_dy = torch.log(torch.abs(depth_grad_pred_dy - depth_grad_gt_dy) + 1).mean()
        
        loss_grad = loss_dx + loss_dy
    
        return loss_grad

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-12)

    def forward(self, normal_pred, normal_gt):
        
        loss_normal = (1 - self.cos(normal_pred, normal_gt)).mean()
        
        return loss_normal


class Sobel(nn.Module):
    """ Soebl operator to calculate depth grad. """

    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """x: depth map (batch_size,1,H,W)"""
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
        return out

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class SNDiscriminator(nn.Module):
    """Defines a DCGAN-like spectral norm discriminator (SNGAN)"""
    def __init__(self, input_nc, ndf=64):
        super(SNDiscriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(input_nc, ndf, 3, stride=1, padding=(1,1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))


        self.fc = SpectralNorm(nn.Linear(4 * 4 * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(0.1)(self.conv1(m))
        m = nn.LeakyReLU(0.1)(self.conv2(m))
        m = nn.LeakyReLU(0.1)(self.conv3(m))
        m = nn.LeakyReLU(0.1)(self.conv4(m))
        m = nn.LeakyReLU(0.1)(self.conv5(m))
        m = nn.LeakyReLU(0.1)(self.conv6(m))
        m = nn.LeakyReLU(0.1)(self.conv7(m))

        return self.fc(m.view(-1, 4 * 4 * 512))
