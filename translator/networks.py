#METHODS ADDED FROM ORIGINAL MUNIT:
#NOTE THAT ALL OF THESE APPEAR IN THE NEW VIDEO COLON PAPER TOO
# DecoderReLU
# DecoderWithLabels
# ResBlockSegmentation
# StylelessGen

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F

##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer    = params.n_layer
        self.gan_type   = params.gan_type
        self.dim        = params.dim
        self.norm       = params.norm
        self.activ      = params.activ
        self.num_scales = params.num_scales
        self.pad_type   = params.pad_type
        self.input_dim  = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

##################################################################################
# Generator
##################################################################################

#class AdaINGen(nn.Module):
#    # AdaIN auto-encoder architecture
#    def __init__(self, input_dim, params):
#        super(AdaINGen, self).__init__()
#        dim = params.dim
#        style_dim = params.style_dim
#        n_downsample = params.n_downsample
#        n_res = params.n_res
#        activ = params.activ
#        pad_type = params.pad_type
#        mlp_dim = params.mlp_dim
#        num_classes = params.num_classes
#        self.num_classes = num_classes
#        self.n_downsample = n_downsample
#
#        # style encoder
#        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)
#
#        # content encoder
#        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
#        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type )
#
#        # MLP to generate AdaIN parameters
#        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)
#
#    def forward(self, images):
#        # reconstruct an image
#        content, style_fake = self.encode(images)
#        images_recon = self.decode(content, style_fake)
#        return images_recon
#
#    def encode(self, images):
#
#        #HERE'S Anonymous'S CHANGE TO MAKE THEM NOT RANDOM
#        # images = images + torch.randn_like( images )*0.1
#        # encode an image to its content and style codes
#        style_fake = self.enc_style(images)
#        content = self.enc_content(images)
#        return content, style_fake
#
#    def decode(self, content, style):
#        # add noise to the style:
#        #style = style + torch.randn_like( style )*0.5
#
#        # decode content and style codes to an image
#        adain_params = self.mlp(style)
#
#        self.assign_adain_params(adain_params, self.dec)
#        images = self.dec(content)
#        return images
#
#    def assign_adain_params(self, adain_params, model):
#        # assign the adain_params to the AdaIN layers in model
#        for m in model.modules():
#            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
#                mean = adain_params[:, :m.num_features]
#                std = adain_params[:, m.num_features:2*m.num_features]
#                m.bias = mean.contiguous().view(-1)
#                m.weight = std.contiguous().view(-1)
#                if adain_params.size(1) > 2*m.num_features:
#                    adain_params = adain_params[:, 2*m.num_features:]
#
#    def get_num_adain_params(self, model):
#        # return the number of AdaIN parameters needed by the model
#        num_adain_params = 0
#        for m in model.modules():
#            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
#                num_adain_params += 2*m.num_features
#        return num_adain_params

class StylelessGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(StylelessGen, self).__init__()
        dim          = params.dim
        #style_dim  = params.style_dim
        n_downsample = params.n_downsample
        n_res        = params.n_res
        activ        = params.activ
        pad_type     = params.pad_type
        self.image_noise_level = .1 #TODO: Make this a parameter

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):
        # reconstruct an image
        content = self.encode(images)
        images_recon = self.decode(content)
        return images_recon

    def encode(self, images):
        if self.training:
            #Add random noise to the input when we're training
            images = images + torch.randn_like( images ) * self.image_noise_level

        # encode an image to its content code
        content = self.enc_content(images)
        return content 

    def decode(self, content):
        # decode content codes to an image
        images = self.dec(content)
        return images

# class VAEGen(nn.Module):
#     # VAE architecture
#     def __init__(self, input_dim, params):
#         super(VAEGen, self).__init__()
#         dim = params.dim
#         n_downsample = params.n_downsample
#         n_res = params.n_res
#         activ = params.activ
#         pad_type = params.pad_type
#
#         # content encoder
#         self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
#         self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)
#
#     def forward(self, images):
#         # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
#         hiddens = self.encode(images)
#         if self.training == True:
#             noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
#             images_recon = self.decode(hiddens + noise)
#         else:
#             images_recon = self.decode(hiddens)
#         return images_recon, hiddens
#
#     def encode(self, images):
#         hiddens = self.enc(images)
#         noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
#         return hiddens, noise
#
#     def decode(self, hiddens):
#         images = self.dec(hiddens)
#         return images

# class ResBlockSegmentation(nn.Module):
#     # AdaIN auto-encoder architecture
#     def __init__(self, input_dim, params):
#         super(ResBlockSegmentation, self).__init__()
#         dim = params.dim
#         n_downsample = 3
#         n_res = 3
#         activ = params.activ
#         pad_type = params.pad_type
#         num_classes = params.num_classes
#         self.n_downsample = n_downsample
#
#         # content encoder
#         self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
#         self.dec = DecoderReLU(n_downsample, n_res, self.enc_content.output_dim, num_classes, res_norm='in', activ=activ, pad_type=pad_type )
#         self.softMax = nn.Softmax(dim=1)
#
#     def forward(self, images):
#         content = self.enc_content(images)
#         segmentation = self.dec(content)
#         return self.softMax( segmentation )
#
#     def set_requires_grad( self, b ):
#         for p in self.parameters():
#             p.requires_grad = b
#         for p in self.enc_content.parameters():
#             p.requires_grad = b
#         for p in self.dec.parameters():
#             p.requires_grad = b


##################################################################################
# Encoder and Decoders
##################################################################################

#class StyleEncoder(nn.Module):
#    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
#        super(StyleEncoder, self).__init__()
#        self.model = []
#        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
#        for i in range(2):
#            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
#            #self.model += [nn.Dropout(p=0.2)]
#            dim *= 2
#        for i in range(n_downsample - 2):
#            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
#            #self.model += [nn.Dropout(p=0.2)]
#
#        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
#        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
#        self.model = nn.Sequential(*self.model)
#        self.output_dim = dim
#
#    def forward(self, x):
#        return self.model(x)

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            #self.model += [nn.Dropout(p=0.2)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, 8, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [Conv2dBlock(8, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        #self.model += [nn.Dropout(p=0.2)]
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            #self.model += [nn.Dropout(p=0.2)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

#class DecoderReLU(nn.Module):
#    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
#        super(DecoderReLU, self).__init__()
#
#        self.model = []
#        self.model += [Conv2dBlock(8, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
#        #self.model += [nn.Dropout(p=0.2)]
#        # AdaIN residual blocks
#        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
#        # upsampling blocks
#        for i in range(n_upsample):
#            self.model += [nn.Upsample(scale_factor=2),
#                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
#            #self.model += [nn.Dropout(p=0.2)]
#            dim //= 2
#        # use reflection padding in the last conv layer
#        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='relu', pad_type=pad_type)]
#        self.model = nn.Sequential(*self.model)
#
#    def forward(self, x):
#        return self.model(x)


# class DecoderWithLabels(nn.Module):
#     def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero', num_classes=7 ):
#         super(DecoderWithLabels, self).__init__()
#
#         self.sharedLayers = []
#         self.decBranch = []
#         self.lblBranch = []
#
#         self.sharedLayers += [Conv2dBlock(8, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
#         # Shared AdaIN residual blocks
#         self.sharedLayers += [ResBlocks(n_res-1, dim, res_norm, activ, pad_type=pad_type)]
#
#         # Separate residual block:
#         self.decBranch += [ResBlock(dim, norm=res_norm, activation=activ, pad_type=pad_type)]
#         self.lblBranch += [ResBlock(dim, norm='in', activation=activ, pad_type=pad_type)]
#
#         # upsampling blocks
#         for i in range(n_upsample):
#             self.decBranch += [nn.Upsample(scale_factor=2),
#                            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
#             self.lblBranch += [nn.Upsample(scale_factor=2),
#                            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
#             dim //= 2
#         # use reflection padding in the last conv layer
#         self.decBranch += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
#         self.lblBranch += [Conv2dBlock(dim, num_classes, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
#
#         self.sharedLayers = nn.Sequential(*self.sharedLayers)
#         self.decBranch = nn.Sequential(*self.decBranch)
#         self.lblBranch = nn.Sequential(*self.lblBranch)
#
#     def forward(self, x):
#         shared = self.sharedLayers( x )
#         return self.decBranch(shared), self.lblBranch(shared)


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
            #self.model += [nn.Dropout(p=0.2)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
#
#         super(MLP, self).__init__()
#         self.model = []
#         self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
#         for i in range(n_blk - 2):
#             self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
#         self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
#         self.model = nn.Sequential(*self.model)
#
#     def forward(self, x):
#         return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            assert False, 'Please uncomment AdaptiveInstanceNorm2d in networks.py, then uncomment the line after this assertion'
            # self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            assert False, 'Please uncomment SpectralNorm in networks.py, then uncomment the line after this assertion'
            # self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

# class LinearBlock(nn.Module):
#     def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
#         super(LinearBlock, self).__init__()
#         use_bias = True
#         # initialize fully connected layer
#         if norm == 'sn':
#             self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
#         else:
#             self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
#
#         # initialize normalization
#         norm_dim = output_dim
#         if norm == 'bn':
#             self.norm = nn.BatchNorm1d(norm_dim)
#         elif norm == 'in':
#             self.norm = nn.InstanceNorm1d(norm_dim)
#         elif norm == 'ln':
#             self.norm = LayerNorm(norm_dim)
#         elif norm == 'none' or norm == 'sn':
#             self.norm = None
#         else:
#             assert 0, "Unsupported normalization: {}".format(norm)
#
#         # initialize activation
#         if activation == 'relu':
#             self.activation = nn.ReLU(inplace=True)
#         elif activation == 'lrelu':
#             self.activation = nn.LeakyReLU(0.2, inplace=True)
#         elif activation == 'prelu':
#             self.activation = nn.PReLU()
#         elif activation == 'selu':
#             self.activation = nn.SELU(inplace=True)
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()
#         elif activation == 'none':
#             self.activation = None
#         else:
#             assert 0, "Unsupported activation: {}".format(activation)
#
#     def forward(self, x):
#         out = self.fc(x)
#         if self.norm:
#             out = self.norm(out)
#         if self.activation:
#             out = self.activation(out)
#         return out

##################################################################################
# VGG network definition
##################################################################################
# class Vgg16(nn.Module):
#     def __init__(self):
#         super(Vgg16, self).__init__()
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, X):
#         h = F.relu(self.conv1_1(X), inplace=True)
#         h = F.relu(self.conv1_2(h), inplace=True)
#         # relu1_2 = h
#         h = F.max_pool2d(h, kernel_size=2, stride=2)
#
#         h = F.relu(self.conv2_1(h), inplace=True)
#         h = F.relu(self.conv2_2(h), inplace=True)
#         # relu2_2 = h
#         h = F.max_pool2d(h, kernel_size=2, stride=2)
#
#         h = F.relu(self.conv3_1(h), inplace=True)
#         h = F.relu(self.conv3_2(h), inplace=True)
#         h = F.relu(self.conv3_3(h), inplace=True)
#         # relu3_3 = h
#         h = F.max_pool2d(h, kernel_size=2, stride=2)
#
#         h = F.relu(self.conv4_1(h), inplace=True)
#         h = F.relu(self.conv4_2(h), inplace=True)
#         h = F.relu(self.conv4_3(h), inplace=True)
#         # relu4_3 = h
#
#         h = F.relu(self.conv5_1(h), inplace=True)
#         h = F.relu(self.conv5_2(h), inplace=True)
#         h = F.relu(self.conv5_3(h), inplace=True)
#         relu5_3 = h
#
#         return relu5_3
#         # return [relu1_2, relu2_2, relu3_3, relu4_3]

##################################################################################
# Normalization layers
##################################################################################
#class AdaptiveInstanceNorm2d(nn.Module):
#    def __init__(self, num_features, eps=1e-5, momentum=0.1):
#        super(AdaptiveInstanceNorm2d, self).__init__()
#        self.num_features = num_features
#        self.eps = eps
#        self.momentum = momentum
#        # weight and bias are dynamically assigned
#        self.weight = None
#        self.bias = None
#        # just dummy buffers, not used
#        self.register_buffer('running_mean', torch.zeros(num_features))
#        self.register_buffer('running_var', torch.ones(num_features))
#
#    def forward(self, x):
#        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
#        b, c = x.size(0), x.size(1)
#        running_mean = self.running_mean.repeat(b)
#        running_var = self.running_var.repeat(b)
#    
#        # Add noise:
#        #x = x + torch.randn_like( x )
#
#        # Apply instance norm
#        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
#
#        out = F.batch_norm(
#            x_reshaped, running_mean, running_var, self.weight, self.bias,
#            True, self.momentum, self.eps)
#
#        out = out.view(b, c, *x.size()[2:])
#
#        return out
#
#        #x = self.norm( x )
#        #self.weight = self.weight.view( self.weight.shape[0], self.weight.shape[1], 1, 1 )
#        #self.bias = self.bias.view( self.bias.shape[0], self.bias.shape[1], 1, 1 )
#       
#        #if self.mask is not None:
#        #    x[mask] = x[mask] * self.weight.expand( x.shape )[mask] + self.bias.expand( x.shape )[mask]
#        #else:
#        #    x = x * self.weight.expand( x.shape ) + self.bias.expand( x.shape )
#        #x = x * self.weight.expand( x.shape ) + self.bias.expand( x.shape )
#
#        #return x
#
#    def __repr__(self):
#        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
 
#def l2normalize(v, eps=1e-12):
#    return v / (v.norm() + eps)


##Usually not used, but is used if dis.norm='spectral' in config.yaml
#class SpectralNorm(nn.Module):
#    """
#    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
#    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
#    """
#    def __init__(self, module, name='weight', power_iterations=1):
#        super(SpectralNorm, self).__init__()
#        self.module = module
#        self.name = name
#        self.power_iterations = power_iterations
#        if not self._made_params():
#            self._make_params()
#
#    def _update_u_v(self):
#        u = getattr(self.module, self.name + "_u")
#        v = getattr(self.module, self.name + "_v")
#        w = getattr(self.module, self.name + "_bar")
#
#        height = w.data.shape[0]
#        for _ in range(self.power_iterations):
#            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
#            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))
#
#        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
#        sigma = u.dot(w.view(height, -1).mv(v))
#        setattr(self.module, self.name, w / sigma.expand_as(w))
#
#    def _made_params(self):
#        try:
#            getattr(self.module, self.name + "_u")
#            getattr(self.module, self.name + "_v")
#            getattr(self.module, self.name + "_bar")
#            return True
#        except AttributeError:
#            return False
#
#
#    def _make_params(self):
#        w = getattr(self.module, self.name)
#
#        height = w.data.shape[0]
#        width = w.view(height, -1).data.shape[1]
#
#        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
#        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
#        u.data = l2normalize(u.data)
#        v.data = l2normalize(v.data)
#        w_bar = nn.Parameter(w.data)
#
#        del self.module._parameters[self.name]
#
#        self.module.register_parameter(self.name + "_u", u)
#        self.module.register_parameter(self.name + "_v", v)
#        self.module.register_parameter(self.name + "_bar", w_bar)
#
#    def forward(self, *args):
#        self._update_u_v()
#        return self.module.forward(*args)
