#THIS ONE HAS BEEN MODIFIED BY THE AUTHORS!
#Well...seeing that I got rid of MSSIM...is this basically just MUNIT now?
"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import MsImageDis
from networks import StylelessGen
from utils import weights_init, get_model_list, get_scheduler
from pytorch_msssim import msssim

import os
import torch
import torch.nn as nn

import upper.source.projector          as projector
import upper.source.unprojector        as unprojector
import upper.source.scene_reader       as scene_reader
import upper.source.view_consistency   as view_consistency
import upper.source.learnable_textures as learnable_textures

import rp

##Twisty BS:
#    Use the following two arrays to permute label values...
#     old_labels=[0,1,2,3,4,5]
#     new_labels=[2,0,3,1,4,5]
#    old_labels=[]
#    new_labels=[]
#     0 Alphabet: 0
#     1 Rubiks: 50
#     2 Garlic: 100
#     3 Apple: 150
#     4 Soda: 200
#     5 Table: 255
#     old_labels=torch.tensor(old_labels)
#     new_labels=torch.tensor(new_labels)


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters, trainable=True):

        super().__init__()

        if trainable:
            #If self.trainable is True, we also allocate discriminators and optimizers etc to vram
            self.trainable=True
        else:
            self.trainable=False

        hyp=self.hyp=hyperparameters


        #Create the learnable texture
        assert hyp.texture.type in 'fourier mlp raster', repr(hyp.texture.type)+' is not a supported type of learnable texture'

        if hyp.texture.type == 'fourier':
            self.texture_pack = learnable_textures.LearnableTexturePackFourier(
                height=hyp.texture.height, 
                width =hyp.texture.width, 
                num_textures=len(hyp.label_values),
                scale=hyp.texture.fourier.scale,
            )

        elif hyp.texture.type == 'mlp':
            assert False, 'Not yet implemented'

        elif hyp.texture.type == 'raster':
            self.texture_pack = learnable_textures.LearnableTexturePackRaster(
                height=hyp.texture.height, 
                width =hyp.texture.width, 
                num_textures=len(hyp.label_values),
            )

        if not self.trainable:
            #If we're not going to train anything, memoize the texture so we don't recalculate it repeatedly
            #Especially with fourier textures, this should speed things up significantly during inference...
            #TODO: Profile this! See if this truly makes it faster (it should, but I don't know by how much!)
            self.texture_pack.forward = rp.memoized(self.texture_pack.forward)

        a_num_channels = hyp.input_dim_a#+self.texture_pack.num_channels
        b_num_channels = hyp.input_dim_b

        self.view_consistency_loss = view_consistency.ViewConsistencyLoss(
            recovery_width =hyp.view_consistency.width,
            recovery_height=hyp.view_consistency.height,
            version        =hyp.view_consistency.version,
        )

        print("BATCH SIZE",hyp.batch_size)

        if hyp.view_consistency_w and not hyp.batch_size>1:
            print( "Using hyp.view_consistency_w! batch_size should be MORE than 1, but its %i"%hyp.batch_size)
            #If hyp.batch_size==1, we can mathematically guarentee that loss_view_consistency==0 (the variance of any length-1 vector is 0)
            hyp.view_consistency_w = 0 #Save time by skipping loss_view_consistency, becuase 0*0=0

        if self.trainable:
            lr = hyp.lr
            tex_lr = hyp.lr * hyp.texture.lr_factor

        # Initiate the networks
        self.gen_a = StylelessGen(a_num_channels, hyp.gen)  # auto-encoder for domain a
        self.gen_b = StylelessGen(b_num_channels, hyp.gen)  # auto-encoder for domain b

        if self.trainable:
            self.dis_a = MsImageDis(a_num_channels, hyp.dis) # discriminator for domain a
            self.dis_b = MsImageDis(b_num_channels, hyp.dis) # discriminator for domain b

        # Setup the optimizers
        if self.trainable:
            beta1 = hyp.beta1
            beta2 = hyp.beta2

            dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
            gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())

            dis_params = [p for p in dis_params if p.requires_grad]
            gen_params = [p for p in gen_params if p.requires_grad]
            tex_params = list(self.texture_pack.parameters())

            self.dis_opt = torch.optim.Adam(dis_params, lr=lr    , betas=(beta1, beta2), weight_decay=hyp.weight_decay)
            self.gen_opt = torch.optim.Adam(gen_params, lr=lr    , betas=(beta1, beta2), weight_decay=hyp.weight_decay)
            self.tex_opt = torch.optim.Adam(tex_params, lr=tex_lr, betas=(beta1, beta2), weight_decay=hyp.weight_decay)

            self.dis_scheduler = get_scheduler(self.dis_opt, hyp)
            self.gen_scheduler = get_scheduler(self.gen_opt, hyp)
            self.tex_scheduler = get_scheduler(self.tex_opt, hyp)

        if self.hyp.color_loss_w:
            assert len(self.hyp.color_loss.label_colors) == len(self.hyp.label_values), 'Must have exactly one color per label'
            self.label_colors = torch.nn.Parameter(torch.Tensor(hyp.color_loss.label_colors), requires_grad=False)
            print("Using color_loss. self.label_colors.shape: ", self.label_colors.shape)

        # Network weight initialization
        if self.trainable:
            self.apply(weights_init(hyp.init))
            self.dis_a.apply(weights_init('gaussian'))
            self.dis_b.apply(weights_init('gaussian'))

            self.label_criterion = nn.CrossEntropyLoss()


    def project_texture_pack(self, x_a):

        hyp=self.hyp

        scene_uvs, scene_labels = scene_reader.extract_scene_uvs_and_scene_labels(scene_images = x_a             ,
                                                                                  label_values = hyp.label_values)

        #Twisty BS:
        # scene_labels = scene_reader.replace_values(scene_labels, old_labels, new_labels)

        texture_pack  = self.texture_pack()
        texture_pack  = (texture_pack - 1/2) * 2 # [0,1] to [-1,1]
        texture_pack *= hyp.texture.multiplier

        if not hyp.texture.multiplier:
            #Instead of multiplying by 0 and keeping track of that gradient, save some time on the backwards pass
            #(0 times anything is 0)
            texture_pack = torch.zeros_like(texture_pack)

        scene_projections = projector.project_textures(scene_uvs, scene_labels, texture_pack) # Range of values: [-1, 1]


        #Twisty BS:
        # x_a_blue = scene_reader.replace_values(scene_labels, torch.tensor(list(range(len(hyp.label_values)))), torch.tensor(hyp.label_values) )
        # x_a[:, 2] = x_a_blue / 255

        #SIMPLE:
        x_a=x_a*2-1

        hint = x_a+0 #Might replace with better content later

        #RESIDUAL:
        x_a = hint*hyp.texture.hint.multiplier + scene_projections #let's try to minimize effort right now...let's just use 3 channels for visualization etc... todo make all 6:

        x_a = torch.cat((x_a,hint),dim=1)#BCHW

        return x_a, scene_uvs, scene_labels


    def recon_criterion(self, input, target):
        output  = torch.mean(torch.abs(input - target))
        output += -msssim(input,target,normalize=True)
        return output


    def forward(self, x_a, x_b):
        x_a, _, _ = self.project_texture_pack(x_a)

        c_a = self.gen_a.encode(x_a)
        c_b = self.gen_b.encode(x_b)

        x_ba = self.gen_a.decode(c_b)
        x_ab = self.gen_b.decode(c_a)

        return x_ab, x_ba


    def color_loss(self, scenes, labels):
        #Assumes scenes are in the range [-1,1] and label_colors are in the range [0,1]
        
        if not self.hyp.color_loss_w:
            return 0

        num_labels = len(self.hyp.color_loss.label_colors)
        mean_colors = unprojector.get_label_averge_colors(scenes, labels, num_labels)
        mean_colors = (mean_colors + 1)/2 #[-1,1] to [0,1]
        return ((self.label_colors - mean_colors)**2).mean()


    def gen_update(self, x_a, x_b):

        assert self.trainable, 'This MUNIT_Trainer is not trainable, and does not have optimizers or discriminators etc (to save memory)'

        hyp=self.hyp
        self.train()

        #Because precise=True, x_a should be in the range (0,1) and x_b should be in the range (-1,1) because precise=False for that domain (see utils.py)

        x_a, scene_uvs, scene_labels = self.project_texture_pack(x_a)

        #Now, after self.project_texture_pack, x_a is in the range (-1,1)

        self.tex_opt.zero_grad()
        self.gen_opt.zero_grad()

        # encode
        c_a = self.gen_a.encode(x_a)
        c_b = self.gen_b.encode(x_b)

        x_ba = self.gen_a.decode(c_b)
        x_ab = self.gen_b.decode(c_a)

        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a)
        x_b_recon = self.gen_b.decode(c_b)

        # encode again
        c_b_recon = self.gen_a.encode(x_ba)
        c_a_recon = self.gen_b.encode(x_ab)

        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon)                         if hyp.recon_x_cyc_w > 0 else None
        x_bab = self.gen_b.decode(c_b_recon)                         if hyp.recon_x_cyc_w > 0 else None

        # reconstruction loss
        loss_gen_cycrecon_x_a = self.recon_criterion(x_aba    , x_a) if hyp.recon_x_cyc_w > 0 else 0
        loss_gen_cycrecon_x_b = self.recon_criterion(x_bab    , x_b) if hyp.recon_x_cyc_w > 0 else 0
        loss_gen_recon_x_a    = self.recon_criterion(x_a_recon, x_a)
        loss_gen_recon_x_b    = self.recon_criterion(x_b_recon, x_b)
        loss_gen_recon_c_a    = self.recon_criterion(c_a_recon, c_a)
        loss_gen_recon_c_b    = self.recon_criterion(c_b_recon, c_b)

        # GAN loss
        loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        
        #Note: The "not weight or loss" pattern below is meant to prevent computing loss when weight=0

        #View Consistency Loss
        loss_view_consistency = not hyp.view_consistency_w or self.view_consistency_loss(x_ab, scene_uvs, scene_labels)
        # if (loss_view_consistency.isnan() | loss_view_consistency.isinf()).any(): print("view consistency has nan or inf")

        #Texture Reality Loss

        loss_texture_reality_a2b = not hyp.texture_reality_a2b_w or self.recon_criterion(x_ab, x_a [:,:3])
        loss_texture_reality_b2a = not hyp.texture_reality_b2a_w or self.recon_criterion(x_b , x_ba[:,:3])


        #Label color loss
        loss_label_colors = not hyp.color_loss_w or self.color_loss(x_ab, scene_labels)        

        #Total loss
        loss_gen_total  = hyp.gan_w                 * loss_gen_adv_a
        loss_gen_total += hyp.gan_w                 * loss_gen_adv_b
        loss_gen_total += hyp.recon_x_w             * loss_gen_recon_x_a
        loss_gen_total += hyp.recon_c_w             * loss_gen_recon_c_a
        loss_gen_total += hyp.recon_x_w             * loss_gen_recon_x_b
        loss_gen_total += hyp.recon_c_w             * loss_gen_recon_c_b
        loss_gen_total += hyp.recon_x_cyc_w         * loss_gen_cycrecon_x_a
        loss_gen_total += hyp.recon_x_cyc_w         * loss_gen_cycrecon_x_b
        loss_gen_total += hyp.view_consistency_w    * loss_view_consistency
        loss_gen_total += hyp.texture_reality_a2b_w * loss_texture_reality_a2b
        loss_gen_total += hyp.texture_reality_b2a_w * loss_texture_reality_b2a
        loss_gen_total += hyp.color_loss_w          * loss_label_colors

        loss_gen_total.backward()

        self.tex_opt.step()
        self.gen_opt.step()

        #Unimportant code:
        self.loss_gen_adv_a        = loss_gen_adv_a       .item()
        self.loss_gen_adv_b        = loss_gen_adv_a       .item()
        self.loss_gen_recon_x_a    = loss_gen_recon_x_a   .item()
        self.loss_gen_recon_c_a    = loss_gen_recon_c_a   .item()
        self.loss_gen_recon_x_b    = loss_gen_recon_x_b   .item()
        self.loss_gen_recon_c_b    = loss_gen_recon_c_b   .item()
        self.loss_gen_cycrecon_x_a = loss_gen_cycrecon_x_a.item()
        self.loss_gen_cycrecon_x_b = loss_gen_cycrecon_x_b.item()
        self.loss_gen_total        = loss_gen_total       .item()


    def sample(self, x_a, x_b, with_grad=False):

        self.eval()

        if not with_grad:
            with torch.no_grad():
                return self.sample(x_a, x_b, with_grad=True)

        x_a_original = x_a #This is the UVL map

        x_a, _, _ = self.project_texture_pack(x_a)

        x_a_recon, x_b_recon, x_ba, x_bab, x_ab, x_aba, x_ab_rand = [], [], [], [], [], [], []
        for i in range(x_a.size(0)):
            # get individual images from list:
            x_a_ = x_a[i].unsqueeze(0)
            x_b_ = x_b[i].unsqueeze(0)

            # a to b:
            c_a        = self.gen_a.encode(x_a_)
            x_a_recon_ = self.gen_a.decode(c_a )     # Reconstruct in same domain

            c_b= self.gen_b.encode(x_b_)

            x_ab_  = self.gen_b.decode(c_a  ) # translate
            c_ab   = self.gen_b.encode(x_ab_) # re-encode
            x_aba_ = self.gen_a.decode(c_ab ) # translate back

            x_a_recon.append(x_a_recon_)
            x_ab     .append(x_ab_     )
            x_aba    .append(x_aba_    )

            # Encode another x_ab2 with a style drawn from b:
            x_ab_rand_ = self.gen_b.decode(c_a)     # translate
            x_ab_rand.append( x_ab_rand_ )

            # b to a:
            x_ba_ = self.gen_a.decode(c_b  ) # translate
            c_ba  = self.gen_a.encode(x_ba_) # re-encode

            x_b_recon_ = self.gen_b.decode(c_b ) # Reconstruct in same domain
            x_bab_     = self.gen_b.decode(c_ba) # translate back

            x_b_recon.append(x_b_recon_)
            x_ba     .append(x_ba_     )
            x_bab    .append(x_bab_    )

        x_a       = (x_a+1)/2
        x_b       = (x_b+1)/2
        x_ba      = (torch.cat(x_ba     )+1)/2
        x_ab      = (torch.cat(x_ab     )+1)/2
        x_bab     = (torch.cat(x_bab    )+1)/2
        x_aba     = (torch.cat(x_aba    )+1)/2
        x_a_recon = (torch.cat(x_a_recon)+1)/2
        x_b_recon = (torch.cat(x_b_recon)+1)/2
        x_ab_rand = (torch.cat(x_ab_rand)+1)/2

        return x_a_original, x_a[:,:3], x_a_recon[:,:3], x_a_recon[:,3:], x_ab, x_aba[:,:3], x_aba[:,3:], \
               x_b, x_b_recon, x_ba[:,:3], x_ba[:,3:], x_bab


    def sample_a2b(self, x_a, with_grad=False):
        #This code is very similar to self.sample(), except it has a few parts removed for the sake of efficiency.
        #If you ever want to modify the functionality of this function, make sure you modify it in self.sample() too
        #TODO: Remove this redundancy lol

        self.eval()

        if not with_grad:
            with torch.no_grad():
                return self.sample_a2b(x_a, with_grad=True)

        x_a, _, _ = self.project_texture_pack(x_a)

        x_ab = []

        for i in range(x_a.size(0)):
            # get individual images from list:
            x_a_ = x_a[i].unsqueeze(0)

            c_a   = self.gen_a.encode(x_a_)
            x_ab_ = self.gen_b.decode(c_a )

            x_ab.append(x_ab_)

        x_ab=(torch.cat(x_ab)+1)/2

        return x_ab


    def sample_b2a(self, x_b, with_grad=False):
        #This function is almost identical to sample_a2b

        self.eval()

        if not with_grad:
            with torch.no_grad():
                return self.sample_b2a(x_b, with_grad=True)

        x_ba = []

        for i in range(x_b.size(0)):
            # get individual images from list:
            x_b_ = x_b[i].unsqueeze(0)

            c_b   = self.gen_b.encode(x_b_)
            x_ba_ = self.gen_a.decode(c_b )

            x_ba.append(x_ba_)

        x_ba=(torch.cat(x_ba)+1)/2

        return x_ba


    def dis_update(self, x_a, x_b):

        assert self.trainable, 'This MUNIT_Trainer is not trainable, and does not have optimizers or discriminators etc'

        hyp=self.hyp
        self.train()

        x_a, _, _ = self.project_texture_pack(x_a)

        self.dis_opt.zero_grad()

        # encode
        c_a = self.gen_a.encode(x_a)
        c_b = self.gen_b.encode(x_b)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b)
        x_ab = self.gen_b.decode(c_a)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyp.gan_w * self.loss_dis_a + hyp.gan_w * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()


    def update_learning_rate(self):
        if self.dis_scheduler is not None: self.dis_scheduler.step()
        if self.gen_scheduler is not None: self.gen_scheduler.step()
        if self.tex_scheduler is not None: self.tex_scheduler.step()


    def resume(self, checkpoint_dir):

        hyp=self.hyp

        def torch_load(path):
            #By default, torch.load might try to load onto GPU
            #But the thing is, it might use the *wrong* GPU and run out of VRAM
            #https://discuss.pytorch.org/t/cuda-error-out-of-memory-when-load-models/38011/3
            #It seems to be a bug. Here's a workaround
            return torch.load(path,map_location='cpu')

        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch_load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])

        # Load textures
        last_model_name = get_model_list(checkpoint_dir, "tex")
        state_dict = torch_load(last_model_name)
        self.texture_pack.load_state_dict(state_dict['tex'])

        if self.trainable:
            # Load discriminators
            last_model_name = get_model_list(checkpoint_dir, "dis")
            state_dict = torch_load(last_model_name)
            self.dis_a.load_state_dict(state_dict['a'])
            self.dis_b.load_state_dict(state_dict['b'])

            # Load optimizers
            checkpoint_path = os.path.join(checkpoint_dir, 'optimizer.pt')
            if rp.file_exists(checkpoint_path):
                state_dict = torch_load(checkpoint_path)
                self.dis_opt.load_state_dict(state_dict['dis'])
                self.gen_opt.load_state_dict(state_dict['gen'])
                self.tex_opt.load_state_dict(state_dict['tex'])
            else:
                #Sometimes you might delete optimizer.pt to recover from NAN errors
                rp.fansi_print("optimizer.pt not found: initializing it without loading it from a checkpoint!",'yellow','bold')

            # Reinitilize schedulers
            self.dis_scheduler = get_scheduler(self.dis_opt, hyp, iterations)
            self.gen_scheduler = get_scheduler(self.gen_opt, hyp, iterations)
            self.tex_scheduler = get_scheduler(self.tex_opt, hyp, iterations)

        print('Resume from iteration %d' % iterations)
        return iterations


    def save(self, snapshot_dir, iterations):

        assert self.trainable, 'This MUNIT_Trainer is not trainable, and does not have optimizers or discriminators etc'

        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        tex_name = os.path.join(snapshot_dir, 'tex_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt'                  )

        torch.save({'a'  : self.gen_a       .state_dict(), 'b'  : self.gen_b  .state_dict()                                  }, gen_name)
        torch.save({'a'  : self.dis_a       .state_dict(), 'b'  : self.dis_b  .state_dict()                                  }, dis_name)
        torch.save({'tex': self.texture_pack.state_dict()                                                                    }, tex_name)
        torch.save({'gen': self.gen_opt     .state_dict(), 'dis': self.dis_opt.state_dict(), 'tex': self.tex_opt.state_dict()}, opt_name)
