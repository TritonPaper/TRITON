"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import argparse
import os
import random
import sys

import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import rp

from trainer import MUNIT_Trainer
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='trained_models', help="outputs path")
parser.add_argument('--device', type=int, default=0, help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

print("Device:",opts.device)
print("Config:",opts.config)
print("Arguments:",opts)
torch.cuda.set_device(opts.device)

cudnn.benchmark = True

# Load experiment setting
config       = get_config(opts.config)
max_iter     = config.max_iter
display_size = config.display_size

# Setup model and data loader
trainer = MUNIT_Trainer(config)
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

random.seed(1)

#Select the images used to generate previews
train_a = rp.random_batch(train_loader_a.dataset, config.display_size)
train_b = rp.random_batch(train_loader_b.dataset, config.display_size)
test_a  = rp.random_batch(test_loader_a .dataset, config.display_size)
test_b  = rp.random_batch(test_loader_b .dataset, config.display_size)

train_display_images_a = torch.stack(train_a).cuda()
train_display_images_b = torch.stack(train_b).cuda()
test_display_images_a  = torch.stack(test_a ).cuda()
test_display_images_b  = torch.stack(test_b ).cuda()

# Setup logger and output folders
model_name = rp.get_file_name(opts.config, include_file_extension=False)
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

run_path = rp.get_parent_folder(__file__)

def backup_code():
    # Backup copy of current settings and scripts:
    
    #Backup the config
    rp.copy_file(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    #Backup the python files
    python_files_dir=os.path.join(output_directory, 'python_files')
    rp.make_directory(python_files_dir)
    for python_file in rp.get_all_files(run_path, file_extension_filter='py'):
        file_name = rp.get_file_name(python_file) + '.old' #Make sure it doesn't have a .py extension, so my refactoring tools don't see it
        new_path = rp.path_join(python_files_dir, file_name)
        rp.copy_file(python_file, new_path)

backup_code()

# Start training
iterations = trainer.resume(checkpoint_directory) if opts.resume else 0

def save_checkpoint():
    trainer.save(checkpoint_directory, iterations)

def get_sample_outputs():
    with torch.no_grad():
        train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
        test_image_outputs  = trainer.sample(test_display_images_a , test_display_images_b )
    return train_image_outputs, test_image_outputs

def write_iter_images():
    train_image_outputs,test_image_outputs = get_sample_outputs()
    write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1 ))
    write_2images(test_image_outputs , display_size, image_directory, 'test_%08d'  % (iterations + 1 ))
    # HTML
    write_html(output_directory + "/index.html", iterations + 1, config.image_save_iter, 'images')

def write_current_images():
    train_image_outputs,test_image_outputs = get_sample_outputs()
    write_2images(train_image_outputs, display_size, image_directory, 'train_current')
    write_2images(test_image_outputs , display_size, image_directory, 'test_current')

try:
    while True:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

            with Timer("Elapsed time in update: %f"):
                # Main training code
                trainer.dis_update(images_a, images_b)
                trainer.gen_update(images_a, images_b)
                torch.cuda.synchronize()
            trainer.update_learning_rate()

            # Dump training stats in log file
            if (iterations + 1) % config.log_iter == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config.image_save_iter == 0:
                write_iter_images()

            if (iterations + 1) % config.image_display_iter == 0:
                write_current_images()

            # Save network weights
            if (iterations + 1) % config.snapshot_save_iter == 0:
                save_checkpoint()

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')

except BaseException as exception:
    from rp import print_verbose_stack_trace
    print_verbose_stack_trace(exception)

for _ in range(5):
    from rp import pseudo_terminal
    print("TRAINING ENDED! RUNNING PSEUDO TERMINAL! (Running %i of 5 times to make sure you don't quit by accident...)"%(_+1))
    pseudo_terminal()
