#Written by Anonymous Author, April 23 2022

import torch
import rp
import translator.trainer
from translator.trainer import MUNIT_Trainer as Trainer
from translator.data import ImageFolder


@rp.memoized
def get_dataset_dimensions(config:dict):
    #Get the size of the images the network was trained on, taking augmentations into account

    aug = {}
    aug["new_size_min"] = config.new_size_min_a
    aug["new_size_max"] = config.new_size_max_a
    aug["output_size" ] = (-1,-1) #Meaningless when image_folder.skip_crop == True

    #Could equivalently use train_real, test_fake, or test_real
    translator_root = rp.get_parent_directory(translator.trainer.__file__)
    image_folder_path = rp.path_join(translator_root, config.data_root, 'train_fake')

    image_folder = ImageFolder(root=image_folder_path, precise=True, augmentation=aug)
    image_folder.skip_crop = True

    o=rp.random_element(image_folder)

    _, height, width = o.shape

    return height, width


class EasyTranslator:

    #This is a wrapper that makes it easy to get translations from numpy images

    def __init__(self,
                 label_values      : list        ,
                 checkpoint_folder : str         ,
                 config_file       : str         ,
                 device            : torch.device,
                ):

        self.label_values      = label_values
        self.checkpoint_folder = checkpoint_folder
        self.config_file       = config_file
        self.device            = device

        config = rp.load_dyaml_file(config_file)
        config = rp.DictReader(config)
        self.config=config

        self.height, self.width = get_dataset_dimensions(config)

        self.trainer=Trainer(config, trainable=False).to(device)
        self.iteration=self.trainer.resume(checkpoint_folder)

        self.print_details()


    def scaled_input(self, image):
        #Rescale the image to the same size the network was trained on
        return rp.cv_resize_image(image, (self.height, self.width), interp='nearest')


    def print_details(self):
        print('Translator():'                                    )
        print('    - Checkpoint Folder:',self.checkpoint_folder  )
        print('    - Label Values:'     ,self.label_values       )
        print('    - Height, Width:'    ,(self.height,self.width))
        print('    - Iteration:'        ,self.iteration          )


    def translate(self, image):
        #Input image is a UV-L Scene

        assert rp.is_image(image)

        #Rescale the image to the same size it was trained on
        image = self.scaled_input(image)

        image = rp.as_rgb_image  (image)
        image = rp.as_float_image(image)

        image = rp.as_torch_image(image)[None] #BCHW
        image = image.to(self.device)

        with torch.no_grad():
            output = self.trainer.sample_a2b(image)
        output = output[0]
        output = rp.as_numpy_image(output)

        #Sometimes the network might change the dimensions.
        #Make sure the output is the same size as the input.
        output = rp.cv_resize_image(output, size=(self.height, self.width))

        assert rp.get_image_dimensions(output) == (self.height, self.width)

        return output

    def __repr__(self):
        return 'EasyTranslator: %s iter %i'%(rp.get_file_name(self.config_file),self.iteration)
