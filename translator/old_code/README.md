[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md)
![Python 3.5](https://img.shields.io/badge/python-3.5-green.svg)
## Generating Large Labeled Data Sets for Laparoscopic Image Processing Tasks Using Unpaired Image-to-Image Translation

[Project Page](http://opencas.dkfz.de/image2image/) - Includes large translated laparoscopic data set with 100 000 translated images, fully labeled.
[Paper](https://arxiv.org/abs/1907.02882)

### License

Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

Note that this is a direct modification of the MUNIT framework ([ProjectPage](https://arxiv.org/abs/1804.04732), [GitHub](https://github.com/NVlabs/MUNIT)). The original Copyright holder is NVIDIA Corporation:

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

Additionally, we use the MS-SSIM implementation (pytorch\_msssim subfolder) by Jorge Pessoa ([GitHub](https://github.com/jorge-pessoa/pytorch-msssim)) which is licensed under the MIT license.

These licenses allow you to use, modify and share the project for non-commercial use as long as you adhere to the conditions of the license above.

### Citation

If you use this work, please cite:
"Generating large labeled data sets for laparoscopic image processing tasks using unpaired image-to-image translation". Micha Pfeiffer, Isabel Funke, Maria R. Robu, Sebastian Bodenstedt, Leon Strenger, Sandy Engelhardt, Tobias Roß, Matthew J. Clarkson, Kurinchi Gurusamy, Brian R. Davidson, Lena Maier-Hein, Carina Riediger, Thilo Welsch, Jürgen Weitz, and Stefanie Speidel. MICCAI 2019.

### Dependencies

pytorch (tested with version 1.2.0), yaml, tensorboard, and tensorboardX (from https://github.com/lanpa/tensorboard-pytorch).

### Example Usage

#### Training
1. Create the data set you want to use. The data set folder should contain 4 subfolders (train\_real, train\_fake, test\_real, test\_fake). Put your images into these four folders. To get started, you can [download our A+ dataset](http://opencas.dkfz.de/image2image/) to use for the fake folders and use any laparoscopic images featuring the liver as your real data set.

3. Setup the yaml file in 'configs/simulation2surgery.yaml'. Change the `data_root` field to the path of your dataset (i.e. the folder containing the four subfolders described above).

3. Start training
    ```
    python3 train.py --config configs/simulation2surgery.yaml --output_path trained_models
    ```
4. Training the net can take a few days (GeForce GTX 1080) with the current setup. However, intermediate image outputs are stored in `trained_models/outputs/simulation2surgery`, as well as model snapshots.

#### Translation

After a model has been trained, you can use it to translate synthetic data. You can either use random styles, or draw style images from a provided folder. For example, to translate all images in the test\_fake from above using random style vectors:

    ```
	python3 translate.py --input_folder data/test_fake --checkpoint trained_models/outputs/simulation2surgery/checkpoints/gen_00500000.pt --output_folder translations/
    ```

And the same translation, but using styles extracted from the test\_real images:

    ```
	python3 translate.py --input_folder data/test_fake --checkpoint trained_models/outputs/simulation2surgery/checkpoints/gen_00500000.pt --output_folder translations/ --style_input_folder data/test_real --store_style_images
    ```

Use 'python3 translate.py --help' for further options.


#### Contact

We welcome your feedback! To contact us, please see the [project page](http://opencas.dkfz.de/image2image/)

