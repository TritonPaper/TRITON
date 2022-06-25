# TRITON
This repository has the code needed to run the TRITON algorithm. This page is meant for reviewers, and so it is anonymous.

To train it, run the following commands:

```
cd translator
python3 train.py --config configs/alphabet_five_base.yaml
```

You will also need to install some python packages, including:
```
pip3 install torch rp einops numpy opencv-python OpenEXR
```

# Citations
The code under 'translator' was originally based on ```https://gitlab.com/nct_tso_public/surgical-video-sim2real```, from the following paper:

```
"Long-Term Temporally Consistent Unpaired Video Translation from Simulated Surgical 3D Data".
Dominik Rivoir, Micha Pfeiffer, Reuben Docea, Fiona Kolbinger, Carina Riediger, Jürgen Weitz, Stefanie Speidel.
International Conference on Computer Vision 2021.
```

Note that that repository was based on MUNIT from ```https://github.com/NVlabs/MUNIT```:
```
@inproceedings{huang2018munit,
  title={Multimodal Unsupervised Image-to-image Translation},
  author={Huang, Xun and Liu, Ming-Yu and Belongie, Serge and Kautz, Jan},
  booktitle={ECCV},
  year={2018}
}
```

The code used in `translator/pytorch_msssim.py` is based on  ```https://github.com/jorge-pessoa/pytorch-msssim```

The code used in `source/learnable_textures.py` is based on `https://github.com/ndahlquist/pytorch-fourier-feature-networks`:
```
@article{DBLP:journals/corr/abs-2006-10739,
  author    = {Matthew Tancik and
               Pratul P. Srinivasan and
               Ben Mildenhall and
               Sara Fridovich{-}Keil and
               Nithin Raghavan and
               Utkarsh Singhal and
               Ravi Ramamoorthi and
               Jonathan T. Barron and
               Ren Ng},
  title     = {Fourier Features Let Networks Learn High Frequency Functions in Low
               Dimensional Domains},
  journal   = {CoRR},
  volume    = {abs/2006.10739},
  year      = {2020},
  url       = {https://arxiv.org/abs/2006.10739},
  eprinttype = {arXiv},
  eprint    = {2006.10739},
  timestamp = {Thu, 14 Oct 2021 09:16:16 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2006-10739.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```





