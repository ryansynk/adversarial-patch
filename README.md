## Adversarial Patch Attacks on Face Recognition Neural Networks

As part of the 2018 NSF-Funded Computer Science REU at the University of Maryland, me (Ryan Synk) and a team of undergraduates 
studied adversarial attacks on facial recognition neural networks. In September 2020, I heavily updated the repository &mdash; originally 
located on the University of Maryland Institute for Advanced Computer Studies (UMIACS) Gitlab &mdash; and migrated it onto my personal 
github page.

This repository contains code written by Ryan Synk, Lara Shonkwiler, Kate Morrison, Xinyi Wang and Carlos Castillo. Credit for this work is spread evenly among 
those names listed, as well as Prof. Tom Goldstein, who directed the group.

# Directions for Generating Images

This software uses [UMDFaces](https://www.umdfaces.io) and [Pytorch](https://www.pytorch.org) to train deep networks for face recognition.

The steps are:
1. Generate thumbnails to train. Use `compute_aligned_images.py` for this task, point to the three batches of UMDFaces, run it three times.
2. Copy the create.py script to the val directory and run it there. This will create the missing directories in val that are required for validation to work.
3. Train. I used `python main.py --pretrained --epochs 200 --lr 0.1 --print-freq 1 /scratch2/umdfaces-thumbnails/`. You should come back 12 hours later.
4. Generate features, for example. I used the `compute_features.py` script for this task.
5. Generate plots/statistics, whatever you want, I used `run_lfw.py` for this.

