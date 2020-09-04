#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 2 10:26:10 2018
@author: Tom Goldstein, Kate Morrison, Lara Shonkwiler, Ryan Synk, Xinyi Wang
"""

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from numpy import linalg
from scipy.optimize import minimize
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from baseImg_targetImg import RajeevNet, NoOp, load_model, compare, image_loader, correlate, tensor_to_array
from baseBatch_targetBatch import loss_fn, scale_range, create_batch_list, optimization
import os
import os.path

import torch.optim

resnet18model, resnet18preprocessor = load_model('resnet18')
#Load other model architectures here

#If you want to, say, test a specific architecture, just modify
#the models list so that it includes only that architecture, and
#everything will work fine.
models = [resnet18model]

    
def create_lr_list(type_lr, num_iterations, stepsize, baseLR, maxLR, currentLR):
    
    lr_list = list()
    
    for iteration in range(num_iterations):
        if (type_lr == 'constant'):
            lr = currentLR
            if (iteration >= 300):
                lr = 1.0
        elif (type_lr == 'tri'):
            lr = get_triangular_lr(iteration, stepsize,baseLR,maxLR)
        elif (type_lr == 'tri_up'):
            lr = get_triangular_lr_up(iteration, stepsize, baseLR, maxLR)
        elif (type_lr == 'discrete'):
            cycles_wanted=num_iterations/stepsize
            if (iteration < 250):
                lr = get_discrete_lr(cycles_wanted, iteration, stepsize, baseLR, maxLR, currentLR)
            else:
                lr = get_triangular_lr(iteration,num_iterations-250,baseLR,maxLR)
        lr_list.append(lr)
    
    #plt.plot(lr_list)
    
    return lr_list

 

def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
    cycle=np.floor(1+iteration/(2*stepsize))
    x=np.abs(iteration/stepsize-2*cycle+1)
    scale_factor=1/cycle
    variation_range=max_lr - base_lr
    variation_range=scale_factor*variation_range
    lr=base_lr+variation_range*np.maximum(0,(1-x))
    return lr


#This method starts from the top to go down. 
#Note, if you want the learning rate to end at the base_lr (instead of max_lr) 
#after several cycles, make maxIters=an odd number * stepsize
def get_triangular_lr_up(iteration, stepsize, base_lr, max_lr):
    iteration=iteration+stepsize
    count=iteration//stepsize
    if count !=0:
        cycle=count//2 +1
        max_lr=(1/cycle)*max_lr
    x=np.abs((iteration % stepsize)/stepsize)
    variation_range=max_lr - base_lr
    if ((iteration//stepsize) %2)==0: #it should be on a down curve
        lr=max_lr-variation_range*np.maximum(0,(1-x))
    else:
        lr=base_lr+variation_range*np.maximum(0,(1-x))
    return lr

def get_discrete_lr(cycle_wanted, iteration,stepsize, base_lr, max_lr, current_lr):
    cycle=np.floor(1+iteration/stepsize)
    variation_range=max_lr - base_lr
    if cycle==cycle_wanted:
        x=np.abs(iteration//stepsize)/stepsize
        lr=max_lr - variation_range*x
    else:
        interval=variation_range/(cycle_wanted-1)
        lr=base_lr+interval*(cycle-1)
    return lr



######## code to actually execute the entire program
def main():
    
    
    
    """
    ryan_smallpatch = resnet18preprocessor(Image.open("/Users/katemorrison/Downloads/ryan_test_01_set.jpg"))
    ryan_bigpatch = resnet18preprocessor(Image.open("/Users/katemorrison/Downloads/ryan_test_02_set.jpg"))
    
    ryan_random_everything = resnet18preprocessor(Image.open("/Users/katemorrison/Downloads/ryan_random_everything_set.jpg"))
    
    
    ryan_rand_move = resnet18preprocessor(Image.open("/Users/katemorrison/Downloads/ryan_random_movement.jpg"))
    
    
    ryan = resnet18preprocessor(Image.open("/Users/katemorrison/Downloads/ryan_standard/ryan_standard.jpg"))
    
    
    kate = resnet18preprocessor(Image.open("./Images/Test_Images/GroupMemberImages/kate_images/pic1kate.jpg"))
    
    
    ryan_small = torch.unsqueeze(ryan_smallpatch, 0)
    ryan_small_vec = resnet18model(ryan_small)
    ryan_small_vec = torch.squeeze(ryan_small_vec, 0)
    
    ryan_big = torch.unsqueeze(ryan_bigpatch, 0)
    ryan_big_vec = resnet18model(ryan_big)
    ryan_big_vec = torch.squeeze(ryan_big_vec, 0)
    
    
    ryan_rand = torch.unsqueeze(ryan_random_everything, 0)
    ryan_rand_vec = resnet18model(ryan_rand)
    ryan_rand_vec = torch.squeeze(ryan_rand_vec, 0)
    
    
    ryan_rand_move = torch.unsqueeze(ryan_rand_move, 0)
    ryan_move_vec = resnet18model(ryan_rand_move)
    ryan_move_vec = torch.squeeze(ryan_move_vec, 0)

    kate = torch.unsqueeze(kate, 0)
    kate_vec = resnet18model(kate)
    kate_vec = torch.squeeze(kate_vec, 0)
    
    
    
    ryan = torch.unsqueeze(ryan, 0)
    ryan_vec = resnet18model(ryan)
    ryan_vec = torch.squeeze(ryan_vec, 0)
    
    
    
    corr = torch.dot(ryan_small_vec, kate_vec) / (torch.norm(ryan_small_vec) * torch.norm(kate_vec))
    loss = 1 - corr
    print("loss of test small ryan: ", loss)
    
    corr2 = torch.dot(ryan_big_vec, kate_vec) / (torch.norm(ryan_big_vec) * torch.norm(kate_vec))
    loss2 = 1 - corr2
    print("loss of test big ryan: ", loss2)
    
    
    corr3 = torch.dot(ryan_vec, kate_vec) / (torch.norm(ryan_vec) * torch.norm(kate_vec))
    loss3 = 1 - corr3
    print("loss of ryan and kate: ", loss3)
    
    
    corr4 = torch.dot(ryan_rand_vec, kate_vec) / (torch.norm(ryan_rand_vec) * torch.norm(kate_vec))
    loss4 = 1 - corr4
    print("loss of ryan w/ random everything: ", loss4)
    
    
    corr5 = torch.dot(ryan_move_vec, kate_vec) / (torch.norm(ryan_move_vec) * torch.norm(kate_vec))
    loss5 = 1 - corr5
    print("loss of ryan w/ random movement: ", loss5)    
    """

    ryan = resnet18preprocessor(Image.open("/Users/katemorrison/Downloads/ryan_standard/ryan_standard.jpg"))
    
    ryan = torch.unsqueeze(ryan, 0)
    ryan_vec = resnet18model(ryan)
    ryan_vec = torch.squeeze(ryan_vec, 0)
    
    ryan_triup = resnet18preprocessor(Image.open("/Users/katemorrison/Downloads/ryan_standard_wpatch.jpg"))

    ryan_triup = torch.unsqueeze(ryan_triup, 0)
    ryan_triup_vec = resnet18model(ryan_triup)
    ryan_triup_vec = torch.squeeze(ryan_triup_vec, 0)
    
    kate = resnet18preprocessor(Image.open("./Images/Test_Images/GroupMemberImages/kate_images/pic1kate.jpg"))
    
    kate = torch.unsqueeze(kate, 0)
    kate_vec = resnet18model(kate)
    kate_vec = torch.squeeze(kate_vec, 0)
    
    corr7 = torch.dot(ryan_vec, kate_vec) / (torch.norm(ryan_vec) * torch.norm(kate_vec))
    loss7 = 1 - corr7
    print("loss of ryan standard and kate: ", loss7)
    
    corr6 = torch.dot(ryan_triup_vec, kate_vec) / (torch.norm(ryan_triup_vec) * torch.norm(kate_vec))
    loss6 = 1 - corr6
    print("loss of ryan standard with patch: ", loss6)
    
    
    
    #base_path = "./Images/Test_Images/GroupMemberImages/ryan_images"
    
    base_path = "/Users/katemorrison/Downloads/ryan_standard"
    target_batch = create_batch_list("./Images/Test_Images/GroupMemberImages/kate_images")
    
    cuda = torch.cuda.is_available()

    # run optimization to generate patches
    #  NOTES:
    #   With learn=2, test_mode=True, maxIters-500, LBFGS optimizer:  loss goes down to  0.35
    #   With learn=.5, test_mode=True, maxIters-500, Adam optimizer:  loss goes down to .40
    #   With learn=.1, test_mode=True, maxIters-500, Adam optimizer:  loss goes down to .36

    type_lr = 'tri_up'
    maxIters = 525
    stepsize = 25
    baseLR = .01
    maxLR = 4
    currentLR = .01
    
    #lr_list = create_lr_list('tri', maxIters, stepsize, baseLR, maxLR, currentLR)

    lr_list = create_lr_list(type_lr, maxIters, stepsize, baseLR, maxLR, currentLR)


    # TODO: we dont use the learn or cutOff parameters anymore; should get rid of those
    misclassImage, patch, finalAvgLoss = optimization(base_path, target_batch, lr_list, learn=.1, maxIters=525, 
                                                     cutOff=0.2, modelStr=models, patchSize=50, cuda=cuda,
                                                     testing_mode=True, normal=True)
    
    
    # print out base with patches
    tensor_to_array(misclassImage, 'Ryan With Patch')
    #
    #   -AFTER IMPLEMENTING RENORMALIZATION OF PATCH to within (0, 1)-
    #   With learn=.1, test_mode=True, maxIters-500, Adam optimizer: loss goes down to .5
    #   With learn=.1, test_mode=True, maxIters-250, Adam optimizer: loss goes down to .49(L2)
    #   With learn=.1, test_mode=True, maxIters-500, Adam optimizer: loss goes down to .55(L2)
    #   -AFTER IMPLEMENTING RENORMALIZATION OF PATCH to within (-.5, 1.5)
    #   With learn=.05, test_mode=True, maxIters-300, Adam optimizer: loss goes down to .37(cosine)
    #misclassImage, patch, finalAvgLoss = optimization(base_path, target_batch, learn=.05, maxIters=300, cutOff=0.3,
    #                                                modelStr=models, patchSize=50, cuda=cuda, testing_mode=True)
    
    
    torch.save(patch[0], './patches/patch1_adam_500iters_triupLR25_r18_toprint_size25.pt')
    torch.save(patch[1], './patches/patch2_adam_500iters_triupLR25_r18_toprint_size25.pt')
    
    patch1 = torch.load('./patches/patch1_adam_500iters_triupLR25_r18_toprint_size25.pt')
    patch2 = torch.load('./patches/patch2_adam_500iters_triupLR25_r18_toprint_size25.pt')

    #patch1 = torch.unsqueeze(patch1, 0)
    #patch2 = torch.unsqueeze(patch2, 0)
    
    tensor_to_array(patch1, "patch1")
    tensor_to_array(patch2, "patch2")
    
    
    """
    ryanImg = resnet18preprocessor(Image.open('./Images/Test_Images/GroupMemberImages/ryan_images/ryanface_01.jpg'))
    kateImg = resnet18preprocessor(Image.open('./Images/Test_Images/GroupMemberImages/kate_images/pic1kate.jpg'))
    ryanImg = torch.unsqueeze(ryanImg, 0)
    kateImg = torch.unsqueeze(kateImg, 0)
    
    patch1 = torch.load('./patches/05_15pixelrange_lr05_patch1_ryantokate_singleImage_300iter_ADAM.pt')
    patch2 = torch.load('./patches/05_15pixelrange_lr05_patch2_ryantokate_singleImage_300iter_ADAM.pt')
    """
    
    #TODO: keep this part?
    """
    # saving patch and base+patch
    torch.save(patch, 'patches/patch.pt')
    torch.save(misclassWithPatch, 'patches/misclassWithPatch.pt')

    # printing out the image
    misclassWithPatch = torch.load('patches/misclassWithPatch.pt')
    tensor_to_array(misclassWithPatch, 'Pic of Ryan With Patch')
    """

    



if __name__=='__main__':
    main()




#KATE OPTIMIZATINO WORK

