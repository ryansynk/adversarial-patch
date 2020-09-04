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
#from kate_optim import create_lr_list, get_triangular_lr, get_triangular_lr_up, get_discrete_lr
import os
import os.path

import torch.optim

resnet18model, resnet18preprocessor = load_model('resnet18')
#Load other model architectures here

#If you want to, say, test a specific architecture, just modify
#the models list so that it includes only that architecture, and
#everything will work fine.
models = [resnet18model]

# Creates the base_batch from the base_path. 
# Then either the first images from base_batch and target_batch are selected, 
# or random images are selected.
# The patch is applied to the base.
# The loss is computed and returned.
def loss_fn(base_path, patch1, target_batch, patch2, base_index=None, target_index=None, model_index=None, testing_mode=False, loss_type = 'cosine'):
    '''
    ### INPUTS ###
    base_path: the path to the folder of base images
        
    patch1, patch2: the perturbations to be applied to base at given coordinates
        
    x1val, y1val: the randomly generated upper-left coordinates of patch1
        
    target_batch: batch of images for 1 person to be iterated over
    
    modelStr: the model to be loaded
        
    x2val, y2val: the randomly generated upper-left coordinates of patch2
    
    testing_mode: boolean distinguishing when to iterate over the batches

    ### OUTPUTS ###
    loss: loss between peturbed base and the target
    '''

    # location of patches is random, but centered on cheeks
    x1val = random.randint(105, (105 + round(0.05 * 226)))
    y1val = random.randint(30, (30 + round(0.05 * 226)))
    x2val = random.randint(105, (105 + round(0.05 * 226)))
    y2val = random.randint(130, (130 + round(0.05 * 226)))

    # Don't randomize anything in testing mode
    if(testing_mode):
        model_index=base_index=target_index=0
        x1val = 105
        y1val = 30
        x2val = 105
        y2val = 130

    model = random.choice(models)
    if(model_index):
        model = models[model_index]
    
    # load in the base_batch
    base_batch = create_batch_list(base_path)
    
    # Gets the size of the patch from the input tensor
    patch1Size = list(patch1.size())[2]
    patch2Size = list(patch2.size())[2]   

    # getting random base image

    if(base_index is None):
        base_batch_size = base_batch.size()[0]
        base_index = random.randint(0, base_batch_size-1)

    baseWithPatch = torch.unsqueeze(base_batch[base_index].clone(),0)

    # plug patches into base
    baseWithPatch[0,0:3,x1val:(patch1Size + x1val),y1val:(patch1Size + y1val)] = patch1
    baseWithPatch[0,0:3,x2val:(patch2Size + x2val),y2val:(patch2Size + y2val)] = patch2

    # plug base+patches into the neural net
    baseFeatVec = model(baseWithPatch)
    baseFeatVec = torch.squeeze(baseFeatVec, 0)

    # getting random target image
    if(target_index is None):
        target_batch_size = target_batch.size()[0]
        target_index = random.randint(0, target_batch_size-1)
    targetImg = torch.unsqueeze(target_batch[target_index],0)
    targetFeatVec = model(targetImg)
    targetFeatVec = torch.squeeze(targetFeatVec, 0)

    # computing the loss between random base and random target
    #Try L2 norm as loss
    if loss_type == 'l2':
        loss = torch.dist(baseFeatVec, targetFeatVec)
    else:
        loss = torch.dot(baseFeatVec, targetFeatVec) / (torch.norm(baseFeatVec) * torch.norm(targetFeatVec))
        loss = 1 - loss

    return loss, baseWithPatch
        
def scale_range (input):
    copy = np.copy(input.data.numpy())
    copy = np.maximum(np.minimum(copy,1),-1)
    input.data = torch.Tensor(copy)
    return input

# Optimization method used for generating adversarial instances to a given
# neural network.
def optimization(base_path, target_batch, lr_list, learn, maxIters, cutOff, modelStr, patchSize, cuda=False, testing_mode=False, normal=False, loss_type = 'cosine'):

    '''
    ### INPUTS ###
    base_path: The path to the folder of base images.

    target_batch: A batch of images that will be iterated over to create the 
    patches for the baseImg. Assumes
    this is a preprocessed 1,3,227,227 tensor.

    learn: Learning rate for Adam

    maxIters: Max number of iterations for the loop (~1000 ??)

    cutOff: The cutoff value for the correlation between two vectors.
    I.e, how close you'd like the perturbed baseImg to be to the target
    in feature space. Should be between 0 and 1

    modelStr: String used to decide which model is loaded in load_model
    
    patchSize: Currently hardcoded size of square patch

    cuda: Whether or not you want cuda

    ### OUTPUTS ###
    misclassImageWithPatch: The base image with the two patches applied
    
    patch: list that contains the two randomly placed patches

    finalAvgLoss: the final average loss between the misclassImageWithPatch
    and all the images in the target_batch
    '''
    
    #Initializations
    torch.set_printoptions(precision=10)

    if(testing_mode):
        print('\nWARNING: running in testing mode (no randomization)\n')

    # generate the patches based on parameters
    patch1 = Variable(torch.zeros(1,3,patchSize,patchSize), requires_grad = True)
    patch2 = Variable(torch.zeros(1,3,patchSize,patchSize), requires_grad = True)
    
    #Iterates over parameters to make the optimization quicker
    for model in models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    # Cuda stuff
    if cuda:
        for model in models:
            model.cuda()
        patch1.cuda()
        patch2.cuda()
        # TODO: not sure what to do with cuda and the base batch images here
        #base_batch.cuda()
        #baseVar.cuda()
        target_batch.cuda()

    # use Adam for optimization
    #optimizer = torch.optim.LBFGS([patch1,patch2], lr = learn)
    optimizer = torch.optim.Adam([patch1, patch2], lr=learn)

    #Begin optimization
    for iteration in range(maxIters):
        # The LBFGS optimizer requires this closure, but its optional for SGD and Adam.
        # I added it so that it's easier to swap in different optimizer.
        #     -Tom
        def closure():

            loss, _ = loss_fn(base_path, patch1, target_batch, patch2, testing_mode=testing_mode, loss_type=loss_type)
            print('(%d) current loss: %f'%(iteration,loss.item()) )

            optimizer.zero_grad()
            loss.backward(retain_graph = True) #It needs this for some reason
            return loss

        optimizer.step(closure)
        
        #print("current LR: ", lr_list[iteration])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_list[iteration]


        #Normalize if the normal paramater is good
        if (normal == True):
            patch1 = scale_range(patch1)
            patch2 = scale_range(patch2)
        
    #Warning message
    if (iteration >= maxIters - 1):
        print('\nWarning: optimization loop ran for the maximum number of' +
            ' iterations. The result may not be correct')
    
    base_batch = create_batch_list(base_path)

    # compute average loss between all base and all target images
    if testing_mode == False:
        totalLoss = 0
        for i in range(base_batch.size()[0]):
            for j in range(target_batch.size()[0]):
                loss, baseWithPatch = loss_fn(base_path, patch1, target_batch, patch2, base_index=i, 
                                              target_index=j, model_index=0, testing_mode=testing_mode, 
                                              loss_type=loss_type)
            totalLoss += loss.item()
            
        # compute avg loss
        finalAvgLoss = totalLoss/(base_batch.size()[0]*target_batch.size()[0])
        print("Final Avg Loss: ", finalAvgLoss)
    else:
        finalAvgLoss, baseWithPatch = loss_fn(base_path, patch1, target_batch, patch2, testing_mode=testing_mode, loss_type=loss_type)
        print("Final Avg Loss: ", finalAvgLoss)
    
    return baseWithPatch, [patch1,patch2], finalAvgLoss


# take in batch_path name, iterates through that folder, filling batch_list
# with the images of that person
def create_batch_list(batch_path):
    
    # get size of batch
    size_batch = 0
    for filename in os.listdir(batch_path):
        if filename.endswith(".jpg"):
            size_batch += 1
    
    # create hollow batch list
    batch_list = torch.zeros(size_batch, 3, 227, 227)
    batch_list = Variable(batch_list)

    # fill in batch_list
    i = 0
    for filename in os.listdir(batch_path):
        if filename.endswith(".jpg"):
            image_path = batch_path+'/'+filename
            image = resnet18preprocessor(Image.open(image_path))
            batch_list[i] = Variable(torch.unsqueeze(image, 0), requires_grad = False)

            i += 1
            
    return batch_list

######## code to actually execute the entire program
def main():

    base_path = "./Images/Test_Images/GroupMemberImages/ryan_images"
    target_batch = create_batch_list("./Images/Test_Images/GroupMemberImages/kate_images")
    
    cuda = torch.cuda.is_available()


    # run optimization to generate patches
    #  NOTES:
    #   With learn=2, test_mode=True, maxIters-500, LBFGS optimizer:  loss goes down to  0.35
    #   With learn=.5, test_mode=True, maxIters-500, Adam optimizer:  loss goes down to .40
    #   With learn=.1, test_mode=True, maxIters-500, Adam optimizer:  loss goes down to .36

    
    #   -AFTER IMPLEMENTING RENORMALIZATION OF PATCH to within (0, 1)-
    #   With learn=.1, test_mode=True, maxIters-500, Adam optimizer: loss goes down to .5
    #   With learn=.1, test_mode=True, maxIters-250, Adam optimizer: loss goes down to .49(L2)
    #   With learn=.1, test_mode=True, maxIters-500, Adam optimizer: loss goes down to .55(L2)
    #   -AFTER IMPLEMENTING RENORMALIZATION OF PATCH to within (-.5, 1.5)
    #   With learn=.05, test_mode=True, maxIters-300, Adam optimizer: loss goes down to .37(cosine)

"""
if __name__=='__main__':
    main()
"""

