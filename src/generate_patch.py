'''
Authors: Tom Goldstein, Lara Shonkwiler, Ryan Synk, Kathleen Morrison, Xinyi Wang
06.15.2018

Updated: Ryan Synk, September 2020
This script takes in two images a base and a target, and optimizes 
adversarial patches to be added onto the base in order to increase the
correlation of the resulting image to the target.
'''

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from numpy import linalg
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Model class
class RajeevNet(nn.Module):
    def __init__(self):
        super(RajeevNet, self).__init__()

    def forward(self, input):
        x = nn.AdaptiveAvgPool2d(1)(input)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        x = 20 * F.normalize(x)
        x = x.contiguous()
        return x

class NoOp(nn.Module):
    def __init__(self):
        super(NoOp, self).__init__()

    def forward(self, input):
        return input

# Creates a verification neural net model from a pre-loaded classification model
# Input: model_version (string which is a key in the torchvision.models dict) (specifies which pre-loaded classification model)
def load_model(model_version, model_path):
    # Load a pre-trained resnet model.  This model is designed for
    # imagenet, so it outputs 1000 classes.
    model = models.__dict__[model_version](pretrained=False)

    # Change the architecture to be compatible with the face recognition stuff.
    # The UMD Faces dataset has 8277 classes, so we need to output that many classes.
    model.avgpool = RajeevNet()
    model.fc = nn.Linear(512, 8277)

    # Load the model parameters from the Carlos' file. To do this, we need to
    # create a DataParallel model of the same format the model was saved in.
    # The DataParallel model is just a wrapper for the original net, and when
    # we load the parameters into the parallel model, it will fill the parameters
    # of the original model that it wraps.
    par_model = nn.DataParallel(model)
    # The model was saved on a GPU. We need to tell the model to load to CPU so we don't have issues
    checkpoint = torch.load(model_path + model_version + '_best.pth.tar', map_location='cpu')
    best_prec1 = checkpoint['best_prec1']
    par_model.load_state_dict(checkpoint['state_dict'])

    # Rip off the last layer so the net spits out the hidden features
    model.fc = NoOp()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessor = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        normalize
    ])

    return model, preprocessor

# Takes in two feature vectors (output from network as variable) 
# and compares them using the correlation
def compare(vec1, vec2):
    cuda = torch.cuda.is_available()

    if cuda:
        vec1 = vec1.cpu().data.numpy()
        vec2 = vec2.cpu().data.numpy()

    else:
        vec1 = vec1.data.numpy()
        vec2 = vec2.data.numpy()
    
    product = np.dot(vec1,np.transpose(vec2)).astype(float)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return product/(norm1*norm2)

# Method that takes in an image and returns a tensor
def image_loader(image_name, preprocess):
    image = Image.open(image_name)
    image = preprocess(image)
    image = image.unsqueeze(0) 
    image = Variable(image, requires_grad=True)
    return image

# Takes two images, runs them through the network and compares the
# feature vectors. Also takes in the model and the preprocess
def correlate(model, preprocess, image1, image2):
    if torch.cuda.is_available():
    	model = nn.DataParallel(model).cuda()

    im1 = image_loader(image1, preprocess)
    im2 = image_loader(image2, preprocess)
    
    return compare(model(im1),model(im2))

# Loss function defined for do_optimization method. Takes in a base image, 
# a patch, and a target image. Plugs patch into the base. Placement location 
# is hard coded because the patch is plugged into the aggregate base image, 
# so many layered patches have technically been applied to the base image
def loss_fn(base, patch1, x1val, y1val, target, modelStr, modelPath, patch2, x2val, y2val):
    
    #model, preprocessor = load_model('resnet18')
    cuda = torch.cuda.is_available()
    model,_ = load_model(modelStr, modelPath)
    
    # Gets the size of the patch from the input tensor
    patch1Size = list(patch1.size())[2]
    patch2Size = list(patch2.size())[2]
    
    # plug patches into the base image
    baseWithPatch = base
    baseWithPatch[0,0:3,x1val:(patch1Size + x1val),y1val:(patch1Size + y1val)] = patch1
    baseWithPatch[0,0:3,x2val:(patch2Size + x2val),y2val:(patch2Size + y2val)] = patch2

    # plug base+patches into the neural net
    predFeatVec = model(baseWithPatch)
    predFeatVec = torch.squeeze(predFeatVec, 0)

    # plug target into the neural net
    targetFeatVec = model(target)
    targetFeatVec = torch.squeeze(targetFeatVec, 0)

    # Computes correlation of two feature vectors. 
    # Returns 1 - loss because that's what we want to minimze, 
    # i.e we want the correlation to be 1
    loss = torch.dot(predFeatVec, targetFeatVec) / (torch.norm(predFeatVec) * torch.norm(targetFeatVec))
    
    return 1 - loss

# Optimization method used for generating adversarial instances to a given neural network
# Input:
#      - baseImg: image to be misclassified. assumes a [1,3,227,227] tensor
#      - targetImg: image for patches to be optimized toward. [1,3,227,227] tensor
#      - learn: learning rate for Adam
#      - maxIters: max number of iterations for optimization
#      - cutOff: Cutoff value for correlation between images. Between (0,1)
#      - modelStr: String used to decied which model is loaded
#      - patchSize: Integer, size of square patch
#      - cuda: boolean, whether or not cuda is being used
#
# Output:
#      - misclassImageWithPatch: The base image with two patches applied
#      - patch: list that contains two randomly placed patches
#      - finalLoss: loss between misclassImg and targetImg
def do_optimization(baseImg, targetImg, modelPath, learn=0.01, maxIters=500, cutOff=0.5, modelStr='resnet18', patchSize=50, cuda=False):

    #Initializations
    torch.set_printoptions(precision=10)
    model,_ = load_model(modelStr, modelPath)

    # generate the patches based on parameters
    patch1 = Variable(torch.rand(1,3,patchSize,patchSize), requires_grad = True)
    patch2 = Variable(torch.rand(1,3,patchSize,patchSize), requires_grad = True)
    
    # location of patches is random, but centered on the cheeks
    x1val = random.randint(105, (105+round(0.1*226)))
    y1val = random.randint(30, (30+round(0.1*226)))
    x2val = random.randint(105, (105+round(0.1*226)))
    y2val = random.randint(130, (130+round(0.1*226)))
  
    baseVar = Variable(baseImg, requires_grad = False)
    targetVar = Variable(targetImg, requires_grad = False)
    
    #Iterates over parameters to make the optimization quicker
    for param in model.parameters():
        param = Variable(param, requires_grad = False)
    
    if cuda:
        model.cuda()
        patch1.cuda()
        patch2.cuda()
        baseImg.cuda()
        targetImg.cuda()

    #optim.SGD expects an iterable
    patch = [patch1,patch2]
    optimizer = torch.optim.Adam(patch, lr = learn)

    #Initial correlation
    loss = loss_fn(baseImg, patch[0], x1val, y1val, targetImg, modelStr, modelPath, patch[1], x2val, y2val)
    print('=> Initial correlation: ' + str((-(loss - 1)).item()))

    #Begins optimization
    for iteration in range(maxIters):
        
        if (loss < cutOff):
            break
        else:
            # location of patches is random, but centered on the cheeks
            x1val = random.randint(105, (105+round(0.1*226)))
            y1val = random.randint(30, (30+round(0.1*226)))
            x2val = random.randint(105, (105+round(0.1*226)))
            y2val = random.randint(130, (130+round(0.1*226)))
            
            loss = loss_fn(baseVar, patch[0], x1val, y1val, targetVar, modelStr, modelPath, patch[1], x2val, y2val)
            print('=> iteration: ' + str(iteration) + ' | current loss: ' + str(loss.item()))
            
            if (loss < cutOff):
                break
            
            optimizer.zero_grad()
            loss.backward(retain_graph = True) #It needs this for some reason
            optimizer.step()
    
    misclassImageWithPatch = baseImg
    misclassImageWithPatch[0,0:3,x1val:(patchSize + x1val),y1val:(patchSize + y1val)] = patch[0]
    misclassImageWithPatch[0,0:3,x2val:(patchSize + x2val),y2val:(patchSize + y2val)] = patch[1]
    
    patch = [patch[0],patch[1]]
    finalLoss = loss
    print("\n")
    print("finalLoss: ", finalLoss)

    return misclassImageWithPatch, patch, finalLoss


# Converts a 1x3x227x227 tensor into a numpyarray that can
# have plt.imshow() called on it. It then displays the figure.
# Note that each time the method is called, you first have
# to exit out of the figure to continue running code
def tensor_to_array(tens, figTitle):
    tens = tens.data.numpy() #Converts to a numpyarray
    tens = tens[0,:,:,:] #Gets rid of 1st dimension
    tens = tens.transpose(1,2,0) #Changes dimension to 227x227x3
    # rescale image to [0,1]
    tens = tens-np.min(tens)
    tens = tens/np.max(tens)

    plt.imshow(tens)
    plt.title(figTitle)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Give two images')
    parser.add_argument('--im1', '--image_path_1',
                        default='../images/Test_Images/GroupMemberImages/ryan_images/ryanface_01.jpg', 
                        help='path to base image')
    parser.add_argument('--im2', '--image_path_2',
                        default='../images/Test_Images/CelebrityImages/john_mayer/john_mayer_0001.jpg', 
                        help='path to target image')
    parser.add_argument('model_path')
    args = parser.parse_args()
    
    # load model and images
    model, preprocessor = load_model('resnet18', args.model_path)
    cuda = torch.cuda.is_available()
    base = preprocessor(Image.open(args.im1))
    target = preprocessor(Image.open(args.im2))

    # perform optimization
    misclassWithPatch, patch, finalLoss = do_optimization(torch.unsqueeze(base, 0), 
                                                     torch.unsqueeze(target, 0), args.model_path, cuda = cuda)

    # saving patch and base+patch
    torch.save(patch, '../patches/patch.pt')
    torch.save(misclassWithPatch, '../patches/misclassWithPatch.pt')

    # printing out the image
    misclassWithPatch = torch.load('../patches/misclassWithPatch.pt')
    tensor_to_array(misclassWithPatch, 'Base Pic With Patch')

if __name__ == "__main__":
    main()
