"""
Modified version of attack.py from https://github.com/Fangyh09/one-pixel-attack-mnist.pytorch. Should be used with the other files from the original repository.
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from models import *
from utils import progress_bar
from torch.autograd import Variable
from os import path
from PIL import Image
import matplotlib.pyplot as plt

from differential_evolution import differential_evolution

parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')
parser.add_argument('--model', default='vgg16', help='The target model')
parser.add_argument('--pixels', default=1, type=int, help='The number of pixels that can be perturbed.')
parser.add_argument('--maxiter', default=100, type=int, help='The maximum number of iteration in the DE algorithm.')
parser.add_argument('--popsize', default=400, type=int, help='The number of adverisal examples in each iteration.')
parser.add_argument('--samples', default=100, type=int, help='The number of image samples to attack.')
parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
parser.add_argument('--save', default='./results/results.pkl', help='Save location for the results with pickle.')
parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

args = parser.parse_args()

def perturb_image(xs, img):
        if xs.ndim < 2:
                xs = np.array([xs])
        batch = len(xs)
        imgs = img.repeat(batch, 1, 1, 1)
        xs = xs.astype(int)

        count = 0
        for x in xs:
                pixels = np.split(x, len(x)/5)
                
                for pixel in pixels:
                        x_pos, y_pos, r, g, b = pixel
                        imgs[count, 0, x_pos, y_pos] = (r/255.0-0.485)/0.229
                        imgs[count, 1, x_pos, y_pos] = (g/255.0-0.456)/0.224
                        imgs[count, 2, x_pos, y_pos] = (b/255.0-0.406)/0.224
                count += 1

        return imgs

def predict_classes(xs, img, target_calss, net, minimize=True):
        imgs_perturbed = perturb_image(xs, img.clone())
        with torch.no_grad():
            input = imgs_perturbed.cuda()
        predictions = F.softmax(net(input), dim=1).data.cpu().numpy()[:, target_calss]

        return predictions if minimize else 1 - predictions

def attack_success(x, img, target_calss, net, targeted_attack=False, verbose=False):

        attack_image = perturb_image(x, img.clone())
        with torch.no_grad():
            input = attack_image.cuda()
        confidence = F.softmax(net(input), dim=1).data.cpu().numpy()[0]
        predicted_class = np.argmax(confidence)

        if (verbose):
                print ("Confidence: %.4f"%confidence[target_calss])
        if (targeted_attack and predicted_class == target_calss) or (not targeted_attack and predicted_class != target_calss):
                return True


def attack(img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
        # img: 1*3*W*H tensor
        # label: a number

        targeted_attack = target is not None
        target_calss = target if targeted_attack else label

        bounds = [(0,224), (0,224), (0,255), (0,255), (0,255)] * pixels

        popmul = max(1, popsize/len(bounds))

        predict_fn = lambda xs: predict_classes(
                xs, img, target_calss, net, target is None)
        callback_fn = lambda x, convergence: attack_success(
                x, img, target_calss, net, targeted_attack, verbose)

        # print("type.popmul", type(popmul))
        inits = np.zeros([int(popmul*len(bounds)), len(bounds)])
        for init in inits:
                for i in range(pixels):
                        init[i*5+0] = np.random.random()*224
                        init[i*5+1] = np.random.random()*224
                        init[i*5+2] = np.random.normal(128,127)
                        init[i*5+3] = np.random.normal(128,127)
                        init[i*5+4] = np.random.normal(128,127)

        attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
                recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

        attack_image = perturb_image(attack_result.x, img)
        with torch.no_grad():
            attack_var = attack_image.cuda()
        predicted_probs = F.softmax(net(attack_var), dim=1).data.cpu().numpy()[0]

        predicted_class = np.argmax(predicted_probs)

        if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_calss):
                return 1, attack_result.x.astype(int)
        return 0, [None]


def attack_all(net, loader, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False):

        correct = 0
        success = 0

        for batch_idx, (input, target) in enumerate(loader):

                img_var = torch.Tensor(input, requires_grad=False).cuda()
                prior_probs = F.softmax(net(img_var), dim=1)
                _, indices = torch.max(prior_probs, 1)
                
                if target[0] != indices.data.cpu()[0]:
                        continue

                correct += 1
                target = target.numpy()

                targets = [None] if not targeted else range(10)
                print("targeted mode", targeted)

                for target_calss in targets:
                        if (targeted):
                                if (target_calss == target[0]):
                                        continue
                        
                        flag, x = attack(input, target[0], net, target_calss, pixels=pixels, maxiter=maxiter, popsize=popsize, verbose=verbose)
                        print("flag==>", flag)

                        success += flag
                        if (targeted):
                                success_rate = float(success)/(9*correct)
                        else:
                                success_rate = float(success)/correct

                        if flag == 1:
                                print ("success rate: %.4f (%d/%d) [(x,y) = (%d,%d) and (R,G,B)=(%d,%d,%d)]"%(
                                        success_rate, success, correct, x[0],x[1],x[2],x[3],x[4]))
                
                if correct == args.samples:
                        break

        return success_rate

def main():
        transforms_val = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # mean, std of imagenet
                ])
        
        is_imagenet = False
        is_vgg = True
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_vgg:
            net = torchvision.models.vgg16(pretrained=True)
        else:
            net = torchvision.models.googlenet(pretrained=True)
        net.to(device)
        net.eval()
        cudnn.benchmark = True
        
        if is_imagenet:
                print("==> Loading data...")
                
                df = pd.read_csv("../DD2412-Project-Grad-CAM/datasets/res2_2510.csv")
                validationSetOriginal = dataSetILS(df)
                validationSet = dataSetILS(df, transforms_val)
                batch_size = 16
                validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size, shuffle=False)
                
                print ("==> Starting attack...")
                
                it = 0
                for batch_images in validationLoader:
                        batch_images = batch_images.to(device)
                        out = net(batch_images)
                        probs, labels = out.topk(k=1)
                
                        for i in range(batch_size):
                                print(it, attack(batch_images[i], int(labels[i][0]), net, target=None, pixels=1, maxiter=150, popsize=175, verbose=False))
                                it += 1
                        
        else:
                image_path = path.abspath("../DD2412-Project-Grad-CAM/images/cat_dog.png")
                imageOriginal, im_path = getImagePIL(image_path)
                image = processImage(imageOriginal, transforms_val).to(device)
                
                plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
                plt.show()
                
                out = net(image.unsqueeze(0))
                _, label = out.topk(k=1)
                print(attack(image, int(label), net, target=None, pixels=1, maxiter=200, popsize=150, verbose=False))
        
def getImagePIL(image_path, verbose = False):
        """ Works on a SINGLE image, specified by image_path. This code currently works only for the cat_dog experiment.
            Don't use it if you want to load batches of images (instead use the dataloared class).
        """
        # Get image location
        image_folder_path = path.abspath(image_path)
        
        if verbose:
            print('Image path', image_path)

        image = Image.open(image_path).convert('RGB')
        return image, image_path

def processImage(image, transf):
        """ Useful for processsing a single image. Once more, if you wish to process a batch, use the dataloader class instead."""
        image = image.copy()
        image = transf(image)  #  apply some transformations
        return image

class dataSetILS(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms=None):
        self.df = dataframe
        self.transforms = transforms

    def __getitem__(self, index):
        """ Returns the image based on the path stored in the dataframe"""
        row = self.df.iloc[index]
        full_path = row.loc['path']
        img_array = Image.open(full_path).convert('RGB')
        if self.transforms is not None:
            img_array = self.transforms(img_array)
        return img_array

    def __len__(self):
        return self.df.shape[0]


if __name__ == '__main__':
        main()
