
from src.models import *
from src.methods.gradCAM import *
from src.methods.guided_backprop import *
from src.utilities import *
import matplotlib.pyplot as plt
from os import path

def guidedBackPropTest(image_path, result_folder, k = 1):
    """ Performs guided backprop on cat_dog image for k most likely classes """
    model = getResNetModel(152)
    model.eval()
    gbb = guidedBackProp(model)
    imageOriginal = getImagePIL(image_path)
    image = processImage(imageOriginal)

    image = image.unsqueeze(0)  # add 'batch' dimension
    print('image dim', image.size())
    gbb.forward(image)
    imagenetClasses = getImageNetClasses()
    topk = gbb.getTopK(k)
    print('top k', topk)
    for i in range(len(topk)):
        heatmap = gbb.generateMapClass(topk[i])
        gradientNumpy = gradientToImage(heatmap)
        file_name = imagenetClasses[topk[i]] + 'guidedBackprop'
        picture_path = os.path.join(result_folder, file_name)
        plt.imshow(gradientNumpy)
        plt.title(file_name)
        plt.savefig(picture_path)


def gradCamTest(image_path, result_folder, k = 1, layerList = ['layer4.2.conv3']):
    model = getResNetModel(152)
    model.eval()
    gm = gradCAM(model,layerList)
    imageOriginal = getImagePIL(image_path)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    gm.forward(image)
    imagenetClasses = getImageNetClasses()
    topk = gm.getTopK(k)
    for i in range(len(topk)):
        map = gm.generateMapClass(topk[i])
        for layers in layerList:
            className = imagenetClasses[topk[i]]
            heatmap = gm.generateCam(map,layers, imageOriginal,className)
            file_name = imagenetClasses[topk[i]] + 'GradCAM'
            picture_path = os.path.join(result_folder,file_name)
            plt.imshow(heatmap)
            plt.title(file_name)
            plt.savefig(picture_path)

def guidedGradCamTest(image_path, result_folder, k = 1, layerList = ['layer4.2.conv3']):
    model = getResNetModel(152)
    model.eval()
    gbp = guidedBackProp(model)
    gm = gradCAM(model, layerList)
    imageOriginal = getImagePIL(image_path)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    gbp.forward(image)
    gm.forward(image)
    imagenetClasses = getImageNetClasses()
    topk = gbp.getTopK(k)
    print('top k', topk)
    for i in range(len(topk)):
        mapCAM = gm.generateMapClass(topk[i])
        mapGuidedBackProp = gbp.generateMapClass(topk[i])
        gradientNumpy = gradientToImage(mapGuidedBackProp)
        for layers in layerList:
            className = imagenetClasses[topk[i]]
            heatmap = gm.generateCam(mapCAM,layers, imageOriginal,className)
            finalMap = heatmap * gradientNumpy
            print('final shape', finalMap.shape)
            print('max', gradientNumpy.max())
            print('min', gradientNumpy.min())
            file_name = imagenetClasses[topk[i]] + 'GuidedGradCAM'
            picture_path = os.path.join(result_folder,file_name)
            plt.imshow(finalMap)
            plt.title(file_name)
            plt.savefig(picture_path)
            
result_folder = path.abspath("../../../results/catdog")
image_path = path.abspath("../../../images/cat_dog.png")
gradCamTest(image_path, result_folder, k = 5)
guidedBackPropTest(image_path, result_folder, k=5)
guidedGradCamTest(image_path, result_folder, k=5)