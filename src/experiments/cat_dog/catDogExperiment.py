
from src.models import *
from src.methods.gradCAM import *
from src.methods.guided_backprop import *
from src.utilities import *
import matplotlib.pyplot as plt
from os import path

def guidedBackPropTest(image_path, k = 1, model = getResNetModel(152)):
    """ Performs guided backprop on cat_dog image for k most likely classes """
    result_folder = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\results\catdog"  # change for your path
    model.eval()
    gbb = guidedBackProp(model)
    imageOriginal, im_path = getImagePIL(image_path)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    gbb.forward(image)
    imagenetClasses = getImageNetClasses()
    _, topk = gbb.getTopK(k)
    topk = topk[0]

    # loop through top k classes
    for i in range(len(topk)):
        cl = int(topk[i])
        className = imagenetClasses[cl]
        heatmap = gbb.generateMapClass(cl)
        gradientNumpy = gradientToImage(heatmap)
        file_name = className + 'guidedBackprop'
        picture_path = os.path.join(result_folder, file_name)
        plt.imshow(gradientNumpy)
        plt.title(file_name)
        plt.show()
        #plt.savefig(picture_path)


def gradCamTest(image_path, k = 1, model = getResNetModel(152), layerList = ['layer4'], counterFactual = False):
    # check what type it is, so we store them separately
    if counterFactual:
        result_folder = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\results\catdog\counterFactual"
    else:
        result_folder = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\results\catdog"

    model.eval()
    gm = gradCAM(model,layerList)
    imageOriginal, im_path = getImagePIL(image_path)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    gm.forward(image)
    imagenetClasses = getImageNetClasses()
    _, topk = gm.getTopK(k)
    topk = topk[0]
    
    # loop through top k classes
    for i in range(len(topk)):
        cl = int(topk[i])
        map = gm.generateMapClass(cl)
        for layers in layerList:
            className = imagenetClasses[cl]
            heatmap = gm.generateCam(map, layers, im_path, counterFactual= counterFactual)
            file_name = className + 'GradCAM'
            if counterFactual:
                file_name += "CounterFactual"
            picture_path = os.path.join(result_folder,file_name)
            plt.imshow(heatmap)
            plt.title(file_name)
            plt.show()
            #plt.savefig(picture_path)

def guidedGradCamTest(image_path, k = 1, model = getResNetModel(152), layerList = ['layer4']):
    result_folder = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\results\catdog"  # change it accordingly
    model.eval()
    gbp = guidedBackProp(model)
    gm = gradCAM(model, layerList)
    imageOriginal, im_path = getImagePIL(image_path)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    gbp.forward(image)
    gm.forward(image)
    imagenetClasses = getImageNetClasses()
    _, topk = gbp.getTopK(k)
    topk = topk[0]

    # loop through top k classes
    for i in range(len(topk)):
        cl = int(topk[i])
        mapCAM = gm.generateMapClass(cl)
        mapGuidedBackProp = gbp.generateMapClass(cl)
        gradientNumpy = gradientToImage(mapGuidedBackProp)
        for layers in layerList:
            className = imagenetClasses[cl]
            heatmap = gm.generateCam(mapCAM, layers, im_path, mergeWithImage = False)
            finalMap = heatmap * gradientNumpy
            file_name = className + 'GuidedGradCAM'
            picture_path = os.path.join(result_folder,file_name)
            plt.imshow(finalMap)
            plt.title(file_name)
            plt.show()
            #plt.savefig(picture_path)

def getModelDetails(model = getResNetModel(152)):
    """ Useful method to get layer information (which we use to access activations in gradcam) """
    for name, layer in model.named_modules():  # module is of format [name, module]
        print("layer", layer)
        print("name", name)


def runExperiment(layerList = ["features.29"], image_path = "cat_dog.png", model = getVGGModel(16), k=1, counterFactual = False, experiment_name = "GradCam"):
    """ Runs the experiments. By default it runs VGG16 with gradcam"""
    if experiment_name == "GuidedBp":
        guidedBackPropTest(model = model,image_path = image_path, k = k)
    elif experiment_name == "GradCam":
        gradCamTest(model = model,image_path = image_path, k = k, layerList = layerList, counterFactual = counterFactual)
    else:
        guidedGradCamTest(model = model,image_path = image_path, k = k, layerList = layerList)


# result_folder = path.abspath("../../../results/catdog2")
# image_path = path.abspath("../../../images/cat_dog.png")
# gradCamTest(image_path, result_folder, k = 5)
# guidedBackPropTest(image_path, result_folder, k=5)
# guidedGradCamTest(image_path, result_folder, k=5)
# model = getVGGModel(pretrained=True)
# getModelDetails(model)
#path = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\images\cat_dog.png"

# runExperiment(image_path = image_path, model = model, layerList=["features.29"], k = 2, experiment_name = "GradCAM")

