from src.models import *
from src.methods.gradCAM import *
from src.methods.guided_backprop import *
from src.utilities import *
from src.experiments.cat_dog.catDogExperiment import *
from src.dataSetLoader import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import glob
from tqdm import tqdm
from torchvision import transforms
from ast import literal_eval

def testBatchAccuracy(model, df, batch_size=10, k=1):  # .42
    """ Computes IOU for a model using batches (work in progress)."""

    # Get device (so you can use gpu if possible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    df_count_size = df.shape[0]

    # create data loader

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    validationSet = dataSetILS(df, transformations)

    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size, shuffle=False)

    it = 0  # we will use to keep track of the batch location
    # Accuracy
    predictions = np.zeros(df_count_size)

    with torch.no_grad():
        for batch_images in tqdm(validationLoader):
            batch_images = batch_images.to(device)

            # df_view gives us access the dataframe on the batch window
            df_view = df.iloc[it:min(it + batch_size, df_count_size)]
            #print("dataframe view:",df_view.head())
            output = model.forward(batch_images)
            # get top k classes (indeces)
            topk = output.sort(dim=1, descending=True)[1].cpu().numpy()[:, :k]
            for i in range(topk.shape[0]):
                truthLabels = df_view.iloc[i].loc["id"]
                #print("truth labels", truthLabels)
                # loop through top k classes
                for classNumber in range(k):  # loop through likeliest predictions
                    batch_label = topk[i, classNumber]
                    #print('batch label', batch_label)
                    if str(batch_label) in truthLabels:
                        predictions[it + i] = 1
                        break

            it += batch_size

        if batch_size==10:
            name = "vgg_predictions"
        else:
            name = "alexnet_predictions"
        np.save(name, predictions)
    return predictions


def getCommonSuccesses(pred_1, pred_2):
    indeces = np.nonzero(pred_1 == pred_2)[0]
    return indeces


def loadCommonImages(df, model1 = getVGGModel(16,batchNorm=True), model2 = getAlexNet(), number_of_images=10):
    print("Starting training with model 1...")
    try:
        pred_1 = np.load("vgg_predictions.npy")
    except:
        pred_1 = testBatchAccuracy(model = model1, df =df, batch_size=10, k = 5)

    print("Starting training with model 2...")
    try:
        pred_2 = np.load("alexnet_predictions.npy")
    except:
        pred_2 = testBatchAccuracy(model = model2, df =df, batch_size=20, k = 5)

    common_indeces = getCommonSuccesses(pred_1,pred_2)

    print("common_indeces")
    print(common_indeces)
    return common_indeces
models = [getAlexNet(), getVGGModel()]
layers = [['features.11'], ['features.29']]

def generateGuidedBpForClass(model, image_path, targetClass, index, isVGG=True):
    result_folder = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\results\truthfulness\guidedBackProp"  # change it accordingly
    model.eval()
    gbb = guidedBackProp(model)
    imageOriginal, im_path = getImagePIL(image_path)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    gbb.forward(image)
    imagenetClasses = getImageNetClasses()
    # loop through top k classes
    className = imagenetClasses[targetClass]
    heatmap = gbb.generateMapClass(targetClass)
    gradientNumpy = tensorToHeatMap(heatmap)
    file_name = str(index) + '_guidedBackprop'
    if isVGG:
        file_name = file_name + '_VGG'
    else:
        file_name = file_name + '_AlexNet'
    className = imagenetClasses[targetClass]
    file_name = file_name + "_" + className
    picture_path = os.path.join(result_folder, file_name)
    plt.imshow(gradientNumpy)
    plt.title(file_name)
    plt.savefig(picture_path)

def generateGuidedGradCamForClass(model, image_path, targetClass, layerList, index, isVGG=True):
    result_folder = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\results\truthfulness\guidedGradCam"  # change it accordingly
    model.eval()
    gbp = guidedBackProp(model)
    gm = gradCAM(model, layerList)
    imageOriginal, im_path = getImagePIL(image_path)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    gbp.forward(image)
    gm.forward(image)
    imagenetClasses = getImageNetClasses()
    # loop through top k classes
    mapCAM = gm.generateMapClass(targetClass)
    mapGuidedBackProp = gbp.generateMapClass(targetClass)
    gradientNumpy = tensorToHeatMap(mapGuidedBackProp)
    for layers in layerList:
        heatmap = gm.generateCam(mapCAM, layers, im_path, mergeWithImage=False)
        finalMap = heatmap * gradientNumpy
        file_name = str(index) + '_GuidedGradCAM'
        if isVGG:
            file_name = file_name + '_VGG'
        else:
            file_name = file_name + '_AlexNet'

        className = imagenetClasses[targetClass]
        file_name = file_name + "_"+ className
        picture_path = os.path.join(result_folder, file_name)
        plt.imshow(finalMap)
        plt.title(file_name)
        plt.savefig(picture_path)

def showCams(common_indeces, df, models, layers):
    model_1 = models[0]
    model_2 = models[1]
    layer_1 = layers[0]
    layers_2 = layers[1]
    for i in common_indeces:
        path = df['path'].iloc[i]
        target_class = literal_eval(df['id'].iloc[i])[0]
        print('target class', target_class)
        print("Generating guided back prop")
        generateGuidedBpForClass(model_1, path, target_class, index=i, isVGG=False)
        generateGuidedBpForClass(model_2, path, target_class, index=i, isVGG=True)
        print("generating guided grad cam")
        generateGuidedGradCamForClass(model_1, path, target_class, layer_1, index=i, isVGG=False)
        generateGuidedGradCamForClass(model_2, path, target_class, layers_2, index=i, isVGG=True)

def generateGuidedGradCamForClassTogether(model, image_path, targetClass, layerList):
    model.eval()
    gbp = guidedBackProp(model)
    gm = gradCAM(model, layerList)
    imageOriginal, im_path = getImagePIL(image_path)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    gbp.forward(image)
    gm.forward(image)
    imagenetClasses = getImageNetClasses()
    # loop through top k classes
    mapCAM = gm.generateMapClass(targetClass)
    mapGuidedBackProp = gbp.generateMapClass(targetClass)
    gradientNumpy = tensorToHeatMap(mapGuidedBackProp)
    layers = layerList[0]
    heatmap = gm.generateCam(mapCAM, layers, im_path, mergeWithImage=False)
    finalMap = heatmap * gradientNumpy
    return finalMap

def generateGuidedBpForClassTogether(model, image_path, targetClass):
    model.eval()
    gbb = guidedBackProp(model)
    imageOriginal, im_path = getImagePIL(image_path)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    gbb.forward(image)
    imagenetClasses = getImageNetClasses()
    # loop through top k classes
    className = imagenetClasses[targetClass]
    heatmap = gbb.generateMapClass(targetClass)
    gradientNumpy = tensorToHeatMap(heatmap)
    return gradientNumpy



df_csv = pd.read_csv("../../../datasets/resized.csv")


common_indeces = [1,13,26,31,33,34,64,64,66,102,103, 110, 122, 150]

def generateImagesSideBySide(common_indeces, df, models, layers):
    result_folder = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\results\truthfulness\experiment pictures\final"  # change it accordingly
    gpp_folder = os.path.join(result_folder,"guidedBackProp")
    gcm_folder = os.path.join(result_folder,"guidedGradCam")
    model_indeces = np.arange(len(models))
    model_names = [model.__class__.__name__ for model in models]
    imagenetClasses = getImageNetClasses()
    for i in common_indeces:
        permutation = np.random.permutation(model_indeces)
        model_1 = models[permutation[0]]
        model_2 = models[permutation[1]]
        layer_1 = layers[permutation[0]]
        layer_2 = layers[permutation[1]]
        path = df['path'].iloc[i]
        target_class = literal_eval(df['id'].iloc[i])[0]
        file_name = str(i) + "_" + model_names[permutation[0]] + "_" + model_names[permutation[1]] + "_" + str(imagenetClasses[target_class])
        print("Generating guided back prop")
        gbp_1 = generateGuidedBpForClassTogether(model_1, path, target_class)
        gbp_2 = generateGuidedBpForClassTogether(model_2, path, target_class)
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(gbp_1)
        ax1.title.set_text("Agent A")

        ax2 = fig.add_subplot(122)
        ax2.imshow(gbp_2)
        ax2.title.set_text("Agent B")

        fig.savefig(os.path.join(gpp_folder, file_name))
        print("generating guided grad cam")
        gcm_1 = generateGuidedGradCamForClassTogether(model_1, path, target_class, layer_1)
        gcm_2 = generateGuidedGradCamForClassTogether(model_2, path, target_class, layer_2)

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(gcm_1)
        ax1.title.set_text("Agent A")

        ax2 = fig.add_subplot(122)
        ax2.imshow(gcm_2)
        ax2.title.set_text("Agent B")

        fig.savefig(os.path.join(gcm_folder, file_name))


generateImagesSideBySide(common_indeces, df_csv, models, layers)
# df_csv = pd.read_csv("../datasets/res2_120.csv")
# print("Picking Dataframe")
# directoryPath = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\datasets\ILSVRC2012 val"
# df = generateDataframe(directoryPath)
# # print("Updating its image list")
# df_csv = pd.read_csv("../datasets/resized.csv")
# image_list = glob.glob(os.path.join(directoryPath,"resized") + "\*.JPEG")
# df_csv["path"] = pd.Series(image_list)
# df_csv.to_csv("../datasets/resized.csv")
#
# pd.set_option('display.max_colwidth', None)
# print(df_csv.head())
# df["path"] = pd.Series(image_list)
# # print("Starting Localisation experiment on VGG")
# model = getAlexNet()
# testBatchAccuracy(model, df, k=1)
# generateDataframe(directoryPath)

# print("top 5 accuracies:")
# print("googlenet:")
# model = getGoogleModel()
# testBatchAccuracy(model, df, batch_size=20, k=5)
#
# print("Alexnet:")
# model = getAlexNet()
# testBatchAccuracy(model, df, batch_size=20, k=5)
#
# print("VGG-16:")
# model = getVGGModel()
# testBatchAccuracy(model, df,  k=5)
#
# print("VGG-16 with batch norm")
# model = getVGGModel(batchNorm=True)
# testBatchLocalisation(model, df, k=5)

# print("top 1 localisation:")
# print("googlenet:")
# model = getGoogleModel()
# testBatchLocalisation(model, df, layer=['inception5b'], batch_size=20, k=1)
#
# print("Alexnet:")
# model = getAlexNet()
# testBatchLocalisation(model, df, layer=['features.11'],batch_size=20, k=1)

# print("VGG-16:")
# model = getVGGModel()
# testBatchLocalisation(model, df,layer=['features.29'], k=1)

# print("VGG-16 with batch norm")
# model = getVGGModel(batchNorm=True)
# testBatchLocalisation(model, df, layer=['features.42'], k=1)

# print("top 5 localisation:")
# print("googlenet:")
# model = getGoogleModel()
# testBatchLocalisation(model, df, layer=['inception5b'], batch_size=20, k=5)
#
# print("Alexnet:")
# model = getAlexNet()
# testBatchLocalisation(model, df, layer=['features.11'],batch_size=20, k=5)
#
# print("VGG-16:")
# model = getVGGModel()
# testBatchLocalisation(model, df,layer=['features.29'], k=5)

# print("VGG-16 with batch norm")
# model = getVGGModel(batchNorm=True)
# testBatchLocalisation(model, df, layer=['features.42'], k=5)
