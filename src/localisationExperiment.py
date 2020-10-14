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
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from torchvision import transforms

def drawBoundingBoxes(image, label, predicted, truth):
    fig, ax = plt.subplots(1)
    image_numpy = image.cpu().numpy().transpose(1,2,0)
    ax.imshow(image_numpy)
    rectpred = patches.Rectangle((predicted[0], predicted[1]), predicted[2] - predicted[0],
                                 predicted[3] - predicted[1], linewidth=1, edgecolor='r', facecolor='none')

    ax.add_patch(rectpred)

    recttruth = patches.Rectangle((truth[0], truth[1]), truth[2] - truth[0],
                                 truth[3] - truth[1], linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(recttruth)
    plt.title(label)
    plt.show()
def compute_iou(predicted, truth):
    """ Given predicted an true bounding boxes of the form [xmin, ymin, xmax, ymax] computes their IOU. """
    #  reshape box = [x1, y1, x2, y2] to [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
    pred_reshaped = []
    pred_reshaped.append([predicted[0], predicted[1]])
    pred_reshaped.append([predicted[2], predicted[1]])
    pred_reshaped.append([predicted[2], predicted[3]])
    pred_reshaped.append([predicted[0], predicted[3]])

    truth_reshaped = []
    truth_reshaped.append([truth[0], truth[1]])
    truth_reshaped.append([truth[2], truth[1]])
    truth_reshaped.append([truth[2], truth[3]])
    truth_reshaped.append([truth[0], truth[3]])

    # Compute IoU using Polygon Library
    poly_1 = Polygon(pred_reshaped)
    poly_2 = Polygon(truth_reshaped)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area

    return iou

def iouThreshold(iou, thresh = 0.5):
    if iou >= thresh:
        return 1
    else:
        return 0

def heatmapThreshold(heatmap, t = 0.15):
    binarizedHeatmap = np.zeros_like(heatmap)
    maxValue = heatmap.max()
    idx = heatmap > t * maxValue
    binarizedHeatmap[idx] = 255
    return binarizedHeatmap

def generateBoundingBox(heatmap):
    heatmap_bin = heatmapThreshold(heatmap)
    heatmap_bin = heatmap_bin.astype(np.uint8)
    cnts = cv2.findContours(heatmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    best = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if len(best) == 0:
            best = [x, y, w, h]
        else:
            if w * h > best[2] * best[3]:  # found bigger square
                best = [x, y, w, h]
    return [best[0], best[1], best[0] + best[2], best[1] + best[3]]


def gradCamLocalisation(image_path, k = 1, layerList = ['features']):
    """ This is used to test localisation on a specific image only. Do not use for batches.
        It will visualize original image along the predicted bounding box
    """
    result_folder = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\results\catdog"

    # process image
    model = getVGGModel(16)
    model.eval()
    gm = gradCAM(model,layerList)
    imageOriginal, im_path = getImagePIL(image_path)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    gm.forward(image)
    imagenetClasses = getImageNetClasses()
    topk = gm.getTopK(k)
    # loop through top k classes
    for i in range(len(topk)):
        map = gm.generateMapClass(topk[i])
        for layers in layerList:
            className = imagenetClasses[topk[i]]
            heatmap = gm.generateCam(map, layers, im_path, guided=True, counterFactual=False)
            print("heatmap shape", heatmap.shape)
            file_name = className + 'GradCAM'
            heatmap = heatmapThreshold(heatmap)
            heatmap = heatmap.astype(np.uint8)
            cnts = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            fig, ax = plt.subplots(1)
            ax.imshow(imageOriginal)
            best =[]
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if len(best)==0:
                    best = [x, y, w, h]
                else:
                    if w*h > best[2]*best[3]:  # found bigger square
                        best = [x, y, w, h]
                #rect = cv2.rectangle(heatmap, (x, y), (x + w, y + h), (36, 255, 12), 2)

            #x, y, w, h = cv2.boundingRect(heatmap)
            # Create figure and axes
            rect = patches.Rectangle((best[0], best[1]), best[2], best[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Add the patch to the Axes
            plt.title(className)
            plt.show()

def stringToListOfBoxes(stringInput):
    """
        More easily understood with an example. Say you have
        StringInput = "n09193705 45 49 499 162 n09193705 2 69 437 207"
        Then:
        - InitialList = ["09193705 45 49 499 162 ","09193705 2 69 437 207"]
        - IntermidiateList = ["45 49 499 162 ","2 69 437 207"]
        - FinalList = [["45","49","499","162"],["2","69","437", "207"]]
    """
    initialList = stringInput.split("n")[1:]
    intermidiateList = [initialList[i][9:] for i in range(len(initialList))]
    finalList = [intermidiateList[i].split(" ")[:4] for i in range(len(intermidiateList))]
    return finalList


def getAttributesFromXML(root):
    """ Given an xml, returns the size of the image, the true classes and the true bounding boxes present. """
    sizes = []
    for type_tag in root.findall('size'):
        sizes = [type_tag[i].text for i in range(len(list(type_tag)) - 1)]
    names = []
    bbox = []
    for type_tag in root.findall('object'):
        name = type_tag.find('name').text
        xmin, ymin, xmax, ymax = int(type_tag.find("bndbox/xmin").text), int(type_tag.find("bndbox/ymin").text), \
                                 int(type_tag.find("bndbox/xmax").text), int(type_tag.find("bndbox/ymax").text)
        names.append(name)
        bbox.append([xmin, ymin, xmax, ymax])

    return sizes, names, bbox

def generateDataframe(directoryPath):
    """ Scans the directory for xmls (and images) and places useful info in dataframe (pandas).
        In order for this to work the images must be located in the directory path, while the xml files in
        directoryPath/val info.
    """
    pcl_file_name = os.path.join(directoryPath, "dataframe")
    if os.path.exists(pcl_file_name):  # if we've done this before, extract it via the pickle
        return pd.read_pickle(pcl_file_name)
    else:
        # get xml and imagel lists
        image_list = glob.glob(directoryPath + "\*.JPEG")
        xml_directory = os.path.join(directoryPath, "val info")
        xml_list = glob.glob(xml_directory + "/*.xml")
        sizes = []
        names = []
        bboxs = []
        name_to_id = translatewnidToClassNames()
        id = []
        for i, xml in enumerate(xml_list):
            if i % 1000 == 0:
                print("i", i)
            xml_root = ET.parse(xml).getroot()
            size, name, bbox = getAttributesFromXML(xml_root)
            sizes.append(size)
            names.append(name)
            bboxs.append(bbox)
            id.append([name_to_id[n] for n in name])
        df = pd.DataFrame()
        df["path"] = pd.Series(image_list)
        df["name"] = pd.Series(names)
        df["id"] = pd.Series(id)
        df["size"] = pd.Series(sizes)
        df["bounding box"] = pd.Series(bboxs)
        # resize:
        for index in range(df.shape[0]):
            row = df.iloc[index]
            original_width = int(row["size"][0])
            original_hight = int(row["size"][1])
            width_ratio = 224.0 / original_width
            hight_ratio = 224.0 / original_hight
            bboxs = row["bounding box"]
            for bbox in bboxs:
                bbox[0] = int(width_ratio * bbox[0])
                bbox[1] = int(hight_ratio * bbox[1])
                bbox[2] = int(width_ratio * bbox[2])
                bbox[3] = int(hight_ratio * bbox[3])

        pd.to_pickle(df, pcl_file_name)
        return df


def testBatchLocalisation(model, df, layer=['features.42'], k = 1):  #.42
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

    batch_size = 10 # My gpu is running out of memory for more :(
    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size, shuffle=False)

    gcm = gradCAM(model, layer)
    it = 0  # we will use to keep track of the batch location
    iou = np.zeros((df_count_size, k))  # iou[i,k] = 1 iff i-th bounding box correspodning to the k-th class was close to a true box
    for batch_images in tqdm(validationLoader):
        batch_images = batch_images.to(device)

        # df_view gives us access the dataframe on the batch window
        df_view = df.iloc[it:min(it + batch_size, df_count_size)]
        gcm.forward(batch_images)

        # get top k classes (indeces)
        topk = gcm.probs.sort(dim=1, descending=True)[1].cpu().numpy()[:,:k]
        # loop through top k classes
        for classNumber in range(k):  # loop through likeliest predictions
            batch_label = topk[:,classNumber]

            # generate the heatmaps
            map = gcm.generateMapClassBatch(torch.from_numpy(batch_label).to(device))
            heatmap = gcm.generateCam(map, layer[0], image_path = None, mergeWithImage = False, isBatch= True)

            for i in range(heatmap.shape[0]):  # iterate over batch
                truthLabels = df_view.iloc[i].loc["id"]
                truthBoxes = df_view.iloc[i].loc["bounding box"]
                bndbox = generateBoundingBox(heatmap[i])  # generate box based on heatmap
                for index, true_class in enumerate(truthLabels):  # loop through the true labels
                    if int(true_class) != int(batch_label[i]):  # don't compare boxes of different objects
                        continue
                    bndbox_true = truthBoxes[index]

                    # used for debug:
                    # drawBoundingBoxes(batch_images[i], imagenetClasses[int(true_class)], bndbox, bndbox_true)
                    w = bndbox_true[2] - bndbox_true[0]
                    h = bndbox_true[3] - bndbox_true[1]
                    threshold = min(0.5, 1.0*w*h/((w+10)*(h+10)))
                    iou_temp = iouThreshold(compute_iou(bndbox, bndbox_true), thresh=threshold)
                    iou[it + i, classNumber] = max(iou[it + i, classNumber], iou_temp)
        it += batch_size
    perRowMax = iou.max(axis=1)
    print("max", perRowMax.shape)
    success_total = perRowMax.sum()/iou.shape[0]
    print("success percentage", success_total)
    print("error localisation", 1 - success_total)

def testBatchAccuracy(model, df, k = 1):  #.42
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

    batch_size = 20 # My gpu is running out of memory for more :(
    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size, shuffle=False)

    it = 0  # we will use to keep track of the batch location
    # Accuracy
    predictions = np.zeros(df_count_size)
    for batch_images in tqdm(validationLoader):
        batch_images = batch_images.to(device)

        # df_view gives us access the dataframe on the batch window
        df_view = df.iloc[it:min(it + batch_size, df_count_size)]
        output = model.forward(batch_images)
        # get top k classes (indeces)
        topk = output.sort(dim=1, descending=True)[1].cpu().numpy()[:,:k]
        for i in range(topk.shape[0]):
            truthLabels = df_view.iloc[i].loc["id"]
            # loop through top k classes
            for classNumber in range(k):  # loop through likeliest predictions
                batch_label = topk[i, classNumber]
                if batch_label in truthLabels:
                    predictions[it + i] = 1
                    break



        it += batch_size
    print("shape", predictions.shape)
    success_total = predictions.sum()/predictions.shape[0]
    print("success percentage", success_total)
    print("error localisation", 1 - success_total)

def reshapeImagenetImages():
    originalValPath = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\datasets\ILSVRC2012 val"
    reshapedPath = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\datasets\ILSVRC2012 val\resized"
    image_list = glob.glob(originalValPath + "\*.JPEG")
    for i, image_path in enumerate(image_list):
        if i % 1000 == 0:
            print("i", i)
        image = Image.open(image_path)
        title = image_path.split("\\")[-1]
        image_reshaped = transforms.Compose([transforms.Resize((224,224))])(image)
        new_image_path = os.path.join(reshapedPath,title)
        image_reshaped.save(new_image_path)
        image.close()


# print("Picking Dataframe")
directoryPath = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\datasets\ILSVRC2012 val"
df = generateDataframe(directoryPath)
# print("Updating its image list")
image_list = glob.glob(os.path.join(directoryPath,"resized") + "\*.JPEG")
# df_csv["path"] = pd.Series(image_list)
# df_csv.to_csv("../datasets/res.csv")
df["path"] = pd.Series(image_list)
# print("Starting Localisation experiment on VGG")
model = getGoogleModel()
testBatchAccuracy(model, df, k=1)
