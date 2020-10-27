from src.models import *
from src.methods.gradCAM import *
from src.methods.guided_backprop import *
from src.utilities import *
from src.dataSetLoader import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.methods.new_grad_cam as ngc

# Get ImageNet classes
imagenetClasses = getImageNetClasses()

# Initialize model and load data
model = getVGGModel(16)
model.eval()
layer = ['features.29']
image_path = path.abspath("../../../images/cat_dog.png")
imageOriginal, im_path = getImagePIL(image_path)
image = processImage(imageOriginal)

# Grad-CAM
gcm = gradCAM(model, layer)
gcm.forward(image.unsqueeze(0))
probs, labels = gcm.getTopK(k=3)
probs = probs[0]
labels = labels[0]

# Print original probabilities and labels
print("Original predictions:")
for prob, label in zip(probs, labels):
    print(float(prob), imagenetClasses[int(label)])
print()

# Plot original image
# plt.imshow(imageOriginal)
# plt.show()

# Plot heatmap
imageOriginalTensor = to_tensor_transform(imageOriginal)
map = gcm.generateMapClass(labels[0])
heatmap = gcm.generateCam(map, layer[0], image_path=None, isBatch=False, rescale=False, mergeWithImage=False)
ngc.GradCAM.plot_heatmap(imageOriginalTensor, heatmap, ratio=(0.5, 0.5))
plt.title(imagenetClasses[int(labels[0])])
plt.show()

#########################################################################x

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
                    imgs[count, 0, x_pos, y_pos] = (r / 255.0 - mean[0])/std[0]
                    imgs[count, 1, x_pos, y_pos] = (g / 255.0 - mean[1])/std[1]
                    imgs[count, 2, x_pos, y_pos] = (b / 255.0 - mean[2])/std[2]
            count += 1

    return imgs

# Perturb original image
arr = np.array([205, 104, 248, 255, 76])
im_array = np.array(imageOriginal)
im_array[arr[0], arr[1], 0] = arr[2]
im_array[arr[0], arr[1], 1] = arr[3]
im_array[arr[0], arr[1], 2] = arr[4]

# Perturb image
image = perturb_image(arr, image)

# Predict
gcm.forward(image)
# gcm.forward(image.unsqueeze(0))
probs, labels = gcm.getTopK(k=3)
probs = probs[0]
labels = labels[0]

# Print top probabilities and labels
print("New predictions:")
for prob, label in zip(probs, labels):
    print(float(prob), imagenetClasses[int(label)])
print()
    
# Plot new image
# plt.imshow(im_array)
# plt.show()

print("Relevant pixel in heatmap:", heatmap[arr[0], arr[1]])

# Plot heatmap
im_array = to_tensor_transform(im_array)
map = gcm.generateMapClass(labels[0])
heatmap = gcm.generateCam(map, layer[0], image_path=None, isBatch=False, rescale=False, mergeWithImage=False)
ngc.GradCAM.plot_heatmap(im_array, heatmap, ratio=(0.5, 0.5))
plt.title(imagenetClasses[int(labels[0])])
plt.show()