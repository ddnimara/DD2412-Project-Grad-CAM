import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from tqdm import tqdm

from src.methods.gradCAM import *
from src.methods.guided_backprop import *
from src.utilities import *
import shap
from captum.attr import IntegratedGradients


class shapModel(nn.Module):
    def __init__(self, model):
        super(shapModel, self).__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x


def fidelity(model, layer, loader, visualize=False, percentile=50, dataset="MNIST"):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_size = len(loader.dataset)

    accuracy_before = np.zeros(data_size)
    it = 0
    print(accuracy_before.shape)
    # with torch.no_grad():
    #     for data, target in tqdm(loader):
    #         data, target = data.to(device), target.to(device)
    #
    #         output = F.softmax(model(data), dim=1)
    #         output_np = output.detach().cpu().numpy()
    #         ar = np.array([output_np[i][target[i]] for i in range(data.shape[0])])
    #         accuracy_before[it:it + data.shape[0]] = ar
    #         it += data.shape[0]

    print("before", accuracy_before)

    accuracy_after = np.zeros(data_size)
    it = 0
    gcm = gradCAM(model, layer)
    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)
        gcm.forward(data)

        map = gcm.generateMapClassBatch(target)
        heatmaps = gcm.generateCam(map, layer[0], image_path=None, mergeWithImage=False, isBatch=True)
        images = data.cpu().numpy().transpose(0, 2, 3, 1)
        images_copy = data.clone()
        for i in range(heatmaps.shape[0]):
            heatmap = heatmaps[i]
            image = images[i]
            if heatmap.max() == 0:
                images_copy[i] = torch.tensor(image.transpose(2, 0, 1))
            else:
                non_zero = np.nonzero(heatmap > 0)
                # print("shape", non_zero.shape)
                threshold = np.percentile(heatmap[non_zero], percentile)
                idx = heatmap > threshold
                image[idx[:,:,0],:] = 0
                images_copy[i] = torch.tensor(image.transpose(2, 0, 1))

        output = F.softmax(model(images_copy), dim=1)
        output_np = output.detach().cpu().numpy()
        ar = np.array([output_np[i][target[i]] for i in range(data.shape[0])])
        accuracy_after[it:it + data.shape[0]] = ar
        it += data.shape[0]

        if visualize:
            for i in range(heatmaps.shape[0]):
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax1.imshow(images[i])
                ax1.title.set_text("original image")
                heatmap = heatmaps[i]

                ax2 = fig.add_subplot(122)
                ax2.imshow(heatmap)
                ax2.title.set_text("heatmap")
                plt.show()

    print("DONE!")
    print("before", accuracy_before)
    print("after", accuracy_after)
    print("fidelity", (accuracy_before - accuracy_after).mean())


def contrastivity_batch(positive, negative):
    hamming_distance = np.count_nonzero(positive != negative)
    union = np.count_nonzero(np.logical_or(positive, negative))
    if union == 0:
        print("found 0")
        return 0
    return hamming_distance / union


def compute_contrastivity(heatmaps, positive):
    contrastivity_for_images = np.zeros(heatmaps.shape[0])
    for i in range(heatmaps.shape[0]):
        per_example_result = 0
        target = positive[i]
        # print("target", target)
        # plt.imshow(heatmaps[i, :, :, target])
        # plt.title("class {}".format(target))
        # plt.show()
        for j in range(heatmaps.shape[3]):
            if j == target:
                continue
            per_example_result += contrastivity_batch(heatmaps[i, :, :, target], heatmaps[i, :, :, j])
            # plt.imshow(heatmaps[i,:,:, j])
            # plt.title("class {}".format(j))
            # plt.show()
        contrastivity_for_images[i] = per_example_result / (heatmaps.shape[3] - 1)
    return contrastivity_for_images


def contrastivity(model, layer, loader, visualize=True, percentile=50, dataset="MNIST"):
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gcm = gradCAM(model, layer)
    it = 0
    data_size = len(loader.dataset)

    contrastivity = np.zeros(data_size)
    for data, positive in tqdm(loader):
        data, positive = data.to(device), positive.to(device)
        gcm.forward(data)
        batch_size = data.shape[0]
        every_class = np.array([[i for i in range(10)] for _ in range(batch_size)])
        heatmap_for_all_class = np.zeros((batch_size, data.shape[2], data.shape[3], 10))
        for j in range(10):
            labels = torch.tensor(every_class[:, j], dtype=torch.int64, device=device)
            map = gcm.generateMapClassBatch(labels)
            heatmaps = gcm.generateCam(map, layer[0], image_path=None, mergeWithImage=False, isBatch=True)
            binarized_heatmaps = np.zeros_like(heatmaps)
            for i in range(batch_size):

                heatmap = heatmaps[i]
                # threshold = np.nanpercentile(heatmap, 90)
                if heatmap.max() > 0:
                    non_zero = np.nonzero(heatmap > 0)
                    threshold = np.percentile(heatmap[non_zero], percentile)
                    idx = heatmap > threshold
                    binarized_heatmaps[i][idx] = 1
                # print('threshold', threshold)
                # print("max binarized", binarized_heatmaps.max())
                # plt.imshow(binarized_heatmaps[i])
            heatmap_for_all_class[:, :, :, j] = binarized_heatmaps[:, :, :, 0]

        contrastivity[it:it + batch_size] = compute_contrastivity(heatmap_for_all_class, positive)

        it += data.shape[0]

    print("final result", contrastivity.mean())


def fidelityShap(model, fit_loader, loader, visualize=True, percentile=50, dataset="MNIST"):
    shap_model = shapModel(model)
    shap_model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shap_model.to(device)
    data_size = len(loader.dataset)

    accuracy_before = np.zeros(data_size)
    it = 0
    print(accuracy_before.shape)
    with torch.no_grad():
        for data, target in tqdm(loader):
            data, target = data.to(device), target.to(device)

            output = shap_model(data)
            output_np = output.detach().cpu().numpy()
            ar = np.array([output_np[i][target[i]] for i in range(data.shape[0])])
            accuracy_before[it:it + data.shape[0]] = ar
            it += data.shape[0]

    accuracy_after = np.zeros(data_size)
    print('before', accuracy_before)
    # set background
    batch = next(iter(fit_loader))
    images, _ = batch
    background = images[:50].to(device)
    e = shap.DeepExplainer(shap_model, background)
    it = 0
    for data, positive in tqdm(loader):
        data, positive = data.to(device), positive.to(device)
        batch_size = data.shape[0]

        shap_heatmaps = e.shap_values(data)
        shap_heatmaps_np = [s.transpose(0, 2, 3, 1) for s in shap_heatmaps]
        images = data.cpu().numpy().transpose(0, 2, 3, 1)
        images_copy = data.clone()
        for i in range(batch_size):
            for j in range(10):
                if j != positive[i]:
                    continue

                shap_for_class = shap_heatmaps_np[j]
                sample_shap = shap_for_class[i, :, :, 0]
                # plt.imshow(sample_shap)
                # plt.show()
                if sample_shap.max() <= 0:
                    images_copy[i] = torch.tensor(image.transpose(2, 0, 1))
                else:
                    non_zero = np.nonzero(sample_shap > 0)
                    threshold = np.percentile(sample_shap[non_zero], percentile)
                    idx = sample_shap > threshold
                    image = images[i]
                    image[idx,:] = 0
                    images_copy[i] = torch.tensor(image.transpose(2, 0, 1))

                # plt.imshow(binarized_heatmaps[i,:,:])
                # plt.show()
        output = shap_model(images_copy)
        output_np = output.detach().cpu().numpy()
        ar = np.array([output_np[i][target[i]] for i in range(data.shape[0])])
        accuracy_after[it:it + data.shape[0]] = ar
        it += data.shape[0]

    print("DONE!")
    print("before", accuracy_before)
    print("after", accuracy_after)
    print("fidelity", (accuracy_before - accuracy_after).mean())


def contrastivityShap(model, fit_loader, loader, visualize=True, percentile=50, dataset="MNIST"):
    shap_model = shapModel(model)
    shap_model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shap_model.to(device)
    data_size = len(loader.dataset)

    # set background
    batch = next(iter(fit_loader))
    images, _ = batch
    background = images[:50].to(device)
    e = shap.DeepExplainer(shap_model, background)
    contrastivity = np.zeros(data_size)
    it = 0
    for data, positive in tqdm(loader):
        data, positive = data.to(device), positive.to(device)
        batch_size = data.shape[0]

        shap_heatmaps = e.shap_values(data)
        shap_heatmaps_np = [s.transpose(0, 2, 3, 1) for s in shap_heatmaps]
        binarized_heatmaps = np.zeros((batch_size, data.shape[2], data.shape[3], 10))

        for j in range(10):

            shap_for_class = shap_heatmaps_np[j]
            for i in range(batch_size):
                sample_shap = shap_for_class[i, :, :, 0]
                # plt.imshow(sample_shap)
                # plt.show()
                if sample_shap.max() > 0:
                    non_zero = np.nonzero(sample_shap > 0)
                    threshold = np.percentile(sample_shap[non_zero], percentile)

                    # idx = sample_shap < 0
                    # sample_shap[idx] = 0
                    # threshold = np.percentile(sample_shap, 90)
                    idx = sample_shap > threshold
                    # print('threshold', threshold)
                    binarized_heatmaps[i, idx, j] = 1
                    # plt.imshow(binarized_heatmaps[i,:,:,j])
                    # plt.show()
        contrastivity[it:it + batch_size] = compute_contrastivity(binarized_heatmaps, positive)
        it += data.shape[0]
    print("final result", contrastivity.mean())


def fidelityIG(model, loader, visualize=False, percentile=50, dataset="MNIST"):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ig = IntegratedGradients(model)
    data_size = len(loader.dataset)

    accuracy_before = np.zeros(data_size)
    it = 0
    print(accuracy_before.shape)
    with torch.no_grad():
        for data, target in tqdm(loader):
            data, target = data.to(device), target.to(device)

            output = F.softmax(model(data), dim=1)
            output_np = output.detach().cpu().numpy()
            ar = np.array([output_np[i][target[i]] for i in range(data.shape[0])])
            accuracy_before[it:it + data.shape[0]] = ar
            it += data.shape[0]

    print("before", accuracy_before)

    accuracy_after = np.zeros(10000)
    it = 0
    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)
        images = data.cpu().numpy().transpose(0, 2, 3, 1)
        images_copy = data.clone()
        baseline = torch.zeros(data.shape).to(device)
        attributions, _ = ig.attribute(data, baseline, target=target, return_convergence_delta=True)
        for i in range(attributions.shape[0]):
            heatmap = attributions[i].cpu().numpy()[0]
            if heatmap.max() <= 0:
                images_copy[i] = torch.tensor(image.transpose(2, 0, 1))
            else:
                non_zero = np.nonzero(heatmap > 0)
                threshold = np.percentile(heatmap[non_zero], percentile)
                image = images[i]
                idx = heatmap > threshold
                image[idx,:] = 0
                images_copy[i] = torch.tensor(image.transpose(2, 0, 1))

        output = F.softmax(model(images_copy), dim=1)
        output_np = output.detach().cpu().numpy()
        ar = np.array([output_np[i][target[i]] for i in range(data.shape[0])])
        accuracy_after[it:it + data.shape[0]] = ar
        it += data.shape[0]

        if visualize:
            for i in range(attributions.shape[0]):
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax1.imshow(images[i])
                ax1.title.set_text("original image")
                heatmap = attributions[i]

                ax2 = fig.add_subplot(122)
                ax2.imshow(heatmap)
                ax2.title.set_text("heatmap")
                plt.show()

    print("DONE!")
    print("before", accuracy_before)
    print("after", accuracy_after)
    print("fidelity", (accuracy_before - accuracy_after).mean())


def contrastivityIG(model, loader, visualize=True, percentile=50, dataset="MNIST"):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ig = IntegratedGradients(model)
    it = 0
    data_size = len(loader.dataset)

    contrastivity = np.zeros(data_size)
    for data, positive in tqdm(loader):
        data, positive = data.to(device), positive.to(device)
        batch_size = data.shape[0]
        baseline = torch.zeros(data.shape).to(device)
        every_class = np.array([[i for i in range(10)] for _ in range(batch_size)])
        heatmap_for_all_class = np.zeros((batch_size, data.shape[2], data.shape[3], 10))
        for j in range(10):
            labels = torch.tensor(every_class[:, j], dtype=torch.int64, device=device)
            attributions, _ = ig.attribute(data, baseline, target=labels, return_convergence_delta=True)
            binarized_heatmaps = np.zeros((batch_size, data.shape[2], data.shape[3]))
            for i in range(batch_size):

                heatmap = attributions[i].cpu().numpy()[0]
                if heatmap.max() > 0:
                    # plt.imshow(heatmap)
                    # plt.title("original")
                    # plt.show()
                    # threshold = np.nanpercentile(heatmap, 90)
                    non_zero = np.nonzero(heatmap > 0)
                    threshold = np.percentile(heatmap[non_zero], percentile)

                    idx = heatmap > threshold
                    binarized_heatmaps[i][idx] = 1
                    # print('threshold', threshold)
                    # print("max binarized", binarized_heatmaps.max())
                    # plt.imshow(binarized_heatmaps[i])
                    # plt.title("binarized")
                    # plt.show()
            heatmap_for_all_class[:, :, :, j] = binarized_heatmaps

        contrastivity[it:it + batch_size] = compute_contrastivity(heatmap_for_all_class, positive)

        it += data.shape[0]

    print("final result", contrastivity.mean())
