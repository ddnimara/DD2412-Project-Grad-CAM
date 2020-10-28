import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from src.methods.gradCAM import *
from src.methods.guided_backprop import *
from src.utilities import *
import shap
from captum.attr import IntegratedGradients
from src.experiments.contrastivity_and_fidelity.googlenet import *
from src.experiments.contrastivity_and_fidelity.contrastivity_and_fidelity import *
#abs_path = os.path.abspath('C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\src\experiments\contrastivity and fidelity\state_dicts')
model = GoogLeNet()
model.load_state_dict(torch.load('state_dicts/googlenet.pt'))
print(model)
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
batch_size_train = 2
batch_size_test = 2
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)])

trainset = CIFAR10(root='../../../datasets/CIFAR10/train', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                          shuffle=True)

testset = CIFAR10(root='.../../../datasets/CIFAR10/test', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                         shuffle=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class shapModel(GoogLeNet):
    def __init__(self, model):
        super(shapModel, self).__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x


def eval(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    accuracy, loss = evaluate(model, test_loader, device)
    print("accuracy", accuracy)
    print("average loss", loss)

def get_hits(predicted, target):
    return predicted.eq(target.data).sum().item()
def evaluate(model, loader, device):
    model.eval()
    hits = 0
    size = 0
    print("size", size)
    loss = 0
    with torch.no_grad():
        for data, target in tqdm(loader):
            size += data.size(0)
            #print('target', target)
            data, target = data.to(device), target.to(device)
            output = F.softmax(model(data), dim=1)
            loss += F.nll_loss(output.log(), target, reduction='sum')
            _, predicted_class = output.topk(1, dim = 1)
            #print('predicted', predicted_class.squeeze(1))
            hits += get_hits(predicted_class.squeeze(1), target)

        accuracy = hits/size
        loss_avg = loss/size
    return accuracy, loss_avg

def visualize(model,layer,  fit_loader, loader, number_of_images=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # shap
    batch = next(iter(fit_loader))
    images, _ = batch
    background = images[:50].to(device)
    shap_model = shapModel(model)
    shap_model.eval()
    e = shap.DeepExplainer(shap_model, background)

    # gcm
    gcm = gradCAM(model, layer)

    # ig
    ig = IntegratedGradients(model)

    batch = next(iter(loader))
    images, target = batch
    images, target = images.to(device), target.to(device)

    # shap
    shap_heatmaps = e.shap_values(images)
    shap_heatmaps_np = [s.transpose(0, 2, 3, 1) for s in shap_heatmaps]

    # gcm
    gcm.forward(images)
    map = gcm.generateMapClassBatch(target)
    heatmaps = gcm.generateCam(map, layer[0], image_path=None, mergeWithImage=False, isBatch=True)\

    # ig
    baseline = torch.zeros(images.shape).to(device)
    attributions, _ = ig.attribute(images, baseline, target=target, return_convergence_delta=True)

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    for i in range(number_of_images):
        # real
        images[i] = inv_normalize(images[i])
        image = images[i].cpu().numpy().transpose(1,2,0)

        real_class = target[i].cpu().numpy()
        print(real_class)
        # shap
        sample_shap_for_class = shap_heatmaps_np[real_class]
        sample_shap = sample_shap_for_class[i,:,:,0]
        sample_shap = np.clip(sample_shap, a_min = 0, a_max = 1)
        sample_shap-=sample_shap.min()
        sample_shap/=sample_shap.max()
        # gradcam
        heatmap_cam = heatmaps[i]

        # ig
        heatmap_ig = attributions[i].cpu().numpy()[0]
        heatmap_ig= np.clip(heatmap_ig, a_min = 0, a_max = 1)
        heatmap_ig-=heatmap_ig.min()
        heatmap_ig/= heatmap_ig.max()
        fig = plt.figure()
        ax1 = fig.add_subplot(141)
        ax1.imshow(image)

        ax1.title.set_text("Original Image")

        ax2 = fig.add_subplot(142)
        ax2.imshow(sample_shap)
        ax2.title.set_text("SHAP")

        ax3 = fig.add_subplot(143)
        ax3.imshow(heatmap_cam)
        ax3.title.set_text("Grad CAM")

        ax4 = fig.add_subplot(144)
        ax4.imshow(heatmap_ig)
        ax4.title.set_text("Integrated Gradients")

        plt.show()

layer = ['inception5b']
result = np.zeros(2)

contrastivity(model, layer, test_loader, percentile=10)
# for p in [1,10]:
#     print("IG contrastivity")
#     print("p = ", p)
#     contrastivity_ig =  np.array([contrastivityIG(model,test_loader, percentile = p)])
#     np.savetxt("contrastivity_IG{}.csv".format(p), contrastivity_ig, delimiter=",")
