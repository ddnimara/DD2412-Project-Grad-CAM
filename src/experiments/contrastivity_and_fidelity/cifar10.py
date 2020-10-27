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
batch_size_train = 4
batch_size_test = 4
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
                                         shuffle=False)

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


layer = ['inception5b']

fidelityShap(model, train_loader, test_loader, percentile=50)