from __future__ import print_function

import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from tqdm import tqdm

from src.methods.gradCAM import *
from src.methods.guided_backprop import *
from src.utilities import *
import shap
from captum.attr import IntegratedGradients
from src.experiments.contrastivity_and_fidelity.contrastivity_and_fidelity import *
class mnistModel(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(mnistModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(12*12*20, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x


batch_size_train = 1024
batch_size_test = 10

normalisation = transforms.Normalize((0.1307,), (0.3081,))

train_val_dataset = MNIST('../../../datasets/mnist/train', train=True, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               normalisation
                             ]))

test_set = MNIST('../../../datasets/mnist/test', train=False, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               normalisation
                             ]))
train_set, val_set = torch.utils.data.random_split(train_val_dataset, [50000, 10000])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True)


def train(model, device, optimizer, epochs, scheduler):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        model.zero_grad()
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = F.softmax(model(data), dim=1)
            loss = F.nll_loss(output.log(), target)
            loss.backward()
            optimizer.step()

        val_acc, val_loss = evaluate(model, val_loader, device)
        print("after {} epoch we have {} loss and {} accuracy".format(epoch,val_loss,val_acc))
        scheduler.step(val_loss)

    torch.save(model,"model.pt")

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

def train_model():
    model = mnistModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, eps=1e-06)
    train(model,device, optimizer, 20, scheduler)

def eval():
    try:
        model = torch.load("model.pt")
    except:
        train_model()
    model = torch.load("model.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accuracy, loss = evaluate(model, test_loader, device)
    print("accuracy", accuracy)
    print("average loss", loss)

def infoOfModel(model):
    for name, layer in model.named_modules():  # module is of format [name, module]
        print("layer", layer)
        print("name", name)


model = torch.load('model.pt')
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model.to(device)
print(model)
layer = ['conv_layers']

# contrastivityShap(model, train_loader,test_loader, percentile=1)