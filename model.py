import torchvision.models as models

def getResNetModel(number=18, pretrained= True):
    if number == 18:
        return models.resnet18(pretrained=pretrained)
    elif number == 34:
        return models.resnet34(pretrained=pretrained)
    elif number == 50:
        return models.resnet50(pretrained=pretrained)
    else:
        return models.resnet101(pretrained=pretrained)

def getDenseNetModel(number=121, pretrained = True):
    if number == 121:
        return models.densenet121(pretrained=pretrained)
    elif number == 161:
        return models.densenet161(pretrained=pretrained)
    else:
        return models.densenet169(pretrained=pretrained)


def getVGGModel(number=16, pretrained=True):
    if number == 11:
        return models.vgg11(pretrained=pretrained)
    elif number == 13:
        return models.vgg13(pretrained=pretrained)
    elif number == 16:
        return models.vgg16(pretrained=pretrained)
    else:
        return models.vgg19(pretrained=pretrained)

def getGoogleModel(pretrained=True):
    return models.googlenet(pretrained=pretrained)

def getAlexNet(pretrained=True):
    return models.alexnet(pretrained=pretrained)
