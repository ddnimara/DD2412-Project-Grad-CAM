import torchvision.models as models

def getResNetModel(number=18, pretrained=True):
    if number == 18:
        return models.resnet18(pretrained=pretrained)
    elif number == 34:
        return models.resnet34(pretrained=pretrained)
    elif number == 50:
        return models.resnet50(pretrained=pretrained)
    elif number == 101:
        return models.resnet101(pretrained=pretrained)
    else:
        return models.resnet152(pretrained=pretrained)

def getDenseNetModel(number=121, pretrained = True):
    if number == 121:
        return models.densenet121(pretrained=pretrained)
    elif number == 161:
        return models.densenet161(pretrained=pretrained)
    else:
        return models.densenet169(pretrained=pretrained)


def getVGGModel(number=16, pretrained=True, batchNorm=False):
    if number == 11:
        if batchNorm:
            return models.vgg11_bn(pretrained=pretrained)
        else:
            return models.vgg11(pretrained=pretrained)
    elif number == 13:
        if batchNorm:
            return models.vgg13_bn(pretrained=pretrained)
        else:
            return models.vgg13(pretrained=pretrained)
    elif number == 16:
        if batchNorm:
            return models.vgg16_bn(pretrained=pretrained)
        else:
            return models.vgg16(pretrained=pretrained)
    else:
        if batchNorm:
            return models.vgg19_bn(pretrained=pretrained)
        else:
            return models.vgg19(pretrained=pretrained)

def getGoogleModel(pretrained=True):
    return models.googlenet(pretrained=pretrained)

def getAlexNet(pretrained=True):
    return models.alexnet(pretrained=pretrained)
