from lib.uwcosnet_model.uwcosnet_resnet import UWCOSNet_ResNet50
from lib.uwcosnet_model.uwcosnet_pvt import UWCOSNet_PVT


def ModelFunc(name):
    if name == "model_name":
        print("create model")
    elif name == "UWCOSNet_ResNet50":
        model = UWCOSNet_ResNet50(num_classes=1, backbone_path="./pretrained_pth/resnet50-0676ba61.pth")
    elif name == "UWCOSNet_PVTv2B2":
        model = UWCOSNet_PVT(num_classes=1, backbone_path="./pretrained_pth/pvt_v2_b2.pth")
    return model
    
