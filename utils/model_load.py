import torch

def ModelLoadFunc(name, model, pth_path):
    if name == "model_name":
        print("model load_state_dict")
    elif name == "UWCOSNet_ResNet50":
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    elif name == "UWCOSNet_PVTv2B2":
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    
    return model