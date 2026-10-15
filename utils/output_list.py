
class OutputListFunc:
    def __init__(self, name):
        if name == "model_name":
            print("self.calculate = self.Func")
        
        elif name == "UWCOSNet_ResNet50":
            self.calculate = self.UWCOSNet_Func
        elif name == "UWCOSNet_PVTv2B2":
            self.calculate = self.UWCOSNet_Func
    
    def UWCOSNet_Func(self, pre):
        out1, out2, out3, out4, out_global_mask, out_global_edge, out_edge1, out_edge2, out_edge3, out_edge4 =  pre
        return out1
    