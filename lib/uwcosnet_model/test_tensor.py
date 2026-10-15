import torch
from thop import profile
from tqdm import tqdm
import time

from uwcosnet_pvt import UWCOSNet_PVT


model = UWCOSNet_PVT(channel=32).cuda()
model.eval()
image = torch.randn(1,3,352,352).cuda()

out_list = model(image)
for out in out_list:
    print(out.shape)


res = []
for i in tqdm(range(1000)):
    torch.cuda.synchronize()
    start = time.time()
    res_= model(image)
    torch.cuda.synchronize()
    end = time.time()
    res.append(end-start)
time_sum = 0
for i in res:
    time_sum += i
print("FPS: %f"%(1.0/(time_sum/len(res))))


flops, params = profile(model, (image,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))