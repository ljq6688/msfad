import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from models.yolo import Model
import yaml

class Hook_Model_Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.cov1 = nn.Conv2d(3,16,3,2)
        self.bn1 = nn.BatchNorm2d(16)
        self.r1 = nn.ReLU()
        self.cov2 = nn.Conv2d(16,32,3,2)
        self.bn2 = nn.BatchNorm2d(32)
        self.r2 = nn.ReLU()
        self.register_buffer('aaa',torch.tensor([123]))
        self.se = nn.Sequential(nn.Conv2d(32,64,3,2),
                                nn.BatchNorm2d(64),
                                nn.ReLU())
        self.se2 = nn.Sequential(nn.Conv2d(64,128,3,2),
                                nn.BatchNorm2d(128),
                                nn.ReLU())

    def forward(self,x):
        x = self.cov1 (x)
        x = self.bn1(x)
        x = self.r1(x)
        self.res1 = x
        x = self.cov2(x)
        x = self.bn2(x)
        x = self.r2(x)
        x = self.se(x)
        self.res2 = x
        x = self.se2(x)
        return x

module_name = []
features_in_hook = []
features_out_hook = []


def hook(module, fea_in, fea_out):
    print("hooker working")
    module_name.append(module.__class__)
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None


if __name__ == '__main__':
    cfg = 'models/yolov5s.yaml'
    hyp = 'data/hyps/hyp.scratch-low.yaml'
    nc = 20
    img = cv2.imread('data/images/bus.jpg')
    img = transforms.ToTensor()(img).unsqueeze(0)
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors'))  # create
    dic1 = dict(model.named_modules())
    dic2 = dict(model.named_children())
    dic3 = dict(model.named_buffers())
    dic4 = dict(model.named_parameters())
    dic5 = dict(model.state_dict())

    out = model(img)








    # model = Hook_Model_Test()
    # print(model.__class__)
    # dic1 = dict(model.named_modules())
    # dic2 = dict(model.named_children())
    # dic3 = dict(model.named_buffers())
    # dic4 = dict(model.named_parameters())
    # dic5 = dict(model.state_dict())
    # dic1[list(dic1.keys())[-5]].register_forward_hook(hook=hook)
    # out = model(img)
    # print(dic2[list(dic2.keys())[2]])
    # print(model.res2)
    # print(features_out_hook)

