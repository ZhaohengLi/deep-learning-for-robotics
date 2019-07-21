import torch
import torch.nn as nn
from main import run

# Flatten Layer: To flatten a tensor into two-dimension
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

# example rgbNet: using only rgb images as input to train a model
class rgbNet(nn.Module):
    def __init__(self):
        super(rgbNet, self).__init__()
        self.net = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    Flatten(),
                )

        self.mix_net = nn.Sequential(
                    nn.Linear(6*6*64, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 15),
                ) 

    def forward(self, img, dep): 
        feat = self.net(img)
        score = self.mix_net(feat)
        return score

# example depthNet: using only depth image as input to train a model
class depthNet(nn.Module):
    def __init__(self):
        super(depthNet, self).__init__()
        self.net = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    Flatten(),
                )

        self.mix_net = nn.Sequential(
                    nn.Linear(6*6*64, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 15),
                ) 

    def forward(self, img, dep): 
        feat = self.net(dep)
        score = self.mix_net(feat)
        return score

# *****************************IMPORTANT******************************
# YOU NEED TO FILL THE CLASS TO FINISH A RGB-D FUSION NETWORK
# NOTICE THAT YOU ONLY NEED TO DEFINE THE NETWORK, AND WE HAVE ALREADY BUILT THE OTHER PARTS(LIKE DATA LOADER, \
# TRAINING CODE ...)
# AFTER FINISHING THE NETWORK, JUST EXCUTE run(rgbdNet) WILL START TO TRAIN AND YOU CAN OBSERVE THE TRAINING PROCESS AND THE\
# ACCURACY ON VALIDATION SET

# AND ALSO YOU CAN RUN run(rgbNet) AND run(depthNet) TO TRAIN ONLY RGB OR DEPTH MODAL. YOU CAN OBSERVE IF THE FUSION \
# GIVE AN ACCURACY BOOST. 

# IF YOU HAVE ANY TROUBLE, YOU CAN REFER TO THE PREVIOUS rgbNet and depthNet
class rgbdNet(nn.Module):
    def __init__(self):
        super(rgbdNet, self).__init__()
        self.rgb_net = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    Flatten(),
                )

        self.dep_net = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    Flatten(),
                )

        self.mix_net = nn.Sequential(
                    nn.Linear(2*6*6*64, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 15),
                ) 

    def forward(self, img, dep): 
        feat_1 = self.rgb_net(img)
        feat_2 = self.dep_net(dep)
        feat = torch.cat([feat_1, feat_2], 1)
        score = self.mix_net(feat)
        return score

# ********************************************************************

run(rgbdNet)

# run(rgbNet)

# run(depthNet)
