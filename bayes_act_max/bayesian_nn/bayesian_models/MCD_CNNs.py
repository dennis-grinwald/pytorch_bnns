from torch import nn
import torch.nn.functional as F

from .mcd_utils import DropoutConv2d, DropoutLinear

class MCD_CNN_11L_224_3C(nn.Module):
    def __init__(self, p=0.5):
        super(MCD_CNN_11L_224_3C, self).__init__()

        # Feature Extractor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(5,5), padding=(1,1))
        self.conv2 = DropoutConv2d(in_channels=48, out_channels=96, kernel_size=(5,5), padding=(1,1), p=p)
        self.conv3 = DropoutConv2d(in_channels=96, out_channels=125, kernel_size=(5,5), padding=(1,1), p=p)
        self.conv4 = DropoutConv2d(in_channels=125, out_channels=180, kernel_size=(5,5), padding=(1,1), p=p)
        self.conv5 = DropoutConv2d(in_channels=180, out_channels=256, kernel_size=(5,5), padding=(1,1), p=p)
        self.conv6 = DropoutConv2d(in_channels=256, out_channels=300, kernel_size=(5,5), padding=(1,1), p=p)
        self.conv7 = DropoutConv2d(in_channels=300, out_channels=350, kernel_size=(5,5), padding=(1,1), p=p)
        self.conv8 = DropoutConv2d(in_channels=350, out_channels=400, kernel_size=(5,5), padding=(1,1), p=p)
        self.pool = nn.MaxPool2d(2,2)

        # Classifier
        self.fc1 = DropoutLinear(in_features=10*10*400, out_features=512, p=p)
        self.fc2 = DropoutLinear(in_features=512, out_features=64, p=p)
        self.fc3 = DropoutLinear(in_features=64, out_features=10, p=p)

    def forward(self, x):
        # Feature Extractor
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x)) 
        x = self.pool(x) 
        x = F.relu(self.conv3(x)) 
        x = F.relu(self.conv4(x)) 
        x = self.pool(x) 
        x = F.relu(self.conv5(x)) 
        x = F.relu(self.conv6(x)) 
        x = self.pool(x)
        x = F.relu(self.conv7(x)) 
        x = F.relu(self.conv8(x)) 
        x = self.pool(x) 

        # Classifier
        x = x.view(-1, 10*10*400) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MCD_CNN_9L_32_3C(nn.Module):
    def __init__(self, p=0.5):
        super(MCD_CNN_9L_32_3C, self).__init__()

        # Feature Extractor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(5,5), padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(5,5), padding=(1,1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=125, kernel_size=(5,5), padding=(1,1))
        self.conv4 = nn.Conv2d(in_channels=125, out_channels=180, kernel_size=(3,3), padding=(1,1))
        self.conv5 = DropoutConv2d(in_channels=180, out_channels=256, kernel_size=(3,3), padding=(1,1), p=p)
        self.conv6 = DropoutConv2d(in_channels=256, out_channels=300, kernel_size=(3,3), padding=(1,1), p=p)
        self.pool = nn.MaxPool2d(2,2)

        # Classifier
        self.fc1 = DropoutLinear(in_features=3*3*300, out_features=512, p=p)
        self.fc2 = DropoutLinear(in_features=512, out_features=64, p=p)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Feature Extractor
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x)) 
        x = self.pool(x) 
        x = F.relu(self.conv3(x)) 
        x = F.relu(self.conv4(x)) 
        x = self.pool(x) 
        x = F.relu(self.conv5(x)) 
        x = F.relu(self.conv6(x)) 
        x = self.pool(x)
        #x = F.relu(self.conv7(x)) 
        #x = F.relu(self.conv8(x)) 
        #x = self.pool(x) 

        # Classifier
        x = x.view(-1, 300*3*3) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

class MCD_CNN_7L_32_3C(nn.Module):
    def __init__(self, p=0.5):
        super(MCD_CNN_7L_32_3C, self).__init__()

        # Feature Extractor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3,3), padding=(1,1))
        self.conv2 = DropoutConv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1), p=p)
        self.conv3 = DropoutConv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1), p=p)
        self.conv4 = DropoutConv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1), p=p)
        self.pool = nn.MaxPool2d(2,2)

        # Classifier
        self.fc1 = DropoutLinear(in_features=8*8*256, out_features=512, p=p)
        self.fc2 = DropoutLinear(in_features=512, out_features=64, p=p)
        self.fc3 = DropoutLinear(in_features=64, out_features=10, p=p)

    def forward(self, x):
        # Feature Extractor
        x = F.relu(self.conv1(x)) #32*32*48
        x = F.relu(self.conv2(x)) #32*32*96
        x = self.pool(x) #16*16*96
        x = F.relu(self.conv3(x)) #16*16*192
        x = F.relu(self.conv4(x)) #16*16*256
        x = self.pool(x) # 8*8*256

        # Classifier
        x = x.view(-1, 8*8*256) # reshape x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
