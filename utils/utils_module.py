import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AddCoords(nn.Module):
    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, input_tensor):
        batch_size, _, height, width = input_tensor.size()

        xx_channel = torch.arange(width).repeat(1, height, 1)
        yy_channel = torch.arange(height).repeat(1, width, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (width - 1)
        yy_channel = yy_channel.float() / (height - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        input_tensor = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        return input_tensor

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords()
        self.conv = nn.Conv2d(in_channels+2, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x
    
class MotionNet_Coord(nn.Module):
    def __init__(self, in_channel, out_channel, num):
        super(MotionNet_Coord, self).__init__()
        if(num==8):
            self.conv = nn.Sequential(
                CoordConv(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
            )
        if(num==10):
            self.conv = nn.Sequential(
                CoordConv(in_channel, 64, kernel_size=5, stride=1, padding=0),
                CoordConv(64, out_channel, kernel_size=3, stride=1, padding=0),
            )
        if(num==12):
            self.conv = nn.Sequential(
                CoordConv(in_channel, out_channel, kernel_size=5, stride=1, padding=0),
            )
        if(num==14):
            self.conv = nn.Sequential(
                CoordConv(in_channel, out_channel, kernel_size=3, stride=1, padding=0),
            )
        if(num==16):
            self.conv = nn.Sequential(
                CoordConv(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            )
        
    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()
        return out

class PointNet(nn.Module):
    def __init__(self, num_classes=4, grid_h=12, grid_w=12):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(8, 256, 1)
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.h = grid_h
        self.w = grid_w
        self.fc_fea = nn.Linear(16*16, 6)

    def forward(self, pre, fea):
        '''pre-processing the predicted points'''
        x = pre.reshape(-1, 2, (self.h*self.w))
        
        '''pre-processing the prompt features and form the superpoint'''
        fea = nn.MaxPool1d(fea.size(-1))(fea).squeeze(-1)
        fea = F.relu(self.fc_fea(fea))
        fea = fea.unsqueeze(-1).repeat(1, 1, self.h*self.w)
        superpoint = torch.cat((x, fea), dim=1)
        
        '''learn the superpoints' features'''
        x = F.relu(self.conv1(superpoint))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = x.view(-1, 512)
        
        '''classification'''
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        
    def forward(self,x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)
        return prompt