import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torch_better_output
import tqdm
import sys
# Image Preprocessing 

transform = transforms.Compose([
    transforms.Scale(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])
batch_size=250
# CIFAR-10 Dataset
train_dataset = dsets.CIFAR10(root='./torchtestdata/',
                               train=True, 
                               transform=transform,
                               download=False)
test_dataset = dsets.CIFAR10(root='./torchtestdata/',
                              train=False, 
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
print('finish data load')
# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv = conv3x3(3, 32)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 32, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[1], 2)
        self.layer3 = self.make_layer(block, 128, layers[2], 2)
#        self.layer4 = self.make_layer(block, 128, layers[3], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(128, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
#        print('')
#        print ('x1:',x.size())
        out = self.conv(x)
#        print ('x2:',out.size())
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
#        print ('l1:',out.size())
        out = self.layer2(out)
#        print ('l2:',out.size())
        out = self.layer3(out)
#        print ('l3:',out.size())
#        out = self.layer4(out)
#        print ('l4:',out.size())
        
        
        out = self.avg_pool(out)
#        print ('avgpool:',out.size())
        out = out.view(out.size(0), -1)
#        print ('flat:',out.size())
        out = self.fc(out)
#        print ('fc:',out.size())
        return out
#resnet=models.resnet18()
#print (resnet)
resnet = ResNet(ResidualBlock, [2,2,2,2])
#print (resnet)
#sys.exit()
resnet.cuda()
#cnn.load_state_dict(torch.load('pytorch_scene_model.pkl'))
#resnet.load_state_dict(torch.load('resnet.pkl'))
def printacc(model):
    top1=0
    top3=0
    top5=0
    total=0
    for images, labels in test_loader:
        labels=labels.cuda()
        images=images.cuda()
        images = Variable(images)
        outputs = model(images)
        
        ac1,ac3,ac5=torch_better_output.accuracy(outputs.data,labels,topk=(1,2,3))
        top1+=ac1[0]
        top3+=ac3[0]
        top5+=ac5[0]
        total += labels.size(0)
    print('top1:%g%% top2:%g%% top3:%g%%'%(100*top1/total,100*top3/total,100*top5/total))

criterion = nn.CrossEntropyLoss()
criterion.cuda()
lr = 0.03
optimizer = torch.optim.Adam(resnet.parameters(),lr=lr)
# Training 
for epoch in range(80):
    with tqdm.tqdm(total=len(train_dataset)//batch_size,leave=True) as pbar:
#        pbar.update(0)
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                pbar.set_description('Epoch [%d/%d]'%(epoch+1,80))
                pbar.set_postfix(loss=loss.data[0])
                pbar.update(10)
                
    printacc(resnet)
    # Decaying Learning Rate
    if (epoch+1) % 20 == 0:
        lr /= 3
        optimizer = torch.optim.Adam(resnet.parameters(), lr=lr) 
    
# Test


# Save the Model

#torch.save(resnet.state_dict(), 'resnet.pkl')