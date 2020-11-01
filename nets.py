import torch

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
class Encoder(nn.Module):
    def __init__(self,vgg):
        super(Encoder,self).__init__()
        # vgg
        # 224 x 224
        self.conv1 = nn.Conv2d(2,3,1,1,0)
#         self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
#         self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226
    
        # 226 x 226
        self.conv2 = nn.Conv2d(3,64,3,1,0)
        self.conv2.weight = torch.nn.Parameter(vgg.features[0].weight.float())
        self.conv2.bias = torch.nn.Parameter(vgg.features[0].bias.float())
        self.relu2 = nn.ReLU(inplace=True)
        self.reflecPad2 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

#         self.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        self.conv3 = nn.Conv2d(64,64,3,1,0)
        self.conv3.weight = torch.nn.Parameter(vgg.features[2].weight.float())
        self.conv3.bias = torch.nn.Parameter(vgg.features[2].bias.float())
        self.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        self.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
#         # 112 x 112

        self.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        self.conv4 = nn.Conv2d(64,128,3,1,0)
        self.conv4.weight = torch.nn.Parameter(vgg.features[5].weight.float())
        self.conv4.bias = torch.nn.Parameter(vgg.features[5].bias.float())
        self.relu4 = nn.ReLU(inplace=True)
        # 112 x 112
        
        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,128,3,1,0)
        self.conv5.weight = torch.nn.Parameter(vgg.features[7].weight.float())
        self.conv5.bias = torch.nn.Parameter(vgg.features[7].bias.float())
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112
        
        self.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 56 x 56
        
        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(128,256,3,1,0)
        self.conv6.weight = torch.nn.Parameter(vgg.features[10].weight.float())
        self.conv6.bias = torch.nn.Parameter(vgg.features[10].bias.float())
        self.relu6 = nn.ReLU(inplace=True)
        # 56 x 56
        
        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(256,256,3,1,0)
        self.conv7.weight = torch.nn.Parameter(vgg.features[12].weight.float())
        self.conv7.bias = torch.nn.Parameter(vgg.features[12].bias.float())
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56
        
        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(256,256,3,1,0)
        self.conv8.weight = torch.nn.Parameter(vgg.features[14].weight.float())
        self.conv8.bias = torch.nn.Parameter(vgg.features[14].bias.float())
        self.relu8 = nn.ReLU(inplace=True)
        # 56 x 56
        
        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(256,256,3,1,0)
        self.conv9.weight = torch.nn.Parameter(vgg.features[16].weight.float())
        self.conv9.bias = torch.nn.Parameter(vgg.features[16].bias.float())
        self.relu9 = nn.ReLU(inplace=True)
        # 56 x 56
        
        self.maxPool3 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 28 x 28
        
        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(256,512,3,1,0)
        self.conv10.weight = torch.nn.Parameter(vgg.features[19].weight.float())
        self.conv10.bias = torch.nn.Parameter(vgg.features[19].bias.float())
        self.relu10 = nn.ReLU(inplace=True)
        # 28 x 28
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad2(out)
#         out = self.reflecPad3(out)
        out = self.conv3(out)
        pool = self.relu3(out)
        out,pool_idx = self.maxPool(pool)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out,pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        pool3 = self.relu9(out)
        out,pool_idx3 = self.maxPool3(pool3)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        return out

class Decoder(nn.Module):
    def __init__(self,w):
        super(Decoder,self).__init__()        
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(512,256,3,1,0)
        self.conv11.weight = torch.nn.Parameter(w['1.weight'].float())
        self.conv11.bias = torch.nn.Parameter(w['1.bias'].float())
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
#         # 56 x 56

        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = nn.Conv2d(256,256,3,1,0)
        self.conv12.weight = torch.nn.Parameter(w['5.weight'].float())
        self.conv12.bias = torch.nn.Parameter(w['5.bias'].float())
        self.relu12 = nn.ReLU(inplace=True)
#         # 56 x 56

        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = nn.Conv2d(256,256,3,1,0)
        self.conv13.weight = torch.nn.Parameter(w['8.weight'].float())
        self.conv13.bias = torch.nn.Parameter(w['8.bias'].float())
        self.relu13 = nn.ReLU(inplace=True)
#         # 56 x 56

        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = nn.Conv2d(256,256,3,1,0)
        self.conv14.weight = torch.nn.Parameter(w['11.weight'].float())
        self.conv14.bias = torch.nn.Parameter(w['11.bias'].float())
        self.relu14 = nn.ReLU(inplace=True)
#         # 56 x 56

        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(256,128,3,1,0)
        self.conv15.weight = torch.nn.Parameter(w['14.weight'].float())
        self.conv15.bias = torch.nn.Parameter(w['14.bias'].float())
        self.relu15 = nn.ReLU(inplace=True)
#         # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
#         # 112 x 112

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(128,128,3,1,0)
        self.conv16.weight = torch.nn.Parameter(w['18.weight'].float())
        self.conv16.bias = torch.nn.Parameter(w['18.bias'].float())
        self.relu16 = nn.ReLU(inplace=True)
#         # 112 x 112

        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(128,64,3,1,0)
        self.conv17.weight = torch.nn.Parameter(w['21.weight'].float())
        self.conv17.bias = torch.nn.Parameter(w['21.bias'].float())
        self.relu17 = nn.ReLU(inplace=True)
#         # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
#         # 224 x 224

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(64,64,3,1,0)
        self.conv18.weight = torch.nn.Parameter(w['25.weight'].float())
        self.conv18.bias = torch.nn.Parameter(w['21.bias'].float())
        self.relu18 = nn.ReLU(inplace=True)
#         # 224 x 224

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(64,1,3,1,0)
#        self.conv19.weight = torch.nn.Parameter(w['28.weight'].float())
#        self.conv19.bias = torch.nn.Parameter(w['28.bias'].float())



    def forward(self,x):
        # decoder
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out

class Generator(nn.Module):
    def __init__(self, vgg, dec_w):
        """
        - vgg: vgg19 weights (e.g., models.vgg19(pretrained=True))
        - dec_w: pretrained weights for decoder (e.g., torch.load('feature_invertor_conv4_1.pth'))
        """
        super(Generator, self).__init__()
        self.enc = Encoder(vgg)
        self.dec = Decoder(dec_w)
    
    def forward(self,x):
        code = self.enc(x)
        out = self.dec(code)
        return out 

if __name__ == "__main__":
    vgg19 = models.vgg19(pretrained=True)
    dec_w = torch.load('models/pretrained_models/autoencoder_vgg19/vgg19_4/feature_invertor_conv4_1.pth')
    inputs = torch.rand(10,2, 224, 224)
    gen = Generator(vgg19, dec_w)
    outputs = gen(inputs)  
    assert(outputs.shape == (10,1,224,224))
    


    
