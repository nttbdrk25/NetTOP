import torch.nn
import torch.nn.init
import torch.nn.functional as F
from .common import conv1x1_block, conv3x3_block, conv3x3_dw_block, Classifier, conv1x1_group_block
from .SE_Attention import *
class com_bottleneck(torch.nn.Module):
    def __init__(self, in_channels,out_channels, reduction_ratio,stride=1):
        super().__init__()
        self.CBP = nn.Sequential(
    conv1x1_block(in_channels=in_channels, out_channels=(in_channels)//reduction_ratio),
    conv3x3_dw_block(channels=(in_channels)//reduction_ratio, stride=stride),
    conv1x1_block(in_channels=(in_channels)//reduction_ratio, out_channels=out_channels))
    def forward(self, x):
        return self.CBP(x)
class ConvPlane(torch.nn.Module):
    def __init__(self, in_channels, stride=1,groups=2,Temporal=True):
        super().__init__()        
        self.Temporal = Temporal
        if self.Temporal==False:            
            self.stride = stride            
            #Pw_xy
            self.pw1 = conv1x1_group_block(in_channels=in_channels,
                             out_channels=in_channels, use_bn=False, groups=groups,activation=None)            
        else:            
            self.stride = 1            
        self.dw = conv3x3_dw_block(channels=in_channels, stride=self.stride)
    def forward(self, x):
        if self.Temporal == False:
            x = self.pw1(x)
            x = self.dw(x)
        else:
            x = self.dw(x)
        return x
class Conv3P(torch.nn.Module):    
    def __init__(self, in_channels, stride=1,Temporal=True,groups=2,in_size=(224, 224)):
        super().__init__()
        self.in_size = in_size
        self.stride = stride        
        self.spatial = ConvPlane(in_channels,stride=stride,Temporal=False,groups=groups)
        if in_size == (32, 32):#32x32 for CIFAR10/100
            self.ConvPlane_32 = ConvPlane(32,stride=stride,groups=groups)
            self.ConvPlane_16 = ConvPlane(16,stride=stride,groups=groups)
            self.ConvPlane_8 = ConvPlane(8,stride=stride,groups=groups)            
        if in_size == (224, 224):# (224, 224) for ImageNet, Stanford Dogs            
            self.ConvPlane_112 = ConvPlane(112,stride=stride,groups=groups)            
            self.ConvPlane_56 = ConvPlane(56,stride=stride,groups=groups)            
            self.ConvPlane_28 = ConvPlane(28,stride=stride,groups=groups)            
            self.ConvPlane_14 = ConvPlane(14,stride=stride,groups=groups)
            
        if stride == 2:            
            self.pw_temporal = com_bottleneck(in_channels=in_channels, out_channels=in_channels,reduction_ratio=16,stride=stride)
    def forward(self, x):        
        batch_size,num_channel,h,w = x.size()
        xy = self.spatial(x)        
        xz = torch.transpose(x,1,2)
        yz = torch.transpose(x,1,3)
        
        if self.in_size == (32, 32):
            if h == 32 and w == 32:
                xz = self.ConvPlane_32(xz)
                yz = self.ConvPlane_32(yz)                
            if h == 16 and w == 16:
                xz = self.ConvPlane_16(xz)
                yz = self.ConvPlane_16(yz)                
            if h == 8 and w == 8:                
                xz = self.ConvPlane_8(xz)
                yz = self.ConvPlane_8(yz)

        if self.in_size == (224, 224):
            if h == 112 and w == 112:
                xz = self.ConvPlane_112(xz)
                yz = self.ConvPlane_112(yz)
            if h == 56 and w == 56:                
                xz = self.ConvPlane_56(xz)
                yz = self.ConvPlane_56(yz)
            if h == 28 and w == 28:                
                xz = self.ConvPlane_28(xz)
                yz = self.ConvPlane_28(yz)
            if h == 14 and w == 14:                
                xz = self.ConvPlane_14(xz)
                yz = self.ConvPlane_14(yz)
        
        xz = torch.transpose(xz,1,2)
        yz = torch.transpose(yz,1,3)
        if self.stride == 2:
            xz = self.pw_temporal(xz)
            yz = self.pw_temporal(yz)        
        x = F.relu(xy*(F.sigmoid(xz*yz)))
        return x    
class TOP_Block(torch.nn.Module):
    """    
    """
    def __init__(self, in_channels, out_channels, stride, groups=2, in_size=(224, 224)):
        super().__init__()
        #PwOut
        self.pw2 = conv1x1_group_block(in_channels=in_channels,
                                        out_channels=out_channels,groups=groups)
        self.Conv3P = Conv3P(in_channels, stride=stride,Temporal=True,groups=groups,in_size=in_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride == 2 or self.in_channels != self.out_channels:
            self.pointRes = conv1x1_group_block(in_channels=in_channels,
                        out_channels=out_channels,stride=stride,groups=groups)
        self.SE = SE(out_channels, 16)        
        self.stride = stride

    def forward(self, x):
        residual = x        
        x = self.Conv3P(x)        
        x = self.pw2(x)        
        x = self.SE(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            x = x + residual
        else:       
            residual = self.pointRes(residual)        
            x = x + residual        
        return x

class NetTOP(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True,
                 groups=2):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                stage.add_module("unit{}".format(unit_id + 1), TOP_Block(in_channels=in_channels, out_channels=unit_channels, stride=stride,in_size=in_size,groups=groups))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)                
        self.final_conv_channels = 1024
        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        in_channels = self.final_conv_channels

        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():            
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)                        
            elif isinstance(module, torch.nn.Linear):                
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):                
                module.weight.data.fill_(1)
                module.bias.data.zero_()            
        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
def build_NetTOP(num_classes, cifar=False,groups=2):
    init_conv_channels = 32
    channels = [[64], [64, 128,128], [256,256,256], [512, 512, 512], [512]]

    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        strides = [1, 2, 2, 2, 2]

    return NetTOP(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       channels=channels,
                       strides=strides,
                       in_size=in_size,groups=groups)
