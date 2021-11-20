import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.bn_dw = nn.BatchNorm2d(nin, eps=1e-5)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.bn_pw = nn.BatchNorm2d(nout, eps=1e-5)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = F.relu(x, inplace=True) 
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = F.relu(x, inplace=True)
        return x

class depthwise_separable_conv_inv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv_inv, self).__init__()
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.bn_pw = nn.BatchNorm2d(nout, eps=1e-5)
        self.depthwise = nn.Conv2d(nout, nout, kernel_size=3, padding=1, groups=nout)
        self.bn_dw = nn.BatchNorm2d(nout, eps=1e-5)


    def forward(self, x):
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = F.relu(x, inplace=True)
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = F.relu(x, inplace=True) 

        return x


class depthwise_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=3, padding=1, groups=nin)
        self.bn_dw = nn.BatchNorm2d(nout, eps=1e-5)
        #self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        #self.bn_pw = nn.BatchNorm2d(nout, eps=1e-5)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = F.relu(x, inplace=True) 
        #x = self.pointwise(x)
        #x = self.bn_pw(x)
        #x = F.relu(x, inplace=True)
        return x

class depthwise_separable_conv_stride2(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv_stride2, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, stride=2, groups=nin)
        self.bn_dw = nn.BatchNorm2d(nin, eps=1e-5)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.bn_pw = nn.BatchNorm2d(nout, eps=1e-5)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = F.relu(x, inplace=True) 
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = F.relu(x, inplace=True)
        return x

class depthwise_conv_stride2(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_conv_stride2, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, stride=2, groups=nin)
        self.bn_dw = nn.BatchNorm2d(nin, eps=1e-5)
        #self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        #self.bn_pw = nn.BatchNorm2d(nout, eps=1e-5)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = F.relu(x, inplace=True) 
        #x = self.pointwise(x)
        #x = self.bn_pw(x)
        #x = F.relu(x, inplace=True)
        return x

class Inception(nn.Module):

  def __init__(self):
    super(Inception, self).__init__()
    self.branch1x1 = BasicConv2d(128, 43, kernel_size=1, padding=0)
    self.branch1x1_2 = BasicConv2d(128, 43, kernel_size=1, padding=0)
    self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3 = BasicConv2d(24, 42, kernel_size=3, padding=1)
    #self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
    #self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=5, padding=2)
    #self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
  
  def forward(self, x):
    branch1x1 = self.branch1x1(x)
    
    #branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch1x1_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch1x1_2 = self.branch1x1_2(branch1x1_pool)
    
    branch3x3_reduce = self.branch3x3_reduce(x)
    branch3x3 = self.branch3x3(branch3x3_reduce)
    
    #branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
    #branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
    #branch3x3_3 = self.branch3x3_3(branch3x3_2)
    
    #outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
    outputs = [branch1x1, branch1x1_2, branch3x3]
    return torch.cat(outputs, 1)



class stem(nn.Module):
    def __init__(self, in_c, out_c):
        super(stem, self).__init__()
        self.c3x3 = BasicConv2d(in_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.c3x3(x)
        #return channel_shuffle(x, 1)
        return x
      
class ghost_1(nn.Module):
    def __init__(self, in_c, li_c):
    #def __init__(self):
        super(ghost_1, self).__init__()

        #self.s1 = BasicConv2d(64, 128, kernel_size=3, padding=1)        
        self.s2 = BasicConv2d(in_c, li_c, kernel_size=1, padding=0)
        self.dw = depthwise_conv(li_c, li_c)         
        
        #self.s3 = BasicConv2d(li_c*2, out_c, kernel_size=1, padding=0)

        #self.att = Att(64)
      
    def forward(self, x):

        #x = self.s1(x)        
        xp = self.s2(x)
        x = self.dw(xp)

        x = torch.cat((xp, x), 1)
        #x = self.s3(x) 

        return x 

class ghost_2(nn.Module):
    def __init__(self, in_c, li_c, out_c):
    #def __init__(self):
        super(ghost_2, self).__init__()

        #self.s1 = BasicConv2d(64, 128, kernel_size=3, padding=1)        
        self.s2 = BasicConv2d(in_c, li_c, kernel_size=1, padding=0)
        self.dw = depthwise_conv(li_c, li_c)         
        
        self.s3 = BasicConv2d(li_c*2, out_c, kernel_size=1, padding=0)

        #self.att = Att(64)
      
    def forward(self, x):

        #x = self.s1(x)        
        xp = self.s2(x)
        x = self.dw(xp)

        x = torch.cat((xp, x), 1)
        x = self.s3(x) 

        return x      


class eff(nn.Module):

    #def __init__(self, in_c, out_c):
    def __init__(self):
        super(eff, self).__init__()
     
        self.g1 = ghost()
        self.g2 = ghost()         
        #self.c = BasicConv2d(128, 128, kernel_size=1, padding=0)
        
        #self.att = Att(64)
      
    def forward(self, x):
        #split = torch.chunk(x, 2, dim=1)
        xp = x
        x = self.g1(x)
        x = self.g2(x)

        #y = self.att(split[0])

        x = x + xp 
        #x = self.c(x)
        return x  


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            #nn.Linear(channel, channel // reduction, bias=False),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel // reduction, eps=1e-5),
            nn.ReLU(inplace=True),
            #nn.Linear(channel // reduction, channel, bias=False),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.conv(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Face(nn.Module):

  def __init__(self, phase, size, num_classes):
    super(Face, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.size = size
    
    self.conv1 = BasicConv2d(3, 16, kernel_size=5, stride=4, padding=1)

    #self.conv_dw_1_1 = depthwise_separable_conv_inv(16, 8)
    self.conv3 = BasicConv2d(16, 32, kernel_size=3, stride=2, padding=1)

    #self.conv_dw_1_2 = depthwise_separable_conv_inv(32, 16)
    self.conv4a = BasicConv2d(32, 16, kernel_size=1) 
    self.conv4b = BasicConv2d(16, 32, kernel_size=3, stride=1, padding=1)    
    #self.conv4c = BasicConv2d(32, 32, kernel_size=1)
    self.conv5 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)

    #self.conv_dw_1_3 = depthwise_separable_conv_inv(64, 32)
    self.conv6a = BasicConv2d(64, 32, kernel_size=1) 
    self.conv6b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
    #self.conv6c = BasicConv2d(64, 64, kernel_size=1)
    self.se1 = SELayer(64)

    self.conv7 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)  
    self.stem_1 = stem(128, 128)
    self.stem_2 = stem(128, 64)
    self.stem_3 = stem(64, 128)
    self.stem_4 = stem(128, 64)    
    self.stem_5 = stem(64, 128)      
    #self.stem_6 = stem(64, 128)
    self.se2 = SELayer(128)

    self.conv_dw_1 = ghost_1(128,128)
    self.conv_dw_std2_1 = depthwise_conv_stride2(256, 256)
    self.se3 = SELayer(256)

    self.conv_dw_2 = ghost_1(256,128)
    self.conv_dw_std2_2 = depthwise_conv_stride2(256, 256)
    self.se4 = SELayer(256)
    
    self.loc, self.conf = self.multibox(self.num_classes)
    
    if self.phase == 'test':
        self.softmax = nn.Softmax(dim=-1)

    if self.phase == 'train':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

  def multibox(self, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers += [nn.Conv2d(64, 2 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(64, 2 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(128, 4 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(128, 4 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 2 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 2 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 2 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 2 * num_classes, kernel_size=3, padding=1)]
    return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)
    
  def forward(self, x):
  
    sources = list()
    loc = list()
    conf = list()
    detection_dimension = list()

    x = self.conv1(x)

    #x = self.conv_dw_1_1(x)
    x = self.conv3(x)

    #x = self.conv_dw_1_2(x)
    x = self.conv4a(x)
    x = self.conv4b(x)
    #x = self.conv4c(x)
    x = self.conv5(x)

    #x = self.conv_dw_1_3(x)
    x = self.conv6a(x)
    x_1 = self.conv6b(x)
    x = self.se1(x_1)
    detection_dimension.append(x.shape[2:])
    sources.append(x)

    #x = self.conv6c(x)
    x = self.conv7(x_1)

    x = self.stem_1(x)
    x = self.stem_2(x)
    x = self.stem_3(x)
    x = self.stem_4(x)
    x_2 = self.stem_5(x)
    #x = self.stem_6(x)   
    x = self.se2(x_2)
    detection_dimension.append(x.shape[2:])
    sources.append(x)

    x = self.conv_dw_1(x_2)    
    x_3 = self.conv_dw_std2_1(x)
    x = self.se3(x_3)
    detection_dimension.append(x.shape[2:])
    sources.append(x)

    x = self.conv_dw_2(x_3)    
    x = self.conv_dw_std2_2(x)
    x = self.se4(x)
    detection_dimension.append(x.shape[2:])
    sources.append(x)
    
    detection_dimension = torch.tensor(detection_dimension, device=x.device)

    for (x, l, c) in zip(sources, self.loc, self.conf):
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    if self.phase == "test":
      output = (loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(-1, self.num_classes)),
                detection_dimension)
    else:
      output = (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                detection_dimension)
  
    return output
