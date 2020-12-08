import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, 
            groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).cuda()
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True).cuda() if bn else None
        self.relu = nn.ReLU(inplace=True).cuda() if relu else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x.cuda()

class TUM(nn.Module):
    def __init__(self, first_level=True, input_planes=128, is_smooth=True, side_channel=512, scales=6):
        super(TUM, self).__init__()
        self.is_smooth = is_smooth
        self.side_channel = side_channel
        self.input_planes = input_planes
        self.planes = 2 * self.input_planes
        self.first_level = first_level
        self.scales = scales
        self.in1 = input_planes + side_channel if not first_level else input_planes

        self.layers = nn.Sequential()
        self.layers.add_module('{}'.format(len(self.layers)), BasicConv(self.in1, self.planes, 3, 2, 1))
        for i in range(self.scales-2):
            if not i == self.scales - 3:
                self.layers.add_module(
                        '{}'.format(len(self.layers)),
                        BasicConv(self.planes, self.planes, 3, 2, 1)
                        )
            else:
                self.layers.add_module(
                        '{}'.format(len(self.layers)),
                        BasicConv(self.planes, self.planes, 3, 1, 0)
                        )
        self.toplayer = nn.Sequential(BasicConv(self.planes, self.planes, 1, 1, 0))
        
        self.latlayer = nn.Sequential()
        for i in range(self.scales-2):
            self.latlayer.add_module(
                    '{}'.format(len(self.latlayer)),
                    BasicConv(self.planes, self.planes, 3, 1, 1)
                    )
        self.latlayer.add_module('{}'.format(len(self.latlayer)),BasicConv(self.in1, self.planes, 3, 1, 1))

        if self.is_smooth:
            smooth = list()
            for i in range(self.scales-1):
                smooth.append(
                        BasicConv(self.planes, self.planes, 1, 1, 0)
                        )
            self.smooth = nn.Sequential(*smooth)

    def _upsample_add(self, x, y, fuse_type='interp'):
        _,_,H,W = y.size()
        if fuse_type=='interp':
            return F.interpolate(x, size=(H,W), mode='nearest') + y
        else:
            raise NotImplementedError
            

    def forward(self, x, y):
        if not self.first_level:
            x = torch.cat([x,y],1)
        conved_feat = [x]
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            conved_feat.append(x)
        
        deconved_feat = [self.toplayer[0](conved_feat[-1])]
        for i in range(len(self.latlayer)):
            deconved_feat.append(
                    self._upsample_add(
                        deconved_feat[i], self.latlayer[i](conved_feat[len(self.layers)-1-i])
                        )
                    )
        if self.is_smooth:
            smoothed_feat = [deconved_feat[0]]
            for i in range(len(self.smooth)):
                smoothed_feat.append(
                        self.smooth[i](deconved_feat[i+1])
                        )
            return smoothed_feat
        return deconved_feat


class SFAM(nn.Module):

    def __init__(self, planes, num_levels, num_scales, compress_ratio=16):
        super(SFAM, self).__init__()
        self.planes = planes
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.compress_ratio = compress_ratio

        self.fc1 = nn.ModuleList([nn.Conv2d(self.planes*self.num_levels,
                                                 self.planes*self.num_levels // 16,
                                                 1, 1, 0)] * self.num_scales).cuda()
        self.relu = nn.ReLU(inplace=True).cuda()
        self.fc2 = nn.ModuleList([nn.Conv2d(self.planes*self.num_levels // 16,
                                                 self.planes*self.num_levels,
                                                 1, 1, 0)] * self.num_scales).cuda()
        self.fc3 = nn.ModuleList([nn.Conv2d(self.planes*self.num_levels,
                                                 256,1, 1, 0)] * self.num_scales).cuda()#I changed
        self.sigmoid = nn.Sigmoid().cuda()
        self.avgpool = nn.AdaptiveAvgPool2d(1).cuda()

    def forward(self, x):
        
        attention_feat = []
        for i, _mf in enumerate(x):
            _tmp_f = self.avgpool(_mf)
            _tmp_f = self.fc1[i](_tmp_f)
            _tmp_f = self.relu(_tmp_f)
            _tmp_f = self.fc2[i](_tmp_f)
            _tmp_f = self.fc3[i](_tmp_f)#I changed
            _tmp_f = self.sigmoid(_tmp_f)
            _mf = self.fc3[i](_mf)
            attention_feat.append(_mf*_tmp_f)
        return attention_feat

