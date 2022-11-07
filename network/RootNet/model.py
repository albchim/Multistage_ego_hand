import torch
import torch.nn as nn
from torch.nn import functional as F
from ..InterHand.resnet import ResNetBackbone
from config import cfg

class RootNet(nn.Module):

    def __init__(self, inplanes=2048, outplanes=256):
        self.inplanes = inplanes
        self.outplanes = outplanes

        super(RootNet, self).__init__()
        self.deconv_layers = self._make_deconv_layer(4)
        self.xy_layer = nn.Conv2d(
            in_channels=self.outplanes,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.depth_layer = nn.Conv2d(
            in_channels=self.inplanes,
            out_channels=1, 
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers):
        layers = []
        inplanes = self.inplanes
        outplanes = self.outplanes
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=inplanes,
                    out_channels=outplanes,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False))
            layers.append(nn.BatchNorm2d(outplanes))
            layers.append(nn.ReLU(inplace=True))
            inplanes = outplanes

        return nn.Sequential(*layers)

    def forward(self, x, k_value):
        # x,y
        xy = self.deconv_layers(x)
        xy = self.xy_layer(xy)
        xy = xy.view(-1,1,cfg.output_shape[0]*cfg.output_shape[1])
        xy = F.softmax(xy,2)
        xy = xy.view(-1,1,cfg.output_shape[0],cfg.output_shape[1])

        hm_x = xy.sum(dim=(2))
        hm_y = xy.sum(dim=(3))

        coord_x = hm_x * torch.arange(cfg.output_shape[1]).float().cuda()
        coord_y = hm_y * torch.arange(cfg.output_shape[0]).float().cuda()
        
        coord_x = coord_x.sum(dim=2)/cfg.output_shape[0]
        coord_y = coord_y.sum(dim=2)/cfg.output_shape[1]

        # z
        img_feat = torch.mean(x.view(x.size(0), x.size(1), x.size(2)*x.size(3)), dim=2) # global average pooling
        img_feat = torch.unsqueeze(img_feat,2); img_feat = torch.unsqueeze(img_feat,3);
        gamma = self.depth_layer(img_feat)
        gamma = gamma.view(-1,1)
        depth = gamma * k_value.view(-1,1)

        coord = torch.cat((coord_x, coord_y, depth), dim=1)
        return coord

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.xy_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        for m in self.depth_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

def get_root_net(inplanes, outplanes):#, is_train=True):
    
    root_net = RootNet(inplanes, outplanes)
    root_net.init_weights()

    return root_net

