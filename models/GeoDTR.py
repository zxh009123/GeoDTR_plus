import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import random
from thop import profile
from thop import clever_format
if os.environ["USER"] == "xyli1905":
    from models.TR import TransformerEncoder, TransformerEncoderLayer
else:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class LearnablePE(nn.Module):

    def __init__(self, d_model, dropout = 0.3, max_len = 8):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model = 256, max_len = 16, dropout=0.3):
        super().__init__()
        self.dr = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)#batch first
        # print(position.shape)
        # print(div_term.shape)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe[:x.size(0)].shape)
        x = x + self.pe[:x.size(0)]
        return self.dr(x)

# Index-awared (learnable) PE
class IAPE(nn.Module):

    def __init__(self, d_model = 256, max_len = 16, dropout=0.3):
        super().__init__()
        self.dr = torch.nn.Dropout(p=dropout)

        self.linear = torch.empty(d_model*2, max_len, d_model)
        nn.init.normal_(self.linear, mean=0.0, std=0.005)
        self.linear = torch.nn.Parameter(self.linear)
        self.embedding_parameter = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x, pos):
        em_pos = torch.einsum('bi, idj -> bdj', pos, self.linear)
        em_pos = em_pos + self.embedding_parameter
        em_pos = F.hardtanh(em_pos)
        x = x + em_pos
        return self.dr(x)

class TRModule(nn.Module):

    def __init__(self, d_model=256, descriptors = 16, nhead=8, nlayers=6, dropout = 0.3, d_hid=2048):
        super().__init__()

        self.pos_encoder = IAPE(d_model, max_len = descriptors, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, activation='gelu', batch_first=True)
        layer_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer = encoder_layers, num_layers = nlayers, norm=layer_norm)


    def forward(self, src, pos):
        src = self.pos_encoder(src, pos)
        output = self.transformer_encoder(src)
        return output

class ConvNextTiny(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.convnext_tiny(weights='IMAGENET1K_V1')
        layers = list(net.children())[:-2]
        layers = list(layers[0].children())[:-2]
        self.net = torch.nn.Sequential(*layers)
        # print(self.net)
    
    def forward(self, x):
        return self.net(x) # (B, 768, H/16, W/16)

class EfficientNetB3(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.efficientnet_b3(weights='DEFAULT')
        layers = list(net.children())[:-2]
        layers = list(layers[0].children())[:-3]

        end_conv = [torch.nn.Conv2d(136, 128, 1), torch.nn.BatchNorm2d(128), torch.nn.SiLU(inplace=True)]

        self.layers = torch.nn.Sequential(*layers, *end_conv)

    def forward(self, x):
        return self.layers(x)


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet34(weights='IMAGENET1K_V1')
        layers = list(net.children())[:3]
        layers_end = list(net.children())[4:-2]

        # 4096
        self.layers = nn.Sequential(*layers, *layers_end)

        # 1024
        # end_conv = [torch.nn.Conv2d(512, 128, 1), torch.nn.BatchNorm2d(128)]
        # self.layers = torch.nn.Sequential(*layers, *layers_end, *end_conv)

    def forward(self, x):
        return self.layers(x)

class SCGeoLayoutExtractor(nn.Module):
    def __init__(
        self, 
        max_len=60, d_model=768, descriptors=8, tr_heads=4, tr_layers=6, dropout = 0.3, d_hid=2048,
        normalize = False, orthogonalize = False, bottleneck = False
    ):
        super().__init__()
        self.normalize = normalize
        self.orthogonalize = orthogonalize
        self.bottleneck = bottleneck
        self.tr_layers = tr_layers

        self.descriptors = descriptors

        if self.tr_layers != 0:
            # Up/Down sample later
            encoder_layers = TransformerEncoderLayer(d_model, tr_heads, d_hid, dropout, activation='gelu', batch_first=True)
            layer_norm = nn.LayerNorm(d_model)
            self.transformer_encoder = TransformerEncoder(encoder_layer = encoder_layers, num_layers = tr_layers, norm=layer_norm)
            self.pe = LearnablePE(d_model, dropout = dropout, max_len = max_len)

            # Up/Down sample before
            # encoder_layers = TransformerEncoderLayer(descriptors, tr_heads, d_hid, dropout, activation='gelu', batch_first=True)
            # layer_norm = nn.LayerNorm(descriptors)
            # self.transformer_encoder = TransformerEncoder(encoder_layer = encoder_layers, num_layers = tr_layers, norm=layer_norm)
            # self.pe = LearnablePE(descriptors, dropout = dropout, max_len = max_len)

        self.pointwise = nn.Linear(d_model, descriptors)

        hid_dim = int(max_len / 2.0)
        self.w1, self.b1 = self.init_weights_(max_len, hid_dim, descriptors)
        self.w2, self.b2 = self.init_weights_(hid_dim, max_len, descriptors)

    def init_weights_(self, din, dout, dnum):
        # weight = torch.empty(din, dout, dnum)
        weight = torch.empty(dnum, din, dout)
        nn.init.normal_(weight, mean=0.0, std=0.005)
        # bias = torch.empty(1, dout, dnum)
        bias = torch.empty(1, dnum, dout)
        nn.init.constant_(bias, val=0.1)
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        return weight, bias

    def forward(self, x):
        B,C,H,W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        # Up/Down sample later
        x = x.view(B, C, H*W)
        x = x.permute(0, 2, 1) # B, H*W, C
        if self.tr_layers != 0:
            x = self.pe(x)
            x = self.transformer_encoder(x) # B, H*W, C

        x = self.pointwise(x) # B, H*W, K
        # print(self.w2.shape)
        # print(self.b2.shape)

        if self.bottleneck:
            x = torch.einsum('bdj, jdi -> bji', x, self.w1) + self.b1
            # print(x.shape)
            x = torch.einsum('bji, jid -> bjd', x, self.w2) + self.b2

            x = x.permute(0,2,1) #NOTE also B, H*W, K

        if self.normalize:
            if self.orthogonalize:
                x, _ = torch.linalg.qr(x, mode="reduced") # use q, of dim (B, H*W, K), already normalized
            else:
                x = F.normalize(x, p=2.0, dim=1) # of dim (B, H*W, K); normalized but generally not orthogonalized
        else:
            # x = F.hardtanh(x) # B, H*W, D
            x = torch.sigmoid(x) # B, H*W, D

        return x

        # Up/Down sample before
        # x = x.view(B, C, H*W)
        # x = x.permute(0, 2, 1) # B, H*W, C

        # x = self.pointwise(x) # B, H*W, K
        # # print(self.w2.shape)
        # # print(self.b2.shape)

        # x = torch.einsum('bdj, jdi -> bji', x, self.w1) + self.b1
        # # print(x.shape)
        # x = torch.einsum('bji, jid -> bjd', x, self.w2) + self.b2

        # x = x.permute(0,2,1) # B, H*W, K

        # if self.tr_layers != 0:
        #     x = self.pe(x)
        #     x = self.transformer_encoder(x) # B, H*W, K

        # x = torch.sigmoid(x) # B, H*W, K
        # return x


class GeoLayoutExtractor(nn.Module):
    def __init__(self, in_dim, descriptors=8, tr_heads=8, tr_layers=6, dropout = 0.3, d_hid=2048):
        super().__init__()

        self.tr_layers = tr_layers

        hid_dim = in_dim // 2
        self.w1, self.b1 = self.init_weights_(in_dim, hid_dim, descriptors)
        self.w2, self.b2 = self.init_weights_(hid_dim, in_dim, descriptors)
        if self.tr_layers != 0:
            self.tr_module = TRModule(d_model=hid_dim, descriptors=descriptors, nhead=tr_heads, nlayers=tr_layers, dropout=dropout,d_hid=d_hid)

    def init_weights_(self, din, dout, dnum):
        # weight = torch.empty(din, dout, dnum)
        weight = torch.empty(din, dnum, dout)
        nn.init.normal_(weight, mean=0.0, std=0.005)
        # bias = torch.empty(1, dout, dnum)
        bias = torch.empty(1, dnum, dout)
        nn.init.constant_(bias, val=0.1)
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        return weight, bias

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        channel = x.shape[1]
        mask, pos = x.max(1)

        pos_normalized = pos / channel

        mask = torch.einsum('bi, idj -> bdj', mask, self.w1) + self.b1

        if self.tr_layers != 0:
            mask = self.tr_module(mask, pos_normalized)

        mask = torch.einsum('bdj, jdi -> bdi', mask, self.w2) + self.b2
        mask = mask.permute(0,2,1) # B, H*W, D
        # lxy20230327 test normalized descriptor
        mask = F.hardtanh(mask)
        # mask = F.softmax(mask, dim=1) #now it a projection rather than a mask

        return mask



class GeoDTR(nn.Module):
    def __init__(
        self, 
        descriptors = 16, tr_heads=8, tr_layers=6, dropout = 0.3, d_hid=2048, is_polar=True, backbone='convnext', dataset = "CVUSA",
        normalize = False, orthogonalize = False, bottleneck = False
    ):
        super().__init__()
        self.normalize = normalize

        if backbone == 'convnext':
            self.backbone_grd = ConvNextTiny()
            self.backbone_sat = ConvNextTiny()
            output_channel = 384
            if dataset == "CVUSA" or dataset == "CVACT":
                grd_feature_size = 287 # 7 * 41
                if is_polar:
                    sat_feature_size = 287 # 7 * 41
                else:
                    sat_feature_size = 256 # 16 * 16
            elif dataset == "VIGOR" and is_polar == False:
                grd_feature_size = 800 # 20 * 40
                sat_feature_size = 400 # 20 * 20
            else:
                raise RuntimeError(f'The configuration {dataset} and polar:{is_polar} is not correct!')
        elif backbone == 'resnet':
            self.backbone_grd = ResNet34()
            self.backbone_sat = ResNet34()
            output_channel = 512

            if dataset == "CVUSA" or dataset == "CVACT":
                grd_feature_size = 336 # 8 * 42
                if is_polar:
                    sat_feature_size = 336 # 8 * 42
                else:
                    sat_feature_size = 256 # 16 * 16
            elif dataset == "VIGOR" and is_polar == False:
                grd_feature_size = 800 # 20 * 40
                sat_feature_size = 400 # 20 * 20
            else:
                raise RuntimeError(f'The configuration {dataset} and polar:{is_polar} is not correct!')
        else:
            raise RuntimeError(f'backbone: {backbone} is not implemented')

        self.GLE_grd = SCGeoLayoutExtractor(
                        max_len = grd_feature_size, 
                        d_model = output_channel, 
                        descriptors = descriptors, 
                        tr_heads = tr_heads, 
                        tr_layers = tr_layers, 
                        dropout = dropout, 
                        d_hid = d_hid,
                        normalize = normalize,
                        orthogonalize = orthogonalize, 
                        bottleneck = bottleneck)
        
        self.GLE_sat = SCGeoLayoutExtractor(
                        max_len = sat_feature_size, 
                        d_model = output_channel, 
                        descriptors = descriptors, 
                        tr_heads = tr_heads, 
                        tr_layers = tr_layers, 
                        dropout = dropout, 
                        d_hid = d_hid, 
                        normalize = normalize,
                        orthogonalize = orthogonalize, 
                        bottleneck = bottleneck)

        # self.GLE_grd = GeoLayoutExtractor(grd_feature_size, \
        #     descriptors=8, tr_heads=4, \
        #     tr_layers=2, dropout = 0.3,\
        #     d_hid=2048)
        # self.GLE_sat = GeoLayoutExtractor(sat_feature_size, \
        #     descriptors=8, tr_heads=4, \
        #     tr_layers=2, dropout = 0.3,\
        #     d_hid=2048)


    def forward(self, sat, grd, is_cf):
        b = sat.shape[0]

        sat_x = self.backbone_sat(sat)
        grd_x = self.backbone_grd(grd)

        # print("sat_x shape : ", sat_x.shape) # B, C, H/16, W/16
        # print("grd_x shape : ", grd_x.shape) # B, C, H/16, W/16

        sat_sa = self.GLE_sat(sat_x)
        grd_sa = self.GLE_grd(grd_x) # B, H*W, D

        sat_x = sat_x.view(b, sat_x.shape[1], sat_x.shape[2]*sat_x.shape[3]) # B, C, H*W
        grd_x = grd_x.view(b, grd_x.shape[1], grd_x.shape[2]*grd_x.shape[3])

        # print("sat_sa shape : ", sat_sa.shape)
        # print("grd_sa shape : ", grd_sa.shape)

        if is_cf:
            # fake_sat_x = sat_x.clone().detach()
            # fake_grd_x = grd_x.clone().detach()

            # upper_half_sat, lower_half_sat = torch.split(fake_sat_x, [b // 2, b // 2], dim=0)
            # upper_half_grd, lower_half_grd = torch.split(fake_grd_x, [b // 2, b // 2], dim=0)

            # fake_sat_x = torch.cat([lower_half_sat, upper_half_sat], dim=0)
            # fake_grd_x = torch.cat([lower_half_grd, upper_half_grd], dim=0)

            # sat_global = torch.matmul(sat_x, sat_sa).view(b,-1)
            # grd_global = torch.matmul(grd_x, grd_sa).view(b,-1)

            # sat_global = F.normalize(sat_global, p=2, dim=1)
            # grd_global = F.normalize(grd_global, p=2, dim=1)

            # fake_sat_global = torch.matmul(fake_sat_x, sat_sa).view(b,-1)
            # fake_grd_global = torch.matmul(fake_grd_x, grd_sa).view(b,-1)

            # fake_sat_global = F.normalize(fake_sat_global, p=2, dim=1)
            # fake_grd_global = F.normalize(fake_grd_global, p=2, dim=1)

            # return sat_global, grd_global, fake_sat_global, fake_grd_global, sat_sa, grd_sa

            # fake_sat_sa = sat_sa.clone()
            # fake_grd_sa = grd_sa.clone()
            # upper_half_sat, lower_half_sat = torch.split(fake_sat_sa, [b // 2, b // 2], dim=0)
            # upper_half_grd, lower_half_grd = torch.split(fake_grd_sa, [b // 2, b // 2], dim=0)
            # fake_sat_sa = torch.cat([lower_half_sat, upper_half_sat], dim=0)
            # fake_grd_sa = torch.cat([lower_half_grd, upper_half_grd], dim=0)

            # Old implementation for hardTanh
            # fake_sat_sa = torch.zeros_like(sat_sa).uniform_(-1.0, 1.0)
            # fake_grd_sa = torch.zeros_like(grd_sa).uniform_(-1.0, 1.0)
            
            # New implementation for Sigmoid
            fake_sat_sa = torch.zeros_like(sat_sa).uniform_(0.0, 1.0)
            fake_grd_sa = torch.zeros_like(grd_sa).uniform_(0.0, 1.0)
            if self.normalize:
                fake_sat_sa = F.normalize(fake_sat_sa, p=2, dim=1)
                fake_grd_sa = F.normalize(fake_grd_sa, p=2, dim=1)

            sat_global = torch.matmul(sat_x, sat_sa).view(b,-1)
            grd_global = torch.matmul(grd_x, grd_sa).view(b,-1)

            sat_global = F.normalize(sat_global, p=2, dim=1)
            grd_global = F.normalize(grd_global, p=2, dim=1)

            fake_sat_global = torch.matmul(sat_x, fake_sat_sa).view(b,-1) # B, C*D
            fake_grd_global = torch.matmul(grd_x, fake_grd_sa).view(b,-1) # B, C*D

            fake_sat_global = F.normalize(fake_sat_global, p=2, dim=1)
            fake_grd_global = F.normalize(fake_grd_global, p=2, dim=1)

            return sat_global, grd_global, fake_sat_global, fake_grd_global, sat_sa, grd_sa

        else:
            sat_global = torch.matmul(sat_x, sat_sa).view(b,-1)
            grd_global = torch.matmul(grd_x, grd_sa).view(b,-1)

            sat_global = F.normalize(sat_global, p=2, dim=1)
            grd_global = F.normalize(grd_global, p=2, dim=1)

            return sat_global, grd_global, sat_sa, grd_sa

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #ACT and USA style
    sat = torch.randn(1, 3, 122, 671).to(device)
    # sat = torch.randn(1, 3, 256, 256).to(device)
    grd = torch.randn(1, 3, 122, 671).to(device)

    # VIGOR style
    # sat = torch.randn(32, 3, 320, 320).to(device)
    # grd = torch.randn(32, 3, 320, 640).to(device)

    model = GeoDTR(descriptors=8, \
        tr_heads=4, \
        tr_layers=2, \
        dropout = 0.3, \
        d_hid=512, \
        is_polar=True, \
        backbone='convnext', \
        dataset='CVUSA')
    model = model.to(device)
    
    result = model(sat, grd, True)

    for i in result:
        print(i.shape)

    macs, params = profile(model, inputs=(sat, grd, False, ))
    macs, params = clever_format([macs, params], "%.3f")

    print(macs)
    print(params)

    # net = SCGeoLayoutExtractor(\
    #     max_len=768, 
    #     d_model=60, \
    #     descriptors=8, \
    #     tr_heads=4, \
    #     tr_layers=2, \
    #     dropout = 0.3, \
    #     d_hid=2048)
    # sat = torch.randn(7, 768, 3, 20)
    # x = net(sat)
    # print(x.shape)

    # m = ConvNextTiny()
    # result = m(sat)
    # print(result.shape)



