import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.usa_dataset import USADataset
from dataset.act_dataset import ACTDataset
from dataset.vigor_dataset import VIGOR
from torchvision.transforms.functional import to_pil_image
import json
import os
import numpy as np
import argparse
from utils.utils import ReadConfig
from models.GeoDTR import GeoDTR
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import tqdm


args_do_not_overide = ['data_dir', 'verbose', 'dataset']

sat_gradients = None
sat_activations = None
grd_gradients = None
grd_activations = None

class GeoDTRVis(GeoDTR):
    def __init__(self,descriptors = 16, 
                       tr_heads=8, 
                       tr_layers=6, 
                       dropout = 0.3, 
                       d_hid=2048, 
                       is_polar=True, 
                       backbone='convnext', 
                       dataset = "CVUSA",
                       normalize = False, 
                       orthogonalize = False, 
                       bottleneck = False):
        super(GeoDTRVis,self).__init__(descriptors = descriptors, 
                                        tr_heads=tr_heads, 
                                        tr_layers=tr_layers, 
                                        dropout = dropout, 
                                        d_hid=d_hid, 
                                        is_polar=is_polar, 
                                        backbone=backbone, 
                                        dataset = dataset,
                                        normalize = normalize, 
                                        orthogonalize = orthogonalize, 
                                        bottleneck = bottleneck)
        
    def forward(self, sat, grd):
        b = sat.shape[0]

        sat_local = self.backbone_sat(sat)
        grd_local = self.backbone_grd(grd)
        
        sat_sa = self.GLE_sat(sat_local)
        grd_sa = self.GLE_grd(grd_local) # B, H*W, D

        sat_x = sat_local.view(b, sat_local.shape[1], sat_local.shape[2]*sat_local.shape[3]) # B, C, H*W
        grd_x = grd_local.view(b, grd_local.shape[1], grd_local.shape[2]*grd_local.shape[3])
        
        sat_global = torch.matmul(sat_x, sat_sa).view(b,-1)
        grd_global = torch.matmul(grd_x, grd_sa).view(b,-1)

        sat_global = F.normalize(sat_global, p=2, dim=1)
        grd_global = F.normalize(grd_global, p=2, dim=1)

        return sat_global, grd_global, sat_sa, grd_sa, sat_local, grd_local

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def GetBestModel(path):
    all_files = os.listdir(path)
    if "epoch_last" in all_files:
        all_files.remove("epoch_last")
    config_files =  list(filter(lambda x: x.startswith('epoch_'), all_files))
    config_files = sorted(list(map(lambda x: int(x.split("_")[1]), config_files)), reverse=True)
    best_epoch = config_files[0]
    return os.path.join('epoch_'+str(best_epoch), 'epoch_'+str(best_epoch)+'.pth')

def grd_backward_hook(module, grad_input, grad_output):
    global grd_gradients # refers to the variable in the global scope
    print('Backward hook running...')
    grd_gradients = grad_output
    # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    print(f'Gradients size: {grd_gradients[0].size()}') 
    # We need the 0 index because the tensor containing the gradients comes
    # inside a one element tuple.

def grd_forward_hook(module, args, output):
    global grd_activations # refers to the variable in the global scope
    print('Forward hook running...')
    grd_activations = output
    # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    print(f'Activations size: {grd_activations.size()}')

def sat_backward_hook(module, grad_input, grad_output):
    global sat_gradients # refers to the variable in the global scope
    print('Backward hook running...')
    sat_gradients = grad_output
    # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    print(f'Gradients size: {sat_gradients[0].size()}') 
    # We need the 0 index because the tensor containing the gradients comes
    # inside a one element tuple.

def sat_forward_hook(module, args, output):
    global sat_activations # refers to the variable in the global scope
    print('Forward hook running...')
    sat_activations = output
    # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    print(f'Activations size: {sat_activations.size()}')

def visualization(img, gradients, activations, name, index):
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    img = unorm(img)
    img = img.cpu()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # weight the channels by corresponding gradients
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    original_heatmap = torch.mean(activations, dim=1).squeeze(0)
    # relu on top of the heatmap
    original_heatmap = F.relu(original_heatmap)
    # normalize the heatmap
    original_heatmap /= torch.max(original_heatmap)
    # interpolate to image size
    heatmap = original_heatmap.unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(heatmap, size = (img.shape[2], img.shape[3]), mode = "bicubic", align_corners=True)
    heatmap = heatmap.squeeze(0).squeeze(0)
    heatmap = heatmap.detach().cpu().numpy()


    img = np.array(to_pil_image(img[0]))
    heatmap = heatmap * 255.0
    heatmap = heatmap.astype(np.uint8)
    heatmap = plt.cm.jet(heatmap)
    alpha = 0.3

    blended = ((heatmap[:, :, :3] * 255.0) * alpha + img * (1 - alpha)).astype(np.uint8)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(img)
    axs[0, 1].imshow(original_heatmap.detach().cpu().numpy(), alpha = 1., interpolation = 'gaussian', cmap = plt.cm.jet)
    axs[1, 0].imshow(blended, alpha = 1., interpolation = 'gaussian', cmap = plt.cm.jet)
    axs[1, 1].remove()

    plt.savefig(f"{index}_{name}.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--data_dir", type=str, default='../scratch/', help='dir to the dataset')
    parser.add_argument('--dataset', default='CVUSA', choices=['CVUSA', 'CVACT', 'VIGOR'], help='choose between CVUSA or CVACT')
    parser.add_argument("--descriptors", type=int, default=16, help='number of descriptors')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument("--model", type=str, help='model')
    parser.add_argument('--model_path', type=str, help='path to model weights')
    parser.add_argument('--no_polar', default=False, action='store_true', help='turn off polar transformation')
    parser.add_argument("--TR_heads", type=int, default=8, help='number of heads in Transformer')
    parser.add_argument("--TR_layers", type=int, default=6, help='number of layers in Transformer')
    parser.add_argument("--TR_dim", type=int, default=2048, help='dim of FFD in Transformer')
    parser.add_argument("--dropout", type=float, default=0.2, help='dropout in Transformer')
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone selection')

    opt = parser.parse_args()

    config = ReadConfig(opt.model_path)
    for k,v in config.items():
        if k in args_do_not_overide:
            continue
        setattr(opt, k, v)
    
    print(opt)

    batch_size = opt.batch_size
    number_descriptors = opt.descriptors

    if opt.no_polar:
        SATELLITE_IMG_WIDTH = 256
        SATELLITE_IMG_HEIGHT = 256
        polar_transformation = False
    else:
        SATELLITE_IMG_WIDTH = 671
        SATELLITE_IMG_HEIGHT = 122
        polar_transformation = True
    print("SATELLITE_IMG_WIDTH:",SATELLITE_IMG_WIDTH)
    print("SATELLITE_IMG_HEIGHT:",SATELLITE_IMG_HEIGHT)

    STREET_IMG_WIDTH = 671
    STREET_IMG_HEIGHT = 122

    if opt.backbone == "convnext":
        # feature length for each descriptor
        DESC_LENGTH = 384
    if opt.backbone == "resnet":
        # feature length for each descriptor
        DESC_LENGTH = 512
        
    NUM_SAMPLES = 200


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if opt.dataset == 'CVACT':
        data_path = os.path.join(opt.data_dir, 'CVACT')
        dataset = ACTDataset(data_dir = data_path, geometric_aug='none', sematic_aug='none', is_polar=polar_transformation, mode='val', is_mutual=False)
        validateloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    if opt.dataset == 'CVUSA':
        data_path = os.path.join(opt.data_dir, 'CVUSA', 'dataset')
        dataset = USADataset(data_dir = data_path, geometric_aug='none', sematic_aug='none', mode='val', is_polar=polar_transformation, is_mutual=False)
        validateloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    if opt.dataset == 'VIGOR':
        data_path = os.path.join(opt.data_dir, 'VIGOR')
        dataset = VIGOR(mode="test_all",root = data_path, same_area=True, print_bool=False)
        validateloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    
    print("number of test samples : ", len(dataset))

    model = GeoDTRVis(descriptors=number_descriptors, \
                    tr_heads=opt.TR_heads, \
                    tr_layers=opt.TR_layers, \
                    dropout = opt.dropout, \
                    d_hid=opt.TR_dim, \
                    is_polar=polar_transformation, \
                    backbone=opt.backbone, \
                    dataset = opt.dataset,
                    normalize = opt.normalize,
                    orthogonalize = opt.orthogonalize, 
                    bottleneck = opt.bottleneck)
    embedding_dims = number_descriptors * DESC_LENGTH
    
    model = nn.DataParallel(model)
    model.to(device)

    best_model = GetBestModel(opt.model_path)
    best_model = os.path.join(opt.model_path, best_model)
    print("loading model : ", best_model)
    model.load_state_dict(torch.load(best_model)['model_state_dict'])
    model.eval()
    all_losses = {}
    for orientation in ['none', 'flip', 'right', 'left', 'back']:
        index = 0
        average_losses = []
        previous_sat = None
        unmatched_losses = []
        for batch in tqdm.tqdm(validateloader):
            if opt.dataset == 'CVACT' or opt.dataset == 'CVUSA':
                sat = batch['satellite'].to(device)       
                grd = batch['ground'].to(device)
            else:
                sat = batch[1].to(device)
                grd = batch[0].to(device)
            
            # rotate sat
            grd = grd.squeeze(0)
            height, width = grd.shape[1], grd.shape[2]

            if orientation == 'flip':
                grd = torch.flip(grd, [2])
            elif orientation == 'left':
                left_grd = grd[:, :, 0:int(math.ceil(width * 0.75))]
                right_grd = grd[:, :, int(math.ceil(width * 0.75)):]
                grd = torch.cat([right_grd, left_grd], dim=2)
            elif orientation == 'right':
                left_grd = grd[:, :, 0:int(math.floor(width * 0.25))]
                right_grd = grd[:, :, int(math.floor(width * 0.25)):]
                grd = torch.cat([right_grd, left_grd], dim=2)
            elif orientation == 'back':
                left_grd = grd[:, :, 0:int(width * 0.5)]
                right_grd = grd[:, :, int(width * 0.5):]
                grd = torch.cat([right_grd, left_grd], dim=2)
            else:
                pass
            grd = grd.unsqueeze(0)

            sat_global, grd_global, _ , _, sat_loc, grd_loc = model(sat, grd)

            distance = 2.0 - 2.0 * torch.matmul(sat_global, grd_global.permute(1, 0))
            pos_dists = torch.diag(distance)
            average_losses.append(pos_dists.item())

            if previous_sat is not None:
                unmatched_distance = 2.0 - 2.0 * torch.matmul(previous_sat, grd_global.permute(1, 0))
                prev_pos_dists = torch.diag(unmatched_distance)
                unmatched_losses.append(prev_pos_dists.item())
            previous_sat = sat_global
            
            index += 1
            if index == NUM_SAMPLES:
                break
        # print(orientation)
        # average_loss = float(average_loss / index)
        # print(average_loss)
        # unmatched_loss = float(unmatched_loss / index)
        # print(unmatched_loss)
        all_losses[orientation] = {'average_loss':average_losses,
                                   'unmatched_loss':unmatched_losses}
    with open(f'CHSG.json', 'w') as j:
        j.write(json.dumps(all_losses))
        
    for k,v in all_losses.items():
        print(k)
        print('average_loss', sum(v['average_loss'])/len(v['average_loss']))
        print('unmatched_loss', sum(v['unmatched_loss'])/len(v['unmatched_loss']))

