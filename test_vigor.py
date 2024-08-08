import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.vigor_dataset import VIGOR
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import argparse
import logging
import calendar
import time
import json

from models.GeoDTR import GeoDTR

from utils.utils import WarmupCosineSchedule, ReadConfig

args_do_not_overide = ['verbose', 'resume_from']
backbone_lists = ['resnet', 'convnext']
torch.autograd.set_detect_anomaly(True)

def validate_top(grd_descriptor, sat_descriptor, dataloader):
    accuracy_top1p = 0.0
    accuracy_top1 = 0.0
    accuracy_top5 = 0.0
    accuracy_top10 = 0.0
    accuracy_hit = 0.0

    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))

    top1_percent = int(dist_array.shape[1] * 0.01) + 1
    top1 = 1
    top5 = 5
    top10 = 10
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, dataloader.dataset.test_label[i][0]]
        prediction = np.sum(dist_array[i, :] < gt_dist)

        dist_temp = np.ones(dist_array[i, :].shape[0])
        dist_temp[dataloader.dataset.test_label[i][1:]] = 0
        prediction_hit = np.sum((dist_array[i, :] < gt_dist) * dist_temp)

        if prediction < top1_percent:
            accuracy_top1p += 1.0
        if prediction < top1:
            accuracy_top1 += 1.0
        if prediction < top5:
            accuracy_top5 += 1.0
        if prediction < top10:
            accuracy_top10 += 1.0
        if prediction_hit < top1:
            accuracy_hit += 1.0
        data_amount += 1.0
    accuracy_top1p /= data_amount
    accuracy_top1 /= data_amount
    accuracy_top5 /= data_amount
    accuracy_top10 /= data_amount
    accuracy_hit /= data_amount
    return [accuracy_top1, accuracy_top5, accuracy_top10, accuracy_top1p, accuracy_hit]

def GetBestModel(path):
    all_files = os.listdir(path)
    config_files =  list(filter(lambda x: x.startswith('epoch_'), all_files))
    config_files = sorted(list(map(lambda x: int(x.split("_")[1]), config_files)), reverse=True)
    best_epoch = config_files[0]
    return os.path.join('epoch_'+str(best_epoch), 'epoch_'+str(best_epoch)+'.pth')

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='../scratch', help='dir to the dataset')
    parser.add_argument("--model_path", type=str, default='None', help='resume from folder')
    parser.add_argument('--verbose', default=False, action='store_true')

    opt = parser.parse_args()
    opt.model = 'GeoDTR'

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    config = ReadConfig(opt.model_path)
    for k,v in config.items():
        if k in args_do_not_overide:
            continue
        setattr(opt, k, v)
    
    print(opt)

    batch_size = opt.batch_size
    number_of_epoch = opt.epochs + 1
    learning_rate = opt.lr
    number_descriptors = opt.descriptors
    gamma = opt.gamma
    is_cf = opt.cf
    same_area = not opt.cross_area

    hyper_parameter_dict = vars(opt)
    
    logger.info("Configuration:")
    for k, v in hyper_parameter_dict.items():
        print(f"{k} : {v}")
    

    SATELLITE_IMG_WIDTH = 320
    SATELLITE_IMG_HEIGHT = 320
    polar_transformation = False
    SAT_DESC_HEIGHT = 20
    SAT_DESC_WIDTH = 20

    print("SATELLITE_IMG_WIDTH:",SATELLITE_IMG_WIDTH)
    print("SATELLITE_IMG_HEIGHT:",SATELLITE_IMG_HEIGHT)

    STREET_IMG_WIDTH = 640
    STREET_IMG_HEIGHT = 320

    if opt.backbone == "convnext":
        GRD_DESC_HEIGHT = 20
        GRD_DESC_WIDTH = 40
        # feature length for each descriptor
        DESC_LENGTH = 384
    if opt.backbone == "resnet":
        GRD_DESC_HEIGHT = 20
        GRD_DESC_WIDTH = 40
        # feature length for each descriptor
        DESC_LENGTH = 512

    # generate time stamp
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    train_dataset = VIGOR(mode="train",root = opt.data_dir, same_area=same_area, print_bool=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    val_reference = VIGOR(mode="test_reference",root = opt.data_dir, same_area=same_area, print_bool=False)
    val_reference_loader = DataLoader(val_reference, batch_size=batch_size, shuffle=False, num_workers=8)

    val_query = VIGOR(mode="test_query",root = opt.data_dir, same_area=same_area, print_bool=False)
    val_query_loader = DataLoader(val_query, batch_size=batch_size, shuffle=False, num_workers=8)

    model = GeoDTR(descriptors=number_descriptors,
                    tr_heads=opt.TR_heads,
                    tr_layers=opt.TR_layers,
                    dropout = opt.dropout,
                    d_hid=opt.TR_dim,
                    is_polar=polar_transformation,
                    backbone=opt.backbone,
                    dataset = "VIGOR",
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


    #Evaluation
    sat_global_descriptor = np.zeros([len(val_reference), embedding_dims])
    grd_global_descriptor = np.zeros([len(val_query), embedding_dims])
    query_labels = np.zeros([len(val_query)])

    model.eval()
    with torch.no_grad():
        for (images, indexes, _) in tqdm(val_reference_loader, disable = opt.verbose):

            sat = images.to(device)
            grd = torch.randn(images.shape[0], 3, STREET_IMG_HEIGHT, STREET_IMG_WIDTH).to(device)

            sat_global, grd_global, sat_desc , grd_desc = model(sat, grd, is_cf=False)

            sat_global_descriptor[indexes.numpy(), :] = sat_global.detach().cpu().numpy()

        for (images, indexes, labels) in tqdm(val_query_loader, disable = opt.verbose):
            sat = torch.randn(images.shape[0], 3, SATELLITE_IMG_HEIGHT, SATELLITE_IMG_WIDTH).to(device)
            grd = images.to(device)

            sat_global, grd_global, sat_desc , grd_desc = model(sat, grd, is_cf=False)

            grd_global_descriptor[indexes.numpy(), :] = grd_global.detach().cpu().numpy()
            query_labels[indexes.numpy()] = labels.numpy()

        # valAcc = validateVIGOR(grd_global_descriptor, sat_global_descriptor, query_labels.astype(int))
        valAcc = validate_top(grd_global_descriptor, sat_global_descriptor, val_query_loader)
        logger.info("validation result")
        print(f"------------------------------------")
        try:
            # print recall value
            top1 = valAcc[0]
            print('top1', ':', valAcc[0])
            print('top5', ':', valAcc[1])
            print('top10', ':', valAcc[2])
            print('top1%', ':', valAcc[3])
            print('hit rate', ':', valAcc[4])

        except:
            print(valAcc)
