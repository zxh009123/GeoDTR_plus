import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset.usa_dataset import USADataset
# from dataset.act_dataset import TestDataset, TrainDataset
# if os.environ["USER"] == "xyli1905":
#     from dataset.act_dataset_cluster import ACTDataset
# else:
from dataset.act_dataset import ACTDataset
from dataset.augmentations import Reverse_Rotate_Flip
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import argparse
import logging
import calendar
import time
import json

from models.GeoDTR import GeoDTR

from utils.utils import softMarginTripletLoss,\
     CFLoss, save_model, WarmupCosineSchedule, ReadConfig,\
     validatenp, SaveDescriptors

args_do_not_overide = ['verbose', 'resume_from']
backbone_lists = ['resnet', 'convnext']
torch.autograd.set_detect_anomaly(True)

def GetBestModel(path):
    all_files = os.listdir(path)
    config_files =  list(filter(lambda x: x.startswith('epoch_'), all_files))
    config_files = sorted(list(map(lambda x: int(x.split("_")[1]), config_files)), reverse=True)
    best_epoch = config_files[0]
    return os.path.join('epoch_'+str(best_epoch), 'trans_'+str(best_epoch)+'.pth')

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--save_suffix", type=str, default='_aug_strong', help='name of the model at the end')
    parser.add_argument("--data_dir", type=str, default='../scratch', help='dir to the dataset')
    parser.add_argument('--dataset', default='CVUSA', choices=['CVUSA', 'CVACT'], help='which dataset to use') 
    parser.add_argument("--descriptors", type=int, default=8, help='number of descriptors')
    parser.add_argument("--TR_heads", type=int, default=4, help='number of heads in Transformer')
    parser.add_argument("--TR_layers", type=int, default=2, help='number of layers in Transformer')
    parser.add_argument("--TR_dim", type=int, default=512, help='dim of FFD in Transformer')
    parser.add_argument("--dropout", type=float, default=0.3, help='dropout in Transformer')
    parser.add_argument("--gamma", type=float, default=10.0, help='value for gamma')
    parser.add_argument("--weight_decay", type=float, default=0.03, help='weight decay value for optimizer')
    parser.add_argument('--cf', default=False, action='store_true', help='counter factual loss')
    parser.add_argument('--verbose', default=True, action='store_false', help='turn on progress bar')
    parser.add_argument('--no_polar', default=False, action='store_true', help='turn off polar transformation')
    parser.add_argument("--resume_from", type=str, default='None', help='resume from folder')
    parser.add_argument('--geo_aug', default='strong', choices=['strong', 'weak', 'none', 'same'], help='geometric augmentation strength') 
    parser.add_argument('--sem_aug', default='strong', choices=['strong', 'weak', 'none', 'same'], help='semantic augmentation strength')
    parser.add_argument('--mutual', default=False, action='store_true', help='no mutual learning')
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone selection')

    parser.add_argument('--normalize', default=False, action='store_true', help='whether normalize descriptors')
    parser.add_argument('--orthogonalize', default=False, action='store_true', help='whether orthogonalize descriptors')
    parser.add_argument('--bottleneck', default=False, action='store_true', help='whether use bottleneck for descriptors')


    opt = parser.parse_args()
    opt.model = 'GeoDTR'

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # if given resume from directory read configs and overwrite
    if opt.resume_from != 'None':
        if os.path.isdir(opt.resume_from):
            config = ReadConfig(opt.resume_from)
            for k,v in config.items():
                if k in args_do_not_overide:
                    continue
                setattr(opt, k, v)
        else: # if directory invalid
            raise RuntimeError(f'Cannot find resume model directory{opt.resume_from}')
    
    if opt.backbone not in backbone_lists:
        raise RuntimeError(f'{opt.backbone} is no supported!')


    batch_size = opt.batch_size
    number_of_epoch = opt.epochs + 1
    learning_rate = opt.lr
    number_descriptors = opt.descriptors
    gamma = opt.gamma
    is_cf = opt.cf

    hyper_parameter_dict = vars(opt)
    
    logger.info("Configuration:")
    for k, v in hyper_parameter_dict.items():
        print(f"{k} : {v}")

    
    if opt.no_polar:
        SATELLITE_IMG_WIDTH = 256
        SATELLITE_IMG_HEIGHT = 256
        polar_transformation = False
        SAT_DESC_HEIGHT = 16
        SAT_DESC_WIDTH = 16
    else:
        SATELLITE_IMG_WIDTH = 671
        SATELLITE_IMG_HEIGHT = 122
        polar_transformation = True
        if opt.backbone == "convnext":
            SAT_DESC_HEIGHT = 7
            SAT_DESC_WIDTH = 41
        if opt.backbone == "resnet":
            SAT_DESC_HEIGHT = 8
            SAT_DESC_WIDTH = 42
    print("SATELLITE_IMG_WIDTH:",SATELLITE_IMG_WIDTH)
    print("SATELLITE_IMG_HEIGHT:",SATELLITE_IMG_HEIGHT)

    STREET_IMG_WIDTH = 671
    STREET_IMG_HEIGHT = 122

    if opt.backbone == "convnext":
        GRD_DESC_HEIGHT = 7
        GRD_DESC_WIDTH = 41
        # feature length for each descriptor
        DESC_LENGTH = 384
    if opt.backbone == "resnet":
        GRD_DESC_HEIGHT = 8
        GRD_DESC_WIDTH = 42
        # feature length for each descriptor
        DESC_LENGTH = 512

    # generate time stamp
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    if opt.resume_from == 'None':
        save_name = f"{ts}_{opt.model}_{opt.dataset}_{opt.backbone}_{polar_transformation}_{opt.save_suffix}"
        print("save_name : ", save_name)
        if not os.path.exists(save_name):
            os.makedirs(save_name)
        else:
            logger.info("Note! Saving path existed !")

        with open(os.path.join(save_name,f"{ts}_parameter.json"), "w") as outfile:
            json.dump(hyper_parameter_dict, outfile, indent=4)
    else:
        logger.info(f'loading model from : {opt.resume_from}')
        save_name = opt.resume_from

    writer = SummaryWriter(save_name)  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_workers = 16
    if opt.dataset == "CVUSA":

        dataloader = DataLoader(USADataset(data_dir = opt.data_dir, \
                                geometric_aug=opt.geo_aug, \
                                sematic_aug=opt.sem_aug, mode='train', \
                                is_polar=polar_transformation, \
                                is_mutual=opt.mutual),\
                                batch_size=batch_size, shuffle=True, num_workers=num_workers)

        validateloader = DataLoader(USADataset(data_dir = opt.data_dir, \
                                    geometric_aug='none', \
                                    sematic_aug='none', \
                                    mode='val', \
                                    is_polar=polar_transformation, \
                                    is_mutual=False),\
            batch_size=batch_size, shuffle=False, num_workers=num_workers)


    elif opt.dataset == "CVACT":
        #train
        train_dataset = ACTDataset(data_dir = opt.data_dir, \
                        geometric_aug=opt.geo_aug, \
                        sematic_aug=opt.sem_aug, \
                        is_polar=polar_transformation, \
                        mode='train', \
                        is_mutual=opt.mutual)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        #val
        validate_dataset = ACTDataset(data_dir = opt.data_dir, \
                            geometric_aug='none', \
                            sematic_aug='none', \
                            is_polar=polar_transformation, \
                            mode='val', \
                            is_mutual=False)
        validateloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = GeoDTR(descriptors=number_descriptors,
                    tr_heads=opt.TR_heads,
                    tr_layers=opt.TR_layers,
                    dropout = opt.dropout,
                    d_hid=opt.TR_dim,
                    is_polar=polar_transformation,
                    backbone=opt.backbone,
                    dataset = opt.dataset,
                    normalize = opt.normalize,
                    orthogonalize = opt.orthogonalize, 
                    bottleneck = opt.bottleneck)
    embedding_dims = number_descriptors * DESC_LENGTH

    model = nn.DataParallel(model)
    model.to(device)

    # for name, param in model.named_parameters():
    #     if 'backbone' in name:
    #         print(name)
    #         param.requires_grad = False

    #set optimizer and lr scheduler

    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=opt.weight_decay, eps=1e-6)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=opt.weight_decay, eps=1e-6)
    lrSchedule = WarmupCosineSchedule(optimizer, 5, opt.epochs)


    start_epoch = 1
    if opt.resume_from != "None":
        logger.info("loading checkpoint...")
        model_path = os.path.join(opt.resume_from, "epoch_last", "epoch_last.pth")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lrSchedule.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']

    # Start training
    logger.info("start training...")
    best_epoch = {'acc':0, 'epoch':0}
    for epoch in range(start_epoch, number_of_epoch):

        logger.info(f"start epoch {epoch}")
        epoch_triplet_loss = 0
        if is_cf:
            epoch_cf_loss = 0
        if opt.mutual:
            epoch_mutual_loss = 0

        model.train() # set model to train
        for batch in tqdm(dataloader, disable = opt.verbose):
            
            optimizer.zero_grad()

            if opt.mutual:
                sat_first = batch['satellite_first']
                grd_first = batch['ground_first']
                sat_second = batch['satellite_second']
                grd_second = batch['ground_second']
                batch_perturb = batch["perturb"]

                # print(sat_first.shape)
                # print(grd_first.shape)
                # print(sat_second.shape)
                # print(grd_second.shape)

                sat_all = torch.cat((sat_first, sat_second), 0)
                grd_all = torch.cat((grd_first, grd_second), 0)

                # print(sat_all.shape)
                # print(grd_all.shape)
                # print(perturb[0])
                # print(perturb[1])
                # print([int(perturb[0][0]), perturb[1][0]])
                # print([int(perturb[0][1]), perturb[1][1]])
                # print([int(perturb[0][2]), perturb[1][2]])
            else:
                sat_all = batch["satellite"]
                grd_all = batch["ground"]

            if is_cf:
                sat_global, grd_global, fake_sat_global, fake_grd_global, sat_desc, grd_desc = model(sat_all, grd_all, is_cf)
            else:
                sat_global, grd_global, sat_desc, grd_desc = model(sat_all, grd_all, is_cf)

            triplet_loss = softMarginTripletLoss(sate_vecs=sat_global, pano_vecs=grd_global, loss_weight=gamma)

            loss = triplet_loss

            epoch_triplet_loss += loss.item()
            
            if is_cf:# calculate CF loss
                CFLoss_sat= CFLoss(sat_global, fake_sat_global)
                CFLoss_grd = CFLoss(grd_global, fake_grd_global)
                CFLoss_total = (CFLoss_sat + CFLoss_grd) / 2.0
                loss += CFLoss_total
                epoch_cf_loss += CFLoss_total.item()

            # mutual loss
            if opt.mutual:
                pass
                
                # grd_desc = grd_desc.reshape((grd_desc.shape[0], GRD_DESC_HEIGHT, GRD_DESC_WIDTH, opt.descriptors))
                # sat_desc = sat_desc.reshape((sat_desc.shape[0], SAT_DESC_HEIGHT, SAT_DESC_WIDTH, opt.descriptors))

                # # split into first and second half
                # grd_desc_first, grd_desc_second = torch.tensor_split(grd_desc, 2)
                # sat_desc_first, sat_desc_second = torch.tensor_split(sat_desc, 2)

                # #reverse second half
                # perturb = [[int(batch_perturb[0][i]), batch_perturb[1][i]] for i in range(len(batch_perturb[1]))]
                # reversed_sat_desc_second, reversed_grd_desc_second = Reverse_Rotate_Flip(sat_desc_second, grd_desc_second, perturb, not opt.no_polar)

                # # print(second_sat.shape)
                # # print(second_grd.shape)

                # sat_mutual_loss = torch.nn.functional.smooth_l1_loss(reversed_sat_desc_second, sat_desc_first)
                # grd_mutual_loss = torch.nn.functional.smooth_l1_loss(reversed_grd_desc_second, grd_desc_first)
                # mutual_loss = (sat_mutual_loss + grd_mutual_loss) / 2.0
                # loss += mutual_loss

                # epoch_mutual_loss += mutual_loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        # adjust lr
        lrSchedule.step()

        logger.info(f"Summary of epoch {epoch}")
        print(f"===============================================")
        print("---------loss---------")
        current_triplet_loss = float(epoch_triplet_loss) / float(len(dataloader))
        print(f"Epoch {epoch} TRI_Loss: {current_triplet_loss}")
        writer.add_scalar('triplet_loss', current_triplet_loss, epoch)
        if is_cf:
            current_cf_loss = float(epoch_cf_loss) / float(len(dataloader))
            print(f"Epoch {epoch} CF_Loss: {current_cf_loss}")
            writer.add_scalar('cf_loss', current_cf_loss, epoch)

        if opt.mutual:
            current_mutual_loss = float(epoch_mutual_loss) / float(len(dataloader))
            print(f"Epoch {epoch} MUTUAL_Loss: {current_mutual_loss}")
            writer.add_scalar('mutual_loss', current_mutual_loss, epoch)

        print("----------------------")


        sat_global_descriptor = np.zeros([8884, embedding_dims])
        grd_global_descriptor = np.zeros([8884, embedding_dims])
        val_i = 0

        model.eval()
        with torch.no_grad():
            for batch in tqdm(validateloader, disable = opt.verbose):
                sat = batch['satellite'].to(device)
                grd = batch['ground'].to(device)

                sat_global, grd_global, sat_desc , grd_desc = model(sat, grd, is_cf=False)

                sat_global_descriptor[val_i: val_i + sat_global.shape[0], :] = sat_global.detach().cpu().numpy()
                grd_global_descriptor[val_i: val_i + grd_global.shape[0], :] = grd_global.detach().cpu().numpy()

                val_i += sat_global.shape[0]

            valAcc = validatenp(sat_global_descriptor, grd_global_descriptor)
            logger.info("validation result")
            print(f"------------------------------------")
            try:
                #print recall value
                top1 = valAcc[0, 1]
                print('top1', ':', valAcc[0, 1] * 100.0)
                print('top5', ':', valAcc[0, 5] * 100.0)
                print('top10', ':', valAcc[0, 10] * 100.0)
                print('top1%', ':', valAcc[0, -1] * 100.0)
                # write to tensorboard
                writer.add_scalars('validation recall@k',{
                    'top 1':valAcc[0, 1],
                    'top 5':valAcc[0, 5],
                    'top 10':valAcc[0, 10],
                    'top 1%':valAcc[0, -1]
                }, epoch)
            except:
                print(valAcc)
            # save best model
            if top1 > best_epoch['acc']:
                best_epoch['acc'] = top1
                best_epoch['epoch'] = epoch
                save_model(save_name, model, optimizer, lrSchedule, epoch, last=False)
            # For each 10th epoch save the descriptors
            if epoch % 10 == 0:
                SaveDescriptors(sat_desc, grd_desc, epoch, save_name, last=False)
            # save last model
            save_model(save_name, model, optimizer, lrSchedule, epoch, last=True)
            # save descriptor to last model
            SaveDescriptors(sat_desc, grd_desc, epoch, save_name, last=True)
            print(f"=================================================")

    # get the best model and recall
    print("best acc : ", best_epoch['acc'])
    print("best epoch : ", best_epoch['epoch'])
