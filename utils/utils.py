import torch
import numpy as np
import os
import math
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import json

def SaveDescriptors(sat_des, grd_des, epoch, save_name, last=True):
    np_sat_desc = sat_des.detach().cpu().numpy()
    np_grd_desc = grd_des.detach().cpu().numpy()

    # save to last epoch folder
    if last == True:
        np.save(os.path.join(save_name, "epoch_last", "sat_des.npy"), np_sat_desc)
        np.save(os.path.join(save_name, "epoch_last", "grd_des.npy"), np_grd_desc)
    else: 
        #Save to a specific folder called descriptors. 
        #In there a bunch of subfolder (epoch_X) to save sat and grd desciptors
        descriptors_path = os.path.join(save_name, "descriptors", f"epoch_{epoch}")

        if not os.path.exists(descriptors_path):
            os.makedirs(descriptors_path)

        np.save(os.path.join(descriptors_path, "sat_des.npy"), np_sat_desc)
        np.save(os.path.join(descriptors_path, "grd_des.npy"), np_grd_desc)

def ReadConfig(path):
    all_files = os.listdir(path)
    config_file =  list(filter(lambda x: x.endswith('parameter.json'), all_files))
    with open(os.path.join(path, config_file[0]), 'r') as f:
        p = json.load(f)
        return p

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def softMarginTripletLoss(sate_vecs, pano_vecs, loss_weight=10.0, hard_topk_ratio=1.0):
    dists = 2.0 - 2.0 * torch.matmul(sate_vecs, pano_vecs.permute(1, 0))  # Pairwise matches within batch
    pos_dists = torch.diag(dists)
    N = len(pos_dists)
    diag_ids = np.arange(N)
    
    num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if hard_topk_ratio < 1.0 else N * (N - 1)

    # Match from satellite to street pano
    triplet_dist_s2p = pos_dists.unsqueeze(1) - dists
    loss_s2p = torch.log(1.0 + torch.exp(loss_weight * triplet_dist_s2p))
    loss_s2p[diag_ids, diag_ids] = 0.0  # Ignore diagnal losses

    if hard_topk_ratio < 1.0:  # Hard negative mining
        loss_s2p = loss_s2p.view(-1)
        loss_s2p, s2p_ids = torch.topk(loss_s2p, num_hard_triplets)
    loss_s2p = loss_s2p.sum() / float(num_hard_triplets)

    # Match from street pano to satellite
    triplet_dist_p2s = pos_dists - dists
    loss_p2s = torch.log(1.0 + torch.exp(loss_weight * triplet_dist_p2s))
    loss_p2s[diag_ids, diag_ids] = 0.0  # Ignore diagnal losses

    if hard_topk_ratio < 1.0:  # Hard negative mining
        loss_p2s = loss_p2s.view(-1)
        loss_p2s, p2s_ids = torch.topk(loss_p2s, num_hard_triplets)
    loss_p2s = loss_p2s.sum() / float(num_hard_triplets)
    # Total loss
    loss = (loss_s2p + loss_p2s) / 2.0
    return loss

def softMarginTripletLossACT(sate_vecs, pano_vecs, utm, UTMthres=625,loss_weight=10.0, hard_topk_ratio=1.0):

    in_batch_dis = torch.zeros(utm.shape[0], utm.shape[0]).to(sate_vecs.get_device())
    for k in range(utm.shape[0]):
        for j in range(utm.shape[0]):
            in_batch_dis[k, j] = (utm[k,0] - utm[j,0])*(utm[k,0] - utm[j,0]) + (utm[k, 1] - utm[j, 1])*(utm[k, 1] - utm[j, 1])

    dists = 2.0 - 2.0 * torch.matmul(sate_vecs, pano_vecs.permute(1, 0))  # Pairwise matches within batch
    pos_dists = torch.diag(dists)
    N = len(pos_dists)
    diag_ids = np.arange(N)
    useful_pairs = torch.ge(in_batch_dis[:,:], UTMthres)
    useful_pairs = useful_pairs.float()
    pair_n = useful_pairs.sum()
    # num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if hard_topk_ratio < 1.0 else N * (N - 1)
    num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if int(hard_topk_ratio * (N * (N - 1))) < pair_n else pair_n

    # Match from satellite to street pano
    triplet_dist_s2p = (pos_dists.unsqueeze(1) - dists) * useful_pairs
    loss_s2p = torch.log(1.0 + torch.exp(loss_weight * triplet_dist_s2p))
    loss_s2p[diag_ids, diag_ids] = 0.0  # Ignore diagnal losses

    if num_hard_triplets != pair_n:  # Hard negative mining
        loss_s2p = loss_s2p.view(-1)
        loss_s2p, s2p_ids = torch.topk(loss_s2p, num_hard_triplets)
    loss_s2p = loss_s2p.sum() / float(num_hard_triplets)

    # Match from street pano to satellite
    triplet_dist_p2s = (pos_dists - dists) * useful_pairs
    loss_p2s = torch.log(1.0 + torch.exp(loss_weight * triplet_dist_p2s))
    loss_p2s[diag_ids, diag_ids] = 0.0  # Ignore diagnal losses

    if num_hard_triplets != pair_n:  # Hard negative mining
        loss_p2s = loss_p2s.view(-1)
        loss_p2s, p2s_ids = torch.topk(loss_p2s, num_hard_triplets)
    loss_p2s = loss_p2s.sum() / float(num_hard_triplets)
    # Total loss
    loss = (loss_s2p + loss_p2s) / 2.0
    return loss

def CFLoss(vecs, hat_vecs, loss_weight=5.0):

    dists = 2.0 * torch.matmul(vecs, hat_vecs.permute(1, 0)) - 2.0
    cf_dists = torch.diag(dists)
    loss = torch.log(1.0 + torch.exp(loss_weight * cf_dists))

    loss = loss.sum() / vecs.shape[0]

    return loss



def save_model(savePath, model, optimizer, scheduler, epoch, last=True):
    if last == True:
        save_folder_name = "epoch_last"
        model_name = "epoch_last.pth"
    else:
        save_folder_name = f"epoch_{epoch}"
        model_name = f'epoch_{epoch}.pth'
    modelFolder = os.path.join(savePath, save_folder_name)
    if os.path.isdir(modelFolder):
        pass
    else:
        os.makedirs(modelFolder)
    # torch.save(model.state_dict(), os.path.join(modelFolder, f'trans_{epoch}.pth'))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(modelFolder, model_name))

# do not use unstable
# def ValidateOne(distArray, topK):
#     acc = 0.0
#     dataAmount = 0.0
#     for i in range(distArray.shape[0]):
#         groundTruths = distArray[i,i]
#         pred = torch.sum(distArray[:,i] < groundTruths)
#         if pred < topK:
#             acc += 1.0
#         dataAmount += 1.0
#     return acc / dataAmount

# def ValidateAll(streetFeatures, satelliteFeatures):
#     distArray = 2.0 - 2.0 * torch.matmul(satelliteFeatures, torch.transpose(streetFeatures, 0, 1))
#     topOnePercent = int(distArray.shape[0] * 0.01) + 1
#     valAcc = torch.zeros((1, topOnePercent), dtype=torch.float)
#     for i in range(topOnePercent):
#         valAcc[0,i] = ValidateOne(distArray, i)
    
#     return valAcc

def validatenp(sat_global_descriptor, grd_global_descriptor):
    dist_array = 2.0 - 2.0 * np.matmul(sat_global_descriptor, grd_global_descriptor.T)
    
    top1_percent = int(dist_array.shape[0] * 0.01) + 1
    val_accuracy = np.zeros((1, top1_percent))
    for i in range(top1_percent):
        # val_accuracy[0, i] = validate(dist_array, i)
        accuracy = 0.0
        data_amount = 0.0
        for k in range(dist_array.shape[0]):
            gt_dist = dist_array[k,k]
            prediction = np.sum(dist_array[:, k] < gt_dist)
            if prediction < i:
                accuracy += 1.0
            data_amount += 1.0
        accuracy /= data_amount
        val_accuracy[0, i] = accuracy
    return val_accuracy

def distancestat(sat_global_descriptor, grd_global_descriptor, compute_rrate=True, fname=None):
    dist_array = 2.0 - 2.0 * np.matmul(sat_global_descriptor, grd_global_descriptor.T)

    if compute_rrate:
        top1_percent = int(dist_array.shape[0] * 0.01) + 1
        val_accuracy = np.zeros((2, 4)) # 1, 5, 10, 1%
        for i, num in enumerate([1, 5, 10, top1_percent]):
            # val_accuracy[0, i] = validate(dist_array, i)
            col_acc = 0.0
            row_acc = 0.0
            data_amount = float(dist_array.shape[0])
            for k in range(dist_array.shape[0]):
                gt_dist = dist_array[k,k]
                col_pred = np.sum(dist_array[:, k] < gt_dist)
                if col_pred < num:
                    col_acc += 1.0
                row_pred = np.sum(dist_array[k, :] < gt_dist)
                if row_pred < num:
                    row_acc += 1.0
            # print(i, num, col_pred, row_pred)
            col_acc /= data_amount
            row_acc /= data_amount
            val_accuracy[0, i] = col_acc
            val_accuracy[1, i] = row_acc
    
    if fname is None:
        assert compute_rrate
        return val_accuracy
    
    # dist among: grd-sat, grd-sat_false, sat-sat_false, grd_false-sat
    col_correct_top1 = [] #correct top1; specific col (grd as ref)
    col_wrong_top1 = [] #wrong top1; specific col (grd as ref)
    row_correct_top1 = [] #correct top1; specific row (sat as ref)
    row_wrong_top1 = [] #wrong top1; specific row (sat as ref)
    for k in range(dist_array.shape[0]):
        d_sat_grd = dist_array[k, k]

        col_min_id = np.argmin(dist_array[:, k])
        row_min_id = np.argmin(dist_array[k, :])
        dist_array[k,k] += 9999.

        # grd as ref
        if col_min_id == k:
            min_id = np.argmin(dist_array[:, k])
            d_satF_grd = dist_array[min_id, k]
            d_sat_satF = 2.0 - 2.0 * np.dot(sat_global_descriptor[min_id], sat_global_descriptor[k])
            col_correct_top1.append(
                [float(k), float(min_id), d_sat_grd, d_satF_grd, d_sat_satF]
            ) # idx, second_min_idx, d_sat_grd, d_satF_grd, d_sat_satF
        else:
            d_satF_grd = dist_array[col_min_id, k]
            d_sat_satF = 2.0 - 2.0 * np.dot(sat_global_descriptor[col_min_id], sat_global_descriptor[k])
            col_wrong_top1.append(
                [float(k), float(col_min_id), d_sat_grd, d_satF_grd, d_sat_satF]
            ) # idx, min_idx, d_sat_grd, d_satF_grd, d_sat_satF

        # sat as ref
        if row_min_id == k:
            min_id = np.argmin(dist_array[k, :])
            d_sat_grdF = dist_array[k, min_id]
            d_grd_grdF = 2.0 - 2.0 * np.dot(grd_global_descriptor[k], grd_global_descriptor[min_id])
            row_correct_top1.append(
                [float(k), float(min_id), d_sat_grd, d_sat_grdF, d_grd_grdF]
            ) # idx, second_min_idx, d_sat_grd, d_sat_grdF, d_grd_grdF
        else:
            d_sat_grdF = dist_array[k, row_min_id]
            d_grd_grdF = 2.0 - 2.0 * np.dot(grd_global_descriptor[k], grd_global_descriptor[row_min_id])
            row_wrong_top1.append(
                [float(k), float(col_min_id), d_sat_grd, d_sat_grdF, d_grd_grdF]
            ) # idx, min_idx, d_sat_grd, d_sat_grdF, d_grd_grdF

    # fname = "./distance_dist.npz"
    np.savez_compressed(
        fname,
        col_correct_top1 = np.array(col_correct_top1),
        col_wrong_top1 = np.array(col_wrong_top1),
        row_correct_top1 = np.array(row_correct_top1),
        row_wrong_top1 = np.array(row_wrong_top1)
    )
    print(f"distance dist saved to {fname}", flush=True)

    if compute_rrate:
        return val_accuracy
    else:
        return

def validateVIGOR(query_features, reference_features, query_labels, topk=[1,5,10]):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])
    # for CVUSA, CVACT
    if N < 80000:
        # query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
        # reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
        similarity = np.matmul(query_features, reference_features.transpose())

        for i in range(N):
            ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)

            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.
    else:
        # split the queries if the matrix is too large, e.g. VIGOR
        assert N % 4 == 0
        N_4 = N // 4
        for split in range(4):
            query_features_i = query_features[(split*N_4):((split+1)*N_4), :]
            query_labels_i = query_labels[(split*N_4):((split+1)*N_4)]
            # query_features_norm = np.sqrt(np.sum(query_features_i ** 2, axis=1, keepdims=True))
            # reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
            similarity = np.matmul(query_features_i, reference_features.transpose())
            for i in range(query_features_i.shape[0]):
                ranking = np.sum((similarity[i, :] > similarity[i, query_labels_i[i]])*1.)
                for j, k in enumerate(topk):
                    if ranking < k:
                        results[j] += 1.

    results = (results / N) * 100.
    # print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}'.format(results[0], results[1], results[2], results[-1]))
    return results


if __name__ == "__main__":
    a = torch.rand(10, 4096)
    b = torch.rand(10, 4096)

    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)

    # print(softMarginTripletLossMX(a, b))
    print(IntraLoss(a, b, 0.4))