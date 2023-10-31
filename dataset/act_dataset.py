import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from torchvision.transforms import transforms
import torchvision
from PIL import Image
import scipy.io as sio
from .trans_utils import RandomPosterize
import torchvision.transforms as transforms
from .augmentations import HFlip, Rotate
import random
import numpy as np
import time
# __all__ = ['TrainDataloader','TestDataloader']

if os.path.exists('/mnt/CVACT/ACT_data.mat'):
    ACT_DATA_MAT_PATH = '/mnt/CVACT/ACT_data.mat'
elif os.path.exists('scratch/CVACT/ACT_data.mat'):
    ACT_DATA_MAT_PATH = 'scratch/CVACT/ACT_data.mat'
elif os.path.exists('./Matlab/ACT_data.mat'):
    ACT_DATA_MAT_PATH = './Matlab/ACT_data.mat'
elif os.path.exists('../scratch/CVACT/ACT_data.mat'):
    ACT_DATA_MAT_PATH = '../scratch/CVACT/ACT_data.mat'
else:
    raise RuntimeError('ACT_data mat does not exist')
# try:
#     if os.environ["SERVER_NAME"] == "gpu02" or os.environ["SERVER_NAME"] == "gpu03" or os.environ["SERVER_NAME"] == "cluster":
#         ACT_DATA_MAT_PATH = './ACT_data.mat'
# except:
#     pass


# class ActDataset(Dataset):
#     def __init__(self, data_dir, transforms_sat, transforms_grd, is_polar=True, mode='train'):
#         self.mode = mode
#         if mode == 'train':
#             folder_name = 'ANU_data_small'
#         elif mode == 'val' or mode == 'test':
#             folder_name = 'ANU_data_test'
#         else:
#             raise RuntimeError(f'no such mode: {mode}')


#         self.img_root = data_dir
#         self.transform_sat = transforms.Compose(transforms_sat)
#         self.transform_grd = transforms.Compose(transforms_grd)

#         self.allDataList = ACT_DATA_MAT_PATH

#         __cur_allid = 0  # for training
#         id_alllist = []
#         id_idx_alllist = []

#         # load the mat
#         anuData = sio.loadmat(self.allDataList)

#         idx = 0
#         for i in range(0,len(anuData['panoIds'])):
            
#             grd_id_align = os.path.join(self.img_root, folder_name, 'streetview_processed', anuData['panoIds'][i] + '_grdView.png')
#             if is_polar:
#                 sat_id_ori = os.path.join(self.img_root, folder_name, 'polarmap', anuData['panoIds'][i] + '_satView_polish.png')
#             else:
#                 sat_id_ori = os.path.join(self.img_root, folder_name, 'satview_polish', anuData['panoIds'][i] + '_satView_polish.jpg')
#             id_alllist.append([ grd_id_align, sat_id_ori])
#             id_idx_alllist.append(idx)
#             idx += 1

#         all_data_size = len(id_alllist)
#         print('InputData::__init__: load', self.allDataList, ' data_size =', all_data_size)

#         if mode == 'val':
#             inds = anuData['valSet']['valInd'][0][0] - 1
#         elif mode == 'test':
#             inds = anuData['valSetAll']['valInd'][0][0] - 1
#         elif mode == 'train':
#             inds = anuData['trainSet']['trainInd'][0][0] - 1
#         Num = len(inds)
#         print('Number of samples:' ,Num)
#         self.List = []
#         self.IdList = []

#         for k in range(Num):
#             self.List.append(id_alllist[inds[k][0]])
#             self.IdList.append(k)


#     def __getitem__(self, idx):
#         itmp = 0
#         while(True):
#             local_idx = idx + itmp
#             try:
#                 x = Image.open(self.List[local_idx][0])
#                 x = self.transform_grd(x)

#                 y = Image.open(self.List[local_idx][1])
#                 y = self.transform_sat(y)

#                 break
#             except:
#                 itmp += 1

#         # return x, y
#         return {'satellite':y, 'ground':x}


#     def __len__(self):
#         return len(self.List)


class ACTDataset(Dataset):
    def __init__(self, data_dir, geometric_aug='strong', sematic_aug='strong', is_polar=True, mode='train', is_mutual=True):
        self.mode = mode
        if mode == 'train':
            folder_name = 'ANU_data_small'
        elif mode == 'val' or mode == 'test':
            folder_name = 'ANU_data_test'
        else:
            raise RuntimeError(f'no such mode: {mode}')
        self.img_root = data_dir

        self.is_polar = is_polar
        self.is_mutual = is_mutual

        STREET_IMG_WIDTH = 671
        STREET_IMG_HEIGHT = 122

        if not is_polar:
            SATELLITE_IMG_WIDTH = 256
            SATELLITE_IMG_HEIGHT = 256
        else:
            SATELLITE_IMG_WIDTH = 671
            SATELLITE_IMG_HEIGHT = 122

        transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH))]
        transforms_sat = [transforms.Resize((SATELLITE_IMG_HEIGHT, SATELLITE_IMG_WIDTH))]

        if sematic_aug == 'strong':
            transforms_sat.append(transforms.ColorJitter(0.3, 0.3, 0.3))
            transforms_street.append(transforms.ColorJitter(0.3, 0.3, 0.3))

            transforms_sat.append(transforms.RandomGrayscale(p=0.2))
            transforms_street.append(transforms.RandomGrayscale(p=0.2))

            # transforms_sat.append(transforms.RandomInvert(p=0.2))
            # transforms_street.append(transforms.RandomInvert(p=0.2))

            try:
                transforms_sat.append(transforms.RandomPosterize(p=0.2, bits=4))
                transforms_street.append(transforms.RandomPosterize(p=0.2, bits=4))
            except:
                transforms_sat.append(RandomPosterize(p=0.2, bits=4))
                transforms_street.append(RandomPosterize(p=0.2, bits=4))

            transforms_sat.append(transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 5)))
            transforms_street.append(transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.1, 5)))

        elif sematic_aug == 'weak':
            transforms_sat.append(transforms.ColorJitter(0.1, 0.1, 0.1))
            transforms_street.append(transforms.ColorJitter(0.1, 0.1, 0.1))

            transforms_sat.append(transforms.RandomGrayscale(p=0.1))
            transforms_street.append(transforms.RandomGrayscale(p=0.1))

        elif sematic_aug == 'none':
            pass
        else:
            raise RuntimeError(f"sematic augmentation {sematic_aug} is not implemented")

        transforms_sat.append(transforms.ToTensor())
        transforms_sat.append(transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)))

        transforms_street.append(transforms.ToTensor())
        transforms_street.append(transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)))

        self.transforms_sat = transforms.Compose(transforms_sat)
        self.transforms_grd = transforms.Compose(transforms_street)

        self.geometric_aug = geometric_aug

        self.allDataList = ACT_DATA_MAT_PATH

        __cur_allid = 0  # for training
        id_alllist = []
        id_idx_alllist = []

        # load the mat
        anuData = sio.loadmat(self.allDataList)

        idx = 0
        for i in range(0,len(anuData['panoIds'])):
            
            grd_id_align = os.path.join(self.img_root, folder_name, 'streetview_processed', anuData['panoIds'][i] + '_grdView.png')
            if is_polar:
                sat_id_ori = os.path.join(self.img_root, folder_name, 'polarmap', anuData['panoIds'][i] + '_satView_polish.png')
            else:
                sat_id_ori = os.path.join(self.img_root, folder_name, 'satview_polish', anuData['panoIds'][i] + '_satView_polish.jpg')
            id_alllist.append([ grd_id_align, sat_id_ori, anuData['utm'][i][0], anuData['utm'][i][1]])
            id_idx_alllist.append(idx)
            idx += 1

        all_data_size = len(id_alllist)
        print('InputData::__init__: load', self.allDataList, ' data_size =', all_data_size)

        if mode == 'val':
            inds = anuData['valSet']['valInd'][0][0] - 1
        elif mode == 'test':
            inds = anuData['valSetAll']['valInd'][0][0] - 1
        elif mode == 'train':
            inds = anuData['trainSet']['trainInd'][0][0] - 1
        Num = len(inds)
        print('Number of samples:' ,Num)
        self.List = []
        self.IdList = []

        for k in range(Num):
            self.List.append(id_alllist[inds[k][0]])
            self.IdList.append(k)


    def __getitem__(self, idx):
        itmp = 0
        # while(True):
        #     local_idx = idx + itmp
        #     try:
        ground = Image.open(self.List[idx][0])
        satellite = Image.open(self.List[idx][1])

        ground_first = self.transforms_grd(ground)
        satellite_first = self.transforms_sat(satellite)

        ground_second = self.transforms_grd(ground)
        satellite_second = self.transforms_sat(satellite)

        utm = np.array([self.List[idx][2], self.List[idx][3]])

        # print(f'ground : {self.List[idx][0]}, \n satellite : {self.List[idx][1]}, \n utm : {utm}')

            #     break
            # except:
            #     itmp += 1

        #geometric transform
        if self.geometric_aug == "strong":
            hflip = random.randint(0,1)
            if hflip == 1:
                satellite_first, ground_first = HFlip(satellite_first, ground_first)
                satellite_second, ground_second = HFlip(satellite_second, ground_second)
            else:
                pass

            orientation = random.choice(["left", "right", "back", "none"])
            if orientation == "none":
                pass
            else:
                satellite_first, ground_first = Rotate(satellite_first, ground_first, orientation, self.is_polar)
                satellite_second, ground_second = Rotate(satellite_second, ground_second, orientation, self.is_polar)

        elif self.geometric_aug == "weak":
            hflip = random.randint(0,1)
            if hflip == 1:
                satellite_first, ground_first = HFlip(satellite_first, ground_first)
                satellite_second, ground_second = HFlip(satellite_second, ground_second)
            else:
                pass

        elif self.geometric_aug == "none":
            pass

        else:
            raise RuntimeError(f"geometric augmentation {self.geometric_aug} is not implemented")


        if self.is_mutual == False:
            return {'satellite':satellite_first, 'ground':ground_first, 'utm':utm}
            
        else:

            # generate new different layout
            hflip = random.randint(0,1)
            orientation = random.choice(["left", "right", "back", "none"])

            while hflip == 0 and orientation == "none":
                hflip = random.randint(0,1)
                orientation = random.choice(["left", "right", "back", "none"])

            if hflip == 1:
                satellite_second, ground_second = HFlip(satellite_second, ground_second)
            else:
                pass

            if orientation == "none":
                pass
            else:
                satellite_second, ground_second = Rotate(satellite_second, ground_second, orientation, self.is_polar)

            perturb = [hflip, orientation]

            return {'satellite_first':satellite_first, 
                    'ground_first':ground_first,
                    'satellite_second':satellite_second,
                    'ground_second':ground_second,
                    'perturb':perturb,
                    'utm':utm}


    def __len__(self):
        return len(self.List)



if __name__ == "__main__":
    dataloader = DataLoader(ACTDataset(data_dir='../scratch/CVACT/', geometric_aug='strong', sematic_aug='strong', is_polar=False, mode='train'),batch_size=4, shuffle=False, num_workers=8)

    # print(len(dataloader))
    total_time = 0
    start = time.time()
    for i,b in enumerate(dataloader):
        end = time.time()
        elapse = end - start
        print("===========================")
        print(b["ground_first"].shape)
        print(b["satellite_first"].shape)
        print(b["ground_second"].shape)
        print(b["satellite_second"].shape)
        print(b["perturb"])
        print("===========================")

        grd = b["ground_first"][0]
        sat = b["satellite_first"][0]
        mu_grd = b["ground_second"][0]
        mu_sat = b["satellite_second"][0]

        sat = sat * 0.5 + 0.5
        grd = grd * 0.5 + 0.5
        mu_sat = mu_sat * 0.5 + 0.5
        mu_grd = mu_grd * 0.5 + 0.5

        torchvision.utils.save_image(sat, "sat_f.png")
        torchvision.utils.save_image(grd, "grd_f.png")
        torchvision.utils.save_image(mu_sat, "sat_s.png")
        torchvision.utils.save_image(mu_grd, "grd_s.png")

        # if i == 2:
        #     break
        time.sleep(2)

    print(total_time / i)
    