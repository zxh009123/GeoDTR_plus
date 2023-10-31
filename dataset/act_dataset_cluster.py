import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import scipy.io as sio
from .trans_utils import RandomPosterize
import torchvision.transforms as transforms
from .augmentations import HFlip, Rotate
import random
import numpy as np
# __all__ = ['TrainDataloader','TestDataloader']

if os.path.exists('/mnt/CVACT/ACT_data.mat'):
    ACT_DATA_MAT_PATH = '/mnt/CVACT/ACT_data.mat'
elif os.path.exists('scratch/CVACT/ACT_data.mat'):
    ACT_DATA_MAT_PATH = 'scratch/CVACT/ACT_data.mat'
elif os.path.exists('./Matlab/ACT_data.mat'):
    ACT_DATA_MAT_PATH = './Matlab/ACT_data.mat'
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
    def __init__(self, data_dir, geometric_aug='strong', sematic_aug='strong', is_polar=True, mode='train'):
        self.mode = mode
        if mode == 'train':
            folder_name = 'ANU_data_small'
        elif mode == 'val' or mode == 'test':
            folder_name = 'ANU_data_test'
        else:
            raise RuntimeError(f'no such mode: {mode}')
        self.img_root = data_dir

        self.is_polar = is_polar

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
        transforms_sat.append(transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)))

        transforms_street.append(transforms.ToTensor())
        transforms_street.append(transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)))

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
        while(True):
            local_idx = idx + itmp
            try:
                ground = Image.open(self.List[local_idx][0])
                ground = self.transforms_grd(ground)

                satellite = Image.open(self.List[local_idx][1])
                satellite = self.transforms_sat(satellite)

                utm = np.array([self.List[local_idx][2], self.List[local_idx][3]])

                break
            except:
                itmp += 1

        #geometric transform
        if self.geometric_aug == "strong":
            hflip = random.randint(0,1)
            if hflip == 1:
                satellite, ground = HFlip(satellite, ground)
            else:
                pass

            orientation = random.choice(["left", "right", "back", "none"])
            if orientation == "none":
                pass
            else:
                satellite, ground = Rotate(satellite, ground, orientation, self.is_polar)

        elif self.geometric_aug == "weak":
            hflip = random.randint(0,1)
            if hflip == 1:
                satellite, ground = HFlip(satellite, ground)
            else:
                pass

        elif self.geometric_aug == "none":
            pass

        else:
            raise RuntimeError(f"geometric augmentation {self.geometric_aug} is not implemented")

        # return x, y
        return {'satellite':satellite, 'ground':ground, 'utm':utm}


    def __len__(self):
        return len(self.List)



if __name__ == "__main__":
    transforms_sat = [transforms.Resize((122, 671)),
                        transforms.ColorJitter(0.1, 0.1, 0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                        ]
    transforms_grd = [transforms.Resize((122, 671)),
                        transforms.ColorJitter(0.1, 0.1, 0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                        ]
    dataloader = DataLoader(ACTDataset(data_dir='/mnt/CVACT/', geometric_aug='strong', sematic_aug='strong', is_polar=True, mode='train'),batch_size=4, shuffle=True, num_workers=8)

    i = 0
    for k in dataloader:
        i += 1
        print("---batch---")
        print("satellite : ", k['satellite'].shape)
        print("grd : ", k['ground'].shape)
        print("grd : ", k['utm'])
        print("-----------")
        if i > 2:
            break
    