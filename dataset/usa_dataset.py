import random
import os
import time
from torch.utils.data import Dataset
from .trans_utils import RandomPosterize
from PIL import Image, ImageFile, ImageOps
from .augmentations import HFlip, Rotate
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import torchvision


class ImageDataset(Dataset):
    def __init__(self, data_dir="../scratch/CVUSA/dataset/", transforms_street=[transforms.ToTensor(),],transforms_sat=[transforms.ToTensor(),], mode='train', is_polar=True):
        self.data_dir = data_dir
        self.transforms_street = transforms.Compose(transforms_street)
        self.transforms_sat = transforms.Compose(transforms_sat)

        if mode == "val" or mode == "dev":
            self.file = os.path.join(self.data_dir, "splits", "val-19zl.csv")
        elif mode == "train":
            self.file = os.path.join(self.data_dir, "splits", "train-19zl.csv")
        else:
            raise RuntimeError("no such mode")

        self.data_list = []
        
        csv_file = open(self.file)
        for l in csv_file.readlines():
            data = l.strip().split(",")
            data.pop(2)
            if is_polar:
                data[0] = data[0].replace("bingmap", "polarmap")
                data[0] = data[0].replace("jpg", "png")
            self.data_list.append(data)

        csv_file.close()

        if mode == "dev":
            self.data_list = self.data_list[0:200]

    def __getitem__(self, index):
        satellite_file, ground_file = self.data_list[index]
        # x = Image.open(os.path.join(self.data_dir, satellite_file))
        # y = Image.open(os.path.join(self.data_dir, ground_file))
        # print(x.size)
        # print(y.size)
        satellite = self.transforms_sat(Image.open(os.path.join(self.data_dir, satellite_file)))
        ground = self.transforms_street(Image.open(os.path.join(self.data_dir, ground_file)))

        return {'satellite':satellite, 'ground':ground}


    def __len__(self):
        return len(self.data_list)


class USADataset(Dataset):
    def __init__(self, data_dir="../scratch/CVUSA/dataset/", geometric_aug='strong', sematic_aug='strong', mode='train', is_polar=True, is_mutual=True):
        self.data_dir = data_dir

        STREET_IMG_WIDTH = 671
        STREET_IMG_HEIGHT = 122

        self.is_polar = is_polar
        self.mode = mode
        self.is_mutual = is_mutual

        if not is_polar:
            SATELLITE_IMG_WIDTH = 256
            SATELLITE_IMG_HEIGHT = 256
        else:
            SATELLITE_IMG_WIDTH = 671
            SATELLITE_IMG_HEIGHT = 122

        transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH))]
        transforms_sat = [transforms.Resize((SATELLITE_IMG_HEIGHT, SATELLITE_IMG_WIDTH))]

        if sematic_aug == 'strong' or sematic_aug == 'same':
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
        self.transforms_street = transforms.Compose(transforms_street)

        self.geometric_aug = geometric_aug
        self.sematic_aug = sematic_aug

        if mode == "val" or mode == "dev":
            self.file = os.path.join(self.data_dir, "splits", "val-19zl.csv")
        elif mode == "train":
            self.file = os.path.join(self.data_dir, "splits", "train-19zl.csv")
        else:
            raise RuntimeError("no such mode")

        self.data_list = []
        
        csv_file = open(self.file)
        for l in csv_file.readlines():
            data = l.strip().split(",")
            data.pop(2)
            if is_polar:
                data[0] = data[0].replace("bingmap", "polarmap")
                data[0] = data[0].replace("jpg", "png")
            self.data_list.append(data)

        csv_file.close()

        if mode == "dev":
            self.data_list = self.data_list[0:200]


    def __getitem__(self, index):
        satellite_file, ground_file = self.data_list[index]

        satellite = Image.open(os.path.join(self.data_dir, satellite_file))
        ground = Image.open(os.path.join(self.data_dir, ground_file))

        satellite_first = self.transforms_sat(satellite)
        ground_first = self.transforms_street(ground)

        if self.sematic_aug != 'same':
            satellite_second = self.transforms_sat(satellite)
            ground_second = self.transforms_street(ground)
        else:
            satellite_second = satellite_first.clone().detach()
            ground_second = ground_first.clone().detach()

        # print(satellite_first.shape)
        # print(ground_first.shape)
        # print(satellite_second.shape)
        # print(ground_second.shape)
        # print("-----------------")


        # Generate first view
        if self.geometric_aug == "strong" or self.geometric_aug == 'same':
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
            return {'satellite':satellite_first, 
                    'ground':ground_first}

        else:
            # mutual_satellite = satellite.clone().detach()
            # mutual_ground = ground.clone().detach()

            # generate new different layout (second view)
            if self.geometric_aug == 'same' or self.geometric_aug == 'none':
                return {'satellite_first':satellite_first, 
                        'ground_first':ground_first,
                        'satellite_second':satellite_second,
                        'ground_second':ground_second,
                        'perturb':'none'}
                

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
                    'perturb':perturb}
            

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":

    STREET_IMG_WIDTH = 671
    STREET_IMG_HEIGHT = 122
    # SATELLITE_IMG_WIDTH = 256
    # SATELLITE_IMG_HEIGHT = 256
    SATELLITE_IMG_WIDTH = 671
    SATELLITE_IMG_HEIGHT = 122

    # transforms_sate = [transforms.Resize((SATELLITE_IMG_HEIGHT, SATELLITE_IMG_WIDTH)),
    #                 transforms.ToTensor()
    #                 ]
    # transforms_street = [transforms.Resize((STREET_IMG_HEIGHT, STREET_IMG_WIDTH)),
    #                 transforms.ToTensor()
    #                 ]

    dataloader = DataLoader(
        USADataset(data_dir='../scratch/CVUSA/dataset/', 
                   geometric_aug='strong', 
                   sematic_aug='strong', 
                   mode='train', 
                   is_polar=False),\
         batch_size=4, 
         shuffle=False, 
         num_workers=8)
    
    # print(len(dataloader))
    total_time = 0
    start = time.time()
    for k,b in enumerate(dataloader):
        end = time.time()
        elapse = end - start
        print("===========================")
        print(b["ground_first"].shape)
        print(b["satellite_first"].shape)
        print(b["ground_second"].shape)
        print(b["satellite_second"].shape)
        print("===========================")

        for i in range(4):
            grd = b["ground_first"][i]
            sat = b["satellite_first"][i]
            mu_grd = b["ground_second"][i]
            mu_sat = b["satellite_second"][i]

            sat = sat * 0.5 + 0.5
            grd = grd * 0.5 + 0.5
            mu_sat = mu_sat * 0.5 + 0.5
            mu_grd = mu_grd * 0.5 + 0.5

            torchvision.utils.save_image(sat, f"sat_f_{i}.png")
            torchvision.utils.save_image(grd, f"grd_f_{i}.png")
            torchvision.utils.save_image(mu_sat, f"sat_s_{i}.png")
            torchvision.utils.save_image(mu_grd, f"grd_s_{i}.png")

        if k == 2:
            break
        time.sleep(2)

