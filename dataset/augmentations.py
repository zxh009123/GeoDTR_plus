import torch
import numpy as np
import math
import random

def HFlip(sat, grd):
    sat = torch.flip(sat, [2])
    grd = torch.flip(grd, [2])

    return sat, grd

def Rotate(sat, grd, orientation, is_polar):
    height, width = grd.shape[1], grd.shape[2]
    if orientation == 'left':
        if is_polar:
            left_sat = sat[:, :, 0:int(math.ceil(width * 0.75))]
            right_sat = sat[:, :, int(math.ceil(width * 0.75)):]
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, -1, [1, 2])
        left_grd = grd[:, :, 0:int(math.ceil(width * 0.75))]
        right_grd = grd[:, :, int(math.ceil(width * 0.75)):]
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)

    elif orientation == 'right':
        if is_polar:
            left_sat = sat[:, :, 0:int(math.floor(width * 0.25))]
            right_sat = sat[:, :, int(math.floor(width * 0.25)):]
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, 1, [1,2])
        left_grd = grd[:, :, 0:int(math.floor(width * 0.25))]
        right_grd = grd[:, :, int(math.floor(width * 0.25)):]
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)

    elif orientation == 'back':
        if is_polar:
            left_sat = sat[:, :, 0:int(width * 0.5)]
            right_sat = sat[:, :, int(width * 0.5):]
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, 1, [1,2])
            sat_rotate = torch.rot90(sat_rotate, 1, [1,2])
        left_grd = grd[:, :, 0:int(width * 0.5)]
        right_grd = grd[:, :, int(width * 0.5):]
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)
    
    else:
        raise RuntimeError(f"Orientation {orientation} is not implemented")

    return sat_rotate, grd_rotate

def Rotate_tensor(sat, grd, orientation, is_polar):
    height, width = grd.shape[1], grd.shape[2]
    if orientation == 'left':
        split_width = int(math.ceil(width * 0.75))
        if is_polar:
            # left_sat = sat[:, :, 0:int(math.ceil(width * 0.75))]
            # right_sat = sat[:, :, int(math.ceil(width * 0.75)):]
            left_sat, right_sat = torch.split(sat, [split_width, width - split_width], dim=2)
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, -1, [1, 2])
        # left_grd = grd[:, :, 0:int(math.ceil(width * 0.75))]
        # right_grd = grd[:, :, int(math.ceil(width * 0.75)):]
        left_grd, right_grd = torch.split(grd, [split_width, width - split_width], dim=2)
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)

    elif orientation == 'right':
        split_width = int(math.floor(width * 0.25))
        if is_polar:
            # left_sat = sat[:, :, 0:int(math.floor(width * 0.25))]
            # right_sat = sat[:, :, int(math.floor(width * 0.25)):]
            left_sat, right_sat = torch.split(sat, [split_width, width - split_width], dim=2)
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, 1, [1,2])
        # left_grd = grd[:, :, 0:int(math.floor(width * 0.25))]
        # right_grd = grd[:, :, int(math.floor(width * 0.25)):]
        left_grd, right_grd = torch.split(grd, [split_width, width - split_width], dim=2)
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)

    elif orientation == 'back':
        split_width = int(width * 0.5)
        if is_polar:
            # left_sat = sat[:, :, 0:int(width * 0.5)]
            # right_sat = sat[:, :, int(width * 0.5):]
            left_sat, right_sat = torch.split(sat, [split_width, width - split_width], dim=2)
            sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        else:
            sat_rotate = torch.rot90(sat, 1, [1,2])
            sat_rotate = torch.rot90(sat_rotate, 1, [1,2])
        # left_grd = grd[:, :, 0:int(width * 0.5)]
        # right_grd = grd[:, :, int(width * 0.5):]
        left_grd, right_grd = torch.split(grd, [split_width, width - split_width], dim=2)
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)
    
    else:
        raise RuntimeError(f"Orientation {orientation} is not implemented")

    return sat_rotate, grd_rotate

def Reverse_Rotate_Flip(sat, grd, perturb, polar):
    # Reverse process

    assert sat.shape[0] == grd.shape[0]
    assert sat.shape[0] == len(perturb)

    sat = sat.permute(0,3,1,2)
    grd = grd.permute(0,3,1,2)

    reversed_sat_desc = torch.zeros_like(sat)
    reversed_grd_desc = torch.zeros_like(grd)

    for i in range(len(perturb)):
        reverse_perturb = [None, None]
        reverse_perturb[0] = perturb[i][0]

        if perturb[i][1] == "left":
            reverse_perturb[1] = "right"
        elif perturb[i][1] == "right":
            reverse_perturb[1] = "left"
        else:
            reverse_perturb[1] = perturb[i][1]

        # print(reverse_perturb)

        # reverse process first rotate then flip
        if reverse_perturb[1] != "none":
            rotated_sati, rotated_grdi = Rotate_tensor(sat[i], grd[i], reverse_perturb[1], polar)
        else:
            rotated_sati = sat[i]
            rotated_grdi = grd[i]

        if reverse_perturb[0] == 1:
            reversed_sat_desc[i], reversed_grd_desc[i] = HFlip(rotated_sati, rotated_grdi)
        else:
            reversed_sat_desc[i] = rotated_sati
            reversed_grd_desc[i] = rotated_grdi

    reversed_sat_desc = reversed_sat_desc.permute(0,2,3,1)
    reversed_grd_desc = reversed_grd_desc.permute(0,2,3,1)

    return reversed_sat_desc, reversed_grd_desc


def Free_Rotation(sat, grd, degree, axis="h"):
    """
    only for polar case
    degree will be made in [0., 360.]; clockwise by default; can be negative;
    axis="h" for horizontal and "v" for vertical direction in the polar image;
        - "h" for normal (improper) rotation & flip; rel pos preserved
        - "v" change the distribution; rel pos NOT preserved
    NOTE sat & grd of shape: (bs, c, h, w)
    """
    height, width = grd.shape[2], grd.shape[3]

    degree = np.mod(degree, 360.)
    ratio = 1.- degree / 360.
    if axis == "h":
        bound = int(width * ratio)

        left_sat  = sat[:, :, :, 0:bound]
        right_sat = sat[:, :, :, bound:]

        left_grd  = grd[:, :, :, 0:bound]
        right_grd = grd[:, :, :, bound:]

        sat_rotate = torch.cat([right_sat, left_sat], dim=3)
        grd_rotate = torch.cat([right_grd, left_grd], dim=3)

    elif axis == "v":
        bound = int(height * ratio)

        left_sat  = sat[:, :, 0:bound]
        right_sat = sat[:, :, bound:]

        left_grd  = grd[:, :, 0:bound]
        right_grd = grd[:, :, bound:]

        sat_rotate = torch.cat([right_sat, left_sat], dim=2)
        grd_rotate = torch.cat([right_grd, left_grd], dim=2)

    return sat_rotate, grd_rotate


def Free_Improper_Rotation(sat, grd, degree, axis="h"):
    """
    only for polar case
    degree will be made in [0., 360.]; clockwise by default; can be negative;
    axis="h" for horizontal and "v" for vertical direction in the polar image;
        - "h" for normal (improper) rotation & flip; rel pos preserved
        - "v" change the distribution; rel pos NOT preserved
    NOTE sat & grd of shape: (bs, c, h, w)
    """
    new_sat = torch.flip(sat, [3])
    new_grd = torch.flip(grd, [3])

    sat_rotate, grd_rotate = Free_Rotation(new_sat, new_grd, degree, axis=axis)

    return sat_rotate, grd_rotate


def Free_Flip(sat, grd, degree):
    """
    only for polar case
    (virtually) flip-reference is the non-polar sat-view image
    degree specifies the flip axis
    degree will be made in [0., 360.]; clockwise by default; can be negative;
    """
    # rotate by -degree
    new_sat, new_grd = Free_Rotation(sat, grd, -degree, axis="h")

    # h-flip
    new_sat = torch.flip(new_sat, [3])
    new_grd = torch.flip(new_grd, [3])

    # rotate back by degree
    new_sat, new_grd = Free_Rotation(new_sat, new_grd, degree, axis="h")
    
    return new_sat, new_grd


if __name__ == "__main__":
    # original descriptors
    # sat = torch.rand(32, 16, 16, 8)
    # sat = torch.rand(2, 4, 4, 3)
    sat = torch.rand(32, 8, 42, 8)
    polar = True
    grd = torch.rand(32, 8, 42, 8)
    # grd = torch.rand(2, 2, 6, 3)


    # Copy to generate a new descriptor
    mu_sat = sat.clone().detach()
    mu_grd = grd.clone().detach()
    

    # generate new descriptor by LS
    mu_sat = mu_sat.permute(0,3,1,2)
    mu_grd = mu_grd.permute(0,3,1,2)

    first_sat = torch.zeros_like(mu_sat)
    first_grd = torch.zeros_like(mu_grd)

    # generate # of batch size LS operations
    # perturb = [[1, "none"], [1, "left"], [1, "back"], [0, "right"], [0, "left"], [0, "back"]]
    perturb = []
    for i in range(32):
        hflip = random.randint(0,1)
        orientation = random.choice(["left", "right", "back", "none"])

        while hflip == 0 and orientation == "none":
            hflip = random.randint(0,1)
            orientation = random.choice(["left", "right", "back", "none"])

        perturb.append([hflip, orientation])

    print(perturb)

    # perform LS to generate new layout
    for i in range(len(perturb)):
        orig_sat = mu_sat[i]
        orig_grd = mu_grd[i]
        if perturb[i][0] == 1:
            orig_sat, orig_grd = HFlip(orig_sat, orig_grd)
        if perturb[i][1] != "none": 
            first_sat[i], first_grd[i] = Rotate(orig_sat, orig_grd, perturb[i][1], polar)
        else:
            first_sat[i] = orig_sat
            first_grd[i] = orig_grd

    first_sat = first_sat.permute(0,2,3,1)
    first_grd = first_grd.permute(0,2,3,1)

    print("=====before:")
    # print(grd[0, :, :, :])
    # print(first_grd[0, :, :, :])
    # print(sat[0, :, :, :])
    # print(first_sat[0, :, :, :])
    print(torch.equal(sat, first_sat))
    print(torch.equal(grd, first_grd))

    # Reverse to original layout
    second_sat, second_grd = Reverse_Rotate_Flip(first_sat, first_grd, perturb, polar)

    print("=====after:")
    # print(grd[0, :, :, :])
    # print(second_grd[0, :, :, :])
    # print(sat[0, :, :, :])
    # print(second_sat[0, :, :, :])
    print(torch.equal(sat, second_sat))
    print(torch.equal(grd, second_grd))
    # for i in range(sat.shape[0]):
    #     print(f"=============={i}==============")
    #     print(torch.equal(sat[i], second_sat[i]))
    #     print(torch.equal(grd[i], second_grd[i]))
    #     print(torch.equal(sat[i], second_sat[i]))
    #     print(torch.equal(grd[i], second_grd[i]))
    #     print("================================")
