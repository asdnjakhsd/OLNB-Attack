import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from scipy.fftpack import idct
import math


class permutation_utils(object):


    def expand_vector(x, size, image_size):
        batch_size = x.size(0)
        num_channels = 3
        x = x.view(-1, num_channels, size, size)
        z = torch.zeros(batch_size, num_channels, image_size, image_size)
        z[:, :, :size, :size] = x
        return z

    def cvmat_to_tensor(cv_img):
        transf = transforms.ToTensor()
        return transf(cv_img).unsqueeze(0)

    def tensor_to_cvmat(tensor_img):
        tensor_img = tensor_img.squeeze(0)
        np_array = tensor_img.numpy()
        max_value = np_array.max()

        np_array = np_array * 255/max_value
        mat = np.uint8(np_array)
        mat = mat.transpose(1, 2, 0)

        # mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        return mat

    def get_square_size(tensor_img):
        img_x = tensor_img.shape[0]
        img_y = tensor_img.shape[1]

        if img_x >= img_y:
            square_size = img_y
        else:
            square_size = img_x
        return int(square_size)

    def get_square_pos(tensor_img, center_x, center_y, square_size):
        img_x = tensor_img.size(2)
        img_y = tensor_img.size(3)

        if center_x <= square_size//2:
            bound_x = 0
        else:
            bound_x = center_x - square_size//2

        if center_y <= square_size//2:
            bound_y = 0
        else:
            bound_y = center_y - square_size//2

        if bound_x + square_size > img_x:
            bound_x = img_x - square_size
        if bound_y + square_size > img_y:
            bound_y = img_y - square_size

        return int(bound_x), int(bound_y)

    def get_square_tensor(tensor_img, boundx, boundy, square_size):
        ret_tensor = torch.zeros(
            [1, tensor_img.size(1), square_size, square_size])

        ret_tensor[:, :, 0:square_size, 0:square_size] = tensor_img[:, :, boundx: (
            boundx + square_size), boundy: (boundy + square_size)]
        return ret_tensor
        
    def get_square_cvmat(cv_img, boundx, boundy, square_size):
        ret_mat = np.zeros((square_size, square_size, 3), np.uint8)
        ret_mat[:, :] = cv_img[boundx:(boundx + square_size), boundy:(boundy + square_size)]

        return ret_mat
          
    def set_cvsquare_to_img(cv_img, boundx, boundy, cv_square):
        size = cv_square.shape[0]
        cv_img[boundx:(boundx+size), boundy:(boundy+size)] = cv_square[:,:]
        return cv_img

    def set_square_to_tensor(tensor_img, boundx, boundy, tensor_square):
        res = tensor_img.clone()
        square_size = tensor_square.size(2)
        res[:, :, boundx: (boundx + square_size), boundy: (boundy + square_size)] = tensor_square[:, :, :, :]
        return res

    # project a new perturbation at x to l2 ball if ||x|| = radius. Set geodesic length = magnitude of dir

    def l2_projection(x, dir, radius):
        # x: current pertubation (4D tensor).     Size: batch_size x num_channels x image_size x image_size
        # dir: pertubation direction (4D tensor). Size: either bs or 1 x num_channels x image_size x image_size
        # radius: radius of l2 ball (positive number)
        bsx = x.size(0)
        bsd = dir.size(0)
        normx = (x**2).sum((1, 2, 3)).sqrt()+1.e-10     # norm of x
        normd = (dir**2).sum((1, 2, 3)).sqrt()+1.e-10   # norm of dir
        # divide the image batch into two sets, based on whether the perturbed images are inside l2 ball or on the boundary
        inside_mask = (radius-normx >= normd).float()
        onBound_mask = (radius-normx < normd).float()
        # form tangent direction
        x_norm = x/normx[:, None, None, None]
        dir_norm = (dir/normd[:, None, None, None]).repeat(bsx//bsd, 1, 1, 1)
        tang_dir = (dir_norm - x_norm * torch.einsum('ij, ij->i', torch.reshape(dir_norm,
                    (bsx, -1)), torch.reshape(x_norm, (bsx, -1)))[:, None, None, None])
        normtg = (tang_dir**2).sum((1, 2, 3)).sqrt()+1.e-10
        # auxiliary perturbation on tangent direction
        perturb0 = tang_dir * (normx * torch.tan(normd/normx) /
                               normtg)[:, None, None, None]
        normp0 = ((x+perturb0)**2).sum((1, 2, 3)).sqrt()+1.e-10
        # project to l2 ball for perturbed images on boundary
        x_new1 = ((x+perturb0) * (radius/normp0)
                  [:, None, None, None]) * onBound_mask[:, None, None, None]
        # just add perturbation directly otherwise
        x_new2 = (x + dir) * inside_mask[:, None, None, None]
        return (x_new1 + x_new2) - x

    def block_order(image_size, channels, initial_size, stride):
        order = torch.zeros(channels, image_size, image_size)
        total_elems = channels * initial_size * initial_size
        perm = torch.randperm(total_elems)
        order[:, :initial_size, :initial_size] = perm.view(channels, initial_size, initial_size)
        for i in range(initial_size, image_size, stride):
            #print("i = ",i)
            num_elems = channels * (2 * stride * i + stride * stride)
            perm = torch.randperm(num_elems) + total_elems
            num_first = channels * stride * (stride + i)
            order[:, :(i+stride), i:(i+stride)] = perm[:num_first].view(channels, -1, stride)
            order[:, i:(i+stride),:i] = perm[num_first:].view(channels, stride, -1)
            total_elems += num_elems
        return order.view(1, -1).squeeze().long().sort()[1]
      
    def laplacian_sharpening(image):
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        sharpened = cv2.addWeighted(image, 1.5, laplacian, -0.5, 0)
        return sharpened
    
    def equ_hist(img):
        channels = cv2.split(img)
        eq_channels = []
        for ch in channels:
            eq_channels.append(cv2.equalizeHist(ch))
        eq_image = cv2.merge(eq_channels)
        return eq_image
        
        
    def block_idct(x, block_size=8, masked=False, ratio=0.5, linf_bound=0.0):
        z = torch.zeros(x.size())
        num_blocks = int(x.size(2) / block_size)
        mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
        if type(ratio) != float:
            for i in range(x.size(0)):
                mask[i, :, :int(block_size * ratio[i]),
                     :int(block_size * ratio[i])] = 1
        else:
            mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
        for i in range(num_blocks):
            for j in range(num_blocks):
                submat = x[:, :, (i * block_size):((i + 1) * block_size),
                           (j * block_size):((j + 1) * block_size)].numpy()
                if masked:
                    submat = submat * mask
                z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)
                  ] = torch.from_numpy(idct(idct(submat, axis=3, norm='ortho'), axis=2, norm='ortho'))
        if linf_bound > 0:
            return z.clamp(-linf_bound, linf_bound)
        else:
            return z
