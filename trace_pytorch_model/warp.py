import numpy as np
import torch
import math
import torch.nn.functional as F

D2R = math.pi / 180.0

class WarpImg:

    def __init__(self, img_height, img_width, device, FoV=90*D2R): # FoV(rad) is field of view of the image in width

        self.device = device
        self.img_height = img_height
        self.img_width = img_width

        fx = (img_width-1)/2/math.tan(FoV/2)
        fy = fx
        cx = (img_width-1)/2
        cy = (img_height-1)/2
        camMtrx = \
            [[1*fx,     0,      1*cx],
             [0,        1*fy,   1*cy],
             [0,        0,         1]]

        camMtrx_inverse = \
            [[ 1/fx,    0, -cx/fx],
             [    0, 1/fy, -cy/fy],
             [    0,    0,      1]]
        
        camMtrx_np = np.array(camMtrx, dtype=np.float32) # list to numpy array
        self.camMtrx = torch.from_numpy(camMtrx_np).to(self.device) # numpy array to tensor

        camMtrx_inverse_np = np.array(camMtrx_inverse, dtype=np.float32) # list to numpy array
        self.camMtrx_inverse = torch.from_numpy(camMtrx_inverse_np).to(self.device) # numpy array to tensor
        
        grid_uv1, grid_xy1 = self.generate_grid()  # torch.float32 [3, height*width]
        self.grid_uv1 = grid_uv1.to(self.device)
        self.grid_xy1 = grid_xy1.to(self.device)

        self.sample_grid_factor = torch.FloatTensor([[[2/(img_width-1), 2/(img_height-1)]]]).to(self.device) # torch.Size([1, 1, 2])

        print("Warper for tensors with height {} and width {} has been initialized!".format(str(img_height), str(img_width)))


    def generate_grid(self):

        W = self.img_width
        H = self.img_height

        u = torch.arange(0, W).view(1, -1).repeat(H, 1).unsqueeze(0).float().to(self.device) 
        v = torch.arange(0, H).view(-1, 1).repeat(1, W).unsqueeze(0).float().to(self.device)
        self.grid_uv = torch.cat((u, v), dim=0) # torch.Size([2, 224, 320])

        grid_uv1 = torch.cat((self.grid_uv, torch.ones_like(self.grid_uv[0:1, :, :])), dim=0).view([3, H*W])
        grid_xy1 = torch.mm(self.camMtrx_inverse, grid_uv1)

        return grid_uv1, grid_xy1 


    def warpSingleImage_H_Mtrx(self,batch_imgs,batch_homoMtrx): # NOTE batch_homoMtrx is for pixel location
        
        sample_grid_normed_batch_list = []

        homoMtrx = batch_homoMtrx[0, :, :]
        sample_grid_uvz = torch.mm(homoMtrx, self.grid_uv1) # [3, height*width] 
        sample_grid_uv1 = sample_grid_uvz / sample_grid_uvz[2, :]  # [3, height*width] 
        sample_grid_uv = sample_grid_uv1[0:2, :].view([2, self.img_height, self.img_width])
        sample_grid_uv = torch.transpose(sample_grid_uv, 0, 1)
        sample_grid_uv = torch.transpose(sample_grid_uv, 1, 2) # [H, W, 2]
        sample_grid_normed = sample_grid_uv * self.sample_grid_factor - 1 # [-1, 1] ([H, W, 2])
        sample_grid_normed_batch_list.append(sample_grid_normed.unsqueeze(0)) 

        batch_sample_grid_normed = torch.cat(sample_grid_normed_batch_list, dim=0)
        # if torch.__version__ < '1.2.0': 
        #     batch_warped_imgs = F.grid_sample(batch_imgs, batch_sample_grid_normed, mode='bilinear', padding_mode='zeros')
        # else: # NOTE torch 1.1.0 does not have the input parameter "align_corners", align_corners=True by default
        batch_warped_imgs = F.grid_sample(batch_imgs, batch_sample_grid_normed, mode='bilinear', padding_mode='zeros', align_corners=True)

        return batch_warped_imgs