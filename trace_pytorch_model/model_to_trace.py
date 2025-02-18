import torch
import torch.nn as nn
import numpy as np
import warp


def conv(in_planes, out_planes, kernel_size=3, stride=2, activation=True):
    if activation:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride)


def transfer_mean_var_single(var, batch_H_uv_Mtrx, batch_4pt_warped_img2):

    batch_4pt_warped_img2_uv1 = torch.transpose((torch.cat((batch_4pt_warped_img2, torch.ones_like(batch_4pt_warped_img2)[:, :, 0:1]), dim=2)), 1, 2) # torch.Size([bs, 3, 4])
    _4pt_orig_img2 = torch.bmm(batch_H_uv_Mtrx, batch_4pt_warped_img2_uv1) # the predicted correspondent pixel locations (in original img2) of the 4pt of img1
    scale = _4pt_orig_img2[:, 2:3, :]
    _4pt_orig_img2 = _4pt_orig_img2 / scale

    scale_batch_i = scale[0, 0, :] # torch.Size([4]) # scales for the 4pt of this batch
    H_uv_Mtrx_batch_i = batch_H_uv_Mtrx[0, :, :]
    Cov_Mtrx_pt_list = []
    for pt_i in range (4):
        scale_pt_i = scale_batch_i[pt_i]
        H_uv_Mtrx_batch_i_scaled = H_uv_Mtrx_batch_i / scale_pt_i # scaled H matrix for each point that satisfy H @ [u1 v1 1]' = [u2 v2 1]'
        # form up the (co)variance matrix
        var_Mtrx_pt_i = torch.diag(torch.cat((var[0, pt_i, :], torch.zeros_like(scale_pt_i).unsqueeze(0)))) # torch.Size([3, 3]
        transfered_var_Mtrx_pt_i = torch.mm(torch.mm(H_uv_Mtrx_batch_i_scaled, var_Mtrx_pt_i), H_uv_Mtrx_batch_i_scaled.t())
        Cov_Mtrx = transfered_var_Mtrx_pt_i[0:2, 0:2] # for a single point
        Cov_Mtrx_pt_list.append(Cov_Mtrx.unsqueeze(0))
    Cov_Mtrx_4pt = torch.cat(Cov_Mtrx_pt_list, dim=0).unsqueeze(0) # torch.Size([4, 2, 2]) # 4 for 4pt

    return _4pt_orig_img2, Cov_Mtrx_4pt


# Taken from https://github.com/JirongZhang/DeepHomography
def DLT_solve(batch_origin_4pt, batch_new_4pt):
    bs = batch_origin_4pt.size(0)
    src_ps = batch_origin_4pt
    dst_p = batch_new_4pt
    ones = torch.ones_like(src_ps)[:, :, 0:1]
    xy1 = torch.cat((src_ps, ones), 2)
    zeros = torch.zeros_like(xy1)
    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
    M1 = torch.cat((xyu, xyd), 2).reshape(bs, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1), 
        src_ps.reshape(-1, 1, 2),
    ).reshape(bs, -1, 2)
    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(bs, -1, 1)
    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(bs, 8)
    H = torch.cat((h8, ones[:,0,:]), 1).reshape(bs, 3, 3)
    H = H.reshape(bs, 3, 3)
    return H


class Down_Net_3blocks(nn.Module):
    
    def __init__(self, img_height, img_width, device):

        super().__init__() 

        self.device = device

        self.blocks_to_run = 3 # NOTE when use prior pose from EKF, the number of blocks we want to run
        assert(self.blocks_to_run >=1 and self.blocks_to_run <=3)

        self.img_warper_full_size = warp.WarpImg(img_height=img_height, img_width=img_width, device=self.device)

        # the offset (in pixel, divided by image height or width) of the 4 points relative to the corners of the image. 
        self.cornerOffset_4pt = 0.0 # use the corner pixel
        pt_ul = np.array([self.cornerOffset_4pt * img_width, self.cornerOffset_4pt * img_height]) # u, v
        pt_bl = np.array([self.cornerOffset_4pt * img_width, img_height - 1 - self.cornerOffset_4pt * img_height])
        pt_br = np.array([img_width - 1 - self.cornerOffset_4pt * img_width, img_height - 1 - self.cornerOffset_4pt * img_height])
        pt_ur = np.array([img_width - 1 - self.cornerOffset_4pt * img_width, self.cornerOffset_4pt * img_height])
        self.origin_4pt = torch.from_numpy(np.array([pt_ul, pt_bl, pt_br, pt_ur])).float().to(self.device) # size[4, 2]

        self.img_width = img_width
        self.img_height = img_height

        self.conv_planes = [8, 16, 32, 64, 128, 256, 256]
        self.fc_input = 5120

        # Homo prediction blocks
        # block 1
        self.block_1_1 = conv(                  2, self.conv_planes[4], kernel_size=7) # 28 40 -> 14 20
        self.block_1_2 = conv(self.conv_planes[4], self.conv_planes[4], kernel_size=5) # 7 10
        self.block_1_3 = conv(self.conv_planes[4], self.conv_planes[5]) # 4 5

        self.fc_block_1 = nn.Linear(self.fc_input, 8, bias=True)

        # block 2
        self.block_2_1 = conv(                  2, self.conv_planes[3], kernel_size=7) # 56 80 -> 28 40
        self.block_2_2 = conv(self.conv_planes[3], self.conv_planes[4], kernel_size=5) # 14 20
        self.block_2_3 = conv(self.conv_planes[4], self.conv_planes[5]) # 7 10
        self.block_2_4 = conv(self.conv_planes[5], self.conv_planes[6]) # 4 5

        self.fc_block_2 = nn.Linear(self.fc_input, 8, bias=True)

        # block 3
        self.block_3_0 = conv(                  2, self.conv_planes[1], kernel_size=7, stride=1) # 112 160
        self.block_3_1 = conv(self.conv_planes[1], self.conv_planes[2], kernel_size=5) # 112 160 -> 56 80
        self.block_3_2 = conv(self.conv_planes[2], self.conv_planes[3]) # 28 40
        self.block_3_3 = conv(self.conv_planes[3], self.conv_planes[4]) # 14 20
        self.block_3_4 = conv(self.conv_planes[4], self.conv_planes[5]) # 7 10
        self.block_3_5 = conv(self.conv_planes[5], self.conv_planes[6]) # 4 5

        self.fc_block_3 = nn.Linear(self.fc_input, 8, bias=True)

        self.downSample_block_1 = nn.AvgPool2d(8, stride=8, padding=0)
        self.downSample_block_2 = nn.AvgPool2d(4, stride=4, padding=0)
        self.downSample_block_3 = nn.AvgPool2d(2, stride=2, padding=0)

        print("Part I of the student model (3 HomoNet blocks) have been initialized!")


    def forward(self,batch_img1,batch_img2,batch_4pt_offset_prior=None):

        batch_size = batch_img1.size()[0]

        batch_img1_4pt = self.origin_4pt.unsqueeze(0).repeat(batch_size, 1, 1) # torch.Size([bs, 4, 2]) # pixel
        if batch_4pt_offset_prior is not None: # when use the states of EKF as prior guess
            batch_H_uv_Mtrx_prior = DLT_solve(batch_img1_4pt, batch_img1_4pt+batch_4pt_offset_prior)
            if self.blocks_to_run == 1:
                return batch_H_uv_Mtrx_prior

        # run blocks

        ## block 1st
        if batch_4pt_offset_prior is None:
            block_1_input_tensor = torch.cat((batch_img1, batch_img2), dim=1) # torch.Size([bs, 256, 28, 40])
            block_1_input_tensor = self.downSample_block_1(block_1_input_tensor)
            block_1_1_tensor = self.block_1_1(block_1_input_tensor) # 28 40 -> 14 20
            block_1_2_tensor = self.block_1_2(block_1_1_tensor) # 7 10
            block_1_3_tensor = self.block_1_3(block_1_2_tensor) # 4 5
            batch_fc_1 = self.fc_block_1(block_1_3_tensor.view(batch_size, -1)) # torch.Size([bs, 8])

            batch_img2_4pt_pred_b1 = batch_img1_4pt + batch_fc_1.view(batch_size, 4, 2) # the predicted correspondent pixel locations (in img2) of the 4pt of img1
            batch_H_uv_Mtrx_b1 = DLT_solve(batch_img1_4pt, batch_img2_4pt_pred_b1) 

            batch_H_uv_Mtrx = batch_H_uv_Mtrx_b1 # img1 to img2: img2_4pt = torch.mm(H_Mtrx, img1_4pt) # torch.Size([batch_size, 3, 3])
        else:
            batch_H_uv_Mtrx = batch_H_uv_Mtrx_prior

        ## block 2nd
        if batch_4pt_offset_prior is None or self.blocks_to_run == 3:
            batch_img2_warped = self.img_warper_full_size.warpSingleImage_H_Mtrx(batch_img2, batch_H_uv_Mtrx)

            block_2_input_tensor = torch.cat((batch_img1, batch_img2_warped), dim=1) # torch.Size([bs, 256, 56, 80])
            block_2_input_tensor = self.downSample_block_2(block_2_input_tensor)
            
            block_2_1_tensor = self.block_2_1(block_2_input_tensor) # 56 80 -> 28 40
            block_2_2_tensor = self.block_2_2(block_2_1_tensor) # 14 20
            block_2_3_tensor = self.block_2_3(block_2_2_tensor) # 7 10 
            block_2_4_tensor = self.block_2_4(block_2_3_tensor) # 4 5 
            batch_fc_2 = self.fc_block_2(block_2_4_tensor.view(batch_size, -1))

            batch_img2_4pt_pred_b2 = batch_img1_4pt + batch_fc_2.view(batch_size, 4, 2) # the predicted correspondent pixel locations (in warped_img2) of the 4pt of img1
            batch_H_uv_Mtrx_b2 = DLT_solve(batch_img1_4pt, batch_img2_4pt_pred_b2) 
            
            batch_H_uv_Mtrx = torch.bmm(batch_H_uv_Mtrx, batch_H_uv_Mtrx_b2)

        ## block 3rd
        if batch_4pt_offset_prior is None or self.blocks_to_run >= 2:
            batch_img2_warped = self.img_warper_full_size.warpSingleImage_H_Mtrx(batch_img2,batch_H_uv_Mtrx)

            block_3_input_tensor = torch.cat((batch_img1, batch_img2_warped), dim=1) # torch.Size([bs, 256, 112, 160])
            block_3_input_tensor = self.downSample_block_3(block_3_input_tensor)
            
            block_3_0_tensor = self.block_3_0(block_3_input_tensor) # 112 160
            block_3_1_tensor = self.block_3_1(block_3_0_tensor) # 112 160 -> 56 80
            block_3_2_tensor = self.block_3_2(block_3_1_tensor) # 28 40
            block_3_3_tensor = self.block_3_3(block_3_2_tensor) # 14 20 
            block_3_4_tensor = self.block_3_4(block_3_3_tensor) # 7 10 
            block_3_5_tensor = self.block_3_5(block_3_4_tensor) # 4 5 
            batch_fc_3 = self.fc_block_3(block_3_5_tensor.view(batch_size, -1))

            batch_img2_4pt_pred_b3 = batch_img1_4pt + batch_fc_3.view(batch_size, 4, 2) # the predicted correspondent pixel locations (in warped_img2) of the 4pt of img1
            batch_H_uv_Mtrx_b3 = DLT_solve(batch_img1_4pt, batch_img2_4pt_pred_b3) 
            
            batch_H_uv_Mtrx = torch.bmm(batch_H_uv_Mtrx, batch_H_uv_Mtrx_b3)

        else:
            batch_H_uv_Mtrx = batch_H_uv_Mtrx_prior

        return batch_H_uv_Mtrx

class HomoNet_last_block(nn.Module):

    def __init__(self, img_warper, conv_planes, fc_input, origin_4pt, device, dropout_rate):
        super().__init__()

        self.device = device
        self.num_fc_layer = 2
        self.MC_dropout_num = 16
        var_fc_dim = 8
        
        print("Predict the 8-d variance of the 8-dim 4pt optical flow.")
        print("Dropout rate of the last block is", dropout_rate)
        print("Run dropout part times:", self.MC_dropout_num)

        # block 4
        self.block_4_0 = conv(             2, conv_planes[0], kernel_size=7, stride=1) # 224 320
        self.block_4_1 = conv(conv_planes[0], conv_planes[1], kernel_size=5) # 224 320 -> 112 160
        self.block_4_2 = conv(conv_planes[1], conv_planes[2]) # 56 80
        self.block_4_3 = conv(conv_planes[2], conv_planes[3]) # 28 40
        self.block_4_4 = conv(conv_planes[3], conv_planes[4]) # 14 20
        self.block_4_5 = conv(conv_planes[4], conv_planes[5]) # 7 10
        self.block_4_6 = conv(conv_planes[5], conv_planes[6]) # 4 5

        if self.num_fc_layer == 1:
            self.fc_block_4_mean = nn.Linear(fc_input, 8, bias=True)
            self.fc_block_4_uncertainty = nn.Linear(fc_input, var_fc_dim, bias=True)
        elif self.num_fc_layer == 2:
            self.fc_block_4_mean = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(fc_input, 256, bias=True),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, 8, bias=True),
            )
            self.fc_block_4_uncertainty = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(fc_input, 256, bias=True),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, var_fc_dim, bias=True),
            )

        self.img_warper = img_warper
        self.origin_4pt = origin_4pt
        self.batch_img1_4pt = self.origin_4pt.unsqueeze(0).repeat(1, 1, 1) # torch.Size([bs, 4, 2])

    def run_conv(self, block_4_input_tensor): # NOTE manually change dropout layers for conv (when the input dropout rate > 0, fc layers has dropout for sure)
        block_4_0_tensor = self.block_4_0(block_4_input_tensor) # 224 320    
        block_4_1_tensor = self.block_4_1(block_4_0_tensor) # 224 320 -> 112 160
        block_4_2_tensor = self.block_4_2(block_4_1_tensor) # 56 80 
        block_4_3_tensor = self.block_4_3(block_4_2_tensor) # 28 40 
        block_4_4_tensor = self.block_4_4(block_4_3_tensor) # 14 20 
        block_4_5_tensor = self.block_4_5(block_4_4_tensor) # 7  10 
        block_4_6_tensor = self.block_4_6(block_4_5_tensor) # 4  5

        return block_4_6_tensor

    def run_fc(self, block_4_6_tensor, batch_size): 
        batch_fc_4_mean = self.fc_block_4_mean(block_4_6_tensor.view(batch_size, -1)).view(batch_size, 4, 2)
        batch_fc_4_log_var = self.fc_block_4_uncertainty(block_4_6_tensor.view(batch_size, -1)).view(batch_size, 4, 2)

        return batch_fc_4_mean, batch_fc_4_log_var * 1e-03

    def forward(self, batch_img1, batch_img2, batch_H_uv_Mtrx_input):

        ## block 4th
        batch_img2_warped = self.img_warper.warpSingleImage_H_Mtrx(batch_img2,batch_H_uv_Mtrx_input)

        block_4_input_tensor = torch.cat((batch_img1, batch_img2_warped), dim=1) # torch.Size([bs, 256, 224, 320])
            
        # run multiple time when inference
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        # assert(batch_img1.size()[0]==1) # duplicate the tensor along batch dimension before dropout
        block_4_6_tensor = self.run_conv(block_4_input_tensor)
        block_4_6_tensor = block_4_6_tensor.repeat(self.MC_dropout_num, 1, 1, 1) # NOTE if only dropout fc
        multi_dropout_mean, multi_dropout_log_var = self.run_fc(block_4_6_tensor, self.MC_dropout_num) 
        multi_dropout_var = torch.exp(multi_dropout_log_var)
        multi_dropout_mean_avg = multi_dropout_mean.mean(0).unsqueeze(0)
        multi_dropout_var_avg = multi_dropout_var.mean(0).unsqueeze(0)
        avg_pred_var = multi_dropout_var_avg
        empirical_vars = torch.square(multi_dropout_mean_avg.repeat(self.MC_dropout_num, 1, 1) - multi_dropout_mean)
        avg_empirical_var = empirical_vars.mean(0).unsqueeze(0)
        ensemble_var = avg_empirical_var + avg_pred_var
        batch_4pt_mean_warped_img2 = self.batch_img1_4pt + multi_dropout_mean_avg
        return batch_4pt_mean_warped_img2, ensemble_var


class combined_stu_model(nn.Module):

    def __init__(self, model_part1, model_last_block_list, device, show_photometric_error=False):
        super().__init__()
        self.model_part1 = model_part1
        self.model_last_block_list = model_last_block_list
        self.device = device

        self.save_pred_and_ensem_emp_var = True
        if self.save_pred_and_ensem_emp_var:
            self.ensemble_pred_empirical_var_list = []

        self.show_photometric_error = show_photometric_error

    def forward(self,batch_img1,batch_img2,batch_4pt_offset_prior=None):
        
        self.model_part1.eval()

        with torch.no_grad():
            batch_H_uv_Mtrx_part1 = self.model_part1(batch_img1,batch_img2,batch_4pt_offset_prior=batch_4pt_offset_prior).detach()
        
        model_last_block = self.model_last_block_list[0]
            
        ensemble_4pt_mean_warped_img2, ensemble_4pt_var_warped_img2 = model_last_block(batch_img1, batch_img2, batch_H_uv_Mtrx_part1)
        batch_4pt_uv1_orig_img2, batch_Cov_Mtrx_4pt = transfer_mean_var_single(ensemble_4pt_var_warped_img2, batch_H_uv_Mtrx_part1, ensemble_4pt_mean_warped_img2)
        
        optical_flow_4pt = (torch.transpose((batch_4pt_uv1_orig_img2[:, 0:2, :]), 1, 2) - self.model_last_block_list[0].batch_img1_4pt).squeeze()
        batch_Cov_Mtrx_4pt = batch_Cov_Mtrx_4pt.squeeze()
        Cov_Mtrx_88 = torch.zeros([8, 8])
        Cov_Mtrx_88[0:2, 0:2] = batch_Cov_Mtrx_4pt[0, :, :]
        Cov_Mtrx_88[2:4, 2:4] = batch_Cov_Mtrx_4pt[1, :, :]
        Cov_Mtrx_88[4:6, 4:6] = batch_Cov_Mtrx_4pt[2, :, :]
        Cov_Mtrx_88[6:8, 6:8] = batch_Cov_Mtrx_4pt[3, :, :]

        if self.show_photometric_error:
            # for the traced cpp model to show the photometric error maps while running
            batch_img1_4pt = self.model_last_block_list[0].origin_4pt.unsqueeze(0)
            batch_H_uv_Mtrx_b4 = DLT_solve(batch_img1_4pt, ensemble_4pt_mean_warped_img2) 
            batch_H_uv_Mtrx_total = torch.bmm(batch_H_uv_Mtrx_part1, batch_H_uv_Mtrx_b4)
            batch_img2_warped = self.model_part1.img_warper_full_size.warpSingleImage_H_Mtrx(batch_img2,batch_H_uv_Mtrx_total)
            photometric_error_map = (batch_img2_warped - batch_img1).abs()

            return optical_flow_4pt.reshape(8, 1), Cov_Mtrx_88, photometric_error_map*255.0
        
        else:
            return optical_flow_4pt.reshape(8, 1), Cov_Mtrx_88


def HomoNet_ICSTN_Down_stu(img_height, img_width, device, pretrained_stu_model=None, dropout_rate=0.05, show_photometric_error=False):

    assert(pretrained_stu_model is not None)

    HomoNet_model_part1 = Down_Net_3blocks(img_height, img_width, device)

    last_block_list = nn.ModuleList([])
    last_block = HomoNet_last_block(HomoNet_model_part1.img_warper_full_size, HomoNet_model_part1.conv_planes, HomoNet_model_part1.fc_input, HomoNet_model_part1.origin_4pt, device, dropout_rate=dropout_rate)
    last_block_list.append(last_block)

    HomoNet_model = combined_stu_model(HomoNet_model_part1, last_block_list, device, show_photometric_error)
    HomoNet_model.load_state_dict(pretrained_stu_model['state_dict'], strict=True)
    print("Loaded the pre-trained student model!")

    total_params_count = sum(p.numel() for name, p in HomoNet_model.named_parameters() if p.requires_grad)
    print("Total number of trainable parameters in the student model: {}".format(total_params_count))

    return HomoNet_model