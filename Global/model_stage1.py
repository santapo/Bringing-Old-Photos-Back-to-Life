# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import base64
# from collections import OrderedDict
# from torch.autograd import Variable
# from Global.options.test_options import TestOptions
# from Global.models.models import create_model
from models.mapping_model import Pix2PixHDModel_Mapping
# import Global.util.util as util
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import cv2

def data_transforms(img, method=Image.BILINEAR, scale=False):

    ow, oh = img.size
    pw, ph = ow, oh
    if scale == True:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img

    return img.resize((w, h), method)


def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Resize(256, Image.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)


def irregular_hole_synthesize(img, mask):

    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")

    return hole_img


def parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    opt.checkpoints_dir = "./checkpoints/restoration"
    ##

    if opt.Quality_restore:
        opt.name = "mapping_quality"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
    if opt.Scratch_and_Quality_restore:
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")
        if opt.HR:
            opt.mapping_exp = 1
            opt.inference_optimize = True
            opt.mask_dilation = 3
            opt.name = "mapping_Patch_Attention"

class GlobalRestoration:
    def __init__(self, device):
        # opt = TestOptions().parse(save=False)
        class OPT:
            gpu_ids = [device]
            print(gpu_ids)
            test_mode = "Full"
            Quality_restore = True
            Scratch_and_Quality_restore = False
            isTrain = False
            resize_or_crop = "scale_width"
            input_nc = 3
            output_nc = 3
            ngf = 64
            norm = "instance"
            spatio_size = 64
            feat_dim = -1
            use_segmentation_model = False
            non_local = ""
            NL_use_mask = False
            mapping_net_dilation = 1
            load_pretrain = ""
            no_load_VAE = False
            use_vae_which_epoch = "latest"
            which_epoch = "latest"

        opt = OPT()
        parameter_set(opt)

        # initilize model
        self.model = Pix2PixHDModel_Mapping()
        self.model.initialize(opt)
        self.model.eval()

        #
        self.img_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.mask_transform = transforms.ToTensor()

    def restore_image(self, input: np.ndarray) -> np.ndarray:
        input = Image.fromarray(input)
        input = data_transforms(input, scale=False)
        origin = input
        input = self.img_transform(input)
        input = input.unsqueeze(0)
        mask = torch.zeros_like(input) 

        with torch.no_grad():
            generated = self.model.inference(input, mask)

        grid = vutils.make_grid((generated.data.cpu() + 1.0) / 2.0, nrow=1, padding=0, normalize=True)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        # cv2.imwrite("test.jpg", ndarr)
        # ndarr = np.ascontiguousarray(ndarr)
        # import ipdb; ipdb.set_trace()
        _, im_arr = cv2.imencode('.jpg', ndarr)
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)
        # cv2.imwrite("test.jpg", ndarr)
        # res = base64.b64encode(ndarr)
        # import ipdb; ipdb.set_trace()
        return im_b64.decode("utf-8")

if __name__ == "__main__":

    tmp = GlobalRestoration(device=0)
    
    input_image = cv2.imread("../test_images/old/a.png")
    res = tmp.restore_image(input_image)
    print(res)
    cv2.imwrite("test.jpg", res)



# if __name__ == "__main__":

#     opt = TestOptions().parse(save=False)
#     parameter_set(opt)

#     model = Pix2PixHDModel_Mapping()

#     model.initialize(opt)
#     model.eval()

#     if not os.path.exists(opt.outputs_dir + "/" + "input_image"):
#         os.makedirs(opt.outputs_dir + "/" + "input_image")
#     if not os.path.exists(opt.outputs_dir + "/" + "restored_image"):
#         os.makedirs(opt.outputs_dir + "/" + "restored_image")
#     if not os.path.exists(opt.outputs_dir + "/" + "origin"):
#         os.makedirs(opt.outputs_dir + "/" + "origin")

#     dataset_size = 0

#     input_loader = os.listdir(opt.test_input)
#     dataset_size = len(input_loader)
#     input_loader.sort()

#     if opt.test_mask != "":
#         mask_loader = os.listdir(opt.test_mask)
#         dataset_size = len(os.listdir(opt.test_mask))
#         mask_loader.sort()

#     img_transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )
#     mask_transform = transforms.ToTensor()

#     for i in range(dataset_size):

#         input_name = input_loader[i]
#         input_file = os.path.join(opt.test_input, input_name)
#         if not os.path.isfile(input_file):
#             print("Skipping non-file %s" % input_name)
#             continue
#         input = Image.open(input_file).convert("RGB")

#         print("Now you are processing %s" % (input_name))

#         if opt.NL_use_mask:
#             mask_name = mask_loader[i]
#             mask = Image.open(os.path.join(opt.test_mask, mask_name)).convert("RGB")
#             if opt.mask_dilation != 0:
#                 kernel = np.ones((3,3),np.uint8)
#                 mask = np.array(mask)
#                 mask = cv2.dilate(mask,kernel,iterations = opt.mask_dilation)
#                 mask = Image.fromarray(mask.astype('uint8'))
#             origin = input
#             input = irregular_hole_synthesize(input, mask)
#             mask = mask_transform(mask)
#             mask = mask[:1, :, :]  ## Convert to single channel
#             mask = mask.unsqueeze(0)
#             input = img_transform(input)
#             input = input.unsqueeze(0)
#         else:
#             if opt.test_mode == "Scale":
#                 input = data_transforms(input, scale=True)
#             if opt.test_mode == "Full":
#                 input = data_transforms(input, scale=False)
#             if opt.test_mode == "Crop":
#                 input = data_transforms_rgb_old(input)
#             origin = input
#             input = img_transform(input)
#             input = input.unsqueeze(0)
#             mask = torch.zeros_like(input)
#         ### Necessary input

#         try:
#             with torch.no_grad():
#                 generated = model.inference(input, mask)
#         except Exception as ex:
#             print("Skip %s due to an error:\n%s" % (input_name, str(ex)))
#             continue

#         if input_name.endswith(".jpg"):
#             input_name = input_name[:-4] + ".png"

#         # image_grid = vutils.save_image(
#         #     (input + 1.0) / 2.0,
#         #     opt.outputs_dir + "/input_image/" + input_name,
#         #     nrow=1,
#         #     padding=0,
#         #     normalize=True,
#         # )
#         # image_grid = vutils.save_image(
#         #     (generated.data.cpu() + 1.0) / 2.0,
#         #     opt.outputs_dir + "/restored_image/" + input_name,
#         #     nrow=1,
#         #     padding=0,
#         #     normalize=True,
#         # )


#         origin.save(opt.outputs_dir + "/origin/" + input_name)