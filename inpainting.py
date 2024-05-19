import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import argparse
from utils import LoadTestData, LoadMaskData
from torch.utils.data import Dataset, DataLoader
from torchvision import utils as vutils
import os
from models import MaskGit as VQGANTransformer
import yaml
import torch.nn.functional as F
from tqdm import tqdm

# for MaskGit inpainting
class MaskGIT:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(
            device=args.device
        )
        self.model.load_transformer_checkpoint(args.load_transformer_ckpt_path)
        self.model.eval()
        self.total_iter = args.total_iter
        self.mask_func = args.mask_func
        self.sweet_spot = args.sweet_spot
        self.device = args.device
        self.output_path = args.output_path
        self.prepare()

    def prepare(self):
        os.makedirs(f"{self.output_path}/test_results", exist_ok=True)
        os.makedirs(f"{self.output_path}/mask_scheduling", exist_ok=True)
        os.makedirs(f"{self.output_path}/imga", exist_ok=True)

    # mask_b: iteration decoding initial mask, where mask_b is true means mask
    def inpainting(self, image, mask_b, i):  # MakGIT inference
        # final decoded image
        dec_img_ori = None

        # save all iterations of masks in latent domain
        maska = torch.zeros(self.total_iter, 3, 16, 16)

        # save all iterations of decoded images
        imga = torch.zeros(self.total_iter + 1, 3, 64, 64)

        # normalize the image
        mean = torch.tensor([0.4868, 0.4341, 0.3844], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.2620, 0.2527, 0.2543], device=self.device).view(3, 1, 1)
        ori = (image[0] * std) + mean

        # mask the first image be the ground truth of masked image
        imga[0] = ori

        self.model.eval()
        with torch.no_grad():
            # total number of mask token
            mask_num = mask_b.sum()

            # mask
            current_mask = mask_b.to(device=self.device)

            # get initial masked z_indices: masked tokens (b,16*16)
            _, z_indices = self.model.encode_to_z(image)
            masked_z_indices = z_indices
            masked_z_indices[current_mask] = self.model.mask_token_id

            z_indices_predict = masked_z_indices

            # iterative decoding
            for step in range(self.total_iter):
                if step == self.sweet_spot:
                    break

                # t/T here is the (step + 1)/total_iter, it'll then be scaled using the gamma func (mask scheduling function)
                t_T = (step + 1) / self.total_iter

                z_indices_predict, current_mask = self.model.inpainting(
                    t_T, z_indices_predict, current_mask, mask_num
                )

                # save current_mask
                mask_i = current_mask.view(1, 16, 16)
                mask_image = torch.ones(3, 16, 16)
                indices = torch.nonzero(mask_i, as_tuple=False)  # label mask true
                mask_image[:, indices[:, 1], indices[:, 2]] = 0  # 3,16,16
                maska[step] = mask_image

                # save current z_indices_predict decoded image
                shape = (1, 16, 16, 256)
                z_q = self.model.vqgan.codebook.embedding(z_indices_predict).view(shape)
                z_q = z_q.permute(0, 3, 1, 2)
                decoded_img = self.model.vqgan.decode(z_q)
                dec_img_ori = (decoded_img[0] * std) + mean
                imga[step + 1] = dec_img_ori

            # decoded image of the sweet spot only, the test_results folder path will be the --predicted-path for fid score calculation
            vutils.save_image(
                dec_img_ori,
                os.path.join(f"{self.output_path}/test_results", f"image_{i:03d}.png"),
                nrow=1,
            )

            # demo score
            vutils.save_image(
                maska,
                os.path.join(f"{self.output_path}/mask_scheduling", f"test_{i}.png"),
                nrow=10,
            )
            vutils.save_image(
                imga, os.path.join(f"{self.output_path}/imga", f"test_{i}.png"), nrow=7
            )


class MaskedImage:
    def __init__(self, args):
        mi_ori = LoadTestData(root=args.test_maskedimage_path, partial=args.partial)
        self.mi_ori = DataLoader(
            mi_ori,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=False,
        )
        mask_ori = LoadMaskData(root=args.test_mask_path, partial=args.partial)
        self.mask_ori = DataLoader(
            mask_ori,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=False,
        )
        self.device = args.device

    def get_mask_latent(self, mask):
        downsampled1 = torch.nn.functional.avg_pool2d(mask, kernel_size=2, stride=2)
        resized_mask = torch.nn.functional.avg_pool2d(
            downsampled1, kernel_size=2, stride=2
        )
        resized_mask[resized_mask != 1] = 0  # 1,3,16*16   check use
        mask_tokens = (resized_mask[0][0] // 1).flatten()  ##[256] =16*16 token
        mask_tokens = mask_tokens.unsqueeze(0)
        mask_b = torch.zeros(mask_tokens.shape, dtype=torch.bool, device=self.device)
        mask_b |= mask_tokens == 0  # true means mask
        return mask_b


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="MaskGIT for Inpainting")
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for MaskGIT')
    
    # output path
    parser.add_argument('--output-path', type=str, default='', required=True, help='output path')
    # transformer ckpt path
    parser.add_argument('--load-transformer-ckpt-path', type=str, default='', required=True, help='load ckpt')
    
    # dataset path
    parser.add_argument('--test-maskedimage-path', type=str, default='data/cat_face/masked_image', help='Path to testing image dataset.')
    parser.add_argument('--test-mask-path', type=str, default='data/mask64', help='Path to testing mask dataset.')
    # MVTM parameter
    parser.add_argument('--sweet-spot', type=int, default=8, help='sweet spot: the best step in total iteration')
    parser.add_argument('--total-iter', type=int, default=12, help='total step for mask scheduling')
    parser.add_argument('--mask-func', type=str, default='cosine', choices=["linear", "cosine", "square"], help='mask scheduling function')
    # fmt: on

    args = parser.parse_args()

    # save args
    os.makedirs(args.output_path, exist_ok=True)
    with open(f"{args.output_path}/inpainting_args.yml", "w") as f:
        yaml.safe_dump(vars(args), f)

    t = MaskedImage(args)
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, "r"))

    # update mask schedluing function
    MaskGit_CONFIGS["model_param"]["gamma_type"] = args.mask_func

    maskgit = MaskGIT(args, MaskGit_CONFIGS)

    # inpainting
    i = 0
    for image, mask in tqdm(zip(t.mi_ori, t.mask_ori), total=len(t.mi_ori)):
        image = image.to(device=args.device)
        mask = mask.to(device=args.device)
        mask_b = t.get_mask_latent(mask)
        maskgit.inpainting(image, mask_b, i)
        i += 1
