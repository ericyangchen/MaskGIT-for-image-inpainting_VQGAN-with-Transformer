import torch
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs["VQ_Configs"])

        self.num_image_tokens = configs["num_image_tokens"]
        self.mask_token_id = configs["num_codebook_vectors"]
        self.choice_temperature = configs["choice_temperature"]
        self.gamma = self.gamma_func(configs["gamma_type"])
        print(f"Using {configs['gamma_type']} mask scheduling function")
        self.transformer = BidirectionalTransformer(configs["Transformer_param"])

    def load_transformer_checkpoint(self, load_ckpt_path):
        print(f"Loading transformer checkpoint from {load_ckpt_path}")

        VQGANTransformer_state_dict = torch.load(load_ckpt_path)

        transformer_state_dict = {
            k.replace("transformer.", ""): v
            for k, v in VQGANTransformer_state_dict.items()
            if k.startswith("transformer.")
        }

        self.transformer.load_state_dict(transformer_state_dict)

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs["VQ_config_path"], "r"))
        model = VQGAN(cfg["model_param"])
        model.load_state_dict(torch.load(configs["VQ_CKPT_path"]), strict=True)
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        # z: b,c,h,w, z_indices: b*c
        z, z_indices, _ = self.vqgan.encode(x)

        # reshape z_indices to (b, c)
        z_indices = z_indices.reshape(z.shape[0], z.shape[1])

        return z, z_indices

    def gamma_func(self, mode):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1].
        During training, the input ratio is uniformly sampled;
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.

        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        def linear_schedule(t_T):
            return 1 - t_T

        def cosine_schedule(t_T):
            return (1 + math.cos(math.pi * t_T)) * 0.5

        def square_schedule(t_T):
            return 1 - t_T**2

        if mode == "linear":
            return linear_schedule
        elif mode == "cosine":
            return cosine_schedule
        elif mode == "square":
            return square_schedule
        else:
            raise NotImplementedError

    def forward(self, x):
        z, z_indices = self.encode_to_z(x)

        # randomly mask tokens for training
        ratio = np.random.rand()
        mask = torch.rand_like(z_indices.float()) < ratio
        masked_z_indices = z_indices.clone()
        masked_z_indices[mask] = self.mask_token_id

        logits = self.transformer(masked_z_indices)

        return logits, z_indices

    @torch.no_grad()
    def inpainting(self, t_T, z_indices, mask, initial_mask_num):
        # mask z_indices with current mask
        masked_z_indices = z_indices.clone()
        masked_z_indices[mask] = self.mask_token_id

        # predict new z_indices
        logits = self.transformer(masked_z_indices)

        # convert to probability
        logits = torch.softmax(logits, dim=-1)

        # FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1)

        # turn t/T to mask scheduling ratio
        ratio = self.gamma(t_T)

        # gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob)))

        # predicted probabilities add temperature annealing gumbel noise as confidence
        temperature = self.choice_temperature * ratio
        confidence = z_indices_predict_prob + temperature * gumbel_noise

        confidence[~mask] = float("-inf")

        sorted_confidence, indices = torch.sort(confidence, descending=True)

        # get the number of new tokens that can be updated in this iteration
        total_should_mask_num = initial_mask_num * ratio
        num_of_new_prediction = int(
            (torch.sum(mask == True) - total_should_mask_num).item()
        )

        # update newly predicted tokens with mask = False
        updated_mask = mask.clone()
        updated_mask[0][indices[0][:num_of_new_prediction]] = False

        return z_indices_predict, updated_mask


__MODEL_TYPE__ = {"MaskGit": MaskGit}
