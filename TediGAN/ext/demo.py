import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))

from torch.nn.functional import normalize
from clip_exp.utils import image_tensor_to_pre_clip
from clip_exp.proj_aug_emb import get_pae

import argparse
import math
import numpy as np
import torch
import torchvision
from torch import optim
from tqdm import tqdm

from utils.common import CLIPLoss
from models.stylegan2.model import Generator
from base.models.perceptual_model import PerceptualModel
import clip


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="man",
                        help="man for manipulation, gen for generation.")
    parser.add_argument("--description", type=str, required=True,
                        help="the text description")
    parser.add_argument("--ckpt", type=str, default="../pretrained/ffhq-config-f.pt",
                        help="pretrained StyleGAN2 weights")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=200,
                        help="number of optimization steps")
    parser.add_argument("--loss-pix-weight", type=float, default=1.0,
                        help='The pixel reconstruction loss scale for optimization. (default: 1.0)')
    parser.add_argument("--loss-reg-weight", type=float, default=2.0,
                        help='The latent loss scale for optimization. (default: 2.0)')
    parser.add_argument('--loss-feat-weight', type=float, default=5e-5,
                        help='The perceptual loss scale for optimization. (default: 5e-5)')
    parser.add_argument('--loss-clip-weight', type=float, default=1,
                        help='The clip loss for optimization. (default: 2.0)')
    parser.add_argument('--f-oom', type=bool, default=False,
                        help='if you have the Out-of-Memory problem, set as True. (default: False)')
    parser.add_argument("--latent-path", type=str, required=True,
                        help="starts the optimization from the given latent code. Expects a .pt format")
    parser.add_argument("--save-intermediate-image-every", type=int, default=0,
                        help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--outdir", help="the directory to save the output images", type=str, required=True)
    parser.add_argument("--image-path", help="the path to the image, not including the directory", type=str, required=True)
    parser.add_argument("--target", help="the optimization target",
                        choices=["text", "pae", "pae+", "paeGS+", "paePCA+", "paeAllEx", "paeAllExD", "dpeGS", "dpePCA", "path"], default="text")
    parser.add_argument("--power", help="augmentation power for projected embedding", type=float, default=9.5)
    parser.add_argument("--num-images", help="number of images to edit", type=int, default=1)
    parser.add_argument("--attribute", help="the face attributes to change", choices=["emotion", "eye", "mouth", "gender", "hairstyle"], default="emotion")
    parser.add_argument("--components", help="the number of principle components to approximate the emotion subspace", type=int, default=10)
    return parser.parse_args()


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def _get_tensor_value(tensor):
    """Gets the value of a torch Tensor."""
    return tensor.cpu().detach().numpy()


def main(args):
    text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()
    os.makedirs(args.outdir, exist_ok=True)

    F = PerceptualModel(min_val=-1.0, max_val=1.0)

    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    z_mean = g_ema.mean_latent(4096)
    # z_load = np.load(args.latent_path)
    # z_init = torch.from_numpy(z_load).cuda()
    # print(np.shape(latent_load))
    F_OOM = args.f_oom

    if args.mode == "man":
        z_init = torch.tensor(np.load(args.latent_path)['w']).cuda()
    else:
        z_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            _, z_init = g_ema([z_init_not_trunc], truncation_latent=z_mean, return_latents=True,
                              truncation=0.7)

    x, _ = g_ema([z_init], input_is_latent=True, randomize_noise=False)

    # z = z_init.detach().clone()
    z = z_mean.detach().clone().repeat(1, 18, 1)

    z.requires_grad = True

    clip_loss = CLIPLoss()

    optimizer = optim.Adam([z], lr=args.lr)

    pbar = tqdm(range(args.step))
    cos_criterion = torch.nn.CosineSimilarity()

    if args.target != "text":
        # projected embedding
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model = clip.load("ViT-B/32", device="cpu")[0].to(device)
            text_features = normalize(clip_model.encode_text(text_inputs))
            imgs = image_tensor_to_pre_clip(x, clip_model.visual.input_resolution)
            image_features = clip_model.encode_image(imgs)
            targets, emotion_space_basis = get_pae(args, image_features, text_features)

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        x_rec, _ = g_ema([z], input_is_latent=True, randomize_noise=False)
        if not F_OOM:
            loss = 0.0
            # Reconstruction loss.
            loss_pix = torch.mean((x - x_rec) ** 2)
            loss = loss + loss_pix * args.loss_pix_weight
            log_message = f'loss_pix: {_get_tensor_value(loss_pix):.3f}'

            # Perceptual loss.
            if args.loss_feat_weight:
                x_feat = F.net(x)
                x_rec_feat = F.net(x_rec)
                loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
                loss = loss + loss_feat * args.loss_feat_weight
                log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'

            # Regularization loss.
            if args.loss_reg_weight:
                loss_reg = torch.mean((z_init - z) ** 2)
                # loss_reg = ((z_init - z) ** 2).sum()
                loss = loss + loss_reg * args.loss_reg_weight
                log_message += f', loss_reg: {_get_tensor_value(loss_reg):.3f}'

            # CLIP loss.
            if args.loss_clip_weight:
                if args.target == "text":
                    loss_clip = clip_loss(x_rec, text_inputs)[0][0]
                else:
                    # get image features
                    imgs = image_tensor_to_pre_clip(x_rec, clip_model.visual.input_resolution)
                    image_features = clip_model.encode_image(imgs)

                    loss_clip = (-1 * cos_criterion(image_features @ emotion_space_basis, targets))[0]

                loss = loss + loss_clip * args.loss_clip_weight
                log_message += f', loss_clip: {_get_tensor_value(loss_clip):.3f}'
        else:
            loss_reg = ((z_init - z) ** 2).sum()
            loss_clip = clip_loss(x_rec, text_inputs)
            loss = loss_reg + loss_clip[0][0] * args.loss_clip_weight  # set loss_clip_weight as 200 in my case.

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        pbar.set_description((f"loss: {loss.item():.4f};"))

    # final_result = torch.cat([x, x_rec])
    # return final_result
    return x_rec


if __name__ == "__main__":
    args = parse_args()
    result_image = main(args)
    torchvision.utils.save_image(result_image.detach().cpu(), os.path.join(
        args.outdir, args.image_path), normalize=True, scale_each=True, value_range=(-1, 1))
