import argparse
import math
import os
import pickle
import sys

sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clip
import numpy as np
import torch
import torchvision
from torch import optim
from torch.nn.functional import normalize
from tqdm import tqdm

from clip_exp.proj_aug_emb import get_pae
from clip_exp.utils import image_tensor_to_pre_clip
from criteria.clip_loss import CLIPLoss
from mapper.training.train_utils import STYLESPACE_DIMENSIONS
from models.stylegan2.model import Generator
from utils import ensure_checkpoint_exists

from arcface.id_loss import IDLoss

STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def main(args):
    ensure_checkpoint_exists(args.ckpt)
    text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()
    os.makedirs(args.outdir, exist_ok=True)

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    if args.latent_path:
        latent_code_init = torch.tensor(np.load(args.latent_path)['w']).cuda()
    elif args.seed:
        device = torch.device('cuda')
        with open("../pretrained/ffhq.pkl", "rb") as f:
            G = pickle.load(f)["G_ema"].to(device)
        z = torch.from_numpy(np.random.RandomState(args.seed).randn(1, G.z_dim)).to(device)
        label = torch.zeros([1, G.c_dim], device="cuda").requires_grad_()
        latent_code_init = G.mapping(z, label, truncation_psi=0.7)
    elif args.mode == "edit":
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                           truncation=args.truncation, truncation_latent=mean_latent)
    else:
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

    with torch.no_grad():
        img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

        if args.target != "text":
            # projected embedding
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model = clip.load("ViT-B/32", device="cpu")[0].to(device)
            text_features = normalize(clip_model.encode_text(text_inputs))
            imgs = image_tensor_to_pre_clip(img_orig, clip_model.visual.input_resolution)
            image_features = clip_model.encode_image(imgs)
            targets, emotion_space_basis = get_pae(args, image_features, text_features)

    if args.work_in_stylespace:
        with torch.no_grad():
            _, _, latent_code_init = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
        latent = [s.detach().clone() for s in latent_code_init]
        for c, s in enumerate(latent):
            if c in STYLESPACE_INDICES_WITHOUT_TORGB:
                s.requires_grad = True
    else:
        latent = latent_code_init.detach().clone()
        latent.requires_grad = True

    clip_loss = CLIPLoss(args)
    id_loss = IDLoss()

    if args.work_in_stylespace:
        optimizer = optim.Adam(latent, lr=args.lr)
    else:
        optimizer = optim.Adam([latent], lr=args.lr)

    pbar = tqdm(range(args.step))
    cos_criterion = torch.nn.CosineSimilarity()

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)

        if args.target == "text":
            c_loss = clip_loss(img_gen, text_inputs)
        else:
            # get image features
            imgs = image_tensor_to_pre_clip(img_gen, clip_model.visual.input_resolution)
            image_features = clip_model.encode_image(imgs)

            c_loss = (-1 * cos_criterion(image_features @ emotion_space_basis, targets)).mean()

        if args.id_lambda > 0:
            i_loss = id_loss(img_gen, img_orig).mean()
        else:
            i_loss = 0

        if args.mode == "edit":
            if args.work_in_stylespace:
                l2_loss = sum([((latent_code_init[c] - latent[c]) ** 2).sum() for c in range(len(latent_code_init))])
            else:
                l2_loss = ((latent_code_init - latent) ** 2).sum()
            loss = c_loss + args.l2_lambda * l2_loss + args.id_lambda * i_loss
        else:
            loss = c_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        if args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
            with torch.no_grad():
                img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)

            torchvision.utils.save_image(img_gen, os.path.join(args.outdir, f"{str(i).zfill(5)}.jpg"), normalize=True, value_range=(-1, 1))

    # if args.mode == "edit":
    #     final_result = torch.cat([img_orig, img_gen])
    # else:
    #     final_result = img_gen

    return img_gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default="a person with purple hair", help="the text that guides the editing/generation")
    parser.add_argument("--ckpt", type=str, default="../pretrained/ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan-size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr-rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=300, help="number of optimization steps")
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"],
                        help="choose between edit an image an generate a free one")
    parser.add_argument("--l2-lambda", type=float, default=0.008, help="weight of the latent distance (used for editing only)")
    parser.add_argument("--id-lambda", type=float, default=0.000, help="weight of id loss (used for editing only)")
    parser.add_argument("--latent-path", type=str, default=None, help="starts the optimization from the given latent code if provided."
                                                                      "Otherwise, starts from the seed if provided"
                                                                      "Otherwise, starts from the mean latent in a free generation, and from a random one in editing. "
                                                                      "Expects a .pt format")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--truncation", type=float, default=1.0, help="used only for the initial latent vector, and only when a latent code path is"
                                                                      "not provided")
    parser.add_argument('--work-in-stylespace', default=False, action='store_true')
    parser.add_argument("--save-intermediate-image-every", type=int, default=0, help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--outdir", help="the directory to save the output images", type=str, required=True)
    parser.add_argument("--image-path", help="the path to the image, not including the directory", type=str, required=True)
    parser.add_argument("--target", help="the optimization target",
                        choices=["text", "pae", "pae+", "paeGS+", "paePCA+", "paeAllEx", "paeAllExD", "dpeGS", "dpePCA", "path"], default="text")
    parser.add_argument("--power", help="augmentation power for projected embedding", type=float, default=9.5)
    parser.add_argument("--num-images", help="number of images to edit", type=int, default=1)
    parser.add_argument("--attribute", help="the face attributes to change", choices=["emotion", "eye", "mouth", "gender", "hairstyle"], default="emotion")
    parser.add_argument("--components", help="the number of principle components to approximate the emotion subspace", type=int, default=10)

    args = parser.parse_args()

    result_image = main(args)

    torchvision.utils.save_image(result_image.detach().cpu(), os.path.join(args.outdir, args.image_path), normalize=True, scale_each=True,
                                 value_range=(-1, 1))
