# helper code for semantic face editing using PAE
# should run with run_models.py

import os
import pickle
from math import ceil

import clip
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn.functional import normalize, one_hot

import dnnlib
from arcface import id_loss
from proj_aug_emb import get_pae
from utils import image_tensor_to_pre_clip, run_in_small_batch, parse_args, DEVICE


def main(args):
    print(f"The number of gpus is {torch.cuda.device_count()}")
    print(f"In {__file__}: experiment options: " + " | ".join(f"{k}={v}" for k, v in vars(args).items()))

    force_fp32 = DEVICE == "cpu"
    print("Running experiment on", DEVICE)

    print("Loading CLIP and StyleGAN2 networks ...", end=" ")
    clip_model = clip.load("ViT-B/32", device="cpu")[0].to(DEVICE)
    # freeze CLIP
    for i in clip_model.parameters():
        i.requires_grad = False

    network = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    with dnnlib.util.open_url(network) as f:
        G = pickle.load(f)["G_ema"].to(DEVICE)
        Gsyn = G.synthesis
        Gmap = G.mapping
    model = G if args.domain == "latent" else Gsyn
    if len(args.seeds) >= torch.cuda.device_count():
        model = nn.DataParallel(model)
    truncation_psi = 1.0
    noise_mode = "const"
    print("Done.")

    # generating texts
    num_new_rows = len(args.texts)
    if args.interpolation:  # just use the first two texts (opposite meaning) to show the interpolation
        args.texts = args.texts[:2]

    with torch.no_grad():
        # get labels for CIFAR10
        if args.dataset == "cifar10":
            # class 5 is for dogs
            one_row_labels = one_hot(torch.tensor([5] * len(args.seeds)), num_classes=G.c_dim).to(DEVICE)
            training_labels = one_row_labels.repeat(num_new_rows, 1)
            all_labels = torch.cat([one_row_labels, training_labels])
        else:
            all_labels = one_row_labels = training_labels = None

        print(f"Generating initial {args.domain} codes ...", end=" ")
        initial_codes = torch.cat([torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(DEVICE) for seed in args.seeds])
        if args.domain == "latent":
            model.codes = torch.nn.Parameter(initial_codes.repeat(num_new_rows, 1))
        else:
            initial_codes = run_in_small_batch(Gmap, initial_codes, one_row_labels, truncation_psi=truncation_psi)
            model.codes = torch.nn.Parameter(initial_codes.repeat(num_new_rows, 1, 1))
        print("Done.")

        print(f"Generating target embeddings ...")
        if args.target != "path":
            imgs = run_in_small_batch(model, initial_codes, one_row_labels,
                                      truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=force_fp32)
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
            imgs = image_tensor_to_pre_clip(imgs, clip_model.visual.input_resolution)
            image_features = clip_model.encode_image(imgs)
            text_tokens = clip.tokenize(args.texts).to(DEVICE)
            text_features = normalize(clip_model.encode_text(text_tokens))
            targets, emotion_space_basis = get_pae(args, image_features, text_features)

        print("Done.")

    print(f"Optimizing on {args.domain} codes ...")
    cos_criterion = torch.nn.CosineSimilarity()
    if args.id_loss:
        ori_imgs = run_in_small_batch(model, initial_codes, one_row_labels, truncation_psi=truncation_psi, noise_mode=noise_mode,
                                      force_fp32=force_fp32)
        id_criterion = id_loss.IDLoss().to(DEVICE)
    else:
        ori_imgs = None
        id_criterion = None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-4) if args.optimizer == "adam" else \
        torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    batch_size = 50
    num_batches = ceil(targets.shape[0] / batch_size)
    for epoch in range(args.epoch):
        total_loss = 0
        for batch in range(num_batches):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            gen_imgs = run_in_small_batch(model, model.codes[batch * batch_size:(batch + 1) * batch_size],
                                          training_labels[batch * batch_size:(batch + 1) * batch_size] if training_labels is not None else None,
                                          truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=force_fp32)

            # get image features
            imgs = (gen_imgs * 0.5 + 0.5).clamp(0, 1)  # N, C, H, W, values in [0, 1]
            imgs = image_tensor_to_pre_clip(imgs, clip_model.visual.input_resolution)
            image_features = clip_model.encode_image(imgs)

            # backward in batch
            loss = (-1 * cos_criterion(image_features @ emotion_space_basis, targets[batch * batch_size:(batch + 1) * batch_size])).mean()

            if args.id_loss:
                loss += args.id_loss_coefficient * id_criterion(gen_imgs, ori_imgs).mean()

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        total_loss /= targets.shape[0]
        # print statistics
        if epoch == 0 or (epoch + 1) % args.output_loss_every == 0 or epoch == args.epoch - 1:
            print(f'Epoch {epoch + 1} loss: {total_loss}')
    print("Done.")

    print("Saving the result ...", end=" ")
    with torch.no_grad():
        imgs = run_in_small_batch(model, model.codes, all_labels, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=force_fp32)
        imgs_to_show = (imgs.permute(0, 2, 3, 1) * 0.5 + 0.5).permute(0, 3, 1, 2).clamp(0, 1)[len(args.seeds):]

    for i, img in enumerate(imgs_to_show):
        seed = args.seeds[i % len(args.seeds)]
        image_path = args.out_path_format.format(seed)
        outdir = args.outdir.format(args.texts[i // len(args.seeds)].replace(' ', '_'))
        os.makedirs(outdir, exist_ok=True)
        torchvision.utils.save_image(img, f"{outdir}/{image_path}")
    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
