# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import pickle
import sys

sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip_exp.proj_aug_emb import get_pae
from clip_exp.utils import run_in_small_batch, image_tensor_to_pre_clip

import re
import math
import time
import click
import legacy
from typing import List, Optional
from argparse import Namespace

import clip
import dnnlib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop
from arcface.id_loss import IDLoss
from PIL import Image

from torch_utils import misc
from torch.nn.functional import normalize
from torch_utils.ops import upfirdn2d


def block_forward(self, x, img, ws, shapes, force_fp32=False, fused_modconv=None, **layer_kwargs):
    misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
    w_iter = iter(ws.unbind(dim=1))
    dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
    memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
    if fused_modconv is None:
        with misc.suppress_tracer_warnings():  # this value will be treated as a constant
            fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

    # Input.
    if self.in_channels == 0:
        x = self.const.to(dtype=dtype, memory_format=memory_format)
        x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
    else:
        misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
        x = x.to(dtype=dtype, memory_format=memory_format)

    # Main layers.
    if self.in_channels == 0:
        x = self.conv1(x, next(w_iter)[..., :shapes[0]], fused_modconv=fused_modconv, **layer_kwargs)
    elif self.architecture == 'resnet':
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
        x = y.add_(x)
    else:
        x = self.conv0(x, next(w_iter)[..., :shapes[0]], fused_modconv=fused_modconv, **layer_kwargs)
        x = self.conv1(x, next(w_iter)[..., :shapes[1]], fused_modconv=fused_modconv, **layer_kwargs)

    # ToRGB.
    if img is not None:
        misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
        img = upfirdn2d.upsample2d(img, self.resample_filter)
    if self.is_last or self.architecture == 'skip':
        y = self.torgb(x, next(w_iter)[..., :shapes[2]], fused_modconv=fused_modconv)
        y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        img = img.add_(y) if img is not None else y

    assert x.dtype == dtype
    assert img is None or img.dtype == torch.float32
    return x, img


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def num_range(s: str) -> List[int]:
    """
    Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.
    """

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', default="../pretrained/ffhq.pkl")
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1.0, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--projected-s', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Directory to save the output images', type=str, required=True, metavar='DIR')
@click.option('--path', help='Where to save the output images', type=str, required=True, metavar='FILE')
@click.option('--text-prompt', help='Text', type=str, required=True)
@click.option('--resolution', help='Resolution of output images', type=int, default=256)
@click.option('--batch-size', help='Batch Size', type=int, default=4)
@click.option('--identity-power', help='How much change occurs on the face', type=str, default="high")
@click.option('--target', help='The optimization target',
              type=click.Choice(["text", "pae", "pae+", "paeGS+", "paePCA+", "paeAllEx", "paeAllExD", "dpeGS", "dpePCA"]), default="text")
@click.option('--power', help="augmentation power for projected embedding", type=float, default=9.5)
@click.option("--attribute", help="the face attributes to change", type=click.Choice(["emotion", "eye", "mouth", "gender", "hairstyle"]), default="emotion")
@click.option("--components", help="the number of principle components to approximate the emotion subspace", type=int, default=10)
def generate_images(
        ctx: click.Context,
        network_pkl: str,
        seeds: Optional[List[int]],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        path: str,
        class_idx: Optional[int],
        projected_w: Optional[str],
        projected_s: Optional[str],
        text_prompt: str,
        resolution: int,
        batch_size: int,
        identity_power: str,
        target: str,
        power: float,
        attribute: str,
        components: int
):
    """
    Generate images using pretrained network pickle.

    Examples:
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    num_images = len(seeds)
    args = Namespace(**locals())

    print('Loading networks from "%s"...' % network_pkl)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # Labels
    label = torch.zeros([1, G.c_dim], device=device).requires_grad_()
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model = model.to(device)
    text = clip.tokenize([text_prompt]).to(device)
    text_features = normalize(model.encode_text(text))

    # Generate images
    for i in G.parameters():
        i.requires_grad = True

    mean = torch.as_tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float, device=device)
    std = torch.as_tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float, device=device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    transf = Compose([Resize(224, interpolation=Image.BICUBIC), CenterCrop(224)])

    styles_array = []
    print("seeds:", seeds)
    t1 = time.time()
    for seed_idx, seed in enumerate(seeds):
        if seed == seeds[-1]:
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        ws = G.mapping(z, label, truncation_psi=truncation_psi)

        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, G.synthesis.num_ws, G.synthesis.w_dim])
            ws = ws.to(torch.float32)

            w_idx = 0
            for res in G.synthesis.block_resolutions:
                block = getattr(G.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        styles = torch.zeros(1, 26, 512, device=device)
        styles_idx = 0
        temp_shapes = []
        for res, cur_ws in zip(G.synthesis.block_resolutions, block_ws):
            block = getattr(G.synthesis, f'b{res}')

            if res == 4:
                temp_shape = (block.conv1.affine.weight.shape[0], block.conv1.affine.weight.shape[0], block.torgb.affine.weight.shape[0])
                styles[0, :1, :] = block.conv1.affine(cur_ws[0, :1, :])
                styles[0, 1:2, :] = block.torgb.affine(cur_ws[0, 1:2, :])
                if seed_idx == (len(seeds) - 1):
                    block.conv1.affine = torch.nn.Identity()
                    block.torgb.affine = torch.nn.Identity()
                styles_idx += 2
            else:
                temp_shape = (block.conv0.affine.weight.shape[0], block.conv1.affine.weight.shape[0], block.torgb.affine.weight.shape[0])
                styles[0, styles_idx:styles_idx + 1, :temp_shape[0]] = block.conv0.affine(cur_ws[0, :1, :])
                styles[0, styles_idx + 1:styles_idx + 2, :temp_shape[1]] = block.conv1.affine(cur_ws[0, 1:2, :])
                styles[0, styles_idx + 2:styles_idx + 3, :temp_shape[2]] = block.torgb.affine(cur_ws[0, 2:3, :])
                if seed_idx == (len(seeds) - 1):
                    block.conv0.affine = torch.nn.Identity()
                    block.conv1.affine = torch.nn.Identity()
                    block.torgb.affine = torch.nn.Identity()
                styles_idx += 3
            temp_shapes.append(temp_shape)

        styles = styles.detach()
        styles_array.append(styles)

    resolution_dict = {256: 6, 512: 7, 1024: 8}
    id_coeff_dict = {"high": 2, "medium": 1, "low": 0.1, "none": 0}
    id_coeff = id_coeff_dict[identity_power]
    styles_direction = torch.zeros(1, 26, 512, device=device)
    styles_direction_grad_el2 = torch.zeros(1, 26, 512, device=device)
    styles_direction.requires_grad_()

    print(f"Generating target embeddings ...", end=" ")
    with open(network_pkl, "rb") as f:
        stylegan_model = pickle.load(f)["G_ema"].to(device)

    with torch.no_grad():
        initial_codes = torch.cat([torch.from_numpy(np.random.RandomState(seed).randn(1, stylegan_model.z_dim)).to(device) for seed in seeds])
        imgs = run_in_small_batch(stylegan_model, initial_codes, None, truncation_psi=truncation_psi, noise_mode=noise_mode,
                                  force_fp32=device == "cpu")
        imgs = (imgs.permute(0, 2, 3, 1) * 0.5 + 0.5).permute(0, 3, 1, 2).clamp(0, 1)
        imgs = image_tensor_to_pre_clip(imgs, model.visual.input_resolution)
        image_features = model.encode_image(imgs)
        targets, emotion_space_basis = get_pae(args, image_features, text_features)
        print("Done")

    id_loss = IDLoss().to(device).eval()

    temp_photos = []
    grads = []
    for i in range(math.ceil(len(seeds) / batch_size)):
        # print(i*batch_size, "processed", time.time()-t1)

        styles = torch.vstack(styles_array[i * batch_size:(i + 1) * batch_size]).to(device)
        seed = seeds[i]

        styles_idx = 0
        x2 = img2 = None

        for k, (res, cur_ws) in enumerate(zip(G.synthesis.block_resolutions, block_ws)):
            block = getattr(G.synthesis, f'b{res}')
            if k > resolution_dict[resolution]:
                continue

            if res == 4:
                x2, img2 = block_forward(block, x2, img2, styles[:, styles_idx:styles_idx + 2, :], temp_shapes[k], noise_mode=noise_mode)
                styles_idx += 2
            else:
                x2, img2 = block_forward(block, x2, img2, styles[:, styles_idx:styles_idx + 3, :], temp_shapes[k], noise_mode=noise_mode)
                styles_idx += 3

        img2_cpu = img2.detach().cpu().numpy()
        temp_photos.append(img2_cpu)
        if i > 3:
            continue

        styles2 = styles + styles_direction

        styles_idx = 0
        x = img = None
        for k, (res, cur_ws) in enumerate(zip(G.synthesis.block_resolutions, block_ws)):
            block = getattr(G.synthesis, f'b{res}')
            if k > resolution_dict[resolution]:
                continue
            if res == 4:
                x, img = block_forward(block, x, img, styles2[:, styles_idx:styles_idx + 2, :], temp_shapes[k], noise_mode=noise_mode)
                styles_idx += 2
            else:
                x, img = block_forward(block, x, img, styles2[:, styles_idx:styles_idx + 3, :], temp_shapes[k], noise_mode=noise_mode)
                styles_idx += 3

        identity_loss = id_loss(img, img2)
        identity_loss *= id_coeff
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
        img = (transf(img.permute(0, 3, 1, 2)) / 255).sub_(mean).div_(std)
        image_features = model.encode_image(img)
        cos_sim = -1 * F.cosine_similarity(image_features, (text_features[0]).unsqueeze(0))
        if target == "text":
            (identity_loss.mean() + cos_sim.sum()).backward(retain_graph=True)
        else:
            (identity_loss.mean() + cos_sim.mean()).backward(retain_graph=True)

    t1 = time.time()
    styles_direction.grad[:, list(range(26)), :] = 0
    with torch.no_grad():
        styles_direction *= 0

    for i in range(math.ceil(len(seeds) / batch_size)):
        print(i * batch_size, "processed", time.time() - t1)

        seed = seeds[i]
        styles = torch.vstack(styles_array[i * batch_size:(i + 1) * batch_size]).to(device)
        img2 = torch.tensor(temp_photos[i]).to(device)
        styles2 = styles + styles_direction

        styles_idx = 0
        x = img = None
        for k, (res, cur_ws) in enumerate(zip(G.synthesis.block_resolutions, block_ws)):
            block = getattr(G.synthesis, f'b{res}')
            if k > resolution_dict[resolution]:
                continue

            if res == 4:
                x, img = block_forward(block, x, img, styles2[:, styles_idx:styles_idx + 2, :], temp_shapes[k], noise_mode=noise_mode)
                styles_idx += 2
            else:
                x, img = block_forward(block, x, img, styles2[:, styles_idx:styles_idx + 3, :], temp_shapes[k], noise_mode=noise_mode)
                styles_idx += 3

        identity_loss = id_loss(img, img2)
        identity_loss *= id_coeff
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
        img = (transf(img.permute(0, 3, 1, 2)) / 255).sub_(mean).div_(std)
        image_features = model.encode_image(img)
        cos_sim = -1 * F.cosine_similarity(image_features @ emotion_space_basis, targets[i * batch_size:(i + 1) * batch_size])
        if target == "text":
            (identity_loss.mean() + cos_sim.sum()).backward(retain_graph=True)
        else:
            (identity_loss.mean() + cos_sim.mean()).backward(retain_graph=True)

        styles_direction.grad[:, [0, 1, 4, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], :] = 0

        if i % 2 == 1:
            styles_direction.data = (styles_direction - styles_direction.grad * 5)
            grads.append(styles_direction.grad.clone())
            styles_direction.grad.data.zero_()
            if i > 3:
                styles_direction_grad_el2[grads[-2] * grads[-1] < 0] += 1

    styles_direction = styles_direction.detach()
    styles_direction[styles_direction_grad_el2 > (len(seeds) / batch_size) / 4] = 0

    # output_filepath = f'{outdir}/stylemc_direction_' + text_prompt.replace(" ", "_") + '.npz'
    np.savez(path, s=styles_direction.cpu().numpy())

    print("time passed:", time.time() - t1)


if __name__ == "__main__":
    generate_images()
