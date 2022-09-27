import argparse
import warnings
from argparse import ArgumentTypeError
from os import path

import clip
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, normalize, center_crop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMOTION_DEFAULT_SEEDS = [6600, 6602, 6604, 6605, 6606, 6607, 6608, 6610, 6612, 6613, 6619, 6621]
EMOTION_DEFAULT_TEXTS = ["a happy face", "a sad face", "an angry face", "a surprised face"]
HAIRSTYLE_DEFAULT_SEEDS = [6600, 6604, 6605, 6608, 6609, 6610, 6613, 6614, 6616, 6618, 6622, 6627, 6632, 6633]
HAIRSTYLE_DEFAULT_TEXTS = ["bald", "curly hair", "blonde", "black hair", "grey hair"]
PHYSICAL_DEFAULT_SEEDS = [6604, 6606, 6608, 6610, 6611, 6617, 6618, 6624, 6627, 6629, 6630]
EYE_DEFAULT_TEXTS = ["large eyes", "small eyes"]
MOUTH_DEFAULT_TEXTS = ["large mouth", "small mouth"]
ALL_DEFAULT_SEEDS = list(range(6600, 6634))
DISPLAY_INCH_PER_IMG = 1.2


# project v onto the direction of u
def project_to_vector(v, u):
    return (u.dot(v) / u.dot(u)) * u.clone()


# perform Gram-Schmidt process to make a set of vectors orthonormal
# each row of vv is a given vector
@torch.no_grad()
def gram_schmidt(vv):
    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device, dtype=vv.dtype)
    uu[0] += vv[0]
    for k in range(1, nk):
        uu[k] += vv[k]
        for j in range(0, k):
            uu[k] -= project_to_vector(vv[k], uu[j])
        uu[k] /= uu[k].norm()
    return uu


# given an image in pytorch tensor format (N, C, H, W), preprocess it so that it is ready to be fed into clip.encode_image
def image_tensor_to_pre_clip(imgs, res=224):
    imgs = resize(imgs, size=res, interpolation=InterpolationMode.BICUBIC)
    imgs = center_crop(imgs, output_size=res)
    return normalize(imgs, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])


# This is a wrapper to run the model with arbitrary batch size.
# It calls the actual model with batch_split_size each time (the last batch is larger)
# StyleGAN2 can only take a maximum of 60 FFHQ faces at one time. it will throw an error if exceeding this number
def run_in_small_batch(model, *inputs, batch_split_size=50, **kwargs):
    if type(model).__name__ == "MappingNetwork":  # it is either Generator or SynthesisNetwork or MappingNetwork
        kwargs.pop("noise_mode", None)
        kwargs.pop("force_fp32", None)
    elif type(model).__name__ == "SynthesisNetwork":
        kwargs.pop("truncation_psi", None)
        inputs = inputs[:1]
    if len(inputs) == 2 and inputs[1] is None:  # FFHQ case
        batches = inputs[0].split(batch_split_size)

        # this code is to avoid situations when the size last batch < the number of GPU (typically =4 in data parallel)
        if batches[-1].shape[0] < 10:
            batches = list(batches)
            batches[-2] = torch.cat(batches[-2:])
            batches = batches[:-1]

        return torch.cat([model(b, None, **kwargs) for b in batches])
    return torch.cat([model(*b, **kwargs) for b in zip(*(input.split(batch_split_size) for input in inputs))])


@torch.no_grad()
def get_embeddings_from_text_file(filename_prefix):
    if path.exists(filename_prefix + ".pt"):
        return torch.load(filename_prefix + ".pt", map_location=DEVICE)

    # No precomputed embeddings. Compute now
    with open(filename_prefix + ".txt") as f:
        all_embeddings = [line.lower().rstrip() for line in f]
    clip_model = clip.load("ViT-B/32", device="cpu")[0].to(DEVICE)
    all_embeddings = clip.tokenize(all_embeddings).to(DEVICE)
    all_embeddings = clip_model.encode_text(all_embeddings)
    torch.save(all_embeddings, filename_prefix + ".pt")
    return all_embeddings


@torch.no_grad()
def get_pae_PCA_basis(n_components=10, attribute="emotion"):
    basis_path = f"data/{attribute}_space_basis_{n_components}.pt"
    if path.exists(basis_path):
        return torch.load(basis_path, map_location=DEVICE)

    # No precomputed basis. Compute now
    all_embeddings = get_embeddings_from_text_file(f"data/{attribute}")
    type_before = all_embeddings.dtype
    all_embeddings = StandardScaler().fit_transform(all_embeddings.cpu().numpy())
    pca = PCA(n_components=n_components)
    pca.fit(all_embeddings)
    basis = torch.from_numpy(pca.components_).to(DEVICE).to(type_before)
    torch.save(basis, basis_path)
    return basis


# number range for arg parser
def argparse_numrange(s):
    ret = []
    groups = s.split(",")
    for group in groups:
        try:
            a = [int(n) for n in group.split("-")]
            if len(a) == 1:
                ret.append(a[0])
            else:
                ret += list(range(a[0], a[1] + 1))
        except:
            raise ArgumentTypeError(f"'{s}' is not a range of number. Expected forms like '0-5', '2' or '3,4,6-10'.")
    return ret


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic face editing with PAE")
    parser.add_argument("-t", "--test", help="whether to run in the testing mode", action="store_true")
    parser.add_argument("--method", help="name of the method to run", choices=["stylemc", "styleclip", "ours", "tedigan"], default="ours")
    parser.add_argument("--epoch", help="number of epochs for training", type=int, default=10000)
    parser.add_argument("--seeds", help="the StyleGAN seeds for images to edit", type=argparse_numrange)
    parser.add_argument("--dataset", help="dataset on which the StyleGAN2 is pretrained on", choices=["cifar10", "ffhq"], default="ffhq")
    parser.add_argument("--domain", help="optimization domain", choices=["latent", "style"], default="latent")
    parser.add_argument("--optimizer", help="optimizer to use", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
    parser.add_argument("--target", help="optimization target",
                        choices=["text", "pae", "pae+", "paeGS+", "paePCA+", "paeAllEx", "paeAllExD", "dpeGS", "dpePCA"], default="text")
    parser.add_argument("--target-path", help="the path to the target, pt format (pytorch)", type=str)
    parser.add_argument("--power", help="augmentation power for projected embedding", type=float, default=8.0)
    parser.add_argument("--components", help="the number of principle components to approximate the emotion subspace", type=int, default=10)
    parser.add_argument("-i", "--interpolation", help="whether to display the interpolation of the first two texts", action="store_true")
    parser.add_argument("--attribute", help="the face attributes to change", choices=["emotion", "eye", "mouth", "hairstyle"],
                        default="emotion")
    parser.add_argument("-l", "--id-loss", help="whether to add the ID loss in addition to the CLIP loss", action="store_true")
    parser.add_argument("--id-loss-coefficient", help="the coefficient of the id loss", type=float, default=0.1)
    parser.add_argument('--texts', help="the text prompts", nargs='*')
    parser.add_argument("--outdir", help="the dir of the output image", type=str, default="output")
    parser.add_argument("--out-path-format", help="the format of name of the output image, should contain a {} to be formatted with the seed.",
                        type=str)
    parser.add_argument("--output-loss-every", help="the frequency of outputting loss in the number of iterations", type=int, default=200)
    parser.add_argument("--show-plot", help="whether to show the plot (default: False)", action="store_true")
    args = parser.parse_args()

    # set default texts
    if args.texts is None:
        if args.dataset == "ffhq":
            if args.attribute == "emotion":
                args.texts = EMOTION_DEFAULT_TEXTS
            elif args.attribute == "hairstyle":
                args.texts = HAIRSTYLE_DEFAULT_TEXTS
            elif args.attribute == "eye":
                args.texts = EYE_DEFAULT_TEXTS
            elif args.attribute == "mouth":
                args.texts = MOUTH_DEFAULT_TEXTS
            else:
                raise NotImplementedError
        else:
            args.texts = ["a white dog", "a black dog", "a dog with large eyes", "a dog  with little ears", "a running dog", "a sitting dog"]

    # set default seeds
    if args.seeds is None and args.dataset == "ffhq":
        if args.attribute == "emotion":
            args.seeds = EMOTION_DEFAULT_SEEDS
        elif args.attribute == "hairsytle":
            args.seeds = HAIRSTYLE_DEFAULT_SEEDS
        elif args.attribute == "mouth" or args.attribute == "eye":
            args.seeds = PHYSICAL_DEFAULT_SEEDS
        else:
            args.seeds = ALL_DEFAULT_SEEDS

    # test mode
    if args.test:
        args.seeds = [args.seeds[0]]
        args.texts = [args.texts[0]]
        args.epoch = 3

    # check
    if "ExD" in args.target and args.power % 1:
        warnings.warn(f"When args.target={args.target}, args.power={args.power} is floored to {int(args.power)}")

    return args
