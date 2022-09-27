import clip
import torch
from torch.nn.functional import normalize

from utils import get_embeddings_from_text_file, gram_schmidt, get_pae_PCA_basis, DEVICE


# given batches of image features I and text features T,
# return targets (len(targets)=len(T)*len(I)) and the subspace basis
# targets[i*len(I) + j] is the projected embedding from T[i] and I[j]
@torch.no_grad()
def get_pae(args, image_features, text_features):
    model = clip.load("ViT-B/32", device="cpu")[0].to(DEVICE)
    emotion_space_basis = torch.eye(text_features.shape[1], device=DEVICE)

    if "text" in args.target or "dpe" in args.target:
        targets = text_features.repeat_interleave(image_features.shape[0], dim=0)
        if "dpe" in args.target:
            if args.attribute == "emotion":
                if args.target == "dpeGS":
                    six_emotions_texts = ["a happy face", "a sad face", "an angry face", "a fearful face", "a surprised face", "a disgusted face"]
                    six_emotions = clip.tokenize(six_emotions_texts).to(DEVICE)
                    six_emotions_features = normalize(model.encode_text(six_emotions))
                    emotion_space_basis = gram_schmidt(six_emotions_features).T
                elif args.target == "dpePCA":
                    emotion_space_basis = get_pae_PCA_basis(n_components=args.components, attribute=args.attribute).T
                targets @= emotion_space_basis
            else:
                raise NotImplementedError
    else:
        # projection
        if "All" in args.target:
            all_emotion_embeddings = get_embeddings_from_text_file(f"data/{args.attribute}")
            sim_scores = normalize(image_features) @ normalize(all_emotion_embeddings).T
            if "ExD" in args.target:
                sort_indices = sim_scores.sort(descending=True).indices
                to_deduct_per_img = torch.stack(
                    [all_emotion_embeddings[each_sort_indices[:int(args.power)]].sum() for each_sort_indices in sort_indices])
            else:
                max_indices = sim_scores.max(dim=-1).indices
                to_deduct_per_img = torch.stack(
                    [args.power * all_emotion_embeddings[each_max_indices] for each_max_indices in max_indices])
        elif "PCA" in args.target:
            subspace_basis = get_pae_PCA_basis(n_components=args.components, attribute=args.attribute)
            text_coeff_sum = (text_features @ subspace_basis.T).sum(dim=-1)
        else:
            if args.attribute == "emotion":
                semantic_basis_text = ["a happy face", "a sad face", "an angry face", "a fearful face", "a surprised face", "a disgusted face"]
            elif args.attribute == "eye":
                semantic_basis_text = ["large eyes", "small eyes"]
            elif args.attribute == "mouth":
                semantic_basis_text = ["large mouth", "small mouth"]
            else:
                raise NotImplementedError

            subspace_basis = clip.tokenize(semantic_basis_text).to(DEVICE)
            subspace_basis = normalize(model.encode_text(subspace_basis))
            text_coeff_sum = torch.ones(text_features.shape[0], device=DEVICE, dtype=text_features.dtype)

            if "GS" in args.target:
                subspace_basis = gram_schmidt(subspace_basis)
                text_coeff_sum = (text_features @ subspace_basis.T).sum(dim=-1)
        image_coeff = image_features @ subspace_basis.T
        image_coeff = args.power * abs(image_coeff)

        # augmentation
        targets = []
        for i, text_feature in enumerate(text_features):
            for j, image_feature in enumerate(image_features):
                target = image_feature.clone()
                if "Ex" in args.target:
                    target = image_feature + args.power * text_feature - to_deduct_per_img[j]
                elif "+" in args.target:
                    shift = 0
                    for k, basis_vector in enumerate(subspace_basis):
                        coeff = image_coeff[j][k]
                        shift += coeff
                        target -= coeff * basis_vector
                    target += shift / text_coeff_sum[i] * text_feature
                else: # pae
                    target += args.power * text_feature
                targets.append(target)
        targets = torch.stack(targets)
    return targets, emotion_space_basis
