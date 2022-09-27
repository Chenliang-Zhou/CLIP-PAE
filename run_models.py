# semantic face editing using PAE

import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'

from semantic_editing import main as pae_semantic_editing
from utils import parse_args


def main(args):
    print(f"In {__file__}: experiment options: " + " | ".join(f"{k}={v}" for k, v in vars(args).items()))

    outdir_format = f"{args.outdir}/{{}}"
    image_path_format = f"{args.out_path_format}_{{}}Seed.png"

    if args.method == "ours":
        args.outdir = outdir_format
        args.out_path_format = image_path_format
        pae_semantic_editing(args)
    else:
        for text in args.texts:
            outdir = outdir_format.format(text.replace(' ', '_'))
            os.makedirs(outdir, exist_ok=True)

            if args.method == "stylemc":
                direction_path = f"data/stylemc_direction_{text.replace(' ', '_')}_{args.target}Target.npz"
                if not os.path.exists(direction_path):
                    print("Generating edit direction ...", end=" ")
                    if args.target == "text":
                        os.system(f"python ../stylemc/find_direction.py --text-prompt '{text}'"
                                  f" --outdir 'data' --path {direction_path} --seeds 1-10 --target {args.target} --power {args.power}")
                    else:
                        os.system(f"python ../stylemc/find_direction.py --text-prompt '{text}' --attribute {args.attribute}"
                                  f" --outdir 'data' --path {direction_path} --seeds 1-10 --target {args.target} --power {args.power}"
                                  f" --components {args.components}")
                    print("Done")
                else:
                    print("Edit direction already exists")

            for seed in args.seeds:
                print("Seed", seed)
                image_path = image_path_format.format(seed)
                latent_path = f"data/seed{seed}_latent_code.npz"

                # generate latent code
                if not os.path.exists(latent_path):
                    print("Generating latent code ...", end=" ")
                    os.system(f"python ../stylemc/generate_w.py --seeds {seed}")
                    print("Done")
                else:
                    print("Latent code already exists")

                if args.method == "stylemc":
                    style_path = f"data/seed{seed}_style_code.npz"
                    if not os.path.exists(style_path):
                        print("Generating style code ...", end=" ")
                        os.system(f"python ../stylemc/w_s_converter.py --outdir 'data' --projected-w {latent_path}")
                        print("Done")
                    else:
                        print("Style code already exists")
                    os.system(f"python ../stylemc/generate_fromS.py --text-prompt '{text}'"
                              f" --outdir {outdir} --image-path {image_path} --s-input {style_path} --direction-input {direction_path}")
                elif args.method == "styleclip":
                    os.system(
                        f"python ../StyleCLIP/run_optimization.py --description '{text}' --latent-path {latent_path} --attribute {args.attribute}"
                        f" --outdir {outdir} --image-path {image_path} --target {args.target} --power {args.power} --components {args.components}")
                elif args.method == "tedigan":
                    os.system(f"python ../TediGAN/ext/demo.py --description '{text}' --latent-path {latent_path} --attribute {args.attribute}"
                              f" --outdir {outdir} --image-path {image_path} --target {args.target} --power {args.power} --components {args.components}")
            print(f"Done text: {text}")
    print("Done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
