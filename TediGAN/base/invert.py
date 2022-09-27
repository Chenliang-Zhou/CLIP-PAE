# python 3.6
"""Revised from Inverts given images to latent codes with In-Domain GAN Inversion."""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))

from utils.inverter import StyleGANInverter
from utils.visualizer import save_image, load_image, resize_image


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='styleganinv_ffhq256', help='Name of the GAN model.')
    parser.add_argument('--mode', type=str,
                        default='man', help='Mode (gen for generation, man for manipulation).')
    parser.add_argument('--description', type=str, default='a happy face',
                        help='The description.')
    parser.add_argument('--input-image-path', type=str, required=True, help='Path of images to invert.')
    parser.add_argument('--output-image-path', type=str, required=True, help='Path of images to invert.')
    parser.add_argument('-o', '--output_dir', type=str, default='output/exp14',
                        help='Directory to save the results. If not specified, '
                             '`./results/inversion/test` '
                             'will be used by default.')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate for optimization. (default: 0.01)')
    parser.add_argument('--num-iterations', type=int, default=200,
                        help='Number of optimization iterations. (default: 200)')
    parser.add_argument('--num-results', type=int, default=0,
                        help='Number of intermediate optimization results to '
                             'save for each sample. (default: 5)')
    parser.add_argument('--loss-weight-feat', type=float, default=5e-5,
                        help='The perceptual loss scale for optimization. '
                             '(default: 5e-5)')
    parser.add_argument('--loss-weight-enc', type=float, default=2.0,
                        help='The encoder loss scale for optimization.'
                             '(default: 2.0)')
    parser.add_argument('--loss-weight-clip', type=float, default=2.0,
                        help='The clip loss for optimization. (default: 2.0)')
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='Which GPU(s) to use. (default: `0`)')
    parser.add_argument("--target", help="optimization target",
                        choices=["text", "pae", "pae+", "paeGS+", "paePCA+", "paeAllEx", "paeAllExD", "dpeGS", "dpePCA"], default="text")
    parser.add_argument("--power", help="augmentation power for projected embedding", type=float, default=9.5)
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    assert os.path.isfile(args.input_image_path)
    os.makedirs(args.output_dir, exist_ok=True)

    inverter = StyleGANInverter(
        args.model_name,
        mode=args.mode,
        learning_rate=args.learning_rate,
        iteration=args.num_iterations,
        reconstruction_loss_weight=1.0,
        perceptual_loss_weight=args.loss_weight_feat,
        regularization_loss_weight=args.loss_weight_enc,
        clip_loss_weight=args.loss_weight_clip,
        description=args.description,
        logger=None,
        target=args.target,
        power=args.power)
    image_size = inverter.G.resolution

    # Invert the given image.
    image = resize_image(load_image(args.input_image_path), (image_size, image_size))
    _, viz_results = inverter.easy_invert(image, num_viz=args.num_results)

    # if args.mode == 'man':
    #     image_name = os.path.splitext(os.path.basename(args.output_image_path))[0]
    # else:
    #     image_name = 'gen'
    # save_image(f'{args.output_dir}/{image_name}_ori.png', viz_results[0])
    # save_image(f'{args.output_dir}/{image_name}_enc.png', viz_results[1])
    # save_image(f'{args.output_dir}/{image_name}_inv.png', viz_results[-1])
    save_image(f'{args.output_dir}/{args.output_image_path}', viz_results[-1])
    print(f'save {args.output_image_path} in {args.output_dir}')


if __name__ == '__main__':
    main()
