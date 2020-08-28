import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size for train')
parser.add_argument('--crop_size', type=int, default=178, help='crop image size')
parser.add_argument('--image_size', type=int, default=128, help='image size')
parser.add_argument("--n_critics", type=int, default=5, help="number of training iterations for WGAN discriminator")

parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
parser.add_argument('--plots_path', type=str,  default='./results/plots/', help='plots path')
parser.add_argument('--inference_path', type=str,  default='./results/inference/', help='inference path')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.75, help='decay learning rate')
parser.add_argument('--lr_decay_every', type=int, default=5, help='decay learning rate for every default epoch')
parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler')

parser.add_argument('--num_epochs', type=int, default=10, help='total epoch')
parser.add_argument('--print_every', type=int, default=100, help='print statistics for every default iteration')
parser.add_argument('--save_every', type=int, default=5, help='save model weights for every default epoch')

parser.add_argument('--lambda_cls', type=int, default=1, help='constant for cls')
parser.add_argument('--lambda_gp', type=int, default=10, help='constant for gradient penalty')
parser.add_argument('--lambda_recon', type=int, default=10, help='constant for reconstruction')

parser.add_argument('--selected_attrs', type=int, default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'], help='selected attributions')

config = parser.parse_args()

if __name__ == "__main__":
    print(config)