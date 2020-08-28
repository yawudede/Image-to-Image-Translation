import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size for train')
parser.add_argument('--val_batch_size', type=int, default=1, help='mini-batch size for validation')
parser.add_argument('--crop_size', type=int, default=128, help='resize images')
parser.add_argument('--test_size', type=int, default=20, help='test size')
parser.add_argument('--num_images', type=int, default=5, help='the number of images for generation')

parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
parser.add_argument('--plots_path', type=str,  default='./results/plots/', help='plots path')
parser.add_argument('--inference_path', type=str, default='./results/inference/', help='inference path')

parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.75, help='decay learning rate')
parser.add_argument('--lr_decay_every', type=int, default=50, help='decay learning rate for every default epoch')
parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler')

parser.add_argument('--num_epochs', type=int, default=5, help='total epoch')
parser.add_argument('--print_every', type=int, default=1000, help='print statistics for every default iteration')
parser.add_argument('--save_every', type=int, default=5, help='save model weights for every default epoch')

parser.add_argument('--z_dim', type=int, default=8, help='noise dimension')
parser.add_argument('--lambda_KL', type=float, default=0.01, help='constant for KL Divergence Loss')
parser.add_argument('--lambda_Image', type=int, default=10, help='constant for Image Loss')
parser.add_argument('--lambda_Z', type=float, default=0.5, help='constant for Z Loss')

config = parser.parse_args()

if __name__ == "__main__":
    print(config)