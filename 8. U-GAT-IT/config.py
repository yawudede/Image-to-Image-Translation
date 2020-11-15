import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
parser.add_argument('--val_batch_size', type=int, default=8, help='mini-batch size for validation')
parser.add_argument('--crop_size', type=int, default=128, help='resize images')
parser.add_argument('--num_blocks', type=int, default=4, help='number of blocks for generator')

parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
parser.add_argument('--plots_path', type=str,  default='./results/plots/', help='plots path')
parser.add_argument('--inference_path', type=str, default='./results/inference/', help='inference path')

parser.add_argument('--num_epochs', type=int, default=50, help='total epoch')
parser.add_argument('--print_every', type=int, default=100, help='print statistics for every n iteration')
parser.add_argument('--save_every', type=int, default=5, help='save model weights for every n epoch')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.75, help='decay learning rate')
parser.add_argument('--lr_decay_every', type=int, default=25, help='decay learning rate for every n epoch')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler', choices=['step', 'plateau', 'cosine'])

parser.add_argument('--lambda_cycle', type=int, default=10, help='weight for cycle loss')
parser.add_argument('--lambda_cam', type=int, default=1000, help='weight for CAM')

config = parser.parse_args()


if __name__ == "__main__":
    print(config)