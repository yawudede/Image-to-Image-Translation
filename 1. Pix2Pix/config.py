import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size for train')
parser.add_argument('--val_batch_size', type=int, default=3, help='mini-batch size for validation')
parser.add_argument('--test_batch_size', type=int, default=8, help='mini-batch size for test')
parser.add_argument('--crop_size', type=int, default=256, help='resize images')

parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
parser.add_argument('--plots_path', type=str,  default='./results/plots/', help='plots path')
parser.add_argument('--inference_path', type=str, default='./results/inference/', help='inference path')

parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.75, help='decay learning rate')
parser.add_argument('--lr_decay_every', type=int, default=50, help='decay learning rate for every default epoch')
parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler, options: [Step, Plateau, Cosine]')

parser.add_argument('--num_epochs', type=int, default=200, help='total epoch')
parser.add_argument('--print_every', type=int, default=100, help='print statistics for every default iteration')
parser.add_argument('--save_every', type=int, default=10, help='save model weights for every default epoch')

parser.add_argument('--l1_lambda', type=int, default=200, help='constant for generator loss')

config = parser.parse_args()


if __name__ == "__main__":
    print(config)