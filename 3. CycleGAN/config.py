import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size for train')
parser.add_argument('--val_batch_size', type=int, default=1, help='mini-batch size for validation')
parser.add_argument('--crop_size', type=int, default=256, help='resize images')

parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
parser.add_argument('--plots_path', type=str,  default='./results/plots/', help='plots path')
parser.add_argument('--inference_path_H2Z', type=str, default='./results/inference/Horse2Zebra/', help='inference path for H2Z')
parser.add_argument('--inference_path_Z2H', type=str, default='./results/inference/Zebra2Horse/', help='inference path for Z2H')

parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.75, help='decay learning rate')
parser.add_argument('--lr_decay_every', type=int, default=50, help='decay learning rate for every default epoch')
parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler, options: [Step, Plateau, Cosine]')

parser.add_argument('--num_epochs', type=int, default=200, help='total epoch')
parser.add_argument('--print_every', type=int, default=100, help='print statistics for every default iteration')
parser.add_argument('--save_every', type=int, default=10, help='save model weights for every default epoch')
parser.add_argument('--num_train_gen', type=int, default=3, help='train generator 3 times while train discrinator one time')

parser.add_argument('--lambda_identity', type=int, default=5, help='constant for identity loss')
parser.add_argument('--lambda_cycle', type=int, default=10, help='constant for cycle loss')

config = parser.parse_args()

if __name__ == "__main__":
    print(config)