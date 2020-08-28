import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size for train')
parser.add_argument('--val_batch_size', type=int, default=8, help='mini-batch size for validation')
parser.add_argument('--crop_size', type=int, default=64, help='resize images')

parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
parser.add_argument('--plots_path', type=str,  default='./results/plots/', help='plots path')
parser.add_argument('--inference_path', type=str, default='./data/results/inference/', help='inference path')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.75, help='decay learning rate')
parser.add_argument('--lr_decay_every', type=int, default=50, help='decay learning rate for every default epoch')
parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler, options: [Step, Plateau, Cosine]')

parser.add_argument('--decay_gan_loss', type=int, default=10000)
parser.add_argument('--starting_rate', type=float, default=0.01)
parser.add_argument('--changed_rate', type=float, default=0.5)

parser.add_argument('--num_epochs', type=int, default=100, help='total epoch')
parser.add_argument('--print_every', type=int, default=100, help='print statistics for every default iteration')
parser.add_argument('--save_every', type=int, default=10, help='save model weights for every default epoch')
parser.add_argument('--num_train_gen', type=int, default=3, help='train generator 3 times while train discrinator one time')

parser.add_argument('--lambda_recon', type=float, default=0.01, help='constant for reconstruction loss')
parser.add_argument('--lambda_adversarial', type=float, default=0.1, help='constant for adversarial loss')
parser.add_argument('--lambda_feature', type=float, default=0.99, help='constant for feature loss')

config = parser.parse_args()


if __name__ == "__main__":
    print(config)