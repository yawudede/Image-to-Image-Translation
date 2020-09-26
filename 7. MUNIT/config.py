import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size for train')
parser.add_argument('--val_batch_size', type=int, default=8, help='mini-batch size for train')
parser.add_argument('--image_size', type=int, default=128, help='image crop size')
parser.add_argument('--style_dim', type=int, default=8, help='style dimension')
parser.add_argument('--display_size', type=int, default=8, help='display dimension')

parser.add_argument('--num_inference', type=int, default=8, help='display dimension')
parser.add_argument('--lambda_recon', type=int, default=10, help='constant for cycle loss')

parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for both discriminator and generator networks')
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay learning rate')
parser.add_argument('--lr_decay_every', type=int, default=50, help='decay learning rate for every default epoch')
parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler, options: [Step, Plateau, Cosine]')

parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
parser.add_argument('--plots_path', type=str,  default='./results/plots/', help='plots path')
parser.add_argument('--inference_random_path', type=str, default='./results/inference/random/', help='inference path for random generation')
parser.add_argument('--inference_ex_guided_path', type=str, default='./results/inference/ex_guided/', help='inference path for example-guided generation')

parser.add_argument('--num_epochs', type=int, default=20, help='total epoch')
parser.add_argument('--print_every', type=int, default=100, help='print statistics for every default iteration')
parser.add_argument('--save_every', type=int, default=5, help='save model weights for every default epoch')

parser.add_argument('--style', type=str, default='Random', choices=['Random', 'Ex_Guided'])

config = parser.parse_args()

if __name__ == '__main__':
    print(config)