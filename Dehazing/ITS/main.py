import os
import torch
import argparse
from torch.backends import cudnn
from models.RFR import build_net
from train import _train
from eval import _eval

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    model = build_net(args.version, args.data)
    # print(model)

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'dense':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='RFR_Full', type=str)
    parser.add_argument('--data', type=str, default='DenseHaze', choices=['ITS', 'Haze4K', 'NHR', 'GTA5', 'DenseHaze', 'NHHAZE', 'OHAZE'])
    parser.add_argument('--version', default='base', choices=['base', 'large'], type=str)

    parser.add_argument('--mode', default='dense', choices=['train', 'dense'], type=str)
    parser.add_argument('--data_dir', type=str, default=r'D:\projects\PycharmProjects\ConvIR-main\ConvIR-main\DenseHaze')

    # Train for its
    #parser.add_argument('--batch_size', type=int, default=8)
    #parser.add_argument('--learning_rate', type=float, default=1e-4)
    #parser.add_argument('--weight_decay', type=float, default=0)
    #parser.add_argument('--num_epoch', type=int, default=600)
    #parser.add_argument('--print_freq', type=int, default=100)
    #parser.add_argument('--num_worker', type=int, default=8)
    #parser.add_argument('--save_freq', type=int, default=20)
    #parser.add_argument('--valid_freq', type=int, default=20)
    #parser.add_argument('--resume', type=str, default='')


    # uncomment for OHAZE
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=5000)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')

    # Train for real-hazy
    #parser.add_argument('--batch_size', type=int, default=1)
    #parser.add_argument('--learning_rate', type=float, default=2e-4)
    #parser.add_argument('--weight_decay', type=float, default=0)
    #parser.add_argument('--num_epoch', type=int, default=5000)
    #parser.add_argument('--print_freq', type=int, default=20)
    #parser.add_argument('--num_worker', type=int, default=4)
    #parser.add_argument('--save_freq', type=int, default=10)
    #parser.add_argument('--valid_freq', type=int, default=10)
    #parser.add_argument('--resume', type=str, default='')

    # Train for Haze4k
    # parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--learning_rate', type=float, default=4e-4)
    # parser.add_argument('--weight_decay', type=float, default=0)
    # parser.add_argument('--num_epoch', type=int, default=1000)
    # parser.add_argument('--print_freq', type=int, default=100)
    # parser.add_argument('--num_worker', type=int, default=8)
    # parser.add_argument('--save_freq', type=int, default=20)
    # parser.add_argument('--valid_freq', type=int, default=20)

    # Test
    parser.add_argument('--test_model', type=str, default=r'D:\projects\PycharmProjects\ConvIR-main\ConvIR-main\Dehazing\ITS\DenseHaze.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.data, args.model_name, 'Training-Results/')
    args.result_dir = os.path.join('results/', args.data, args.model_name, 'images', args.data)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    command = 'cp ' + 'models/layers.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'models/RFR.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'train.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'main.py ' + args.model_save_dir
    os.system(command)
    print(args)
    main(args)
