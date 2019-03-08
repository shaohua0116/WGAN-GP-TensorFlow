import argparse
import os

from model import Model


def argparser(is_train=True):

    def str2bool(v):
        return v.lower() == 'true'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint_g', type=str, default=None)
    parser.add_argument('--checkpoint_d', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='celeba',
                        choices=['bedroom', 'celeba', 'ImageNet', 'CityScape',
                                 'CIFAR10', 'CIFAR100', 'SVHN',
                                 'MNIST', 'Fashion_MNIST'])
    parser.add_argument('--dataset_path', type=str, default=None)
    # Model
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gan_type', type=str, default='wgan-gp',
                        choices=['lsgan', 'wgan-gp'])
    parser.add_argument('--n_z', type=int, default=128)
    parser.add_argument('--num_dis_conv', type=int, default=6)
    parser.add_argument('--num_g_res_block', type=int, default=3)
    parser.add_argument('--num_d_res_block', type=int, default=3)
    parser.add_argument('--g_norm_type', type=str, default='batch',
                        choices=['batch', 'instance', 'none'])
    parser.add_argument('--d_norm_type', type=str, default='none',
                        choices=['batch', 'instance', 'none'])
    parser.add_argument('--deconv_type', type=str, default='bilinear',
                        choices=['bilinear', 'nn', 'transpose'])

    # Training config {{{
    # ========
    # log
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--write_summary_step', type=int, default=100)
    parser.add_argument('--ckpt_save_step', type=int, default=10000)
    # learning
    parser.add_argument('--max_training_steps', type=int, default=10000000)
    parser.add_argument('--learning_rate_g', type=float, default=1e-4)
    parser.add_argument('--learning_rate_d', type=float, default=1e-4)
    parser.add_argument('--adam_beta1', type=float, default=0.5)
    parser.add_argument('--adam_beta2', type=float, default=0.9)
    parser.add_argument('--lr_weight_decay', type=str2bool, default=False)
    parser.add_argument('--update_g', type=int, default=1)
    parser.add_argument('--update_d', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=10)
    # }}}

    # Testing config {{{
    # ========
    parser.add_argument(
            '--output_file', type=str, default=None,
            help='dump all generated images to a HDF5 file with the filename specify here')
    parser.add_argument('--write_summary_image', type=str2bool, default=False)
    parser.add_argument('--summary_image_name', type=str, default='summary.png')
    parser.add_argument('--max_evaluation_steps', type=int, default=5)
    # }}}

    config = parser.parse_args()

    if config.dataset_path is None:
        dataset_path = os.path.join('./datasets', config.dataset.lower())
    else:
        dataset_path = config.dataset_path

    if config.dataset in ['CIFAR10', 'CIFAR100', 'SVHN', 'MNIST', 'Fashion_MNIST']:
        import datasets.hdf5_loader as dataset
    elif config.dataset in ['bedroom', 'celeba', 'ImageNet', 'CityScape']:
        import datasets.image_loader as dataset
    dataset_train, dataset_test = dataset.create_default_splits(dataset_path)

    img = dataset_train.get_data(dataset_train.ids[0])
    config.h = img.shape[0]
    config.w = img.shape[1]
    config.c = img.shape[2]

    # --- create model ---
    model = Model(config, debug_information=config.debug, is_train=is_train)

    return config, model, dataset_train, dataset_test
