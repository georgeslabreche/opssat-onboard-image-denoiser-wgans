import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from dir_utils import DirUtils


def plot_losses(df, noise_type, noise_factor, plot_filepath):
    """plot d_loss and g_loss"""

    # Create the plots
    fig = plt.figure(figsize=(12, 10))

    # Plotting d_loss
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(df['epoch'], df['d_loss'], color='tab:blue')
    ax1.set_title('D Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('D Loss')
    ax1.legend()
    ax1.grid(True)
    ax1.legend_ = None

    # Plotting g_loss
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.plot(df['epoch'], df['g_loss'], color='tab:red')
    ax2.set_title('G Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('G Loss')
    ax2.legend()
    ax2.grid(True)
    ax2.legend_ = None

    plt.tight_layout()
    plt.savefig(plot_filepath, format='svg')
    plt.show()


def plot_metrics(df, noise_type, noise_factor, plot_filepath):
    """plot ssim, psnr, and mse"""

    # create the plots
    fig = plt.figure(figsize=(12, 10))

    metrics = ['ssim', 'psnr', 'mse']
    colors = ['tab:blue', 'tab:red', 'tab:green']
    titles = ['SSIM', 'PSNR', 'MSE']

    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(3, 1, i+1)
        ax.plot(df['epoch'], df[metric], color=colors[i])
        ax.set_title(titles[i])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
        ax.legend_ = None

    plt.tight_layout()
    plt.savefig(plot_filepath, format='svg')
    plt.show()


def plot(noise_type, noise_factor):
    """plot the metrics collected during training"""

    # the directory utils
    dir_utils = DirUtils(noise_type, noise_factor)

    # list and sort all matching files
    files = sorted(glob.glob(os.path.join(dir_utils.get_checkpoint_path(), 'metrics_epoch_*.csv')), 
               key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))

    # read and concatenate all files with an additional column for epoch segment
    all_metrics_data = []
    for file in files:
        df = pd.read_csv(file)
        epoch_segment = int(os.path.basename(file).split('_')[2].split('.')[0]) # extract the number from the filename
        df['epoch'] = epoch_segment
        all_metrics_data.append(df)
        
    # combine into a single dataframe
    df_all_metrics = pd.concat(all_metrics_data, ignore_index=True)
    
    # plot losses
    plot_losses(df_all_metrics, noise_type, noise_factor,
        dir_utils.get_checkpoint_path(f'plot_loss_{noise_type}{noise_factor}.svg'))

    # plot metrics
    plot_metrics(df_all_metrics, noise_type, noise_factor,
        dir_utils.get_checkpoint_path(f'plot_metrics_{noise_type}{noise_factor}.svg'))


if __name__ == '__main__':

    # arguments parser
    parser = argparse.ArgumentParser(description="Plot the training metrics")

    # noise type
    parser.add_argument('--noise_type', '-t', type=str, default='fnp', help="Type of noise. Default is 'fnp'")

    # noise factor
    parser.add_argument('--noise_factor', '-f', type=int, default=50, help='Noise factor. Default is 50')

    # parse the arguments
    args = parser.parse_args()

    # plot the metrics
    plot(args.noise_type, args.noise_factor)