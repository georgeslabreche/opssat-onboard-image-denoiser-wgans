# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/21 16:32
@Auth ： kuiju_wang
@File ：main.py
@IDE ：PyCharm

"""
import os
from config import *

# somehow, the Conda environment can't read the required dlls when this path is included in the environment variables.
if INCLUDE_TF_DLL_PATH:
  os.add_dll_directory(TF_DLL_PATH)

import numpy as np
import tensorflow as tf
from model import generatorNet, discriminatorNet
from utils import *
from tqdm import tqdm
from skimage.metrics import *
from dir_utils import DirUtils
from image_operation import make_dir
from convert import convert_to_tflite
import csv
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def save_tflite(model, filepath):
    """convert the given model to tflite and save it as a file"""

    # convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # save the TFLite model to a file
    with open(filepath, 'wb') as f:
        f.write(tflite_model)


def train(epochs, noise_type, noise_factor):
    """Train the discriminator and generator models"""
    
    # the directory utils
    dir_utils = DirUtils(noise_type, noise_factor)

    truth = read_img_2_array(dir_utils.get_train_clean_path())
    noise = read_img_2_array(dir_utils.get_train_noise_path())
    
    if PATCH_ENABLE is True:
        truth, noise = get_patch(truth, noise)

    batch_val_truth = read_img_2_array(dir_utils.get_val_clean_path())
    batch_val_noise = read_img_2_array(dir_utils.get_val_noise_path())

    generator = generatorNet()
    generator.build(input_shape=(None, None, None, BATCH_SHAPE[3]))

    discriminator = discriminatorNet()
    
    if PATCH_ENABLE is True:
        discriminator.build(input_shape=(None, PATCH_SHAPE[1], PATCH_SHAPE[2], PATCH_SHAPE[3]))
    else:
        discriminator.build(input_shape=(None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]))

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=G_LEARNING_RATE, name="g_optimizer")
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LEARNING_RATE, name="d_optimizer")

    # use these to check for best model after each epoch iteration
    max_ssmi = None
    max_psnr = None
    min_mse = None

    # preserve the best models
    generator_best = None
    discriminator_best = None

    # train for given number of epochs
    for epoch in range(1, epochs+1):

        # initialize metric trackers at the start of each epoch
        ssim_list = []
        psnr_list = []
        mse_list = []
        d_loss_list = []
        g_loss_list = []

        for times in tqdm(range(truth.shape[0] // BATCH_SIZE)):

            batch_truth = tf.convert_to_tensor(truth[BATCH_SIZE * times:BATCH_SIZE * (times + 1)])
            batch_noise = tf.convert_to_tensor(noise[BATCH_SIZE * times:BATCH_SIZE * (times + 1)])

            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_noise, batch_truth)
                d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                g_loss = g_loss_fn(generator, discriminator, batch_noise, batch_truth)
                g_grads = tape.gradient(g_loss, generator.trainable_variables)
                g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

            # evaluate
            fake_img = np.squeeze([generator(tf.expand_dims(tf.convert_to_tensor(val_image), axis=0)).numpy() for val_image in batch_val_noise])

            imgraw = batch_val_truth
            n_val = len(imgraw)

            # psnr calculation
            psnr_values = [
                peak_signal_noise_ratio(imgraw[n], fake_img[n])
                for n in range(n_val)
            ]
            psnr = np.mean(np.array(psnr_values))

            # ssim calculation
            ssim_values = [
                structural_similarity(
                    imgraw[n], 
                    fake_img[n], 
                    multichannel=True, 
                    data_range=1, 
                    channel_axis=-1
                ) 
                for n in range(n_val)
            ]
            ssim = np.mean(np.array(ssim_values))

            # mse calculation
            mse_values = [
                mean_squared_error(imgraw[n], fake_img[n]) 
                for n in range(n_val)
            ]
            mse = np.mean(np.array(mse_values))

            # append metrics
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            mse_list.append(mse)

            # append loss
            d_loss = round(d_loss.numpy(), 6)
            d_loss_list.append(d_loss)
            
            g_loss = round(g_loss.numpy(), 6)
            g_loss_list.append(g_loss)

            # print results
            print(f'  EPOCH: {epoch}, d_loss: {d_loss}, g_loss: {g_loss}, ' +
                f'ssim: {ssim}, psnr: {psnr}, mse: {mse}')

        # save weights
        if epoch == 1 or (epoch > 1 and ssim > max_ssmi and psnr > max_psnr and mse < min_mse):
            max_ssmi, max_psnr, min_mse = ssim, psnr, mse

            # create output dir if it doesn't exist
            if epoch == 1:
                make_dir(dir_utils.get_checkpoint_path())

            # save metrics on the best model so far
            with open(dir_utils.get_checkpoint_path('best.txt'), 'w') as file:
                file.write(f"epoch: {epoch}\n")
                file.write(f" ssim: {ssim}\n")
                file.write(f" psnr: {psnr}\n")
                file.write(f"  mse: {mse}")

            # save the best generator
            generator.save_weights(dir_utils.get_checkpoint_path('BEST_ge.h5'))
            generator.save(dir_utils.get_checkpoint_path('BEST_ge'), save_format='tf')
            generator_best = generator

            # save the descriminator
            if SAVE_DIS:
                discriminator.save_weights(dir_utils.get_checkpoint_path('BEST_di.h5'))
                discriminator.save(dir_utils.get_checkpoint_path('BEST_di'), save_format='tf')
                discriminator_best = discriminator

        if epoch % 5 == 0:

            # save the generator
            generator.save_weights(dir_utils.get_checkpoint_path(f'EPOCH_{epoch}_ge.h5'))
            generator.save(dir_utils.get_checkpoint_path(f'EPOCH_{epoch}_ge'), save_format='tf')

            # save the descriminator
            if SAVE_DIS:
                discriminator.save_weights(dir_utils.get_checkpoint_path(f'EPOCH_{epoch}_di.h5'))
                discriminator.save(dir_utils.get_checkpoint_path(f'EPOCH_{epoch}_di'), save_format='tf')


        # save collected metrics to csv after each epoch completion
        with open(dir_utils.get_checkpoint_path(f'metrics_epoch_{"{:02}".format(epoch)}.csv'), 'w', newline='') as csvfile:
            # the csv file writer
            writer = csv.writer(csvfile)

            # write header
            writer.writerow(['epoch', 'd_loss', 'g_loss', 'ssim', 'psnr', 'mse'])

            # write data
            for i in range(len(d_loss_list)):
                writer.writerow([i+1, d_loss_list[i], g_loss_list[i], ssim_list[i], psnr_list[i], mse_list[i]])


    # save the final generator model
    generator.save_weights(dir_utils.get_checkpoint_path('FINAL_ge.h5'))
    generator.save(dir_utils.get_checkpoint_path('FINAL_ge'), save_format='tf')

    # convert the best generator model into the tflite format
    # FIXME: don't load the generator from file. USe object
    #convert_to_tflite(noise_type, noise_factor)
    
    save_tflite(generator_best, dir_utils.get_checkpoint_path(f'wgan_ge_{noise_type}{noise_factor}.tflite'))

    # save the final discriminator model
    if SAVE_DIS:
        discriminator.save_weights(dir_utils.get_checkpoint_path('FINAL_di.h5'))
        discriminator.save(dir_utils.get_checkpoint_path('FINAL_di'), save_format='tf')

        # convert the best discriminator model into the tflite format
        # FIXME: don't load the generator from file. USe object
        #convert_to_tflite(noise_type, noise_factor)
        
        save_tflite(discriminator_best, dir_utils.get_checkpoint_path(f'wgan_di_{noise_type}{noise_factor}.tflite'))

if __name__ == '__main__':

    # arguments parser
    parser = argparse.ArgumentParser(description="Train the model for a specified number of epochs")

    # noise type
    parser.add_argument('--noise_type', '-t', type=str, default='fnp', help="Type of noise. Default is 'fnp'")

    # noise factor
    parser.add_argument('--noise_factor', '-f', type=int, default=50, help='Noise factor. Default is 50')

    # number of epochs
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train the model for')

    # parse the arguments
    args = parser.parse_args()

    # train the models
    train(args.epochs, args.noise_type, args.noise_factor)

