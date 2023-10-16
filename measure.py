# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/15 15:04
@Auth ： kuiju_wang
@File ：mesure.py.py
@IDE ：PyCharm

"""
import os
from skimage.metrics import *
import cv2
import numpy as np
import pandas as pd
import csv
from config import *
from tqdm import tqdm
from dir_utils import DirUtils


def calculate_metrics(imgraw, imgmesure):
    ssim = structural_similarity(imgraw, imgmesure, multichannel=True, data_range=255, channel_axis=-1)
    psnr = peak_signal_noise_ratio(imgraw, imgmesure)
    mse = mean_squared_error(imgraw, imgmesure)
    return ssim, psnr, mse


def measure(noise_type, noise_factor):

    # the directory utils
    dir_utils = DirUtils(noise_type, noise_factor)
    
    # the file path of the metrics csv
    metrics_filepath = dir_utils.get_gen_img_path('metrics.csv')
    
    # delete if it already exists
    # so that if we run the test again it does not attempt to load the metrics file as an image file
    if os.path.exists(metrics_filepath):
        os.remove(metrics_filepath)

    # get list of image filenames
    unnoised_image_filename_list = sorted(os.listdir(dir_utils.get_test_clean_path()))
    denoised_image_filename_list = sorted(os.listdir(dir_utils.get_gen_img_path()))

    # list of original unnoised images
    img_unnoised_list = np.array([np.array(cv2.imread(os.path.join(dir_utils.get_test_clean_path(), name))) for name in unnoised_image_filename_list])
    
    # list of denoised images (only the jpeg ones)
    img_denoised_list = np.array([
        np.array(cv2.imread(os.path.join(dir_utils.get_gen_img_path(), name)))
        for name in denoised_image_filename_list if name.endswith('.jpeg')
    ])

    # save metrics to csv
    with open(dir_utils.get_gen_img_path('metrics.csv'), 'w', newline='') as csvfile:
        # the csv file writer
        writer = csv.writer(csvfile)
       
        # write header
        writer.writerow(['name', 'ssim', 'psnr', 'mse'])

        # write rows
        for i in tqdm(range(img_denoised_list.shape[0])):
            ssim, psnr, mse = calculate_metrics(img_unnoised_list[i], img_denoised_list[i])
            writer.writerow([denoised_image_filename_list[i], ssim, psnr, mse])