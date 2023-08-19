# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/2 21:43
@Auth ： kuiju_wang
@File ：test.py
@IDE ：PyCharm

"""
import os
from config import *

# somehow, the Conda environment can't read the required dlls when this path is included in the environment variables.
if INCLUDE_TF_DLL_PATH:
    os.add_dll_directory(TF_DLL_PATH)

import measure
from utils import *
from dir_utils import DirUtils
from model import generatorNet
from image_operation import make_dir
import argparse
from tqdm import tqdm

import tensorflow as tf
tf.debugging.set_log_device_placement(True)


def test_h5(noise_type, noise_factor):
    """Test the h5 model against the test dataset"""

    # the directory utils
    dir_utils = DirUtils(noise_type, noise_factor)

    # make output dir if it doesn't exit
    make_dir(dir_utils.get_gen_img_path())

    generator = generatorNet()
    generator.build(input_shape=(None, None, None, BATCH_SHAPE[3]))

    # TODO: support option to load weights + architecture, i.e. the BEST_ge folder and not the BEST_ge.h5 file
    generator.load_weights(dir_utils.get_checkpoint_path(f'BEST_ge{".h5" if SAVE_WEIGHTS_ONLY else ""}'))

    batch_test_noise = read_img_2_array(dir_utils.get_test_noise_path())

    # load the test noised images and input them into the generator to denoise them
    denoised_image_list = np.squeeze([generator(tf.expand_dims(tf.convert_to_tensor(val_image), axis=0)).numpy() for val_image in batch_test_noise])

    # filename list of denoised images
    filename_list = os.listdir(dir_utils.get_test_noise_path())

    # save the denoised images
    for i, denoised_image in enumerate(denoised_image_list):

        # save as png
        #cv2.imwrite(os.path.join(dir_utils.get_gen_img_path(), filename_list[i].replace('.jpeg', '.png')), denoised_image * 255.0)

        # save a jpeg
        cv2.imwrite(os.path.join(dir_utils.get_gen_img_path(), filename_list[i]), denoised_image * 255.0, [cv2.IMWRITE_JPEG_QUALITY, 100])


# FIXME: does not work
#        produces black images
def test_tflite(noise_type, noise_factor):
    """Test the tflite model against the test dataset"""

    def pad_image(img, patch_size):
        """Pad the image so that it fits neatly into the patch size."""
        h, w, c = img.shape
        
        # Calculate required padding for height and width
        pad_h = (patch_size - (h % patch_size)) % patch_size
        pad_w = (patch_size - (w % patch_size)) % patch_size

        # Pad with white pixels
        padded_img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=1)
        return padded_img

    def split_into_patches(img, patch_size):
        """Split the image into smaller patches."""
        patches = []
        h, w, _ = img.shape
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = img[i:i+patch_size, j:j+patch_size]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    patches.append(patch)
        return patches

    def reassemble_from_patches(patches, original_shape, patch_size):
        """Stitch patches together into a full image."""
        h, w, _ = original_shape
        img = np.zeros((h, w, 3))
        patch_idx = 0
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                img[i:i+patch_size, j:j+patch_size] = patches[patch_idx]
                patch_idx += 1
        return img

    # the directory utils
    dir_utils = DirUtils(noise_type, noise_factor)
    
    # make output dir if it doesn't exit
    make_dir(dir_utils.get_gen_img_path())

    # load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=dir_utils.get_checkpoint_path(f'wgan_ge_{noise_type}{noise_factor}.tflite'))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    batch_test_noise = read_img_2_array(dir_utils.get_test_noise_path())
    filename_list = os.listdir(dir_utils.get_test_noise_path())
    
    if PATCH_ENABLE is False:
        for i in tqdm(range(0, len(batch_test_noise))):

            noisy_image_with_batch_dim = np.expand_dims(batch_test_noise[i], axis=0)

            # denoise the image
            interpreter.set_tensor(input_details[0]['index'], noisy_image_with_batch_dim)
            interpreter.invoke()

            # get the denoised the image, normalize it
            denoised_image = interpreter.get_tensor(output_details[0]['index'])
            denoised_image = (denoised_image * 255.0).astype(np.uint8)[0]  # Convert from float to uint8 and remove batch dimension

            # save to jpeg
            cv2.imwrite(os.path.join(dir_utils.get_gen_img_path(), filename_list[i]), denoised_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

    else:
        # the expected input size of your model
        PATCH_SIZE = PATCH_SHAPE[1]

        for i in tqdm(range(0, len(batch_test_noise))):

            # pad the image and then split into patches
            # commented out: we don't need this in our case because we make sure that the path size is a multiple of hte image size
            # padded_img = pad_image(batch_test_noise[i], PATCH_SIZE)

            # split into patches
            # if we used padded image then use padded_img instead of batch_test_noise[i]
            patches = split_into_patches(batch_test_noise[i], PATCH_SIZE)
            denoised_patches = []

            for patch in patches:
                patch_with_batch_dim = np.expand_dims(patch, axis=0)

                interpreter.set_tensor(input_details[0]['index'], patch_with_batch_dim)
                interpreter.invoke()

                denoised_patch = interpreter.get_tensor(output_details[0]['index'])
                denoised_patches.append(denoised_patch.squeeze())

            # reassemble patches into full image
            # if we used padded image then use padded_img.shape instead of batch_test_noise[i].shape
            denoised_image = reassemble_from_patches(denoised_patches, batch_test_noise[i].shape, PATCH_SIZE)
            
            # crop the image back to its original size if you don't want the padded region
            # in our case we don't need to pad the image to begin with so we commented this out
            #denoised_image = denoised_image[:batch_test_noise[i].shape[0], :batch_test_noise[i].shape[1], :]

            # save to png
            #cv2.imwrite(os.path.join(dir_utils.get_gen_img_path(), filename_list[i].replace('.jpeg', '.png')), denoised_image * 255.0)

            # save to jpeg
            cv2.imwrite(os.path.join(dir_utils.get_gen_img_path(), filename_list[i]), denoised_image * 255.0, [cv2.IMWRITE_JPEG_QUALITY, 100])


if __name__ == '__main__':
    # arguments parser
    parser = argparse.ArgumentParser(description="Test the model that was trained for the given noise type and factor")

    # noise type
    parser.add_argument('--noise_type', '-t', type=str, default='fnp', help="Type of noise. Default is 'fnp'")

    # noise factor
    parser.add_argument('--noise_factor', '-f', type=int, default=50, help='Noise factor. Default is 50')

    # parse the arguments
    args = parser.parse_args()

    # test the models against the test dataset
    #test_h5(args.noise_type, args.noise_factor)
    test_tflite(args.noise_type, args.noise_factor)

    if GEN_CSV:
        # FIXME: ValueError: Input images must have the same dimensions.
        measure.measure(args.noise_type, args.noise_factor)