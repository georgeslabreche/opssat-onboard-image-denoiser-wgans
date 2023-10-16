import tensorflow as tf

import os
from config import *
from dir_utils import DirUtils
from model import generatorNet
import numpy as np
import argparse

# somehow, the Conda environment can't read the required dlls when this path is included in the environment variables.
#if INCLUDE_TF_DLL_PATH:
#  os.add_dll_directory(TF_DLL_PATH)


def convert_to_tflite(noise_type, noise_factor, input_filepath=None, output_filepath=None):
  # FIXME: What about descriminator?

  # the directory utils
  dir_utils = DirUtils(noise_type, noise_factor)

  if input_filepath is None:
    input_filepath = dir_utils.get_checkpoint_path(f'BEST_ge{".h5" if SAVE_WEIGHTS_ONLY else ""}')

  if output_filepath is None:
    output_filepath = dir_utils.get_checkpoint_path(f'wgan_ge_{noise_type}{noise_factor}.tflite')
    
  print(f'Converting...\n  from: {input_filepath}\n    to: {output_filepath}')

  # the trained model that will be loaded to convert to tflite
  model = None

  if SAVE_WEIGHTS_ONLY is False:
    # load the model
    model = tf.keras.models.load_model(input_filepath)
    dummy_data = np.random.rand(1, PATCH_SHAPE[1], PATCH_SHAPE[2], PATCH_SHAPE[3])
    _ = model(dummy_data)
        
    # Try fetching the input tensor from the first layer
    input_tensor = model.layers[0].input
    
    if input_tensor is not None:
        print("Input tensor name:", input_tensor.name)
    else:
        print("Model does not seem to have a valid input tensor!")
        exit()

  else:
    # if we only saved the weights and not the architecture, i.e. save_weights() instead of save()
    # recreate the model the model's architecture first and then load the saved weights
    model = generatorNet()
  
    # create dummy data with the right shape
    dummy_data = np.random.rand(1, PATCH_SHAPE[1], PATCH_SHAPE[2], PATCH_SHAPE[3])

    # calling the model to initialize variables
    _ = model(dummy_data)

    # load the weights
    model.load_weights(input_filepath)

  # convert the model
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.experimental_new_converter = True

  # set the input shape for the model.
  # note: this ensures the model can accept dynamic sizes, but you still need to resize before inference.
  #converter.input_shapes = {input_tensor.name: [1, None, None, BATCH_SHAPE[3]]}
  
  converter.input_shapes = {input_tensor.name: [1, PATCH_SHAPE[1], PATCH_SHAPE[2], PATCH_SHAPE[3]]}
  tflite_model = converter.convert()
  
  # load TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  # get input and output details
  tflite_input_details = interpreter.get_input_details()
  tflite_output_details = interpreter.get_output_details()

  # save the converted model
  with open(output_filepath, 'wb') as f:
    f.write(tflite_model)


if __name__ == '__main__':
  # arguments parser
  parser = argparse.ArgumentParser(description="Convert to tflite the model trained with the given noise type and factor")

  # noise type
  parser.add_argument('--noise_type', '-t', type=str, default='fnp', help="Type of noise. Default is 'fnp'")

  # noise factor
  parser.add_argument('--noise_factor', '-f', type=int, default=50, help='Noise factor. Default is 50')

  # parse the arguments
  args = parser.parse_args()

  # convert the model
  convert_to_tflite(args.noise_type, args.noise_factor)
