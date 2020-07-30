import os
from io import BytesIO
import time
import random
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

import my_params


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = my_params.INPUT_SIZE
  FROZEN_GRAPH_NAME = 'my_frozen_inference_graph'

  def __init__(self):
    """Creates and loads pretrained deeplab model."""
    # Load TFLite model and allocate tensors.
    self.interpreter = tf.lite.Interpreter(model_path=MY_GRAPH_PATH)
    self.interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = self.interpreter.get_input_details()
    output_details = self.interpreter.get_output_details()

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    # Test model on random input data.
    target_size = (self.INPUT_SIZE, self.INPUT_SIZE)
    resized_image = image.resize(target_size, Image.ANTIALIAS)
    input_data = np.expand_dims(resized_image, axis=0)
    self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)

    self.interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])
    print(np.max(output_data))
    return resized_image, output_data[0]


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()
  #plt.savefig('/home/haimzis/deeplab/my_inference_output/')


def run_visualization(images):
  """Inferences DeepLab model and visualizes result."""
  for image in images:
    try:
      with tf.gfile.FastGFile(image, 'rb') as f:
        jpeg_str = f.read()
        original_im = Image.open(BytesIO(jpeg_str))
    except IOError:
      print('Cannot retrieve image')
      return

    print('running deeplab on image {0}'.format(image))
    resized_im, seg_map = MODEL.run(original_im)

    vis_segmentation(resized_im, seg_map)
    time.sleep(2.0)


if __name__ == '__main__':
  LABEL_NAMES = np.asarray([
      'background', 'Mole'
  ])

  MY_GRAPH_PATH = '/home/haimzis/PycharmProjects/deeplab/deployed/MobileNet_V3_large_ver2.tflite'
  FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
  FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

  print('loading DeepLab model...')
  MODEL = DeepLabModel()
  print('model loaded successfully!')
  #images = tf.gfile.Glob('/home/haimzis/Desktop/Data/inference_data/*')
  images = tf.gfile.Glob('/home/haimzis/PycharmProjects/deeplab/datasets/moles_detector/images/validation/*')
  run_visualization(random.sample(images, 10))
