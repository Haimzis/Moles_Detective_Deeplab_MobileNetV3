# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts moles_detector data to TFRecord file format with Example protos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import random
import sys
import build_data
from six.moves import range
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import cv2
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_image_folder',
    './moles_detector/images/training',
    'Folder containing training images')
tf.app.flags.DEFINE_string(
    'train_image_label_folder',
    './moles_detector/annotations/training',
    'Folder containing annotations for training images')

tf.app.flags.DEFINE_string(
    'val_image_folder',
    './moles_detector/images/validation',
    'Folder containing validation images')

tf.app.flags.DEFINE_string(
    'val_image_label_folder',
    './moles_detector/annotations/validation',
    'Folder containing annotations for validation')

tf.app.flags.DEFINE_string(
    'output_dir', './moles_detector/tfrecord',
    'Path to save converted tfrecord of Tensorflow example')

_NUM_SHARDS = 25
generated_dataset_load = True


def create_dirs(root_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
        os.mkdir(root_dir + '/training')
        os.mkdir(root_dir + '/validation')
    else:
        if not os.path.exists(root_dir + '/training'):
            os.mkdir(root_dir + '/training')
        if not os.path.exists(root_dir + '/validation'):
            os.mkdir(root_dir + '/validation')


def data_preprocess(imgs_dir, annotations_dir):
    #imgs_names = tf.gfile.Glob(os.path.join(imgs_dir + '/*/*.jpg'))
    annotations_names = tf.gfile.Glob(os.path.join(annotations_dir + '/*/*.png'))
    for annotation_name in annotations_names:
        annotation = cv2.imread(annotation_name, -1)
        annotation = annotation // 255
        cv2.imwrite(annotation_name, annotation)


def load_generated_dataset(training_images_dir, training_masks_dir, validation_images_dir, validation_masks_dir):
    annotations_dir = '/home/haimzis/deeplab/datasets/moles_detector/annotations'
    images_dir = '/home/haimzis/deeplab/datasets/moles_detector/images'
    create_dirs(annotations_dir)
    create_dirs(images_dir)
    img_names = tf.gfile.Glob(os.path.join(
        '/home/haimzis/PycharmProjects/DL_training_preprocessing/Output/generated_training_data/images/*.jpg'))
    mask_names = tf.gfile.Glob(os.path.join(
        '/home/haimzis/PycharmProjects/DL_training_preprocessing/Output/generated_training_data/annotations/*.png'))
    img_names.sort()
    mask_names.sort()
    data = {'image': img_names, 'mask': mask_names}
    data_frame = pd.DataFrame(data=data)
    x_train, x_test, y_train, y_test = train_test_split(data_frame['image'], data_frame['mask'], test_size=0.25,
                                                        random_state=101, shuffle=True)

    for target_dir, dataset in {training_images_dir: x_train, validation_images_dir: x_test,
                                training_masks_dir: y_train, validation_masks_dir: y_test}.items():
        for data in dataset:
            data_des = target_dir + '/' + data.split('/')[-1]
            shutil.copy(data, data_des)
    data_preprocess(images_dir, annotations_dir)
    print('dataset has been generated!')


def _convert_dataset(dataset_split, dataset_dir, dataset_label_dir):
    """Converts the moles_detector dataset into tfrecord format.

  Args:
    dataset_split: Dataset split (e.g., train, val).
    dataset_dir: Dir in which the dataset locates.
    dataset_label_dir: Dir in which the annotations locates.

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
    max_depth = 4
    img_names = []
    for depth in range(0, max_depth):
        img_names += tf.gfile.Glob(os.path.join(dataset_dir + '/*' * depth, '*.jpg'))
    random.shuffle(img_names)
    seg_names = []
    for f in img_names:
        # get the filename without the extension
        basename = os.path.basename(f).split('.')[0]
        # cover its corresponding *_seg.png
        seg = os.path.join(dataset_label_dir, basename + '.png')
        seg_names.append(seg)

    num_images = len(img_names)
    num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

    image_reader = build_data.ImageReader('jpg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            FLAGS.output_dir,
            '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                sys.stdout.flush()
                # Read the image.
                image_filename = img_names[i]
                image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
                height, width = image_reader.read_image_dims(image_data)
                # Read the semantic segmentation annotation.
                seg_filename = seg_names[i]
                seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')
                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(
                    image_data, img_names[i], height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


def main(unused_argv):
    tf.gfile.MakeDirs(FLAGS.output_dir)
    if generated_dataset_load:
        load_generated_dataset(FLAGS.train_image_folder, FLAGS.train_image_label_folder, FLAGS.val_image_folder,
                               FLAGS.val_image_label_folder)
    _convert_dataset('train', FLAGS.train_image_folder, FLAGS.train_image_label_folder)
    _convert_dataset('val', FLAGS.val_image_folder, FLAGS.val_image_label_folder)


if __name__ == '__main__':
    tf.app.run()
