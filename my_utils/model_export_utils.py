import tensorflow as tf
import re
import os
import my_params


def find_most_updated_weights_file(checkpoint_dir):
    with tf.gfile.Open(checkpoint_dir + '/checkpoint', 'r') as f:
        text = f.readline()
        model_checkpoint = re.findall(r'model.ckpt-\d*', text)[-1]
        return '/'.join((checkpoint_dir, model_checkpoint))


def export_model_as_tflite(pb_path=my_params.graph_def_file):
    with tf.gfile.Open('./tflite_convert.txt', 'r') as f:
        command = f.read()
        command = command.format(output_file=pb_path.replace('pb', 'tflite'),
                       graph_def_file=pb_path,
                       input_arrays=my_params.input_arrays,
                       output_arrays=my_params.output_arrays,
                       input_shape=my_params.input_shape,
                       inference_input_type=my_params.inference_input_type,
                       inference_type=my_params.inference_type,
                       mean_values=str(my_params.mean_values),
                       std_dev_values=str(my_params.std_dev_values))
        os.system(command)