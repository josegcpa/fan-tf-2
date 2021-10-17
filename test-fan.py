import argparse
import os
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from fan_utilities import *

def interpolate(x,values,fn):
    for value in values:
        yield fn(x,value)

interpolation_factory = {
    'brightness':{
        'function':tf.image.adjust_brightness,
        'values':[0.8,0.9,1.1,1.20]},
    'hue':{
        'function':tf.image.adjust_hue,
        'values':[-0.2,-0.1,0.1,0.2]},
    'contrast':{
        'function':tf.image.adjust_contrast,
        'values':[0.8,0.9,1.1,1.2]},
    'saturation':{
        'function':tf.image.adjust_saturation,
        'values':[0.8,0.9,1.1,1.20]}}

for k in interpolation_factory:
    interpolation_factory[k]['metrics'] = [
        tf.keras.metrics.MeanAbsoluteError() 
        for _ in range(len(interpolation_factory[k]['values']))]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains a feature-aware normalization   model.')

    parser.add_argument('--dataset_path',dest='dataset_path',
                        action='store',type=str,default=None)
    parser.add_argument('--hdf5',dest = 'hdf5',
                        action = 'store_true',
                        help = 'The dataset is a HDF5 file.')

    parser.add_argument('--padding',dest = 'padding',
                        action = 'store',default = 'VALID',
                        help = 'Define padding.',
                        choices = ['VALID','SAME'])
    parser.add_argument('--n_features',dest='n_features',
                        action='store',type=int,default=16,
                        help='Features for the latent space.')
    parser.add_argument('--upscaling',dest = 'upscaling',
                        action = 'store',type = str,default = 'standard',
                        help = 'Type of upscaling (standard or transpose).')


    parser.add_argument('--input_height',dest='input_height',
                        action='store',type=int,default=256,
                        help='The file extension for all images.')
    parser.add_argument('--input_width',dest='input_width',
                        action='store',type=int,default=256,
                        help='The file extension for all images.')

    parser.add_argument('--checkpoint_path',dest='checkpoint_path',
                        action='store',type=str,default='checkpoints',
                        help='Path to saved model.')

    args = parser.parse_args()

    mobilenet_v2 = keras.applications.MobileNetV2(include_top=False)
    sub_model = keras.Model(
        inputs=mobilenet_v2.input,
        outputs=[mobilenet_v2.get_layer(x).output for x in output_layer_ids])
    fan_model = FAN(sub_model,args.n_features,
                    args.input_height,args.input_width,
                    batch_size=1,upscaling=args.upscaling)
    fan_model.load_weights(args.checkpoint_path)

    if args.hdf5 == False:
        data_generator = DataGenerator(args.dataset_path)
    else:
        data_generator = DataGeneratorHDF5(args.dataset_path)
    
    pbar = tqdm(data_generator.n_images)
    for k in interpolation_factory:
        fn = interpolation_factory[k]['function']  
        values = interpolation_factory[k]['values']
        fan_model.encoder_decoder.update_batch_size(len(values))
        for image in data_generator.generate():
            condition_h = image.shape[0] <= args.input_height
            condition_w = image.shape[1] <= args.input_width
            if np.all([condition_h,condition_w]):
                image_tensor = tf.convert_to_tensor(image)
                image_list = []
                for i,aug_im in enumerate(interpolate(image,values,fn)):
                    image_list.append(aug_im)
                image_batch = tf.stack(image_list,axis=0)
                image_predictions = fan_model.predict(image_batch)
                print(tf.reduce_min(image_predictions),tf.reduce_max(image_predictions))
                for i,image_prediction in enumerate(image_predictions):
                    interpolation_factory[k]['metrics'][i].update_state(
                        image_tensor[9:-9,9:-9,:],
                        image_prediction[9:-9,9:-9,:])
            pbar.update(1)
    
    for k in interpolation_factory:
        values = interpolation_factory[k]['values']
        for i in range(len(values)):
            v = values[i]
            metric = interpolation_factory[k]['metrics'][i]
            metric_value = float(metric.result().numpy())
            print('{},{},{}'.format(k,v,metric_value))