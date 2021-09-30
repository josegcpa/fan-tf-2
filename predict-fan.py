import argparse
import os
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from fan_utilities import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains a feature-aware normalization   model.')

    parser.add_argument('--input_path',dest='input_path',
                        action='store',type=str,default=None)
    parser.add_argument('--output_path',dest='output_path',
                        action='store',type=str,default=None)
    parser.add_argument(
        '--example',dest = 'example',
        action = 'store_true',
        help = 'Combines input and prediction when saving the output.')

    parser.add_argument('--padding',dest = 'padding',
                        action = 'store',default = 'VALID',
                        help = 'Define padding.',
                        choices = ['VALID','SAME'])
    parser.add_argument('--n_features',dest='n_features',
                        action='store',type=int,default=16,
                        help='Features for the latent space.')

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

    try: os.makedirs(args.output_path)
    except: pass

    mobilenet_v2 = keras.applications.MobileNetV2(include_top=False)
    output_layer_ids = ['block_3_depthwise','block_5_depthwise',
                        'block_7_depthwise','block_9_depthwise']
    sub_model = keras.Model(
        inputs=mobilenet_v2.input,
        outputs=[mobilenet_v2.get_layer(x).output for x in output_layer_ids])
    fan_model = FAN(sub_model,args.n_features,
                    args.input_height,args.input_width)
    fan_model.load_weights(args.checkpoint_path)

    data_generator = DataGenerator(args.input_path,False,None)
    
    pbar = tqdm(data_generator.n_images)
    for image,image_path in data_generator.generate(with_path=True):
        root = os.path.split(image_path)[-1]
        output_path = '{}/prediction_{}'.format(args.output_path,root)
        condition_h = image.shape[0] <= args.input_height
        condition_w = image.shape[1] <= args.input_width
        if np.all([condition_h,condition_w]):
            image_tensor = tf.expand_dims(
                tf.convert_to_tensor(image),axis=0)
            image_prediction = fan_model.predict(image_tensor)[0]
        else:
            large_image = LargeImage(
                image,tile_size=[args.input_height,args.input_width])
            for tile,coords in large_image.tile_image():
                tile_tensor = tf.expand_dims(
                    tf.convert_to_tensor(tile),axis=0)
                tile_prediction = fan_model.predict(tile_tensor)
                large_image.update_output(tile_prediction,coords)
            image_prediction = large_image.return_output()
        
        if args.example == True:
            image_prediction = np.concatenate([image,image_prediction],axis=1)
        image_prediction = Image.fromarray(np.uint8(image_prediction*255))
        image_prediction.save(output_path)
        pbar.update(1)