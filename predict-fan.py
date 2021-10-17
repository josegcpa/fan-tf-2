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
    parser.add_argument('--hdf5',dest = 'hdf5',
                        action = 'store_true',
                        help = 'The dataset is a HDF5 file.')
    parser.add_argument('--hdf5_output',dest = 'hdf5_output',
                        action = 'store_true',
                        help = 'The output is saved as a HDF5 file.')

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
                        help='Image input height.')
    parser.add_argument('--input_width',dest='input_width',
                        action='store',type=int,default=256,
                        help='Image input width.')
    parser.add_argument('--resize_size',dest='resize_size',
                        action='store',type=int,default=256,
                        help='Target size for images.')

    parser.add_argument('--checkpoint_path',dest='checkpoint_path',
                        action='store',type=str,default='checkpoints',
                        help='Path to saved model.')

    args = parser.parse_args()

    if args.hdf5_output == False:
        try: os.makedirs(args.output_path)
        except: pass

    if args.resize_size != args.input_height:
        h,w = args.resize_size,args.resize_size
    else:
        h,w = args.input_height,args.input_width

    mobilenet_v2 = keras.applications.MobileNetV2(include_top=False)
    sub_model = keras.Model(
        inputs=mobilenet_v2.input,
        outputs=[mobilenet_v2.get_layer(x).output for x in output_layer_ids])
    fan_model = FAN(sub_model,args.n_features,h,w,
                    batch_size=1,upscaling=args.upscaling)
    fan_model.load_weights(args.checkpoint_path)

    if args.hdf5 == False:
        data_generator = DataGenerator(args.input_path)
    else:
        data_generator = DataGeneratorHDF5(args.input_path)
    
    if args.hdf5_output == True:
        F = h5py.File(args.output_path,'w')

    pbar = tqdm(data_generator.n_images)
    for image,image_path in data_generator.generate(with_path=True):
        root = os.path.split(image_path)[-1]
        output_path = '{}/prediction_{}'.format(args.output_path,root)
        condition_h = image.shape[0] <= args.input_height
        condition_w = image.shape[1] <= args.input_width
        if np.all([condition_h,condition_w]):
            image = tf.convert_to_tensor(image)
            if args.resize_size != args.input_height:
                image = tf.image.resize(
                    image,[args.resize_size,args.resize_size])
            image_tensor = tf.expand_dims(image,axis=0)
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
        
        image_prediction = image_prediction - image_prediction.min() / (image_prediction.max() - image_prediction.min())
        
        if args.hdf5_output == False:
            if args.example == True:
                image_prediction = np.concatenate([image,image_prediction],axis=1)

            image_prediction = Image.fromarray(np.uint8(image_prediction*255))
            image_prediction.save(output_path)
        else:
            g = F.create_group(root)
            g['image'] = image
            g['prediction'] = image_prediction
        pbar.update(1)