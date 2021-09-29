import os
import numpy as np
from math import inf
from PIL import Image
from glob import glob

import tensorflow as tf
from tensorflow import keras

class FANLayer(keras.layers.Layer):
    def __init__(self,n_features,ratio,epsilon=1e-8,
                 upscaling='standard'):
        super(FANLayer,self).__init__()
        self.n_features = n_features
        self.ratio = ratio
        self.epsilon = epsilon
        self.upscaling = upscaling
        self.setup_layers()
    
    def setup_layers(self):
        self.sigmoid = keras.Sequential()
        self.sigmoid.add(keras.layers.Conv2D(self.n_features,1))
        self.sigmoid.add(keras.layers.Activation('sigmoid'))
        
        self.relu = keras.Sequential()
        self.relu.add(keras.layers.Conv2D(self.n_features,1))
        self.relu.add(keras.layers.Activation('relu'))

        if self.upscaling == 'standard':
            self.sigmoid.add(keras.layers.UpSampling2D(self.ratio))
            self.relu.add(keras.layers.UpSampling2D(self.ratio))
        elif self.upscaling == 'transpose':
            self.sigmoid.add(
                keras.layers.Conv2D(self.n_features,self.ratio,padding='same',
                                    strides=self.ratio))
            self.relu.add(
                keras.layers.Conv2D(self.n_features,self.ratio,padding='same',
                                    strides=self.ratio))
    
    def call(self,x,z):
        mean_x = tf.reduce_mean(x,axis=[1,2],keepdims=True)
        std_x =  tf.math.reduce_std(x,axis=[1,2],keepdims=True)
        x = (x - mean_x) / (std_x**2 + self.epsilon)
        sigmoid_output = self.sigmoid(z)
        relu_output = self.relu(z)
        return x * sigmoid_output + relu_output

class FAN(keras.Model):
    def __init__(self,sub_model,n_features,
                 input_height,input_width,epsilon=1e-8):
        super(FAN,self).__init__()
        self.n_features = n_features
        self.epsilon = epsilon
        self.sub_model = sub_model
        self.input_height = input_height
        self.input_width = input_width
        self.n_z = len(self.sub_model.output)
        self.infer_ratios()
        self.setup_layers()
    
    def setup_layers(self):
        self.latent_space_encoder = keras.layers.Conv2D(
            self.n_features,1)
        self.linear_layers = {
            'fan_'+str(i): FANLayer(self.n_features,ratio) 
            for i,ratio in zip(range(self.n_z),self.ratios[-1::-1])}
        self.latent_space_decoder = keras.layers.Conv2D(3,1)
        
    def infer_ratios(self):
        tmp = tf.ones([1,self.input_height,self.input_width,3])
        outs = self.sub_model(tmp)
        self.ratios = [self.input_height//x.shape[1] for x in outs]

    def call(self,x):
        feature_space = self.latent_space_encoder(x)
        zs = self.sub_model(x)
        tr = feature_space
        for lin,z in zip(self.linear_layers,zs[-1::-1]):
            tr = self.linear_layers[lin](tr,z)
        return self.latent_space_decoder(tr)

class ColourAugmentation(keras.layers.Layer):
    def __init__(self,
                 brightness_delta,
                 contrast_lower,contrast_upper,
                 hue_delta,
                 saturation_lower,saturation_upper,
                 probability=0.1):
        super(ColourAugmentation,self).__init__()
        self.probability = probability
        self.brightness_delta = brightness_delta
        self.contrast_lower = contrast_lower
        self.contrast_upper = contrast_upper
        self.hue_delta = hue_delta
        self.saturation_lower = saturation_lower
        self.saturation_upper = saturation_upper
        
    def brightness(self,x):
        return tf.image.random_brightness(
            x,self.brightness_delta)
    
    def contrast(self,x):
        return tf.image.random_contrast(
            x,self.contrast_lower,self.contrast_upper)
    
    def hue(self,x):
        return tf.image.random_hue(
            x,self.hue_delta)
    
    def saturation(self,x):
        return tf.image.random_saturation(
            x,self.saturation_lower,self.saturation_upper)
    
    def call(self,x):
        fn_list = [self.brightness,self.contrast,
                   self.hue,self.saturation]
        np.random.shuffle(fn_list)
        for fn in fn_list:
            if np.random.uniform() < self.probability:
                x = fn(x)
        x = tf.clip_by_value(x,0,1)
        return x
    
class Flipper(keras.layers.Layer):
    def __init__(self,probability=0.1):
        super(Flipper,self).__init__()
        self.probability = probability
            
    def call(self,x):
        if np.random.uniform() < self.probability:
            x = tf.image.flip_left_right(x)
        if np.random.uniform() < self.probability:
            x = tf.image.flip_up_down(x)
        return x

class DataGenerator:
    def __init__(self,image_folder_path,transform=None):
        self.image_folder_path = image_folder_path
        self.image_paths = glob('{}/*'.format(self.image_folder_path))
        self.n_images = len(self.image_paths)
        self.transform = transform
    
    def generate(self):
        image_idx = [x for x in range(len(self.image_paths))]
        np.random.shuffle(image_idx)
        for idx in image_idx:
            x = np.array(Image.open(self.image_paths[idx]))[:,:,:3]
            x = tf.convert_to_tensor(x) / 255
            if self.transform is not None:
                x = self.transform(x)
            yield x