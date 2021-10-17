import argparse
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from fan_utilities import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains a feature-aware normalization model.')

    parser.add_argument('--dataset_path',dest='dataset_path',
                        action='store',type=str,default=None)
    parser.add_argument('--hdf5',dest = 'hdf5',
                        action = 'store_true',
                        help = 'The dataset is a HDF5 file.')
    parser.add_argument('--padding',dest = 'padding',
                        action = 'store',default = 'VALID',
                        help = 'Define padding.',
                        choices = ['VALID','SAME'])
    parser.add_argument('--input_height',dest = 'input_height',
                        action = 'store',type = int,default = 256,
                        help = 'The file extension for all images.')
    parser.add_argument('--input_width',dest = 'input_width',
                        action = 'store',type = int,default = 256,
                        help = 'The file extension for all images.')
    parser.add_argument('--n_features',dest = 'n_features',
                        action = 'store',type = int,default = 16,
                        help = 'Features for the latent space.')
    parser.add_argument('--upscaling',dest = 'upscaling',
                        action = 'store',type = str,default = 'standard',
                        help = 'Type of upscaling (standard or transpose).')

    parser.add_argument('--log_every_n_steps',dest = 'log_every_n_steps',
                        action = 'store',type = int,default = 100,
                        help = 'How often are the loss and global step logged.')
    parser.add_argument('--save_summary_folder',dest = 'save_summary_folder',
                        action = 'store',type = str,default = 'summaries',
                        help = 'Directory where summaries are saved.')
    parser.add_argument('--save_checkpoint_folder',dest = 'save_checkpoint_folder',
                        action = 'store',type = str,default = 'checkpoints',
                        help = 'Directory where checkpoints are saved.')
    parser.add_argument('--batch_size',dest = 'batch_size',
                        action = 'store',type = int,default = 4,
                        help = 'Size of mini batch.')
    parser.add_argument('--number_of_epochs',dest = 'number_of_epochs',
                        action = 'store',type = int,default = 50,
                        help = 'Number of epochs in the training process.')
    parser.add_argument('--learning_rate',dest = 'learning_rate',
                        action = 'store',type = float,default = 0.001,
                        help = 'Learning rate for the SGD optimizer.')

    #Data augmentation
    for arg in [
        ['brightness_max_delta',16. / 255.,float],
        ['saturation_lower',0.8,float],['saturation_upper',1.2,float],
        ['hue_max_delta',0.2,float],
        ['contrast_lower',0.8,float],['contrast_upper',1.2,float]]:
        parser.add_argument('--{}'.format(arg[0]),dest=arg[0],
                            action='store',type=arg[2],default=arg[1])

    args = parser.parse_args()

    data_augmentation_params = {
        'brightness_delta':args.brightness_max_delta,
        'saturation_lower':args.saturation_lower,
        'saturation_upper':args.saturation_upper,
        'hue_delta':args.hue_max_delta,
        'contrast_lower':args.contrast_lower,
        'contrast_upper':args.contrast_upper,
        'probability':1}

    print("Setting up network...")
    mobilenet_v2 = keras.applications.MobileNetV2(include_top=False)
    sub_model = keras.Model(
        inputs=mobilenet_v2.input,
        outputs=[mobilenet_v2.get_layer(x).output for x in output_layer_ids])
    fan_model = FAN(sub_model,args.n_features,
                    args.input_height,args.input_width,
                    batch_size=args.batch_size,
                    upscaling=args.upscaling)

    print("Setting up data generator...")
    flipper = Flipper()
    colour_augmenter = ColourAugmentation(**data_augmentation_params)
    if args.hdf5 == False:
        data_generator = DataGenerator(args.dataset_path,flipper)
    else:
        data_generator = DataGeneratorHDF5(args.dataset_path,flipper)
    def load_generator():
        for image in data_generator.generate():
            yield colour_augmenter(image),image
    generator = load_generator
    output_types = (tf.float32,tf.float32)
    output_shapes = (
        tf.TensorShape((args.input_height,args.input_width,3)),
        tf.TensorShape((args.input_height,args.input_width,3)))
    tf_dataset = tf.data.Dataset.from_generator(
        generator,output_types=output_types,output_shapes=output_shapes)
    tf_dataset = tf_dataset.repeat()
    tf_dataset = tf_dataset.batch(args.batch_size)
    tf_dataset = tf_dataset.prefetch(args.batch_size*5)

    print("Setting up training...")
    mae = tf.keras.metrics.MeanAbsoluteError()
    loss_fn = keras.losses.MeanSquaredError()
    fan_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss_fn, metrics=[mae])
    fan_model.build(input_shape=(args.batch_size,args.input_height,args.input_width,3))
    steps_per_epoch = data_generator.n_images // args.batch_size

    print("Training...")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=args.save_summary_folder)
    image_callback = ImageCallBack(
        args.log_every_n_steps,tf_dataset,log_dir=args.save_summary_folder)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        args.save_checkpoint_folder + '-{epoch:02d}')
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        'loss',min_lr=1e-6)

    all_callbacks = [
        tensorboard_callback,image_callback,checkpoint_callback,lr_callback]

    fan_model.fit(
        x=tf_dataset,batch_size=args.batch_size,epochs=args.number_of_epochs,
        callbacks=all_callbacks,steps_per_epoch=steps_per_epoch)
