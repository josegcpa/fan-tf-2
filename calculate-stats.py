import h5py
import numpy as np
import pickle
import argparse
from skimage import color
from skimage.metrics import structural_similarity as ssim
from itertools import product
from tqdm import tqdm

def rescale(x,x_min=None,x_max=None):
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()
    x = (x - x_min)/(x_max - x_min)
    x = np.clip(x,0,1)
    return x

def colour_histogram(x):
    r = np.histogram(x[:,:,0],255,[0,255])
    g = np.histogram(x[:,:,1],255,[0,255])
    b = np.histogram(x[:,:,2],255,[0,255])
    
    return r,g,b

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains a feature-aware normalization model.')

    parser.add_argument('--dataset_path',dest='dataset_path',
                        action='store',type=str,default=None)
    parser.add_argument('--output_path',dest='output_path',
                        action='store',type=str,default=None)
    parser.add_argument('--N',dest='N',
                        action='store',type=int,default=1000)

    args = parser.parse_args()

    curr_dataset = h5py.File(args.dataset_path,'r')
    N = args.N

    all_histograms = {'original':[],'prediction':[]}
    all_lab = {'original':[],'prediction':[]}
    all_ssim = []
    all_min_max = []

    all_keys = list(curr_dataset.keys())
    all_keys = np.random.choice(all_keys,N,replace=False)
    print(curr_dataset.filename)

    all_images = {}

    print("Retrieving all images...")
    for key in tqdm(all_keys):
        image = curr_dataset[key]['image'][()]
        prediction = curr_dataset[key]['prediction'][()]
        all_images[key] = [image,prediction]

    x_min = np.quantile([all_images[x][1].min() for x in all_images],0.05)
    x_max = np.quantile([all_images[x][1].max() for x in all_images],0.95)
    all_min_max = [x_min,x_max]
    print("Rescaling...")
    for k in tqdm(all_keys):
        image,prediction = all_images[k]
        all_images[k] = [
            np.uint8(image),
            np.uint8(rescale(prediction,x_min,x_max)*255)]

    print("Calculating histograms...")
    for key in tqdm(all_keys):
        image = all_images[key][0]
        prediction = all_images[key][1]

        image_histogram = colour_histogram(image)
        prediction_histogram = colour_histogram(prediction)

        all_histograms['original'].append(
            image_histogram)
        all_histograms['prediction'].append(
            prediction_histogram)

    print("Calculating std(L), std(a) and std(b)...")
    for key in tqdm(all_keys):
        image = all_images[key][0]
        prediction = all_images[key][1]

        lab_image = color.rgb2lab(image)
        lab_prediction = color.rgb2lab(prediction)

        all_lab['original'].append(
            np.std(lab_image.reshape([-1,3]),axis=0))
        all_lab['prediction'].append(
            np.std(lab_prediction.reshape([-1,3]),axis=0))

    print("Calculating structural similarity with original...")
    for key in tqdm(all_keys):
        image = all_images[key][0]
        prediction = all_images[key][1]

        all_ssim.append(
            ssim(image,prediction,multichannel=True))
    
    with open(args.output_path,'wb') as o:
        all_output = {
            'histograms':all_histograms,
            'lab':all_lab,
            'ssim':all_ssim,
            'min_max':all_min_max}
        pickle.dump(all_output,o)