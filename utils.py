import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


HEIGHT = 137
WIDTH = 236
SIZE = 128
DATASET_PATH = '/home/rauf/datasets/bengali/'
train_file_names = ['train_image_data_{}.parquet'.format(i) for i in range(4)]
test_file_names = ['test_image_data_{}.parquet'.format(i) for i in range(4)]

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect rsatio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

def create_dataset(file_names):
    file_paths = [os.path.join(DATASET_PATH, file_name) for file_name in file_names]
    data_list = [pd.read_parquet(file_path, engine='pyarrow') for file_path in file_paths]
    data = [df.iloc[:,1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8) for df in data_list]
    data = np.concatenate(data, axis=0)
    del data_list
    return data

def create_cropped_dataset(file_names, size=SIZE):
    data = create_dataset(file_names)
    num_images = data.shape[0]
    data_cropped = np.zeros((num_images, size, size), dtype=np.uint8)
    for idx, image in enumerate(data):
        img = 255 - image
        img = (img*(255.0/img.max())).astype(np.uint8)
        img = crop_resize(img)
        data_cropped[idx,:,:] = img
    del data
    return data_cropped

def save_pngs(data, save_path, file_name_prefix):
    os.makedirs(save_path, exist_ok=True)
    for idx, img in enumerate(data):
        file_name = '{}_{}.png'.format(file_name_prefix, str(idx))
        file_save_path = os.path.join(save_path, file_name)
        cv2.imwrite(file_save_path, img)


def show_random_samples(data, n_samples):
    fig, axs = plt.subplots(n_samples, 1, figsize=(10, 5*n_samples))
    total_n_samples = data.shape[0]
    selected_data_idxs = np.random.choice(total_n_samples, n_samples, replace=False)
    selected_data = data[selected_data_idxs,:,:]
    for idx, sample in enumerate(selected_data):
        axs[idx].imshow(sample)
        axs[idx].set_title('Crop & resize')
        axs[idx].axis('off')
    plt.show()