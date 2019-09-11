"""Module to parse dataset."""

# Imports.
import os
from os.path import join
import glob

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
#libraries added
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

def read_image_paths(data_dir):
    """Read image paths from data directory.

    Args:
        data_dir (str): path to folder with images.

    Returns:
        image_paths (list): list of image paths.

    """
    image_extension_pattern = '*.jpg'
    image_paths = sorted((y for x in os.walk(data_dir) for y in
                          glob.glob(join(x[0], image_extension_pattern))))
    return image_paths


def get_image_paths_dict(data_dir):
    """Create and return dict that maps image IDs to image paths.

    Args:
        data_dir (str): path to folder with images

    Returns:
        image_paths_dict (dict): dict to map image IDs to image paths.

    """
    image_paths = read_image_paths(data_dir)
    image_paths_dict = {}
    for image_path in image_paths:
        image_id = image_path.split('/')[-1].split('.jpg')[0]
        image_paths_dict[image_id] = image_path

    return image_paths_dict


def read_meta_data(data_dir,index_flag =1):
    """Read meta data file using Pandas.

    Returns:
        meta_data (pandas.core.frame.DataFrame): meta-data object.

    """
    if index_flag == 1:
        meta_data = pd.read_csv(join(data_dir, 'HAM10000_metadata.csv'),
                            index_col='image_id')
    else:
        meta_data = pd.read_csv(join(data_dir, 'HAM10000_metadata.csv'))
    return meta_data


def load_image(image_path):
    """Load image as numpy array.

    Args:
        image_path (str): path to image.

    Returns:
        (numpy.ndarray): image as numpy array.

    """
    return np.array(Image.open(image_path))


def show_images(images, cols = 1, titles = None):
    """Display multiple images arranged as a table.

    Args:
        images (list): list of images to display as numpy arrays.
        cols (int, optional): number of columns.
        titles (list, optional): list of title strings for each image.

    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def compute_img_mean_std(data_dir):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """
    image_paths = list(get_image_paths_dict(data_dir).values())
    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs

def undup_data(data_dir):
    meta_data = read_meta_data(data_dir,index_flag=0)
    # this will tell us how many images are associated with each lesion_id
    df_undup = meta_data.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    meta_data['duplicates'] = meta_data['lesion_id']
    # apply the function to this new column
    meta_data['duplicates'] = meta_data['duplicates'].apply(get_duplicates)
    df_undup = meta_data[meta_data['duplicates'] == 'unduplicated']
    df_undup = df_undup.drop(columns='duplicates')
    df_undup = df_undup.set_index('image_id')
    return df_undup



def create_train_val_split(data_dir, train_fraction, val_fraction):
    """Split data into training and validation sets, based on given fractions.

    Args:
        train_fraction (float): fraction of data to use for training.
        val_fraction (float): fraction of data to use for training.

    Returns:
        (tuple): tuple with training image IDs and validation image IDs.

    """
    assert(train_fraction + val_fraction <= 1.0)

    # TODO: Implement a proper training/validation split
    # Created a stratified sampling train test split
    
    meta_data_undup = undup_data(data_dir)
    meta_data = read_meta_data(data_dir)
    train, val = train_test_split(meta_data_undup,
                                test_size = val_fraction,random_state=42, 
                                stratify=meta_data_undup.dx)
    train = meta_data.drop(val.index,axis=0)
    train['image_id'] = train.index
    train = train.reset_index(drop=True)
    # Copy fewer class to balance the number of 7 classes
    data_aug_rate = [15,10,5,50,5,0,40]
    dx_list = ['akiec','bcc','bkl','df','mel','nv','vasc']
    for i in dx_list:
        if data_aug_rate[dx_list.index(i)]:
            train=train.append([train.loc[train['dx'] == i,:]]*(data_aug_rate
                               [dx_list.index(i)]-1), ignore_index=True)
    train = train.set_index('image_id')
#    train['dx'].value_counts()
#    image_ids = meta_data.index.tolist()
#    num_images = len(image_ids)
#    num_train_ids = int(num_images * train_fraction)
#    num_val_ids = int(num_images * val_fraction)
    train_ids = train.index.tolist()
    val_ids = val.index.tolist()
    return train_ids, val_ids


class HAM10000(Dataset):
    """HAM10000 dataset.

    Attributes:
        sampling_list (list): list of image IDs to use.
        image_paths_dict (dict): dict to map image IDs to image paths.
        meta_data (pandas.core.frame.DataFrame): meta data object.
        class_map_dict (dict): dict to map label strings to label indices.

    """

    def __init__(self, data_dir, sampling_list):
        """Constructor.

        Args:
            data_dir (str): path to images and metadata file
            sampling_list (list): list of image IDs to use.

        """
        self.data_dir = data_dir
        self.sampling_list = sampling_list
        self.image_paths_dict = get_image_paths_dict(self.data_dir)
        self.meta_data = read_meta_data(self.data_dir)
        self.class_map_dict = self.get_class_map_dict()

    def get_labels(self):
        """Get labels of dataset and return them as list.

        Returns:
            (list): list of all labels.

        """
        labels = [self.meta_data.loc[image_id]['dx'] for image_id in self.sampling_list]

        return labels

    def get_num_classes(self):
        """Get number of classes.

        Returns:
            (int): number of classes.

        """
        return len(self.class_map_dict)

    def get_class_map_dict(self):
        """Get dict to map label strings to label indices.

        Returns:
            class_map_dict (dict): dict to map label strings to label indices.

        """
        classes_list = list(self.meta_data.groupby('dx')['lesion_id'].nunique().keys())
        classes_list = sorted(classes_list)
        class_map_dict = {}
        for i, cls in enumerate(classes_list):
            class_map_dict[cls] = i

        return class_map_dict

    def __len__(self):
        """Get size of dataset.

        Returns:
            (int): size of dataset, i.e. number of samples.

        """
        return len(self.sampling_list)

    def __getitem__(self, index):
        """Get item.

        Args:
            index (int): index.

        Returns:
            (tuple): tuple with image and label.

        """
        image_id = self.sampling_list[index]
        img = Image.open(self.image_paths_dict.get(image_id))
        assert(image_id in self.meta_data.index)
        label = self.class_map_dict[self.meta_data.loc[image_id]['dx']]
        # Normalizing the image pixels
        #providing the mean and std directly ... calculated previously
#        norm_mean,norm_std = compute_img_mean_std(self.data_dir)
        norm_mean = [0.7630329, 0.54564583, 0.5700466]
        norm_std = [0.14092815, 0.15261224, 0.1699708]
        trans = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(norm_mean,norm_std )])
        img = trans(img)
#        img = transforms.ToTensor()(img)

        return img, label
