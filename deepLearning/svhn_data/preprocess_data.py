# coding: utf-8
import os
import sys
import json
import pickle
import random

import tqdm
import optparse
import numpy as np
import PIL.Image as Image


def crop_image(img_path, img_boxes, shape=(32, 32)):
    """
    crop image into 32x32 size

    :param img_path: image path
    :param img_boxes: image box data
    :param shape: croped shape
    """
    img = Image.open(img_path)
    left, top, width, height = \
        img_boxes['left'], img_boxes['top'], img_boxes['width'], img_boxes['height']
    # get the heights and width
    min_top, max_top = np.amin(top), np.amax(top)
    im_height = max_top + height[np.argmax(top)] - min_top
    min_left, max_left = np.amin(left), np.amax(left)
    im_width = max_left + width[np.argmax(left)] - min_left
    # get top, left, bottom, right
    im_top = np.floor(min_top - 0.1 * im_height)
    im_left = np.floor(min_left - 0.1 * im_width)
    im_bottom = np.amin([np.ceil(im_top + 1.2 * im_height), img.size[1]])
    im_right = np.amin([np.ceil(im_left + 1.2 * im_width), img.size[0]])
    # crop image
    cropped = np.array(img.crop(box=map(int, (im_left, im_top, im_right, im_bottom))).resize(shape), dtype=np.float32)
    img.close()
    return cropped


def to_gray_scale(img):
    """
    convert image to gray scale

    :param img: image object
    :return: gray scaled image object
    """
    return np.dot(np.array(img, dtype=np.float32), [[0.2989], [0.5870], [0.1140]])


def z_score_normalize(img):
    """
    z score normalize image

    :param img: image object
    :return: z score normalized image
    """
    mean = np.mean(img, dtype=np.float32)
    std = np.std(img, dtype=np.float32)
    if std < 1e-4:
        std = 1.
    return (img - mean) / std


def fetch_all_data(folder, shape=(32, 32)):
    """
    extract train, test, extra dateset, load them into numpy array

    :param folder: train, test or extra folder
    :param shape: reshape shape
    :return: dataset and lables tuple
    """
    with open(os.path.join(folder, 'digitStruct.json')) as f:
        json_data = json.load(f)
    size = len(json_data)
    # load RGB image
    dataset = np.ndarray([size, shape[0], shape[1], 3], dtype=np.float32)
    labels = np.ones([size, 6], dtype=np.int) * 10

    for _i, _struct in tqdm.tqdm(enumerate(json_data)):
        _pic = _struct['filename']
        _box = _struct['bbox']
        croped = crop_image(os.path.join(folder, _pic), _box, shape=shape)
        # gray scale
        gray = to_gray_scale(croped)
        # normalize data
        norm_img = z_score_normalize(gray)
        num_digits = len(_box['label'])
        if num_digits > 5:
            print('image %s has more than 5 digits, skip...' % _i)
        else:
            labels[_i, 0] = num_digits
            for j in range(num_digits):
                labels[_i, j + 1] = _box['label'][j] if _box['label'][j] != 10 else 0
            dataset[_i, ...] = norm_img[...]
    return dataset, labels


def preprocess_data(train_folfer, test_folder, extra_folder, shape=(32, 32),
                    pickle_path=''):
    print('staring preprocess data...')
    train_data, train_labels = fetch_all_data(folder=train_folfer, shape=shape)
    test_data, test_labels = fetch_all_data(folder=test_folder, shape=shape)
    extra_data, extra_labels = fetch_all_data(folder=extra_folder, shape=shape)
    print('end fetch...')
    # delete 29929th image
    train_data = np.delete(train_data, 29929, axis=0)
    train_labels = np.delete(train_labels, 29929, axis=0)
    # delete 703th image
    train_data = np.delete(train_data, 703, axis=0)
    train_labels = np.delete(train_labels, 703, axis=0)
    # shuffle data
    np.random.seed(42)
    n_labels = 10
    valid_index_1, valid_index_2 = list(), list()
    train_index_1, train_index_2 = list(), list()
    for i in np.arange(n_labels):
        valid_index_1.extend(np.where(train_labels[:, 1] == i)[0][:500].tolist())
        valid_index_2.extend(np.where(extra_labels[:, 1] == i)[0][:250].tolist())
        train_index_1.extend(np.where(train_labels[:, 1] == i)[0][500:].tolist())
        train_index_2.extend(np.where(extra_labels[:, 1] == i)[0][250:].tolist())
    # shuffle data
    for idx in [valid_index_1, valid_index_2, train_index_1, train_index_2]:
        random.shuffle(idx)
    # concatenate data
    valid_dataset = np.concatenate((extra_data[valid_index_2, ...], train_data[valid_index_1, ...]), axis=0)
    valid_labels = np.concatenate((extra_labels[valid_index_2, :], train_labels[valid_index_1, :]), axis=0)
    train_dataset = np.concatenate((extra_data[train_index_2, ...], train_data[train_index_1, ...]), axis=0)
    train_labels_ = np.concatenate((extra_labels[train_index_2, :], train_labels[train_index_1, :]), axis=0)
    print(train_dataset.shape, train_labels_.shape)
    print(test_data.shape, test_labels.shape)
    print(valid_dataset.shape, valid_labels.shape)
    # free memory
    del train_data
    del train_labels
    del extra_data
    del extra_labels
    save = {
        'train_data': train_dataset,
        'train_labels': train_labels_,
        'test_data': test_data,
        'test_labels': test_labels,
        'valid_data': valid_dataset,
        'valid_labels': valid_labels
    }
    if len(pickle_path) != 0:
        print('starting pickle data')
        with open(pickle_path, 'wb') as f:
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            file_stat = os.stat(pickle_path)
            info = 'Compresse success, pickle size: %s' % file_stat.st_size
            return info
    else:
        return save


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option("--train", dest="train_folder", help="train data folder")
    parser.add_option("--test", dest="test_folder", help="test data folder")
    parser.add_option("--extra", dest="extra_folder", help="extra data folder")
    opts, args = parser.parse_args()
    if None in [opts.train_folder, opts.test_folder, opts.extra_folder]:
        print('Usage: --train=/path/to/train_folfer --test=/path/to/test_folder --extra=/path/to/extra_folder')
        sys.exit(-1)
    data = preprocess_data(opts.train_folder, opts.test_folder, opts.extra_folder, shape=(54, 54),
                           pickle_path='SVHN_54.pickle')
