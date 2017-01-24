# coding: utf-8
import os
import sys
import pickle
import random

import tqdm
import h5py
import numpy as np
import PIL.Image as Image


def crop_image(img_path, img_boxes, shape=(32, 32)):
    """
    对每张图片按照64x64的尺寸以及img_boxes的范围进行截取
    :param img_path: image path
    :param img_boxes: image box data
    :param shape: croped shape
    """
    img = Image.open(img_path)
    left, top, width, height = \
        img_boxes['left'], img_boxes['top'], img_boxes['width'], img_boxes['height']
    # 获取截取后的图片的高度和宽度
    min_top, max_top = np.amin(top), np.amax(top)
    im_height = max_top + height[np.argmax(top)] - min_top
    min_left, max_left = np.amin(left), np.amax(left)
    im_width = max_left + width[np.argmax(left)] - min_left
    # 扩大图片，确定top, left, bottom, right
    im_top = np.floor(min_top - 0.1 * im_height)
    im_left = np.floor(min_left - 0.1 * im_width)
    im_bottom = np.amin([np.ceil(im_top + 1.2 * im_height), img.size[1]])
    im_right = np.amin([np.ceil(im_left + 1.2 * im_width), img.size[0]])
    # 剪切图片
    return img.crop(box=map(int, (im_left, im_top, im_right, im_bottom))).resize(shape)


def to_gray_scale(img):
    """
    把图片转为灰度值
    :param img: image object
    :return: gray scaled image object
    """
    return np.dot(np.array(img, dtype=np.float32), [[0.2989], [0.5870], [0.1140]])


def z_score_normalize(img):
    """
    对灰度图片进行归一化
    :param img: image object
    :return: z score normalized image
    """
    mean = np.mean(img, dtype=np.float32)
    std = np.std(img, dtype=np.float32)
    if std < 1e-4:
        std = 1.
    return (img - mean) / std


def get_name(index, hdf5_data):
    """
    get name of picture
    :param index: picture index
    :param hdf5_data: hdf5 data
    :return: file name
    """
    name = hdf5_data['/digitStruct/name']
    file_name = ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])
    return file_name


def get_box_data(index, hdf5_data):
    """
    获取left, top, width, height
    :param index: index of box
    :param hdf5_data: hdf5 data
    :return: box data
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data


def fetch_all_data(folder):
    """
    对train, test, extra的数据进行提取, 以numpy的array的形式存储
    :param folder: train, test or extra folder
    :return: dataset and lables tuple
    """
    mat_data = h5py.File(os.path.join(folder, 'digitStruct.mat'))
    size = mat_data['/digitStruct/name'].size
    dataset = np.ndarray([size, 32, 32, 1], dtype=np.float32)
    labels = np.ones([size, 6], dtype=np.int) * 10

    for _i in tqdm.tqdm(range(size)):
        _pic = get_name(_i, mat_data)
        # 缩放图片
        _box = get_box_data(_i, mat_data)
        croped = crop_image(os.path.join(folder, _pic), _box)
        # 灰度化
        gray = to_gray_scale(croped)
        # 归一化之后的图像
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


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('help: command line args, train_folfer, test_folder and extra_folder')
        sys.exit(-1)
    train_folfer, test_folder, extra_folder = sys.argv[1:]
    print('staring preprocess data...')
    train_data, train_labels = fetch_all_data(folder=train_folfer)
    test_data, test_labels = fetch_all_data(folder=test_folder)
    extra_data, extra_labels = fetch_all_data(folder=extra_folder)
    print('end fetch...')
    # 删除第29929张图片的数据
    train_data = np.delete(train_data, 29929, axis=0)
    train_labels = np.delete(train_labels, 29929, axis=0)
    # 混洗数据，让类分布更加均匀
    random.seed()
    n_labels = 10
    valid_index_1, valid_index_2 = list(), list()
    train_index_1, train_index_2 = list(), list()
    for i in np.arange(n_labels):
        # 得到训练和验证数据，按照图片中的第一个数字分别得到500个
        # 目的是为了让数字显得更均匀，让类分布更加均匀
        valid_index_1.extend(np.where(train_labels[:, 1] == i)[0][:500].tolist())
        valid_index_2.extend(np.where(extra_labels[:, 1] == i)[0][:250].tolist())
        train_index_1.extend(np.where(train_labels[:, 1] == i)[0][500:].tolist())
        train_index_2.extend(np.where(extra_labels[:, 1] == i)[0][250:].tolist())
    # 混洗数据
    random.shuffle(valid_index_1)
    random.shuffle(valid_index_2)
    random.shuffle(train_index_1)
    random.shuffle(train_index_2)
    # 拼接训练数据和验证数据
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
    picked_data = 'SVHN.pickle'
    with open(picked_data, 'wb') as f:
        save = {
            'train_data': train_dataset,
            'train_labels': train_labels_,
            'test_data': test_data,
            'test_labels': test_labels,
            'valid_data': valid_dataset,
            'valid_labels': valid_labels
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        file_stat = os.stat(picked_data)
        print('Compresse success, pickle size: %s' % file_stat.st_size)
