# coding: utf-8
# Process SVHN dataset
import os
import json

import tqdm
import h5py
import optparse


def hdf5_to_json(folder, out_json_path, return_value=False):
    """
    convert hdfd5 data format to json data format

    :param folder: matlab data folder
    :param out_json_path: json format data path
    :param return_value: is return result?
    :return: return result or None
    """
    hdf5_data = h5py.File(os.path.join(folder, 'digitStruct.mat'))
    size = hdf5_data['/digitStruct/name'].size
    result = []
    for i in tqdm.tqdm(range(size)):
        bbox = dict()
        bbox['filename'] = get_name(i, hdf5_data)
        bbox['bbox'] = get_box_data(i, hdf5_data)
        result.append(bbox)
    try:
        with open(out_json_path, mode='w') as f:
            f.write(json.JSONEncoder(indent=True).encode(result))
            if return_value:
                return result
    except Exception as e:
        print(e.args)


def get_name(index, hdf5_data):
    """
    get the image file name

    :param index: image file index
    :param hdf5_data: hdf5 file object
    :return: file name
    """
    name = hdf5_data['/digitStruct/name']
    file_name = ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])
    return file_name


def get_box_data(index, hdf5_data):
    """
    get bbox info: left, top, width, height

    :param index:
    :param hdf5_data:
    :return: image's bboxes data in dict type
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def pack_values(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(pack_values)
    # convert 10 to 0
    meta_data['label'] = [0 if v == 10 else v for v in meta_data['label']]
    return meta_data


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('f', dest='mat_dir', help='Matlab full number SVHN input file', default='digitStruct.mat')
    parser.add_option('-o', dest='json_path', help='name for the json output file', default='digitStruct')
    options, args = parser.parse_args()

    hdf5_to_json(options.mat_dir, options.json_path)
