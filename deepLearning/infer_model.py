# coding: utf-8
import os
import platform

import multi_digits.muti_digits_model as tf_model


def infer(input_data, ckpt_data):
    """
    input data and return its numbers
    :param input_data: numpy arrary
    :param ckpt_data: ckpt data path
    :return:
    """
    model = tf_model.MultiDigits()
    if platform.system() == 'Windows':
        dl_path = 'deepLearning\\'
    else:
        dl_path = 'deepLearning/'
    result, sotfmax_output = model.infer_model(input_data=input_data,
                                               ckpt_path=os.path.join(os.getcwd(), dl_path, ckpt_data))
    return to_number(result[0].tolist(), index=1), sotfmax_output


def to_number(result, index=1):
    """
    results to number
    :param result: numpy array result, like [[1, 2, 9, 10]]
    :param index: 0
    :return: number, int type
    """
    output = map(str, result[index:result.index(10) if 10 in result else len(result)])
    return int(''.join(output))
