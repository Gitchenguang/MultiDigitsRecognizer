# coding: utf-8
import sys

import multi_digits.muti_digits_model


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('help: command line argv: picked_file: SVHN.pickle, save_path: ckpt_data/SVHN.ckpt')
        sys.exit(-1)
    picked_file, save_path = sys.argv[1:]
    train_model = multi_digits.muti_digits_model.MultiDigits(picked_file=picked_file)
    train_model.define_graph()
    train_model.train_model(save_path=save_path, epoch=150000)
