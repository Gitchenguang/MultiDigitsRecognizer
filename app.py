# coding: utf-8
import os
import uuid

import PIL.Image as Image
from werkzeug import secure_filename
from flask import Flask, url_for, render_template, request, url_for, redirect, send_from_directory

import deepLearning.infer_model as infer_model
import deepLearning.preprocess_data as preprocess


ALLOWED_EXTENSIONS = set(list(['png', 'jpg', 'jpeg']))
UPLOAD_FOLDER = 'static/img/upload'
app = Flask(__name__, template_folder='template', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def convert_to_rgb(img):
    newimg = Image.new('RGB', img.size)
    newimg.paste(img, mask=img.split()[3])
    return newimg


def process_image(file_path):
    """
    resize image to 32x32
    :param file_path: file path
    :return:
    """
    img = Image.open(file_path, mode='r')
    if img.mode == 'RGBA':
        img = convert_to_rgb(img)
    img = img.resize([32, 32], Image.ANTIALIAS)
    # to gray scale
    img = preprocess.to_gray_scale(img)
    # normalize
    img = preprocess.z_score_normalize(img)
    return img


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        _file = request.files['file']
        if _file and allowed_file(_file.filename):
            filename = secure_filename(_file.filename)
            _path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            _file.save(_path)
            img = process_image(_path)
            number, softmax_output = infer_model.infer(
                input_data=img.reshape((1, 32, 32, 1)), ckpt_data='ckpt_data/SVHN.ckpt')
            return render_template('result.html', image_path='/static/img/upload/'+filename,
                                   string_variable=number,
                                   table=softmax_output.to_html(index=False).replace('&lt;', '<').replace('&gt;', '>'))
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
