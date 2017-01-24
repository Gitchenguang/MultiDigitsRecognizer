A simple CNN multi-digits recognizer based on TensorFlow
---

This is a simple CNN multi-digits recognizer based on TensorFlow, web project on http://goingmyway.cn:5000

### Dependencies
* Python 2.7
* tensorflow-gpu==0.12.0
* Flask==0.12
* h5py==2.6.0
* numpy==1.11.3
* matplotlib==1.5.3
* six==1.10.0
* pandas==0.18.1
* scikit-learn==0.18.1
* scipy==0.18.1

### Show the tree map of the project

	capstone/
	├── app.py  # app file, run it                     
	├── deepLearning   # deep learning model
	│   ├── ckpt_data  # ckpt data path
	│   │   ├── checkpoint
	│   │   ├── SVHN.ckpt.data-00000-of-00001
	│   │   ├── SVHN.ckpt.index
	│   │   └── SVHN.ckpt.meta
	│   ├── infer_model.py  # infer model call by the app
	│   ├── __init__.py
	│   ├── multi_digits  # model dir
	│   │   ├── __init__.py
	│   │   └── muti_digits_model.py  # model file
	│   ├── preprocess_data.py
	│   └── train_model.py
	├── __init__.py
	├── README.md
	├── static
	│   └── img
	│       ├── 46.png
	│       ├── cmd.jpg
	│       ├── ico.ico
	│       ├── icon.ico
	│       ├── meme.png
	│       ├── myid.jpg
	│       ├── pic.jpg
	│       └── sample
	│           ├── 10.png
	│           ├── 11.png
	│           ├── 1.png
	│           ├── 2.png
	│           ├── 3.png
	│           ├── 4.png
	│           ├── 5.png
	│           ├── 6.png
	│           ├── 7.png
	│           ├── 8.png
	│           └── 9.png
	└── template
	    ├── index.html
	    └── result.html

### Run the app

To run the app, just go the home directory of the app, and type
    
    python app.py 1>>app.log 2>&1

Attention, to run the app, `ckpt` data of the CNN model must be in the `APP_HOME/deepLearning/ckpt_data` directory.

### Train the model

Trainning data is from [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/)

Before train the model, you must preprocess the data, in the home directory of the app, type

    $ cd deepLearning
    $ python preprocess_data.py $train_path $test_path $extra_path

and after that, there is a pickled file named `SVHN.pickle` in the directory, to train the model, just type

    $ python train_model.py pickled_data_path ckpt_data_path

or you can just type the following command to train the model

    $ python train_model.py SVHN.pickle ckpt_data/SVHN.ckpt

on GTX 660 graph card with 8G memory, 8 cores cpu, it costs about 35min to train the model.
