import numpy as np
import time
import os
import cv2
import pickle
import mxnet as mx
import argparse
import sys
from sklearn.preprocessing import normalize
import tensorflow as tf
import larq
from scipy import interpolate
from verification import evaluate
from sklearn import metrics
from scipy.optimize import brentq


use_flip = True
# emb_size = 128


def load_caffe_model(model_dir):
    files = os.listdir(model_dir)
    deploy_files = []
    model_files = []
    for f in files:
        if f.endswith('.prototxt'):
            deploy_files.append(f)
        elif f.endswith('.caffemodel'):
            model_files.append(f)

    if len(deploy_files) == 0:
        raise ValueError('No deploy file found in the model directory (%s)' % model_dir)
    elif len(deploy_files) > 1:
        raise ValueError('There should not be more than one deploy file in the model directory (%s)' % model_dir)
    deploy_file = model_dir + deploy_files[0]

    if len(model_files) == 0:
        raise ValueError('No caffemodel file found in the model directory (%s)' % model_dir)
    elif len(model_files) > 1:
        raise ValueError('There should not be more than one caffemodel file in the model directory (%s)' % model_dir)
    model_file = model_dir + model_files[0]

    net = cv2.dnn.readNetFromCaffe(deploy_file, model_file)
    return net


def load_mxnet_model(image_size, model_str, layer):
    ctx = mx.cpu()
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer+'_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def load_data(db_name, image_size, args):
    bins, issame_list = pickle.load(open(os.path.join(args.eval_db_path, db_name+'.bin'), 'rb'), encoding='bytes')
    datasets = np.empty((len(issame_list)*2, image_size[0], image_size[1], 3))

    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        # img = cv2.imdecode(np.fromstring(_bin, np.uint8), -1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        datasets[i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(datasets.shape)

    return datasets, issame_list


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_feat_caffe(args, buffer, model):
    blob = cv2.dnn.blobFromImages(buffer, 1, args.image_size, args.image_mean)
    model.setInput(blob)
    # feat = np.zeros((len(buffer), args.emb_size), dtype=np.float32)
    feat = model.forward("feat_extract")  # ("feat_extract") ("pool5/7x7_s1")
    feat = feat[:, :, 0, 0]
    return feat


def get_feat_keras(args, buffer, model):
    data = tf.convert_to_tensor(buffer)
    emb = model(data)
    
    if use_flip:
        emb_f = model(tf.image.flip_left_right(data))
        emb = emb + emb_f

    emb = (np.array(emb))
    return emb


def get_feat_mxnet(args, buffer, model):

    if use_flip:
        input_blob = np.zeros((len(buffer) * 2, 3, args.image_size[0], args.image_size[1]))
    else:
        input_blob = np.zeros((len(buffer), 3, args.image_size[0], args.image_size[1]))
    idx = 0
    for item in buffer:
        img = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)  # cv2.imread(item)[:, :, ::-1]  # to rgb
        img = np.transpose(img, (2, 0, 1))
        attempts = [0, 1] if use_flip else [0]
        for flipid in attempts:
            _img = np.copy(img)
            if flipid == 1:
                do_flip(_img)
            input_blob[idx] = _img
            idx += 1
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    _embedding = model.get_outputs()[0].asnumpy()
    embedding = np.zeros((len(buffer), args.emb_size), dtype=np.float32)
    if use_flip:
        embedding1 = _embedding[0::2]
        embedding2 = _embedding[1::2]
        embedding = embedding1 + embedding2
    else:
        embedding = _embedding
    return embedding


def get_feature(args, buffer, model):

    if args.model_type == 'caffe':
        embedding = get_feat_caffe(args, buffer, model)
    elif args.model_type == 'mxnet':
        embedding = get_feat_mxnet(args, buffer, model)
    elif args.model_type == 'h5':
        embedding = get_feat_keras(args, buffer, model)

    return embedding


def main(args):
    # Load the model
    if args.model_type == 'caffe':
        model = load_caffe_model(args.model)
        read_image_func = cv2.imread
    elif args.model_type == 'mxnet':
        model = load_mxnet_model(args.image_size, args.model, 'fc1')
        read_image_func = cv2.imread
    elif args.model_type == 'h5':
        model = tf.keras.models.load_model(args.model, compile=False)
        #read_image_func = lambda img: (tf.cast(tf.image.decode_image(tf.io.read_file(img), channels=3, expand_animations=False),"float32") - 127.5) * 0.0078125
        read_image_func = lambda img: tf.cast(tf.image.decode_image(tf.io.read_file(img), channels=3, expand_animations=False),"float32")
        
    for db in args.eval_datasets:
        print('begin db %s convert.' % db)

        path = os.path.join(args.eval_db_path, db, '_align_MTCNN/images/')

        start_time = time.time()

        filename = open(os.path.join(args.eval_db_path, db, "names_all.txt"), 'r')
        f = filename.readlines()

        features_all = np.zeros((len(f), args.emb_size), dtype=np.float32)

        ii = 0
        fstart = 0
        buffer = []
        for i in f:
            if ii % 1000 == 0:
                print("processing ", ii)
            ii += 1
            line = i.split()
            name = path + line[0]
            img = read_image_func(name)
            buffer.append(img)
            if len(buffer) == args.batch_size:
                embedding = get_feature(args, buffer, model)
                buffer = []
                fend = fstart + embedding.shape[0]
                features_all[fstart:fend, :] = embedding
                fstart = fend
        if len(buffer) > 0:
            embedding = get_feature(args, buffer, model)
            fend = fstart + embedding.shape[0]
            features_all[fstart:fend, :] = embedding

        features_all = normalize(features_all)

        if args.save_feat:
            np.savetxt(os.path.join(args.eval_db_path, db, '_align_MTCNN/features', args.feat_name),
                       features_all, fmt="%s")

        duration = time.time() - start_time
        print(duration)


def parse_arguments(argv):
    """test parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the proto_file and model_file',
                        default='models/Luis/QuickNet/quicknet_FT_scface/'
                                'quicknet_FT_scface_50_lr_112x112_folders_basic_lfw_14_hr2lr_interArea_epoch_1_binary_0.752667')
    parser.add_argument('--model_type', default='h5', help='type of pretrained model')  # caffe, keras (h5) or mxnet
    parser.add_argument('--image_size', default=(112, 112), help='the image size')
    parser.add_argument('--image_mean', default=(91.4953, 103.8827, 131.0912), help='the image mean')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=32)
    parser.add_argument('--emb_size', type=int, default=128, help='Size of the feature vector')
    parser.add_argument('--eval_datasets', default=['SCFace'], help='evaluation datasets')
    parser.add_argument('--eval_db_path', default='/media/yoanna/Data/CENATAV/TEORICO/deep_learning/experiments/low_resolution/datasets/',
                                                  help='evaluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--feat_name', type=str, help='feature_name for the database',
                        default='quicknet_FT_scface.txt')
    parser.add_argument('--save_feat', type=bool, help='feature_name for the database',
                        default=True)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))








