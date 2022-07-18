from tensorflow import keras
import losses, train, models
import tensorflow_addons as tfa
import tensorflow as tf
import os

    # basic_model = models.buildin_models("ResNet101V2", dropout=0.4, emb_shape=512, output_layer="E")
model_name="/home/lusantlueg/Documents/Keras_insightface/checkpoints/binaryfacenet_best2/keras_binaryfacenet_retina_basic_agedb_30_epoch_19_binary_0.751167.h5"
#basic_model = models.buildin_models(model_name, dropout=0, emb_shape=128, output_layer="E")
#basic_model = model_name

data_path = '/home/lusantlueg/Documents/Keras_insightface/datasets/scface_50_lr_112x112_folders/'
eval_paths = [os.path.join(data_path,'lfw_14.bin'), os.path.join(data_path,'lfw_14_hr2lr_interArea.bin'),os.path.join(data_path,'lfw_7.bin'), os.path.join(data_path,'lfw_7_hr2lr_interArea.bin')]

tt = train.Train(data_path, save_path='keras_binaryfacenet_scface50.h5', eval_paths=eval_paths,
                model=model_name, batch_size=64, random_status=0,
                lr_base=0.001, lr_decay=0.1, lr_decay_steps=[9,15,18], lr_min=1e-5)
optimizer = tfa.optimizers.SGDW(learning_rate=0.1, momentum=0.9, weight_decay=5e-5)
#optimizer = tfa.optimizers.AdamW(weight_decay=5e-5)
sch = [
 {"loss": losses.ArcfaceLoss(scale=16), "epoch": 20, "optimizer": optimizer},
 #{"loss": losses.ArcfaceLoss(scale=32), "epoch": 15, "optimizer": optimizer},
 #{"loss": losses.ArcfaceLoss(scale=64), "epoch": 15, "optimizer": optimizer},
 # {"loss": losses.ArcfaceLoss(), "epoch": 20, "triplet": 64, "alpha": 0.35},
]
tt.train(sch, 0)


