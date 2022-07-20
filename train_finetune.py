from tensorflow import keras
import losses, train, models
import tensorflow_addons as tfa
import tensorflow as tf
import os

datasets_root = './datasets'
dataset_name = 'scface_50_lr_112x112_folders'

model_root = './checkpoints'
model_name='quicknet'
checkpoint_path=os.path.join(model_root,'quicknet_best','keras_quicknet_adaptativelr_retina_basic_agedb_30_epoch_17_binary_0.890000.h5')

data_path = os.path.join(datasets_root,dataset_name)

eval_paths = [os.path.join(data_path,'lfw_7.bin'), os.path.join(data_path,'lfw_7_hr2lr_interArea.bin'),os.path.join(data_path,'lfw_14.bin'), os.path.join(data_path,'lfw_14_hr2lr_interArea.bin')]

tt = train.Train(data_path, save_path=model_name+'_FT_'+dataset_name+'.h5', eval_paths=eval_paths,
                model=checkpoint_path, batch_size=128, random_status=0,
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


