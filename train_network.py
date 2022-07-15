from tensorflow import keras
import losses, train, models
import tensorflow_addons as tfa
import tensorflow as tf

    # basic_model = models.buildin_models("ResNet101V2", dropout=0.4, emb_shape=512, output_layer="E")
model_name="binaryfacenet"
basic_model = models.buildin_models(model_name, dropout=0, emb_shape=128, output_layer="E")
data_path = '/home/lusantlueg/Documents/Keras_insightface/datasets/ms1m-retinaface-t1'
eval_paths = ['/home/lusantlueg/Documents/Keras_insightface/datasets/ms1m-retinaface-t1/lfw.bin', '/home/lusantlueg/Documents/Keras_insightface/datasets/ms1m-retinaface-t1/cfp_fp.bin', '/home/lusantlueg/Documents/Keras_insightface/datasets/ms1m-retinaface-t1/agedb_30.bin']

tt = train.Train(data_path, save_path='keras_binaryfacenet_retina.h5', eval_paths=eval_paths,
                basic_model=basic_model, batch_size=64, random_status=0,
                lr_base=0.01, lr_decay=0.1, lr_decay_steps=[9,15,18], lr_min=1e-5)
optimizer = tfa.optimizers.SGDW(learning_rate=0.1, momentum=0.9, weight_decay=5e-5)
#optimizer = tfa.optimizers.AdamW(weight_decay=5e-5)
sch = [
 {"loss": losses.ArcfaceLoss(scale=16), "epoch": 20, "optimizer": optimizer},
 #{"loss": losses.ArcfaceLoss(scale=32), "epoch": 15, "optimizer": optimizer},
 #{"loss": losses.ArcfaceLoss(scale=64), "epoch": 15, "optimizer": optimizer},
 # {"loss": losses.ArcfaceLoss(), "epoch": 20, "triplet": 64, "alpha": 0.35},
]
tt.train(sch, 0)


