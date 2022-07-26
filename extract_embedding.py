import tensorflow as tf
import os
import glob2
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize

import larq

#Load model

checkpoint_directory = os.path.join(".","checkpoints","binaryfacenet_best2")
model_name = "keras_binaryfacenet_retina_basic_agedb_30_epoch_19_binary_0.751167.h5"
flip = False
batch_size = 32
data_path = './datasets/test_data'

basic_model = tf.keras.models.load_model(os.path.join(checkpoint_directory,model_name),compile=False)

#load data
class ImageClassesRule_map:
    def __init__(self, dir, dir_rule="*", excludes=[]):
        raw_classes = [os.path.basename(ii) for ii in glob2.glob(os.path.join(dir, dir_rule))]
        self.raw_classes = sorted([ii for ii in raw_classes if ii not in excludes])
        self.classes_2_indices = {ii: id for id, ii in enumerate(self.raw_classes)}
        self.indices_2_classes = {vv: kk for kk, vv in self.classes_2_indices.items()}

    def __call__(self, image_name):
        raw_image_class = os.path.basename(os.path.dirname(image_name))
        return self.classes_2_indices[raw_image_class]

def tf_imread(file_path):
    # tf.print('Reading file:', file_path)
    img = tf.io.read_file(file_path)
    # img = tf.image.decode_jpeg(img, channels=3)  # [0, 255]
    img = tf.image.decode_image(img, channels=3, expand_animations=False)  # [0, 255]
    img = tf.cast(img, "float32")  # [0, 255]
    return img


AUTOTUNE = tf.data.experimental.AUTOTUNE

image_classes_rule = ImageClassesRule_map(data_path)

image_names = glob2.glob(os.path.join(data_path, "*", "*.jpg"))
image_names += glob2.glob(os.path.join(data_path, "*", "*.png"))
image_names = np.random.permutation(image_names).tolist()
image_classes = [image_classes_rule(ii) for ii in image_names]

classes = np.max(image_classes) + 1 if len(image_classes) > 0 else 0
total_images = len(image_names)


ds = tf.data.Dataset.from_tensor_slices((image_names)).shuffle(buffer_size=total_images)
process_func = lambda imm: (tf_imread(imm))
ds = ds.map(process_func, num_parallel_calls=AUTOTUNE)
ds = ds.batch(batch_size) 
#ds = ds.prefetch(buffer_size=AUTOTUNE)

steps = int(total_images / batch_size)

#extract embeddings
embs = []
for img_batch in tqdm(ds, "Evaluating ", total=steps):
    emb = basic_model(img_batch)
    if flip:
        emb_f = basic_model(tf.image.flip_left_right(img_batch))
        emb = emb + emb_f
    embs.extend(np.array(emb))
embs= np.array(embs)
#embs = normalize(embs)
#embs_a = embs[::2]
#embs_b = embs[1::2]
#dists = (embs_a * embs_b).sum(1)
print(embs.shape)
print(embs)
print("Done!")

