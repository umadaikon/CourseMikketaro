
# coding: utf-8

# In[3]:


import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import csv

from ssd import SSD300
from ssd_utils import BBoxUtility

# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))


# In[4]:


voc_classes = ['Hold']
NUM_CLASSES = len(voc_classes) + 2


# In[5]:


input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights.99-2.68.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)


# In[6]:


inputs = []
images = []
# img_path = './hold_pinakuru2IMG_94991.JPG'
# img = image.load_img(img_path, target_size=(300, 300))
# img = image.img_to_array(img)
# images.append(imread(img_path))
# inputs.append(img.copy())
img_path = './hold_pinakuru2/IMG_20181218_204546.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())


inputs = preprocess_input(np.array(inputs))


# In[7]:


preds = model.predict(inputs, batch_size=1, verbose=1)


# In[8]:


results = bbox_util.detection_out(preds)


# In[9]:


# get_ipython().run_cell_magic('time', '', 'a = model.predict(inputs, batch_size=1)\nb = bbox_util.detection_out(preds)')


# In[10]:


for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.75]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    with open('pi_9505.csv', 'w', newline='') as csv_file:
        fieldnames = ['centerx', 'centery', 'p1_x', 'p1_y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = voc_classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
            centerx = (xmin + xmax)/2
            centery = (ymin + ymax)/2

            writer.writerow({'centerx': str(centerx), 'centery': str(centery), 'p1_x': str(xmax), 'p1_y': str(ymax), 'p2_x': str(xmin), 'p2_y': str(ymax), 'p3_x': str(xmin), 'p3_y': str(ymin), 'p4_x': str(xmax), 'p4_y': str(ymin)})

            print("center_x = " + str(centerx) + ", center_y = " + str(centery))
            print("p1_x = " + str(xmax) + ", p1_y = " + str(ymax))
            print("p2_x = " + str(xmin) + ", p2_y = " + str(ymax))
            print("p3_x = " + str(xmin) + ", p3_y = " + str(ymin))
            print("p4_x = " + str(xmin) + ", p4_y = " + str(ymin))
    plt.show()
