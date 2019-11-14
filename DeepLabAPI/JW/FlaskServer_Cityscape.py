import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from PIL import Image
import json, argparse, time
from flask import Flask, request
from flask_cors import CORS

from DeepLabV3 import DeepLabModel
from six.moves import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt


LABEL_NAMES_ADE20K = np.array(['wall' ,'building' ,'sky' ,'floor' ,'tree' ,'ceiling' ,'road' ,'bed' ,'windowpane' ,'grass' ,'cabinet' ,'sidewalk' ,'person' ,'earth' ,'door' ,'table' ,'mountain' ,'plant' ,'curtain' ,'chair' ,'car' ,'water' ,'painting' ,'sofa' ,'shelf' ,'house' ,'sea' ,'mirror' ,'rug' ,'field' ,'armchair' ,'seat' ,'fence' ,'desk' ,'rock' ,'wardrobe' ,'lamp' ,'bathtub' ,'railing' ,'cushion' ,'base' ,'box' ,'column' ,'signboard' ,'chest of drawers' ,'counter' ,'sand' ,'sink' ,'skyscraper' ,'fireplace' ,'refrigerator' ,'grandstand' ,'path' ,'stairs' ,'runway' ,'case' ,'pool table' ,'pillow' ,'screen door' ,'stairway' ,'river' ,'bridge' ,'bookcase' ,'blind' ,'coffee table' ,'toilet' ,'flower' ,'book' ,'hill' ,'bench' ,'countertop' ,'stove' ,'palm' ,'kitchen island' ,'computer' ,'swivel chair' ,'boat' ,'bar' ,'arcade machine' ,'hovel' ,'bus' ,'towel' ,'light' ,'truck' ,'tower' ,'chandelier' ,'awning' ,'streetlight' ,'booth' ,'television' ,'airplane' ,'dirt track' ,'apparel' ,'pole' ,'land' ,'bannister' ,'escalator' ,'ottoman' ,'bottle' ,'buffet' ,'poster' ,'stage' ,'van' ,'ship' ,'fountain' ,'conveyer belt' ,'canopy' ,'washer' ,'plaything' ,'swimming pool' ,'stool' ,'barrel' ,'basket' ,'waterfall' ,'tent' ,'bag' ,'minibike' ,'cradle' ,'oven' ,'ball' ,'food' ,'step' ,'tank' ,'trade name' ,'microwave' ,'pot' ,'animal' ,'bicycle' ,'lake' ,'dishwasher' ,'screen' ,'blanket' ,'sculpture' ,'hood' ,'sconce' ,'vase' ,'traffic light' ,'tray' ,'ashcan' ,'fan' ,'pier' ,'crt screen' ,'plate' ,'monitor' ,'bulletin board' ,'shower' ,'radiator' ,'glass' ,'clock' ,'flag'])
#LABEL_NAMES_CITYSCAPES = np.array(['unlabeled','ego vehicle','rectification border','out of roi','static' ,'dynamic','ground','road','sidewalk','parking','rail track','building','wall','fence' , 'guard rail' ,'bridge','tunnel' ,'pole','polegroup','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus'  ,'caravan','trailer' ,'train','motorcycle','bicycle','license plate']);
LABEL_NAMES_CITYSCAPES = np.array(['road','sidewalk','parking','rail track','person'
                                      ,'rider','car','truck','bus' ,'on rails'
                                      ,'motorcycle','bicycle','caravan','trailer','building'
                                      ,'wall','fence', 'guard rail' ,'bridge','tunnel'
                                      ,'pole','pole group','traffic sign','traffic light','vegetation'
                                      ,'terrain','sky','ground','static', 'dynamic' ]);
LABEL_NAMES_PASCAL = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])



download_path1 = "../model/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz"
download_path2 = "../model/deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz"
download_path_cityscape1 = "../model/deeplabv3_cityscapes_train_2018_02_06.tar.gz"
download_path_cityscape2 = "../model/deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz"
download_path_cityscape3 = "../model/deeplab_cityscapes_xception71_trainvalfine_2018_09_08.tar.gz"
download_path_pascal1 = "../model/deeplabv3_pascal_trainval_2018_01_04.tar.gz"
download_path_pascal2 = "../model/deeplabv3_pascal_train_aug_2018_01_04.tar.gz"


LABEL_NAMES = LABEL_NAMES_CITYSCAPES;
model_path = download_path_cityscape2;

print(len(LABEL_NAMES))

#MODEL = DeepLabModel(download_path1)
#MODEL.store(saved_model_path)
#resized_img, res =MODEL.runWithCV('./data/ki_corridor/1.png')
#plt.figure(figsize=(20, 15))
#plt.imshow(res)
#plt.show()


###########################
# ADE20K Label names & Color map
###########################

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()

def GetColorMap(image, seg_map):
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    unique_labels = np.unique(seg_map)
    print(FULL_LABEL_MAP[unique_labels].astype(np.uint8))
    #print(unique_labels)
    return seg_image

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)
@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()

    data = request.data.decode("utf-8")
    print(0);
    if data == "":
        params = request.form
        x_in = json.loads(params['image'])

    else:
        params = json.loads(data)
        x_in = params['image']

    ############
    #Convert PIL Image
    ######
    width = len(x_in[0])
    height = len(x_in)

    print(1);
    na = np.array(x_in, dtype=np.uint8)

    img = Image.fromarray(na, 'RGB')

    img.save('./data/target.jpg')
    im = img.load()
    print(2);
    #img = Image.new('RGB', (width, height))
    #img.putdata(tuple(x_in))

    #plt.figure(figsize=(20, 15))
    #plt.imshow(img)
    #plt.show()

    ##################################################
    # Tensorflow part
    ##################################################
    resized_img, seg_map = MODEL.run(img)
    print(3);
    seg_image = GetColorMap(resized_img, seg_map)
    print(4);
    #y_out = persistent_sess.run(y, feed_dict={
    #    x: x_in
    #})
    ##################################################
    # END Tensorflow part
    ##################################################

    json_data = json.dumps({'seg_img': seg_image.tolist()})
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data
##################################################
# END API part
##################################################

if __name__ == "__main__":

    ##################################################
    # Tensorflow part
    ##################################################
    MODEL = DeepLabModel(model_path)
    graph = MODEL.graph
    x = graph.get_tensor_by_name(MODEL.INPUT_TENSOR_NAME)
    y = graph.get_tensor_by_name(MODEL.OUTPUT_TENSOR_NAME)
    print(x)
    print(y)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(graph=graph, config=sess_config)
    ##################################################
    # END Tensorflow part
    ##################################################

    print('Starting the API')
    app.run(host='143.248.96.81', port = 35006)