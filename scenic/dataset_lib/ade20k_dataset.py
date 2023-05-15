# Copyright 2023 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generators for the Cityscapes dataset."""

import collections
import functools
from typing import Optional
from .ade20_tfds import Ade20k

from absl import logging
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf

# Based on https://github.com/mcordts/cityscapesScripts
ADE20KClass = collections.namedtuple(
    'ADE20KClass',
    ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances',
     'ignore_in_eval', 'color'])

ADE_CLASSES = ("wall", "building, edifice", "sky", "floor, flooring", "tree",
               "ceiling", "road, route", "bed", "windowpane, window", "grass",
               "cabinet", "sidewalk, pavement",
               "person, individual, someone, somebody, mortal, soul",
               "earth, ground", "door, double door", "table", "mountain, mount",
               "plant, flora, plant life", "curtain, drape, drapery, mantle, pall",
               "chair", "car, auto, automobile, machine, motorcar",
               "water", "painting, picture", "sofa, couch, lounge", "shelf",
               "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair",
               "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press",
               "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion",
               "base, pedestal, stand", "box", "column, pillar", "signboard, sign",
               "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink",
               "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox",
               "grandstand, covered stand", "path", "stairs, steps", "runway",
               "case, display case, showcase, vitrine",
               "pool table, billiard table, snooker table", "pillow",
               "screen door, screen", "stairway, staircase", "river", "bridge, span",
               "bookcase", "blind, screen", "coffee table, cocktail table",
               "toilet, can, commode, crapper, pot, potty, stool, throne",
               "flower", "book", "hill", "bench", "countertop",
               "stove, kitchen stove, range, kitchen range, cooking stove",
               "palm, palm tree", "kitchen island",
               "computer, computing machine, computing device, data processor, "
               "electronic computer, information processing system",
               "swivel chair", "boat", "bar", "arcade machine",
               "hovel, hut, hutch, shack, shanty",
               "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, "
               "motorcoach, omnibus, passenger vehicle",
               "towel", "light, light source", "truck, motortruck", "tower",
               "chandelier, pendant, pendent", "awning, sunshade, sunblind",
               "streetlight, street lamp", "booth, cubicle, stall, kiosk",
               "television receiver, television, television set, tv, tv set, idiot "
               "box, boob tube, telly, goggle box",
               "airplane, aeroplane, plane", "dirt track",
               "apparel, wearing apparel, dress, clothes",
               "pole", "land, ground, soil",
               "bannister, banister, balustrade, balusters, handrail",
               "escalator, moving staircase, moving stairway",
               "ottoman, pouf, pouffe, puff, hassock",
               "bottle", "buffet, counter, sideboard",
               "poster, posting, placard, notice, bill, card",
               "stage", "van", "ship", "fountain",
               "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
               "canopy", "washer, automatic washer, washing machine",
               "plaything, toy", "swimming pool, swimming bath, natatorium",
               "stool", "barrel, cask", "basket, handbasket", "waterfall, falls",
               "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle",
               "oven", "ball", "food, solid food", "step, stair", "tank, storage tank",
               "trade name, brand name, brand, marque", "microwave, microwave oven",
               "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna",
               "bicycle, bike, wheel, cycle", "lake",
               "dishwasher, dish washer, dishwashing machine",
               "screen, silver screen, projection screen",
               "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase",
               "traffic light, traffic signal, stoplight", "tray",
               "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, "
               "dustbin, trash barrel, trash bin",
               "fan", "pier, wharf, wharfage, dock", "crt screen",
               "plate", "monitor, monitoring device", "bulletin board, notice board",
               "shower", "radiator", "glass, drinking glass", "clock", "flag")

palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                 [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                 [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                 [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                 [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                 [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                 [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                 [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                 [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                 [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                 [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                 [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                 [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                 [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                 [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                 [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                 [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                 [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                 [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                 [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                 [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                 [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                 [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                 [102, 255, 0], [92, 0, 255]]
CLASSES = []
for i, name in enumerate(ADE_CLASSES):
    CLASSES.append(
        ADE20KClass(
            name=name,
            id=i,
            train_id=i,
            category='none',
            category_id=0,
            has_instances=False,
            ignore_in_eval=False,
            color=palette[i]
    )
        )



# Number of pixels per Cityscapes class ID in the training set:
PIXELS_PER_CID = {
    7: 3806423808,
    8: 629490880,
    11: 2354443008,
    12: 67089092,
    13: 91210616,
    17: 126753000,
    19: 21555918,
    20: 57031712,
    21: 1647446144,
    22: 119165328,
    23: 415038624,
    24: 126403824,
    25: 13856368,
    26: 725164864,
    27: 27588982,
    28: 24276994,
    31: 24195352,
    32: 10207740,
    33: 42616088
}


def preprocess_example(example, train, dtype=tf.float32, resize=None):
  """Preprocesses the given image.

  Args:
    example: dict; Example coming from TFDS.
    train: bool; Whether to apply training-specific preprocessing or not.
    dtype: Tensorflow data type; Data type of the image.
    resize: sequence; [H, W] to which image and labels should be resized.

  Returns:
    An example dict as required by the model.
  """
  image = dataset_utils.normalize(example['image'], dtype)
  mask = example['segmentation']

  # Resize test images (train images are cropped/resized during augmentation):
  if not train:
    if resize is not None:
      image = tf.image.resize(image, resize, 'bilinear')
      mask = tf.image.resize(mask, resize, 'nearest')

  image = tf.cast(image, dtype)
  mask = tf.cast(mask, dtype)
  mask = tf.squeeze(mask, axis=2)
  return {'inputs': image, 'label': mask}


def augment_example(
    example, dtype=tf.float32, resize=None, **inception_crop_kws):
  """Augments the given train image.

  Args:
    example: dict; Example coming from TFDS.
    dtype: Tensorflow data type; Data type of the image.
    resize: sequence; [H, W] to which image and labels should be resized.
    **inception_crop_kws: Keyword arguments passed on to
      inception_crop_with_mask.

  Returns:
    An example dict as required by the model.
  """
  image = example['inputs']
  mask = example['label'][..., tf.newaxis]

  # Random crop and resize ("Inception crop"):
  image, mask = dataset_utils.inception_crop_with_mask(
      image,
      mask,
      resize_size=image.shape[-3:-1] if resize is None else resize,
      **inception_crop_kws)

  # Random LR flip:
  seed = tf.random.uniform(shape=[2], maxval=2**31 - 1, dtype=tf.int32)
  image = tf.image.stateless_random_flip_left_right(image, seed)
  mask = tf.image.stateless_random_flip_left_right(mask, seed)

  image = tf.cast(image, dtype)
  mask = tf.cast(mask, dtype)
  mask = tf.squeeze(mask, axis=2)
  return {'inputs': image, 'label': mask}


def get_post_exclusion_labels():
  """Determines new labels after excluding bad classes.

  See Figure 1 in https://arxiv.org/abs/1604.01685 for which classes are
  excluded. Excluded classes get the new label -1.

  Returns:
    An array of length num_old_classes, containing new labels.
  """
  old_to_new_labels = np.array(
      [-1 if c.ignore_in_eval else c.train_id for c in CLASSES])
  assert np.all(np.diff([i for i in old_to_new_labels if i >= 0]) == 1)
  return old_to_new_labels


def get_class_colors():
  """Returns a [num_classes, 3] array of colors for the model output labels."""
  cm = np.stack([c.color for c in CLASSES if not c.ignore_in_eval], axis=0)
  return cm / 255.0


def get_class_names():
  """Returns a list with the class names of the model output labels."""
  return [c.name for c in CLASSES if not c.ignore_in_eval]


def get_class_proportions():
  """Returns a [num_classes] array of pixel frequency proportions."""
  p = [PIXELS_PER_CID[c.id] for c in CLASSES if not c.ignore_in_eval]
  return np.array(p) / np.sum(p)


def exclude_bad_classes(batch, new_labels):
  """Adjusts masks and batch_masks to exclude void and rare classes.

  This must be applied after dataset_utils.maybe_pad_batch() because we also
  update the batch_mask. Note that the data is already converted to Numpy by
  then.

  Args:
    batch: dict; Batch of data examples.
    new_labels: nd-array; array of length num_old_classes, containing new
      labels.

  Returns:
    Updated batch dict.
  """
  # Convert old labels to new labels:
  batch['label'] = batch['label'] - 1
  batch['label'] = new_labels[batch['label'].astype(np.int32)]

  # Set batch_mask to 0 at pixels that have an excluded label:
  mask_dtype = batch['batch_mask'].dtype
  batch['batch_mask'] = (
      batch['batch_mask'].astype(np.bool_) & (batch['label'] != -1))
  batch['batch_mask'] = batch['batch_mask'].astype(mask_dtype)

  return batch


@datasets.add_dataset('ade20k')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns generators for the Cityscapes train, validation, and test set.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  del rng
  dtype = getattr(tf, dtype_str)
  dataset_configs = dataset_configs or {}
  target_size = dataset_configs.get('target_size', None)

  logging.info('Loading train split of the Cityscapes dataset.')
  preprocess_ex_train = functools.partial(
      preprocess_example, train=True, dtype=dtype, resize=None)
  augment_ex = functools.partial(
      augment_example, dtype=dtype, resize=target_size, area_min=30,
      area_max=100)

  train_split = dataset_configs.get('train_split', 'train')
  train_ds, _ = dataset_utils.load_split_from_tfds(
      'Ade20k',
      batch_size,
      split=train_split,
      preprocess_example=preprocess_ex_train,
      augment_train_example=augment_ex,
      shuffle_seed=shuffle_seed)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

  logging.info('Loading validation split of the Cityscapes dataset.')
  preprocess_ex_eval = functools.partial(
      preprocess_example, train=False, dtype=dtype, resize=target_size)
  eval_ds, _ = dataset_utils.load_split_from_tfds(
      'Ade20k', eval_batch_size, split='validation',
      preprocess_example=preprocess_ex_eval)

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size,
      pixel_level=True)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size,
      pixel_level=True)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  exclude_classes = functools.partial(
      exclude_bad_classes, new_labels=get_post_exclusion_labels())

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(exclude_classes, train_iter)
  train_iter = map(shard_batches, train_iter)

  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(maybe_pad_batches_eval, eval_iter)
  eval_iter = map(exclude_classes, eval_iter)
  eval_iter = map(shard_batches, eval_iter)

  if target_size is None:
    input_shape = (-1, 1024, 2048, 3)
  else:
    input_shape = (-1,) + tuple(target_size) + (3,)

  meta_data = {
      'num_classes':
          len([c.id for c in CLASSES if not c.ignore_in_eval]),
      'input_shape':
          input_shape,
      'num_train_examples':
          dataset_utils.get_num_examples('cityscapes', train_split),
      'num_eval_examples':
          dataset_utils.get_num_examples('cityscapes', 'validation'),
      'input_dtype':
          getattr(jnp, dtype_str),
      'target_is_onehot':
          False,
      'class_names':
          get_class_names(),
      'class_colors':
          get_class_colors(),
    #   'class_proportions':
    #       get_class_proportions(),
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
