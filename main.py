import os
import xml.etree.ElementTree
import random
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
import numpy as np
from numpy import zeros, asarray
import matplotlib.pyplot as plt


# class that defines and loads the human dataset
datadir = "C:/Users/praga/OneDrive/Desktop/Master thesis docs/RADARTHESISSAMPLE/loadradardatsetinpython/dataset"


class HumanDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, datadir, is_train=True):
        # define classes
        self.add_class("dataset", 1, "Human")
        # define data locations
        images_dir = datadir + '/JPEGImages/'
        annotations_dir = datadir + '/Annotations/'
        # find all images
        for filename in os.listdir(images_dir):
            # print(filename)
            # extract image id
            image_ids = filename[:-4]
            # print('IMAGE ID: ',image_id)

            # if is_train and int(image_id) >= 150:
            #     continue
            # if not is_train and int(image_id) < 150:
            #     continue

            img_path = images_dir + filename
            ann_path = annotations_dir + image_ids + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_ids, path=img_path, annotation=ann_path)

# A helper method to extract the bounding boxes from the annotation file
    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)
        print(filename)
        root = tree.getroot()

        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(float(box.find('xmin').text))
            ymin = int(float(box.find('ymin').text))
            xmax = int(float(box.find('xmax').text))
            ymax = int(float(box.find('ymax').text))
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
            print(coors,"Coordinates")
            # print(filename,"Filename")
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width,height

        # Loads the binary masks for an image.
    def load_mask(self, image_ids):
         info = self.image_info[image_ids]
         path = info['annotation']
         boxes, w, h = self.extract_boxes(path)
         masks = zeros([h, w, len(boxes)], dtype='uint8')

         class_ids = list()
         for i in range(len(boxes)):
             box = boxes[i]
             row_s, row_e = box[1], box[3]
             col_s, col_e = box[0], box[2]
             masks[row_s:row_e, col_s:col_e, i] = 1
             class_ids.append(self.class_names.index('Human'))
         return masks, asarray(class_ids, dtype='int32')


        # # load an image reference
        # def image_reference(self, image_ids):
        #     info = self.image_info[image_ids]
        # return info['path']


# Train
train_set = HumanDataset()
train_set.load_dataset(datadir, is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

num=random.randint(0, len(train_set.image_ids))
# define image id
image_ids = num
# load the image
image = train_set.load_image(image_ids)
print (image_ids, "ImageID")
# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_ids)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names)


