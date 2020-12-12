import pandas as pd
import numpy as np
import os
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
import csv
import shutil


# if you are using CUB dataset, use this function
def make_cub_dataset(folder):

    split = {}

    train_count = {}
    test_count = {}

    with open('{}/classes.txt'.format(folder)) as fp:
        line = fp.readline()
        while line:

            line = line.strip()
            class_id, class_name = line.split(' ')
            folder_name = class_name.split('.')[0]
            train_count[folder_name] = 1
            test_count[folder_name] = 1

            cwd = os.getcwd()
            train_path = os.path.join(cwd, '{}/train'.format(folder))
            test_path = os.path.join(cwd, '{}/test'.format(folder))

            train_folder = '{}/{}'.format(train_path,
                                          'class_' + str(folder_name))
            test_folder = '{}/{}'.format(test_path,
                                         'class_' + str(folder_name))

            if not os.path.exists(train_folder):
                os.makedirs(train_folder)

            if not os.path.exists(test_folder):
                os.makedirs(test_folder)

            line = fp.readline()

        print('done making folders')

    with open('{}/train_test_split.txt'.format(folder)) as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            image_id, image_split = line.split(' ')
            split[int(image_id)] = 'train' if int(image_split) else 'test'
            line = fp.readline()

    with open('{}/images.txt'.format(folder)) as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            image_id, image_path = line.split(' ')
            img_class = image_path.split('.')[0]
            image_split = split[int(image_id)]
            full_image_path = r'{}/images/{}'.format(folder, image_path)

            if image_split == 'train':
                iter = train_count[img_class]
                os.system(
                    'cp {} {}/train/class_{}/{}.jpg'.format(full_image_path, folder, img_class, str(iter)))
                train_count[img_class] += 1
            else:
                iter = test_count[img_class]
                os.system(
                    'cp {} {}/test/class_{}/{}.jpg'.format(full_image_path, folder, img_class, str(iter)))
                test_count[img_class] += 1

            line = fp.readline()

        print('completed setting up image dataset')


def make_custom_dataset(folder):


    with open('datasets/classes.csv') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for row in csv_reader:
                
                print('working on {}th element'.format(line_count))
                line_count += 1

                train_folder = '{}/{}'.format('datasets/train', row[1])
                test_folder = '{}/{}'.format('datasets/test', row[1])

                if not os.path.exists(train_folder):
                    os.makedirs(train_folder)

                if not os.path.exists(test_folder):
                    os.makedirs(test_folder)


    with open('datasets/train.txt') as fp:
        line = fp.readline()

        while line:
            line = line.strip()

            image_path, img_id = line.split(' ')

            full_image_path = os.path.join(folder, image_path)

            new_path = image_path.split('images/')[1]
            new_path = os.path.join(folder, 'train', new_path)

            print('processing image from:{}'.format(new_path))


            shutil.copyfile(full_image_path, new_path )

            line = fp.readline()


    with open('datasets/test.txt') as fp:

        line = fp.readline()

        while line:
            line = line.strip()

            image_path, img_id = line.split(' ')

            full_image_path = os.path.join(folder, image_path)

            new_path = image_path.split('images/')[1]
            new_path = os.path.join(folder, 'test', new_path)

            print('processing image from:{}'.format(new_path))

            shutil.copyfile(full_image_path, new_path)

            line = fp.readline()



if __name__ == '__main__':

    folder = 'datasets'

    make_cub_dataset(folder)

   