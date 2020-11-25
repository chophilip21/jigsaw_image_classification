import os


def make_dataset(folder):

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


if __name__ == "__main__":

    make_dataset('bird')  # for CUB
