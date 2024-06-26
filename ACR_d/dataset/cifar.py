import logging
import math

import os
import sys
import pickle
import random

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch
import torchvision
import torch.utils.data as data

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)

SVHN_mean = (0.4377, 0.4438, 0.4728)
SVHN_std = (0.1980, 0.2010, 0.1970)


def transpose(x, source='NCHW', target='NHWC'):
    return x.transpose([source.index(d) for d in target])


def compute_adjustment_list(label_list, tro, args):
    label_freq_array = np.array(label_list)
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    l_samples = make_imb_data(args.num_max, args.num_classes, args.imb_ratio_label, 1, 0)
    u_samples = make_imb_data(args.num_max_u, args.num_classes, args.imb_ratio_unlabel, 0, args.flag_reverse_LT)




    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, args)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)


    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train_labeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, l_samples, u_samples


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    l_samples = make_imb_data(args.num_max, args.num_classes, args.imb_ratio_label, 1, 0)
    u_samples = make_imb_data(args.num_max_u, args.num_classes, args.imb_ratio_unlabel, 0, args.flag_reverse_LT)

    #steps_l_per_epoch = sum(l_samples) // args.batch_size
    #steps_u_per_epoch = sum(u_samples) // (args.batch_size * args.mu)


    #dataset_l_repeat_times = args.eval_step // steps_l_per_epoch + 1
    #dataset_u_repeat_times = args.eval_step // steps_u_per_epoch + 1

    #times = max(dataset_l_repeat_times, dataset_u_repeat_times)



    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, args)



    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)


    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train_labeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    #return train_labeled_dataset, train_unlabeled_dataset, test_dataset
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, l_samples, u_samples


def get_stl10(args, root):

    transform_labeled = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)])

    transform_val = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)])

    # base_dataset = datasets.STL10(
    #     root, split="train", download=True)

    train_labeled_dataset = datasets.STL10(root, split="train", transform=transform_labeled, download=True)
    train_unlabeled_dataset = datasets.STL10(root, split="unlabeled",
                                                         transform=TransformFixMatchSTL(mean=stl10_mean, std=stl10_std),
                                                         download=True)
    test_dataset = datasets.STL10(root, split="test", transform=transform_val, download=True)


    l_samples = make_imb_data(args.num_max, args.num_classes, args.imb_ratio_label, 1, 0)
    train_labeled_idxs = train_split_l(train_labeled_dataset.labels, l_samples, args)
    train_labeled_dataset = make_imbalance(train_labeled_dataset, train_labeled_idxs)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, l_samples, l_samples


def get_svhn(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_mean, SVHN_std)
    ])


    base_dataset = datasets.SVHN(root, split='train', download=True)
    test_dataset = datasets.SVHN(root, split='test', download=True)

    test_idxs = testsplit(test_dataset.labels)

    l_samples = make_imb_data(args.num_max, args.num_classes, args.imb_ratio_label, 1, 0)
    u_samples = make_imb_data(args.num_max_u, args.num_classes, args.imb_ratio_unlabel, 0, args.flag_reverse_LT)

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.labels, l_samples, u_samples, args)

    train_labeled_dataset = SVHN_labeled(root, train_labeled_idxs, split='train', transform=transform_labeled)
    train_unlabeled_dataset = SVHN_unlabeled(root, train_unlabeled_idxs, train_labeled_idxs, split='train',
                                             transform=TransformFixMatch(mean=SVHN_mean, std=SVHN_std))
    test_dataset = SVHN_labeled(root, test_idxs, split='test', transform=transform_val, download=True)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_smallimagenet(args, root):
    assert args.img_size == 32 or args.img_size == 64, 'img size should only be 32 or 64!!!'

    root = os.path.join(root, 'small_imagenet_127_{}'.format(args.img_size))

    base_dataset = SmallImageNet(root, args.img_size, True)
    base_test_dataset = SmallImageNet(root, args.img_size, False)

    labeled_percent = 0.1

    dataset_mean = (0.48109809, 0.45747185, 0.40785507)  # np.mean(base_dataset.data, axis=(0, 1, 2)) / 255

    dataset_std = (0.26040889, 0.2532126, 0.26820634)  # np.std(base_dataset.data, axis=(0, 1, 2)) / 255

    transform_train = transforms.Compose([
        transforms.RandomCrop(args.img_size, padding=int(args.img_size / 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])


    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    # select labeled data and construct labeled dataset
    num_classes = len(set(base_dataset.targets))
    num_data_per_cls = [0 for _ in range(num_classes)]
    for l in base_dataset.targets:
        num_data_per_cls[l] += 1

    num_labeled_data_per_cls = [int(np.around(n * labeled_percent)) for n in num_data_per_cls]
    num_unlabeled_data_per_cls = [n - l for n, l in zip(num_data_per_cls, num_labeled_data_per_cls)]

    #print('total number of labeled data is ', sum(num_labeled_data_per_cls))
    # train_labeled_idxs = train_split(base_dataset.targets, num_labeled_data_per_cls, num_classes, args.seed)
    train_labeled_idxs = train_split_l(base_dataset.targets, num_labeled_data_per_cls, args)
    train_unlabeled_idxs = list(set(range(len(base_dataset.targets))) - set(train_labeled_idxs))

    test_idx = testsplit(base_test_dataset.targets, num_classes, 50)


    train_labeled_dataset = SmallImageNet(root, args.img_size, True,
                                          transform=transform_train,
                                          indexs=train_labeled_idxs,
                                          )

    train_unlabeled_dataset = SmallImageNet(root, args.img_size, True,
                                            transform=TransformFixMatch(mean=dataset_mean, std=dataset_std, img_size=args.img_size),
                                            indexs=train_unlabeled_idxs,
                                            )

    test_dataset = SmallImageNet(root, args.img_size, False, transform=transform_val, indexs=test_idx)

    arr = np.array(num_labeled_data_per_cls)
    tar_index = np.argsort(-arr)
    tar_index = tar_index.tolist()

    for idx in range(len(train_labeled_dataset.targets)):
        train_labeled_dataset.targets[idx] = tar_index.index(train_labeled_dataset.targets[idx])

    for idx in range(len(train_unlabeled_dataset.targets)):
        train_unlabeled_dataset.targets[idx] = tar_index.index(train_unlabeled_dataset.targets[idx])

    for idx in range(len(test_dataset.targets)):
        test_dataset.targets[idx] = tar_index.index(test_dataset.targets[idx])

    train_unlabeled_dataset.targets = np.array(train_unlabeled_dataset.targets)
    #train_unlabeled_dataset.targets[train_labeled_idxs] = -2
    cls_num_list_l = num_labeled_data_per_cls
    cls_num_list_u = num_unlabeled_data_per_cls

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, cls_num_list_l, cls_num_list_u

def get_smallimagenet_1k(args, root):
    assert args.img_size == 32 or args.img_size == 64, 'img size should only be 32 or 64!!!'

    root = os.path.join(root, 'small_imagenet_1k_{}'.format(args.img_size))

    dataset_mean = (0.48109809, 0.45747185, 0.40785507)  # np.mean(base_dataset.data, axis=(0, 1, 2)) / 255

    dataset_std = (0.26040889, 0.2532126, 0.26820634)  # np.std(base_dataset.data, axis=(0, 1, 2)) / 255

    transform_train = transforms.Compose([
        transforms.RandomCrop(args.img_size, padding=int(args.img_size / 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])


    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])


    base_dataset = SmallImageNet(root, args.img_size, True)


    num_max = 1280
    args.imb_ratio_label = 256
    args.imb_ratio_unlabel = 256
    labeled_percent = 0.2
    args.num_classes = 1000
    #args.num_max_l = int(num_max * labeled_percent)
    #args.num_max_u = num_max - args.num_max_l

    samples = make_imb_data(num_max, args.num_classes, args.imb_ratio_label, 1, 0)
    l_samples = []
    u_samples = []
    for i in samples:
        l_samples.append(int(i * labeled_percent))
        u_samples.append(i - int(i * labeled_percent))

    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, args)


    train_labeled_dataset = SmallImageNet(root, args.img_size, True,
                                          transform=transform_train,
                                          indexs=train_labeled_idxs,
                                          )

    train_unlabeled_dataset = SmallImageNet(root, args.img_size, True,
                                            transform=TransformFixMatch(mean=dataset_mean, std=dataset_std, img_size=args.img_size),
                                            indexs=train_unlabeled_idxs,
                                            )

    test_dataset = SmallImageNet(root, args.img_size, False, transform=transform_val)


    return train_labeled_dataset, train_unlabeled_dataset, test_dataset, l_samples, u_samples








def train_split(labels, n_labeled_per_class, n_unlabeled_per_class, args):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    for i in range(args.num_classes):
        idxs = np.where(labels == i)[0]
        #if len(idxs) < (n_labeled_per_class[i] + n_unlabeled_per_class[i]):
        #    print('Warning: class {} has less samples than expected'.format(i))
            #print('n_labeled_per_class[{}]: {}'.format(i, n_labeled_per_class[i]))
            #print('n_unlabeled_per_class[{}]: {}'.format(i, n_unlabeled_per_class[i]))
            #print('len(idxs): {}'.format(len(idxs)))

        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
        # train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs, train_unlabeled_idxs


def train_split_l(labels, n_labeled_per_class, args):
    labels = np.array(labels)
    train_labeled_idxs = []
    # train_unlabeled_idxs = []
    for i in range(args.num_classes):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        # train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
    return train_labeled_idxs


def testsplit(labels, num_classes=10, num_samples=1500):
    labels = np.array(labels)
    test_idxs=[]
    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        print(idxs)
        test_idxs.extend(idxs[:num_samples])
    np.random.shuffle(test_idxs)
    return test_idxs


def make_imbalance(dataset, indexs):
    dataset.data = dataset.data[indexs]
    try:
        dataset.labels = dataset.labels[indexs]
    except:
        dataset.targets = dataset.targets[indexs]

    return dataset


def make_imb_data(max_num, class_num, gamma, flag = 1, flag_LT = 0):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))

    if flag == 0 and flag_LT == 1:
        class_num_list = list(reversed(class_num_list))
    #if flag == 0 and flag_LT == 2:
    #    random.shuffle(class_num_list)

    if flag == 0 and flag_LT == 3:
        #centric
        cls_num_array = np.array(class_num_list)
        idx_centric = list(range(0,class_num,2))[::-1] + list(range(1,class_num,2))
        class_num_list = cls_num_array[idx_centric].tolist()

    if flag == 0 and flag_LT == 4:
        #off-centric
        cls_num_array = np.array(class_num_list)
        idx_off_centric = list(range(1,class_num,2)) + list(range(0,class_num,2))[::-1]
        class_num_list = cls_num_array[idx_off_centric].tolist()


    #logger.info('class_num_list: {}'.format(class_num_list))
    #print(f'class_num_list: {class_num_list}')
    return list(class_num_list)


class TransformFixMatch(object):
    def __init__(self, mean, std, img_size=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong1)
        # return self.normalize(weak), self.normalize(strong)


class TransformFixMatchSTL(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        strong1 = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.normalize(strong1)
        # return self.normalize(weak), self.normalize(strong)





class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, exindexs = [], train=True,
                 transform=None, target_transform=None,
                 download=False, times=1):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)
            self.targets[exindexs] = -2
            self.targets = self.targets[indexs]
            # self.targets = np.array(self.targets)[indexs]
            self.data = self.data.repeat(times, axis=0)
            self.targets = self.targets.repeat(times,axis=0)


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, exindexs = [], train=True,
                 transform=None, target_transform=None,
                 download=False, times=1):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)
            self.targets[exindexs] = -2
            self.targets = self.targets[indexs]
            # self.targets = np.array(self.targets)[indexs]

            self.data = self.data.repeat(times, axis=0)
            self.targets = self.targets.repeat(times,axis=0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHN_labeled(torchvision.datasets.SVHN):

    def __init__(self, root, indexs=None, exindexs=[], split='train',
                 transform=None, target_transform=None,
                 download=True):
        super(SVHN_labeled, self).__init__(root, split=split,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)
            self.labels[exindexs] = -2
            self.labels = self.labels[indexs]
            # self.labels = np.array(self.labels)[indexs]
        self.data = transpose(self.data)
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target, index
        return img, target


class SVHN_unlabeled(SVHN_labeled):

    def __init__(self, root, indexs, exindexs=[], split='train',
                 transform=None, target_transform=None,
                 download=True):
        super(SVHN_unlabeled, self).__init__(root, indexs, exindexs, split=split,
                                                transform=transform, target_transform=target_transform,
                                                download=download)


class SmallImageNet(data.Dataset):
    train_list = ['train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3', 'train_data_batch_4',
                  'train_data_batch_5', 'train_data_batch_6', 'train_data_batch_7', 'train_data_batch_8',
                  'train_data_batch_9', 'train_data_batch_10']
    test_list = ['val_data']

    def __init__(self, file_path, imgsize, train, transform=None, target_transform=None, indexs=None):
        # assert imgsize == 32 or imgsize == 64, 'imgsize should only be 32 or 64'
        self.imgsize = imgsize
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        # now load the picked numpy arrays
        for filename in downloaded_list:
            file = os.path.join(file_path, filename)
            with open(file, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])  # Labels are indexed from 1
        self.targets = [i - 1 for i in self.targets]
        arr_tmp = np.zeros((len(self.targets), 3*imgsize*imgsize), dtype=np.uint8)
        arr_num = 0
        for data_i in self.data:
            arr_tmp[arr_num:arr_num+len(data_i)] = data_i
            arr_num += len(data_i)

        self.data = arr_tmp.reshape((len(self.targets), 3, imgsize, imgsize))
        #self.data = self.data.reshape((len(self.targets), 3, self.imgsize, self.imgsize))
        self.data = self.data.transpose(0, 2, 3, 1)  # convert to HWC  shape(-1, 32, 32, 3)


        # if len(exindexs) > 0:
        #     self.targets = np.array(self.targets)
        #     self.targets[exindexs] = -2
            # self.targets = self.targets[indexs]

        self.indexs = indexs


            #self.data = self.data[indexs]
            #self.targets = np.array(self.targets)[indexs]
        #logger.info('data shape: {}'.format(self.data.shape))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.indexs is not None:
            index = self.indexs[index]

        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target, index
        return img, target


    def __len__(self):
        if self.indexs is not None:
            return len(self.indexs)
        else:
            return len(self.data)



DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'stl10': get_stl10,
                   'svhn': get_svhn,
                   'smallimagenet': get_smallimagenet, # 127 classes
                   'smallimagenet_1k': get_smallimagenet_1k,
                   }

if __name__ == '__main__':

    #u_samples = make_imb_data(400, 10, 0.01, 0, 2)
    #print(u_samples)
    dataset = SmallImageNet('/home/data/imagenet32/', 32, False)
    targets = dataset.targets
    # get the number of samples per class
    num_classes = len(set(targets))
    print(num_classes)
    num_data_per_cls = [0 for _ in range(num_classes)]
    for l in targets:
        num_data_per_cls[l] += 1
    print(num_data_per_cls)



    #import numpy as np
    #class_num = 1000
    #gamma = 1280/5
    #max_num = 1280
    #mu = np.power(1/gamma, 1/(class_num - 1))
    #class_num_list = []
    #for i in range(class_num):
    #    if i == (class_num - 1):
    #        class_num_list.append(int(max_num / gamma))
    #    else:
    #        class_num_list.append(int(max_num * np.power(mu, i)))
    #print(class_num_list)




    #def __init__(self, file_path, imgsize, train, transform=None, target_transform=None, indexs=None):



