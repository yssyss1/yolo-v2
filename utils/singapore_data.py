import sys
sys.path.append("..")

from glob import glob
from tqdm import tqdm
import os
import time
import cv2
import baker
from scipy.io import loadmat
from lxml import etree
from utils.coco2voc import xml_root, instance_to_xml
import numpy as np
import random
import shutil


labels = ['Ferry', 'Buoy', 'Vessel/ship', 'Speed boat', 'Boat', 'Kayak', 'Sail boat', 'Swimming person', 'Flying bird/plane', 'Other']


@baker.command(
    params={
        "video_path": "folder path which containing videos",
        "out_path": "destination path which will contain frames extracted from videos",
    }
)
def frame_extraction(video_path, out_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError('{} path is not exists'.format(video_path))

    for path in tqdm(glob(os.path.join(video_path, '*.avi')), desc='Frame extraction'):
        print('Start frame extraction {}'.format(path))
        destination_path = os.path.join(out_path, path.split('/')[-1].split('.')[0])
        os.makedirs(destination_path, exist_ok=True)

        time_start = time.time()
        cnt = 0

        cap = cv2.VideoCapture(path)

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                cv2.imwrite(os.path.join(destination_path, '{}_{}.jpg'.format(path.split('/')[-1].split('.')[0], cnt)), frame)
                cnt += 1
            else:
                break

        time_end = time.time()
        print ("Done extracting frames.\n%d frames extracted" % cnt)
        print ("It took %d seconds for extraction" % (time_end-time_start))
        cap.release()


@baker.command(
    params={
        "mat_path": "folder path which containing mat files",
        "image_path": "folder path which containing frames' folders that are extracted from videos",
        "out_path": "destination path in which parsed annotation xml files will be saved"
    }
)
def mat_to_xml(mat_path, image_path, out_path):
    if not os.path.exists(mat_path):
        raise FileNotFoundError('{} path is not exists'.format(mat_path))

    os.makedirs(out_path, exist_ok=True)
    dict_key = ['bbox', 'category_id']

    for path in tqdm(glob(os.path.join(mat_path, '*.mat')), desc='Mat file to xml file'):
        mat = loadmat(path)

        obj_id = mat['structXML']['Object'][0]
        bounding_box = mat['structXML']['BB'][0]
        base_name = path[:-len('_ObjectGT.mat')].split('/')[-1]

        frame = cv2.imread(os.path.join(image_path, base_name, base_name + '_0.jpg'))
        h, w, c = frame.shape
        frame_num = len(obj_id)

        for i in tqdm(range(frame_num), desc='Mat file to xml file {}'.format(path.split('/')[-1])):
            image_name = base_name + '_{}.jpg'.format(i)
            annotation = xml_root(image_name, h, w)
            # TODO empty frame
            instances = [{dict_key[0]: a, dict_key[1]: labels[b[0]-1]} for a, b in zip(bounding_box[i], obj_id[i])]

            for instance in instances:
                annotation.append(instance_to_xml(instance))
            etree.ElementTree(annotation).write(os.path.join(out_path, '{}_{}.xml'.format(base_name, i)))


@baker.command(
    params={
        "annotation_path": "folder path which containing annotation xml files",
        "frame_path": "folder path which containing frames' folders that are extracted from videos",
        "out_path": "destination path in which all train, test datasets are saved",
        "using_data_ratio": "frames are very similar to each other so we'll use small portion of frames that are sampled with uniform random sampling - default: 0.2",
        "test_ratio": "train and test dataset splitting ratio - default: 0.1"
    }
)
def make_dataset(annotation_path, frame_path, out_path, using_data_ratio=0.1, test_ratio=0.1):
    if not os.path.exists(annotation_path):
        raise FileNotFoundError('{} is not exists'.format(annotation_path))

    if not os.path.exists(frame_path):
        raise FileNotFoundError('{} is not exists'.format(frame_path))

    if using_data_ratio > 1:
        raise ValueError('using_data_ratio must be lower than 1')

    if test_ratio > 1:
        raise ValueError('test_ratio must be lower than 1')

    directory_list = ['train/image', 'train/annotation', 'valid/image', 'valid/annotation']
    path_list =[os.path.join(out_path, directory) for directory in directory_list]

    for path in path_list:
        os.makedirs(path, exist_ok=True)

    data_list = list(filter(lambda x: '.' not in x, os.listdir(frame_path)))
    annotation_list = glob(annotation_path + '/*.xml')

    test_data_list, train_data_list = split_list(data_list, test_ratio)

    for idx, dataset in tqdm(enumerate([train_data_list, test_data_list])):
        for data in tqdm(dataset):
            data_annotation_list = np.array(list((filter(lambda x: data in x, annotation_list))))
            data_length = len(data_annotation_list)
            sampled_data_annotation_list = data_annotation_list[random.sample(range(data_length), int(using_data_ratio*data_length))]

            for one_frame_annotation in sampled_data_annotation_list:
                one_frame_name = one_frame_annotation.split('/')[-1].split('.')[0] + '.jpg'
                one_frame_path = os.path.join(frame_path, data, one_frame_name)

                shutil.copy2(one_frame_path, path_list[idx*2])
                shutil.copy2(one_frame_annotation, path_list[idx*2+1])


def split_list(raw_list, split_ratio):
    split_num = int(split_ratio*len(raw_list))
    shuffled = raw_list[:]
    random.shuffle(shuffled)
    return shuffled[:split_num], shuffled[split_num:]


def extract_detection_dataset(mat_dir, video_dir, out_path):
    if not os.path.exists(mat_dir):
        raise FileNotFoundError('{} is not exists'.format(mat_dir))

    if not os.path.exists(video_dir):
        raise FileNotFoundError('{} is not exists'.format(video_dir))

    dst_video_path = os.path.join(out_path, 'video')
    dst_mat_path = os.path.join(out_path, 'gt')

    os.makedirs(dst_video_path, exist_ok=True)
    os.makedirs(dst_mat_path, exist_ok=True)

    mat_list = glob(os.path.join(mat_dir, '*.mat'))

    for mat_path in tqdm(mat_list):
        base_name = mat_path[:-len('_ObjectGT.mat')].split('/')[-1]
        video_path = os.path.join(video_dir, base_name + '.avi')

        if os.path.exists(video_path):
            shutil.copy2(video_path, dst_video_path)
            shutil.copy2(mat_path, dst_mat_path)


def check_valid_dataset(mat_dir, video_dir, invalid_out_dir):
    if not os.path.exists(mat_dir):
        raise FileNotFoundError('{} is not exists'.format(mat_dir))

    if not os.path.exists(video_dir):
        raise FileNotFoundError('{} is not exists'.format(video_dir))

    gt_out = os.path.join(invalid_out_dir, 'gt')
    videos_out = os.path.join(invalid_out_dir, 'videos')

    os.makedirs(gt_out, exist_ok=True)
    os.makedirs(videos_out, exist_ok=True)

    video_list = glob(os.path.join(video_dir, '*.avi'))
    invalid_list = video_list.copy()

    for video_path in tqdm(video_list):
        base_name = video_path.split('/')[-1].split('.')[0]
        cap = cv2.VideoCapture(video_path)
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        mat_path = os.path.join(mat_dir, base_name + '_ObjectGT.mat')
        mat = loadmat(mat_path)
        mat_length = len(mat['structXML']['Object'][0])

        if frame_length == mat_length:
            invalid_list.remove(video_path)
        else:
            print('Different Length {}'.format(base_name))
            print('Frame length {}'.format(frame_length))
            print('Mat length {}'.format(mat_length))

            shutil.move(video_path, videos_out)
            shutil.move(mat_path, gt_out)

    print('\nList - frame length and mat length different')
    print(invalid_list)


if __name__ == '__main__':
    baker.run()
