from glob import glob
from tqdm import tqdm
import os
import time
import cv2
import baker
from scipy.io import loadmat
from lxml import etree
from utils.coco2voc import xml_root, instance_to_xml


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
        "out_path": "destination path which will contain frames extracted from videos"
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

        for i in tqdm(range(len(obj_id)), desc='Mat file to xml file {}'.format(path.split('/')[-1])):
            image_name = base_name + '_{}.jpg'.format(i)
            frame = cv2.imread(os.path.join(image_path, base_name, image_name))
            h, w, c = frame.shape

            annotation = xml_root(image_name, h, w)
            instances = [{dict_key[0]: a, dict_key[1]: labels[b[0]-1]} for a, b in zip(bounding_box[i], obj_id[i])]

            for instance in instances:
                annotation.append(instance_to_xml(instance))
            etree.ElementTree(annotation).write(os.path.join(out_path, '{}_{}.xml'.format(base_name, i)))


if __name__ == '__main__':
    mat_to_xml('/home/seok/singapore_dataset/gt', '/home/seok/frames', '/home/seok/xmls')
    # baker.run()