import baker
from cytoolz import merge, join, groupby
from cytoolz.compatibility import iteritems
from scipy.ndimage import imread
from itertools import starmap
from lxml import etree, objectify
from tqdm import tqdm
import json
import os


def xml_root(folder, filename, width, height):
    E = objectify.ElementMaker(annotate=False)
    return E.annotation(
        E.folder(folder),
        E.filename(filename),
        E.source(
            E.database('MS COCO 2014'),
            E.annotation('MS COCO 2014'),
            E.image('Flickr'),
        ),
        E.size(
            E.width(width),
            E.height(height),
            E.depth(3),
        ),
        E.segmented(0)
    )


def instance_to_xml(anno):
    E = objectify.ElementMaker(annotate=False)
    xmin, ymin, width, height = anno['bbox']
    return E.object(
            E.name(anno['category_id']),
            E.bndbox(
                E.xmin(xmin),
                E.ymin(ymin),
                E.xmax(xmin+width),
                E.ymax(ymin+height),
                ),
            )


def key_join(leftkey, leftseq, rightkey, rightseq):
    return starmap(merge, join(leftkey, leftseq, rightkey, rightseq))


def create_annotation_instance(coco_annotation):
    json_data = open(coco_annotation).read()
    content = json.loads(json_data)
    categories = {d['id']: d['name'] for d in content['categories']}
    return categories, tuple(key_join('id', content['images'], 'image_id', content['annotations']))


@baker.command(
    params={
        "data_path": 'annotations, images 폴더를 포함하는 루트 폴더',
        'subset': 'train, val 구별자',
        'destination_path': '새로운 annotation 저장 폴더'
    }
)
def change_annotations(data_path, subset, destination_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError('{} path is not exist'.format(data_path))

    os.makedirs(destination_path, exist_ok=True)

    annotation_path = os.path.join(data_path, 'annotations/instances_{}2014.json'.format(subset))
    image_path = os.path.join(data_path, 'images/{}2014'.format(subset))

    if not os.path.exists(annotation_path):
        raise FileNotFoundError('{} annotation is not exist'.format(annotation_path))

    if not os.path.exists(image_path):
        raise FileNotFoundError('{} image is not exist'.format(image_path))

    categories, instances = create_annotation_instance(annotation_path)

    for i, instance in enumerate(instances):
        instances[i]['category_id'] = categories[instance['category_id']]

    for name, group in tqdm(iteritems(groupby('file_name', instances)), desc='Create annotation xml files'):
        out_name = name.split('.')[-2]
        img = imread(os.path.join(image_path, name))

        if img.ndim == 3:
            annotation = xml_root('VOC2014', '{}.jpg'.format(out_name), group[0]['height'], group[0]['width'])

            for instance in group:
                annotation.append(instance_to_xml(instance))
            etree.ElementTree(annotation).write(os.path.join(destination_path, '{}.xml'.format(out_name)))


if __name__ == '__main__':
    baker.run()