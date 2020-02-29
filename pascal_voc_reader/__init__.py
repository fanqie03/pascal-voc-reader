import pandas as pd
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

class Reader:
    def __init__(self, voc_root):
        """Pascal VOC format data reader

        Args:
            voc_root (str) - pascal voc format data root

        """
        self.voc_root = Path(voc_root)
        self.xml_root = self.voc_root / 'Annotations'
        self.txt_root = self.voc_root / 'ImageSets' / 'Main'
        self.img_root = self.voc_root / 'JPEGImages'
        self.df = pd.DataFrame(self.parse_xml(self.xml_root))
        self.df['img_path'] = self.df.stem.map(lambda x: self.img_root / (x + '.jpg'))
        train_array, val_array, train_val_array, test_array = self.parse_txt(self.txt_root)
        self.df['train'] = self.df.stem.isin(train_array)
        self.df['val'] = self.df.stem.isin(val_array)
        self.df['trainval'] = self.df.stem.isin(train_val_array)
        self.df['test'] = self.df.stem.isin(test_array)

    def parse_xml(self, xml_root):
        ret = []
        for i in tqdm(xml_root.glob('*.xml')):
            ret.extend(self.parse_single_xml(i))
        return ret

    def parse_single_xml(self, xml_path):
        ret = []

        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        size = root.find('size')
        if size:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            depth = int(size.find('depth').text)
        else:
            width, height, depth = -1, -1, -1

        for obj in root.findall('object'):
            name = obj.find('name').text
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
            ratio = h / w
            ret.append(dict(name=name, difficult=difficult,
                x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                xml_path=xml_path, stem=xml_path.stem,
                width=width, height=height, depth=depth,
                h=h, w=w, ratio=ratio))
        return ret

    def parse_txt(self, txt_root):
        train_array=self.parse_single_txt(txt_root / 'train.txt')
        test_array=self.parse_single_txt(txt_root / 'test.txt')
        val_array=self.parse_single_txt(txt_root / 'val.txt')
        train_val_array=self.parse_single_txt(txt_root / 'trainval.txt')
        return train_array, val_array, train_val_array, test_array

    def parse_single_txt(self, txt_path):
        if not os.path.exists(txt_path):
            return []
        with open(txt_path) as f:
            return list(f.readlines())

    def show(self):
        import cv2
        grouped = self.df.groupby('stem')
        # length = self.df.
        length = self.df.stem.unique().size
        indexes = self.df.stem.unique()
        def nothing(x):
            index = indexes[x]
            sub_table = grouped.get_group(index)
            img_path = str(sub_table.iloc[0]['img_path'])
            img = cv2.imread(img_path)
            for index, row in sub_table.iterrows():
                cv2.rectangle(img, (row['x1'], row['y1']), (row['x2'], row['y2']),
                              (0, 255, 0))
            # img = cv2.resize(img, (800, 800))
            cv2.imshow('image_win', img)
            # cv2.waitKey()

        cv2.namedWindow('image_win')
        cv2.namedWindow('tracker_win')
        cv2.createTrackbar('tracker', 'tracker_win', 0, length - 1, nothing)

        if cv2.waitKey() == 'q':
            cv2.destroyAllWindows()


if __name__ == '__main__':
    reader = Reader('C:\\Users\\mengf\\datasets\\mask_detecting\\VOCdevkit\\VOC2020')