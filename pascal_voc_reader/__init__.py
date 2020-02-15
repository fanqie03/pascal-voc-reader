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
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)

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
            ret.append(dict(name=name, difficult=difficult,
                x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                filename=xml_path.name, stem=xml_path.stem,
                width=width, height=height, depth=depth))
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

if __name__ == '__main__':
    reader = Reader('C:\\Users\\mengf\\datasets\\mask_detecting\\VOCdevkit\\VOC2020')