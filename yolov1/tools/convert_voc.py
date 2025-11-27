import xml.etree.ElementTree as ET
import os
import shutil
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default='./Dataset')
    args = parser.parse_args()
    return args


sets = [('2012', 'train'), ('2012', 'val')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_xml(file_path, out_file):
    out_file = open(out_file, 'w')
    tree = ET.parse(file_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')

        #bb = (max(1, float(xmlbox.find('xmin').text)), max(1, float(xmlbox.find('ymin').text))
        #      , min(w - 1, float(xmlbox.find('xmax').text)), min(h - 1, float(xmlbox.find('ymax').text)))
        bb = (max(1, int(xmlbox.find('xmin').text)), max(1, int(xmlbox.find('ymin').text))
              , min(w - 1, int(xmlbox.find('xmax').text)), min(h - 1, int(xmlbox.find('ymax').text)))


        #out_file.write(",".join([str(a) for a in bb]) + ',' + str(cls_id) + '\n')
        out_file.write(str(cls_id)+' '+ " ".join([str(a) for a in bb]) + '\n')

    out_file.close()


if __name__ == '__main__':

    args = parse_args()
    root_dir = args.dir_path
    
    if not os.path.exists(root_dir + '/Labels'):
        os.makedirs(root_dir + '/Labels')

    if not os.path.exists(root_dir + '/Images'):
        os.makedirs(root_dir + '/Images')

    train_list_file = open(root_dir + '/train.txt', 'w')
    test_list_file = open(root_dir + '/test.txt', 'w')
        
    for data_ in sets:
        if(data_[1]=='val'):
            data_list_file = test_list_file
        else:
            data_list_file = train_list_file
        
        #name_list = open('./VOCdevkit'+ './VOC%s/ImageSets/Main/%s.txt' % (data_[0], data_[1])).read().strip().split()
        name_list = open('./VOC%s/ImageSets/Main/%s.txt' % (data_[0], data_[1])).read().strip().split()

        print(len(name_list))
        name_list = tqdm(name_list)
        
        file_writer = ''
        for i, xml_name in enumerate(name_list):
            #file_path = './VOCdevkit' + '/VOC%s/Annotations/%s.xml' % (data_[0], xml_name)
            file_path = './VOC%s/Annotations/%s.xml' % (data_[0], xml_name)
            label_file = root_dir + '/Labels/%s.txt' % (xml_name)
            convert_xml(file_path, label_file)
            
            #img_file = './VOCdevkit' + '/VOC%s/JPEGImages/%s.jpg' % (data_[0],xml_name)
            img_file = './VOC%s/JPEGImages/%s.jpg' % (data_[0],xml_name)
            copy_img_file = root_dir + '/Images/%s.jpg' % (xml_name)
            shutil.copy(img_file,copy_img_file)
            file_writer += xml_name + '\n'

        data_list_file.write(file_writer)
        file_writer = ''

    train_list_file.close()
    test_list_file.close()
    
    #./Datasets/train.txt의  train file list를 정렬시킴
    train_list = open(root_dir + '/train.txt').read().strip().split()
    train_list.sort()
    
    train_list_file = open(root_dir + '/train.txt', 'w+')
    
    file_writer = ''
    for i, file_name in enumerate(train_list):
        file_writer += file_name + '\n'
        
    train_list_file.write(file_writer)
    train_list_file.close()

    # ./Datasets/test.txt의  train file list를 정렬시킴    
    test_list = open(root_dir + '/test.txt').read().strip().split()
    
    test_list.sort()
    test_list_file = open(root_dir + '/test.txt', 'w+')
    
    file_writer = ''
    for i, file_name in enumerate(test_list):
        file_writer += file_name + '\n'
    
    test_list_file.write(file_writer)
    test_list_file.close()