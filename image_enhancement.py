# coding=utf-8

'''
对测试集进行增强,存储于训练集
'''
import os
import cv2 as cv
import sys
import numpy as np
import random
import re
import show_tools
from xml.dom.minidom import parse
from lxml import etree

image_save_path = './VOCdevkit/VOCdevkit_train/JPEGImages/'
xml_save_path = './VOCdevkit/VOCdevkit_train/Annotation/'
image_read_path = './VOCdevkit/VOCdevkit_test/JPEGImages/'
xml_read_path = './VOCdevkit/VOCdevkit_test/Annotation/'

'''
voc  1--->28

第一组扩展	(该类自带)
原图片 ---> 	_flip_x
			_flip_y
			_flip_x_y

第二组扩展	gaussian_blur_fun
第一组 --->	第一组全体高斯模糊

第三组扩展	change_exposure_fun
第一组 --->	第一组全体进行曝光、和欠曝光

第三组扩展	add_salt_noise
第一组 --->  第一组全体加入椒盐噪声

第四组扩展	add_Translate
第一组 --->	第一组进行平移

第五组扩展	add_Resize
第一组 -->	第一组进行缩放
'''
class DataAugment(object):

    def __init__(self, image_path='a'):
        self.add_saltNoise = True
        self.gaussianBlur = True
        self.changeExposure = True
        self.translate = False
        self.translate_x = 0
        self.translate_y = 0
        self.add_resize = False
        self.resize_proportion = 1
        self.id = image_save_path + image_path
        self.name = image_path
        img = cv.imread(image_read_path+str(self.name)+'.jpg')

        try:
            self.img_h, self.img_w, self.img_c = img.shape
            self.img_h = int(self.img_h)
            self.img_w = int(self.img_w)
            self.img_c = int(self.img_c)
            # print(img.shape)
        except:
            print('No Such image!---'+str(id)+'.jpg')
            sys.exit(0)
        self.src = img
        dst1 = cv.flip(img, 0, dst=None)
        dst2 = cv.flip(img, 1, dst=None)
        dst3 = cv.flip(img, -1, dst=None)
        self.flip_x = dst1
        self.flip_y = dst2
        self.flip_x_y = dst3
        cv.imwrite(str(self.id)+'_flip_x'+'.jpg', self.flip_x)
        cv.imwrite(str(self.id)+'_flip_y'+'.jpg', self.flip_y)
        cv.imwrite(str(self.id)+'_flip_x_y'+'.jpg', self.flip_x_y)

        self.root = ''

    def gaussian_blur_fun(self):
        if self.gaussianBlur:
            dst1 = cv.GaussianBlur(self.src, (5, 5), 0)
            dst2 = cv.GaussianBlur(self.flip_x, (5, 5), 0)
            dst3 = cv.GaussianBlur(self.flip_y, (5, 5), 0)
            dst4 = cv.GaussianBlur(self.flip_x_y, (5, 5), 0)
            self.src_gaussian = dst1
            self.flip_x_gaussian = dst2
            self.flip_y_gaussian = dst3
            self.flip_x_y_gaussian = dst4
            cv.imwrite(str(self.id)+'_Gaussian'+'.jpg', self.src_gaussian)
            cv.imwrite(str(self.id)+'_flip_x'+'_Gaussian'+'.jpg', self.flip_x_gaussian)
            cv.imwrite(str(self.id)+'_flip_y'+'_Gaussian'+'.jpg', self.flip_y_gaussian)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_Gaussian'+'.jpg', self.flip_x_y_gaussian)

    def change_exposure_fun(self):
        if self.changeExposure:
            # contrast
            reduce = 0.5
            increase = 1.4
            # brightness
            g = 10
            h, w, ch = self.src.shape
            add = np.zeros([h, w, ch], self.src.dtype)
            dst1 = cv.addWeighted(self.src, reduce, add, 1-reduce, g)
            dst2 = cv.addWeighted(self.src, increase, add, 1-increase, g)
            dst3 = cv.addWeighted(self.flip_x, reduce, add, 1 - reduce, g)
            dst4 = cv.addWeighted(self.flip_x, increase, add, 1 - increase, g)
            dst5 = cv.addWeighted(self.flip_y, reduce, add, 1 - reduce, g)
            dst6 = cv.addWeighted(self.flip_y, increase, add, 1 - increase, g)
            dst7 = cv.addWeighted(self.flip_x_y, reduce, add, 1 - reduce, g)
            dst8 = cv.addWeighted(self.flip_x_y, increase, add, 1 - increase, g)
            self.src_reduce_ep = dst1
            self.flip_x_reduce_ep = dst3
            self.flip_y_reduce_ep = dst5
            self.flip_x_y_reduce_ep = dst7
            self.src_increase_ep = dst2
            self.flip_x_increase_ep = dst4
            self.flip_y_increase_ep = dst6
            self.flip_x_y_increase_ep = dst8
            cv.imwrite(str(self.id)+'_ReduceEp'+'.jpg', self.src_reduce_ep)
            cv.imwrite(str(self.id)+'_flip_x'+'_ReduceEp'+'.jpg', self.flip_x_reduce_ep)
            cv.imwrite(str(self.id)+'_flip_y'+'_ReduceEp'+'.jpg', self.flip_y_reduce_ep)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_ReduceEp'+'.jpg', self.flip_x_y_reduce_ep)
            cv.imwrite(str(self.id)+'_IncreaseEp'+'.jpg', self.src_increase_ep)
            cv.imwrite(str(self.id)+'_flip_x'+'_IncreaseEp'+'.jpg', self.flip_x_increase_ep)
            cv.imwrite(str(self.id)+'_flip_y'+'_IncreaseEp'+'.jpg', self.flip_y_increase_ep)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_IncreaseEp'+'.jpg', self.flip_x_y_increase_ep)

    def add_salt_noise(self):
        if self.add_saltNoise:
            percentage = 0.005
            dst1 = self.src
            dst2 = self.flip_x
            dst3 = self.flip_y
            dst4 = self.flip_x_y
            num = int(percentage * self.src.shape[0] * self.src.shape[1])
            for i in range(num):
                rand_x = random.randint(0, self.src.shape[0] - 1)
                rand_y = random.randint(0, self.src.shape[1] - 1)
                if random.randint(0, 1) == 0:
                    dst1[rand_x, rand_y] = 0
                    dst2[rand_x, rand_y] = 0
                    dst3[rand_x, rand_y] = 0
                    dst4[rand_x, rand_y] = 0
                else:
                    dst1[rand_x, rand_y] = 255
                    dst2[rand_x, rand_y] = 255
                    dst3[rand_x, rand_y] = 255
                    dst4[rand_x, rand_y] = 255
            self.src_salt = dst1
            self.flip_x_salt = dst2
            self.flip_y_salt = dst3
            self.flip_x_y_salt = dst4
            cv.imwrite(str(self.id)+'_Salt'+'.jpg', self.src_salt)
            cv.imwrite(str(self.id)+'_flip_x'+'_Salt'+'.jpg', self.flip_x_salt)
            cv.imwrite(str(self.id)+'_flip_y'+'_Salt'+'.jpg', self.flip_y_salt)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_Salt'+'.jpg', self.flip_x_y_salt)

    def add_Translate(self, x_translate_proportion=0, y_translate_proportion=0):

        if x_translate_proportion is not 0 or y_translate_proportion is not 0:
            self.translate = True
            self.translate_x = int(x_translate_proportion * self.img_w)
            self.translate_y = int(y_translate_proportion * self.img_h)
            M = np.float32([[1, 0, self.translate_x], [0, 1, self.translate_y]])
            dst1 = cv.warpAffine(self.src, M, (self.img_w, self.img_h))
            dst2 = cv.warpAffine(self.flip_x, M, (self.img_w, self.img_h))
            dst3 = cv.warpAffine(self.flip_y, M, (self.img_w, self.img_h))
            dst4 = cv.warpAffine(self.flip_x_y, M, (self.img_w, self.img_h))

            cv.imwrite(str(self.id) + '_Translate' + '.jpg', dst1)
            cv.imwrite(str(self.id) + '_flip_x' + '_Translate' + '.jpg', dst2)
            cv.imwrite(str(self.id) + '_flip_y' + '_Translate' + '.jpg', dst3)
            cv.imwrite(str(self.id) + '_flip_x_y' + '_Translate' + '.jpg', dst4)

    def add_Resize(self, proportion=1):
        if proportion is not 1:
            self.add_resize = True
            self.resize_proportion = proportion
            cv.imwrite(str(self.id) + '_Resize' + '.jpg',
                       cv.resize(self.src, None, fx=proportion, fy=proportion, interpolation=cv.INTER_CUBIC))
            cv.imwrite(str(self.id) + '_flip_x' +'_Resize' + '.jpg',
                       cv.resize(self.flip_x, None, fx=proportion, fy=proportion, interpolation=cv.INTER_CUBIC))
            cv.imwrite(str(self.id) + '_flip_y' + '_Resize' + '.jpg',
                       cv.resize(self.flip_y, None, fx=proportion, fy=proportion, interpolation=cv.INTER_CUBIC))
            cv.imwrite(str(self.id) + '_flip_x_y' + '_Resize' + '.jpg',
                       cv.resize(self.flip_x_y, None, fx=proportion, fy=proportion, interpolation=cv.INTER_CUBIC))

    def voc_generation(self):
        image_names = [str(self.id)+'_flip_x', str(self.id)+'_flip_y', str(self.id)+'_flip_x_y']
        if self.gaussianBlur:
            image_names.append(str(self.id)+'_Gaussian')
            image_names.append(str(self.id)+'_flip_x'+'_Gaussian')
            image_names.append(str(self.id)+'_flip_y' + '_Gaussian')
            image_names.append(str(self.id)+'_flip_x_y'+'_Gaussian')
        if self.changeExposure:
            image_names.append(str(self.id)+'_ReduceEp')
            image_names.append(str(self.id)+'_flip_x'+'_ReduceEp')
            image_names.append(str(self.id)+'_flip_y'+'_ReduceEp')
            image_names.append(str(self.id)+'_flip_x_y'+'_ReduceEp')
            image_names.append(str(self.id)+'_IncreaseEp')
            image_names.append(str(self.id)+'_flip_x'+'_IncreaseEp')
            image_names.append(str(self.id)+'_flip_y'+'_IncreaseEp')
            image_names.append(str(self.id)+'_flip_x_y'+'_IncreaseEp')
        if self.add_saltNoise:
            image_names.append(str(self.id)+'_Salt')
            image_names.append(str(self.id)+'_flip_x' + '_Salt')
            image_names.append(str(self.id)+'_flip_y' + '_Salt')
            image_names.append(str(self.id)+'_flip_x_y' + '_Salt')
        if self.translate:
            image_names.append(str(self.id) + '_Translate')
            image_names.append(str(self.id) + '_flip_x' + '_Translate')
            image_names.append(str(self.id) + '_flip_y' + '_Translate')
            image_names.append(str(self.id) + '_flip_x_y' + '_Translate')
        if self.add_resize:
            image_names.append(str(self.id) + '_Resize')
            image_names.append(str(self.id) + '_flip_x' + '_Resize')
            image_names.append(str(self.id) + '_flip_y' + '_Resize')
            image_names.append(str(self.id) + '_flip_x_y' + '_Resize')

        for image_name in image_names:
            # print(xml_read_path+self.name+'.xml')
            # / VOCdevkit / VOCdevkit_test / Annotation / XCM - 307A_Bottom.xml
            doc = parse(xml_read_path+str(self.name)+'.xml')
            root = doc.documentElement
            objects = root.getElementsByTagName('object')

            objname_list = []
            xmin_list = []
            ymin_list = []
            xmax_list = []
            ymax_list = []
            for object in objects:
                objname = object.getElementsByTagName("name")[0].childNodes[0].data
                xmin = object.getElementsByTagName("xmin")[0].childNodes[0].data
                ymin = object.getElementsByTagName("ymin")[0].childNodes[0].data
                xmax = object.getElementsByTagName("xmax")[0].childNodes[0].data
                ymax = object.getElementsByTagName("ymax")[0].childNodes[0].data
                objname_list.append(objname)
                xmin_list.append(int(float(xmin)))
                ymin_list.append(int(float(ymin)))
                xmax_list.append(int(float(xmax)))
                ymax_list.append(int(float(ymax)))

            match_pattern2 = re.compile(r'(.*)_x(.*)')
            match_pattern3 = re.compile(r'(.*)_y(.*)')
            match_pattern4 = re.compile(r'(.*)_x_y(.*)')
            # print(objname_list)
            if match_pattern4.match(image_name):
                for i in range(len(objname_list)):
                    # print(xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i])
                    xmin_list[i] = self.img_w - xmin_list[i]
                    ymin_list[i] = self.img_h - ymin_list[i]
                    xmax_list[i] = self.img_w - xmax_list[i]
                    ymax_list[i] = self.img_h - ymax_list[i]
                    if xmax_list[i] < xmin_list[i]:
                        d = xmax_list[i]
                        xmax_list[i] = xmin_list[i]
                        xmin_list[i] = d
                    if ymax_list[i] < ymin_list[i]:
                        d = ymax_list[i]
                        ymax_list[i] = ymin_list[i]
                        ymin_list[i] = d
                    # print('->', xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i])
            elif match_pattern3.match(image_name):
                for i in range(len(objname_list)):
                    # print(xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i])
                    xmin_list[i] = self.img_w - xmin_list[i]
                    ymin_list[i] = ymin_list[i]
                    xmax_list[i] = self.img_w - xmax_list[i]
                    ymax_list[i] = ymax_list[i]
                    if xmax_list[i] < xmin_list[i]:
                        d = xmax_list[i]
                        xmax_list[i] = xmin_list[i]
                        xmin_list[i] = d
                    # print('->',xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i])
            elif match_pattern2.match(image_name):
                for i in range(len(objname_list)):
                    # print(xmin_list[i],ymin_list[i],xmax_list[i],ymax_list[i])
                    xmin_list[i] = xmin_list[i]
                    ymin_list[i] = self.img_h - ymin_list[i]
                    xmax_list[i] = xmax_list[i]
                    ymax_list[i] = self.img_h - ymax_list[i]
                    if ymax_list[i] < ymin_list[i]:
                        d = ymax_list[i]
                        ymax_list[i] = ymin_list[i]
                        ymin_list[i] = d
                    # print('->',xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i])
            else:
                for i in range(len(objname_list)):
                    xmin_list[i] = xmin_list[i]
                    ymin_list[i] = ymin_list[i]
                    xmax_list[i] = xmax_list[i]
                    ymax_list[i] = ymax_list[i]

            if self.translate:
                match_pattern = re.compile(r'(.*)_Translate(.*)')
                if match_pattern.match(image_name):
                    for i in range(len(objname_list)):
                        xmin_list[i] = int(xmin_list[i] + self.translate_x)
                        ymin_list[i] = int(ymin_list[i] + self.translate_y)
                        xmax_list[i] = int(xmax_list[i] + self.translate_x)
                        ymax_list[i] = int(ymax_list[i] + self.translate_y)

            if self.resize_proportion:
                match_pattern = re.compile(r'(.*)_Resize(.*)')
                if match_pattern.match(image_name):
                    for i in range(len(objname_list)):
                        xmin_list[i] = int(self.resize_proportion * xmin_list[i])
                        ymin_list[i] = int(self.resize_proportion * ymin_list[i])
                        xmax_list[i] = int(self.resize_proportion * xmax_list[i])
                        ymax_list[i] = int(self.resize_proportion * ymax_list[i])

            file_name = image_name + '.jpg'
            root = self.set_filename(file_name)
            self.set_size(self.img_w, self.img_h, self.img_c, root)
            for i in range(len(objname_list)):
                if xmin_list[i] > self.img_w or ymin_list[i] > self.img_h or ymax_list[i] > self.img_h or xmax_list[i] > self.img_w:
                   continue
                self.add_pic_attr(str(objname_list[i]), xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i], root)
            # print(xml_save_path+self.name+'.xml')
            self.savefile(xml_save_path+image_name.split('/')[4]+'.xml', root)

    '''以下都是生成VOC的功能性函数'''
    def set_filename(self, file_name):
        root = etree.Element("annotation")
        child1 = etree.SubElement(root, "folder")
        child1.text = "VOC2007"

        child2 = etree.SubElement(root, "filename")
        child2.text = file_name
        child3 = etree.SubElement(root, "source")
        child4 = etree.SubElement(child3, "annotation")
        child4.text = "PASCAL VOC2007"
        child5 = etree.SubElement(child3, "database")
        child6 = etree.SubElement(child3, "image")
        child6.text = "flickr"
        child7 = etree.SubElement(child3, "flickrid")
        child7.text = "35435"
        return root

    def set_size(self, witdh, height, channel,root):
        size = etree.SubElement(root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "channel")
        channeln.text = str(channel)

    def savefile(self, filename,root):
        tree = etree.ElementTree(root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

    def add_pic_attr(self, label, x, y, w, h, root):
        object = etree.SubElement(root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(x)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(y)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(w)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(h)

if __name__ == "__main__":
    image_names = os.listdir(image_read_path)
    index = 0
    for i in image_names:
        index += 1
        show_tools.view_bar('image_enhancement', index, len(image_names))
        dataAugmentObject = DataAugment(i.replace('.jpg', ''))
        dataAugmentObject.gaussian_blur_fun()
        dataAugmentObject.change_exposure_fun()
        dataAugmentObject.add_salt_noise()
        dataAugmentObject.add_Translate(x_translate_proportion=0.2, y_translate_proportion=0.2)
        dataAugmentObject.add_Resize(proportion=0.8)
        dataAugmentObject.voc_generation()
