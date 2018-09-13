#coding:utf-8
import cv2
import math
import numpy as np
import os
import random
import os,shutil
import xml.etree.cElementTree as ET
import sys

#用于过滤或删除特定标签
counts=0
if __name__=='__main__':
    image_suff=".jpg"

    srcGT="/media/nizhengqi/0007912600089656/fpn.pytorch-master/data/VOCdevkit2007/VOC2007/Annotations/"
    srcimage = "/media/nizhengqi/0007912600089656/fpn.pytorch-master/data/VOCdevkit2007/VOC2007/JPEGImages/"
    desGT="/media/nizhengqi/0007912600089656/fpn.pytorch-master/data/VOCdevkit2007/VOC2007/new_gt/"
    desimage = "/media/nizhengqi/0007912600089656/fpn.pytorch-master/data/VOCdevkit2007/VOC2007/new_image/"
    tags=["xiaochicun"]
    pos=1#0代表过滤出来上面的标签，1代表删除上面的标签
    for root, dirs, files in os.walk(srcGT):
        print(len(files))
        count = 0
        for path in files:
            fullpath=srcGT+path
            if os.path.isfile(fullpath) and ".xml" in path:
                tree = ET.parse(fullpath)
                root = tree.getroot()
                r_name=path.replace(".xml","")
                need_delete = []
                count=0
                for object in root.getchildren():
                    if object.tag=="object":
                        count=count+1
                        name = object.find('name').text
                        if pos==0:
                            if name not in tags:
                                need_delete.append(object)
                        else:
                            if name  in tags:
                                need_delete.append(object)
                if count!=len(need_delete) and count!=0:
                    while len(need_delete) > 0:
                        root.remove(need_delete[0])
                        need_delete.remove(need_delete[0])

                    tree2 = ET.ElementTree(root)
                    new_full_path = desGT + r_name
                    print(new_full_path)
                    if os.path.isfile(new_full_path+".xml"):
                        r_name=r_name+"_"
                    new_full_path = desGT + r_name
                    tree2.write(new_full_path+".xml", encoding='utf-8')
                    shutil.copyfile(srcimage+path.replace(".xml",image_suff), desimage+r_name+image_suff)

    cv2.waitKey(0)