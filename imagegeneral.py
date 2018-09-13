#coding:utf-8
import cv2
import math
import numpy as np
import os
import random

import xml.etree.cElementTree as ET
import sys
#用于抠图。
counts=0
class GEN_Annotations:
    def __init__(self,filename):
        self.root=ET.Element("annotation")
        child1=ET.SubElement(self.root,"folder")
        child1.text="VOC2012"
        child2=ET.SubElement(self.root,"filename")
        child2.text=filename
        child22=ET.SubElement(self.root,"path")
        child22.text=filename
        child3=ET.SubElement(self.root,"source")
        child6=ET.SubElement(self.root,"segmented")
        child6.text="0"
        child4=ET.SubElement(child3,"annotation")
        child4.text="PASCAL VOC2012"
        child5=ET.SubElement(child3,"database")
    def set_size(self,width,height,channel):
        size=ET.SubElement(self.root,"size")
        widthn=ET.SubElement(size,"width")
        widthn.text=str(width)
        heightn=ET.SubElement(size,"height")
        heightn.text=str(height)
        channeln=ET.SubElement(size,"channel")
        channeln.text=str(channel)
    def savefile(self,filename):
        tree=ET.ElementTree(self.root)
        tree.write(filename,encoding='utf-8')
    def add_pic_attr(self,label,x,y,w,h):
        object=ET.SubElement(self.root,"object")
        namen=ET.SubElement(object,"name")
        namen.text=label
        pos=ET.SubElement(object,"pos")
        pos.text="Unspecified"
        truncated=ET.SubElement(object,"truncated")
        truncated.text="0"
        difficult=ET.SubElement(object,"difficult")
        difficult.text="0"
        bndbox=ET.SubElement(object,"bndbox")
        xminn=ET.SubElement(bndbox,"xmin")
        xminn.text=str(int(x))
        yminn=ET.SubElement(bndbox,"ymin")
        yminn.text=str(int(y))
        xmaxn=ET.SubElement(bndbox,"xmax")
        xmaxn.text=str(int(w))
        ymaxn=ET.SubElement(bndbox,"ymax")
        ymaxn.text=str(int(h))
def readxml(path,tags):
    if os.path.isfile(path) and ".xml" in path:
        tree=ET.parse(path)
        root=tree.getroot()
        filename=root.find('filename').text
        filename=filename[:-4]
        for size in root.findall('size'):
            width=int(size.find('width').text)
            height=int(size.find('height').text)
        fp=np.array([])
        count=0
        for object in root.findall('object'):
            name=object.find('name').text
            if name==tags:
                fp=np.append(fp,count)
            count=count+1
        count=0
        choose=random.randint(0,len(fp)-1)
        for object in root.findall('object'):
            name=object.find('name').text
            if count==fp[choose]:
                bndbox=object.find('bndbox')
                xmin=int(bndbox.find('xmin').text)
                ymin=int(bndbox.find('ymin').text)
                xmax=int(bndbox.find('xmax').text)
                ymax=int(bndbox.find('ymax').text)
                p1=[xmin,ymin,1]
                p2=[xmax,ymax,1]
                p3=[xmin,ymax,1]
                p4=[xmax,ymin,1]
                return p1,p2,p3,p4,tree,width,height
            count=count+1

def loopxml(path,tags):
    list=os.listdir(path)
    files=[[None,[None]]]
    for i in range(0,len(list)):
        lpath=path+"/"+list[i]
        if os.path.isfile(lpath) and ".xml" in lpath:
            tree=ET.parse(lpath)
            root=tree.getroot()
            filename=root.find('filename').text
            filename=filename[:-4]
            for size in root.findall('size'):
                width=size.find('width').text
                height=size.find('height').text
                for object in root.findall('object'):
                    name=object.find('name').text

                    for i in range(len(tags)):
                        if tags[i][0]==name:
                            have=name in [x[0] for x in files]
                            count=0;
                            for x in files:
                                if x[0]==name:
                                    break
                                count=count+1
                            if have==True:

                                files[count][1]=np.append(files[count][1],lpath)
                            else:
                                tem_file=[[name,[None]]]
                                tem_file[0][1]=lpath
                                files=np.concatenate((files,tem_file),axis=0)
    return files
def random_op(op,path,filename,tags,saveimageorgtPath):
    print(path)
    #image=cv2.imread(path)
    image=cv2.imdecode(np.fromfile(path),1)
    #cv2.imshow("test",image)
    #cv2.waitKey(0)
    #print(image)
    rows,cols=image.shape[:2]
    #print("1",rows,cols)
    # deltax=random.randint(-int(rows/2),int(rows/2))
    # deltay=random.randint(-int(cols/2),int(cols/2))
    deltax =0# random.randint(-int(rows / 2), int(rows / 2))
    deltay =0# random.randint(-int(cols / 2), int(cols / 2))
    tp=[[0,0,1]]
    #print("op",op)
    op=0
    global transform
    if op==0:
        M=np.float32([[1,0,deltax],[0,1,deltay]])
        shifted=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
        transform=np.concatenate((M,tp))
    # elif op==1:
    #     M=cv2.getRotationMatrix2D((cols/2,rows/2),random.randint(-60,60),1)
    #     #print(M)
    #     transform=np.concatenate((M,tp))
    #     shifted=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
    p1,p2,p3,p4,tree,w,h=readxml(path.replace(".jpg",".xml"),tags)
    p11=np.dot(transform,p1)
    p11=p11.tolist()
    p22=np.dot(transform,p2)
    p22=p22.tolist()
    p33=np.dot(transform,p3)
    p33=p33.tolist()
    p44=np.dot(transform,p4)
    p44=p44.tolist()
    sud=np.array([p11,p22,p33,p44])
    xmin=min(sud[:,0])
    ymin=min(sud[:,1])
    xmax=max(sud[:,0])
    ymax=max(sud[:,1])
    w=shifted.shape[1]
    h=shifted.shape[0]
    #print("2",len(shifted[0]),h)
    imagesavepath=saveimageorgtPath+"/"+filename+"_"+tags+"_"+str(counts)+"_.jpg"
    gtsavepath=saveimageorgtPath+"/"+filename+"_"+tags+"_"+str(counts)+"_.xml"

    kxHeight=int(ymax-ymin)
    kxWidth=int(xmax-xmin)
    print(kxHeight,kxWidth)
    center_x=int(kxWidth/2+xmin)
    center_y=int(kxHeight/2+ymin)
    w = 600
    h = 600
    shifted2 = shifted[center_y-int(h/2):center_y+int(h/2),
                   center_x-int(w/2):center_x+int(w/2)]




    print(shifted2.shape[:2])
    if shifted2.shape[0]>0 and shifted2.shape[1]>0:
        cv2.imencode('.jpg',shifted2)[1].tofile(imagesavepath)
        anno=GEN_Annotations(filename)
        anno.set_size(w,h,3)
        anno.add_pic_attr(tags,int(300-kxWidth/2),int(300-kxHeight/2),int(300+kxWidth/2),int(300+kxHeight/2))
        anno.savefile(gtsavepath)
if __name__=='__main__':
    ImagePaht="/media/nizhengqi/0007912600089656/fpn.pytorch-master/data/VOCdevkit2007/VOC2007_back/VOC2007/JPEGImages"
    GtPath="/media/nizhengqi/0007912600089656/fpn.pytorch-master/data/VOCdevkit2007/VOC2007_back/VOC2007/JPEGImages"

    tags=[["xiaochicun",1]]

    # tags=[["dg",246],["sg",229],
    # ["jyzzb",157],["jyhsh",274],["jyhtl",298],["jyhyw",287],
    # ["fzchy",224],["fzcpx",297],["fzcsh",279],
    # ["fzcxs",294],["bmqk",296],["lslmqk",287],
    # ["lsqbm",289],["lsqdp",297],["lsqlm",289],
    # ["lsqxz",232],["lsxs",275],["wtxztc",292],
    # ["tjbm",296],["tjjs",274],["tjtf",296],
    # ["tjwl",297],["tjxx",243],["fnsssh",258],
    # ["bspsh",287],["bsptl",263],["nc",271],["yw",157]]
    SaveImagePath="/media/nizhengqi/0007912600089656/fpn.pytorch-master/data/VOCdevkit2007/VOC2007_back/VOC2007/new_image_gt"
    addGtPath="/media/nizhengqi/0007912600089656/fpn.pytorch-master/data/VOCdevkit2007/VOC2007_back/VOC2007/new_image_gt"
    l_files=loopxml(GtPath,tags)
    print(l_files)
    for i in range(len(tags)):
        for j in range(tags[i][1]):
            for f in range(len(l_files)):
                if l_files[f][0]==tags[i][0]:
                    rd=counts#random.randint(0,len(l_files[f][1])-1)
                    if counts>=len(l_files[f][1])-1:
                        print(counts)
                        counts = 0
                        continue

                    op=random.randint(0,1)
                    imagefull=l_files[f][1][rd].replace(".xml",".jpg")
                    sps=imagefull.split('/')
                    imagename=sps[len(sps)-1].replace(".xml","").replace(".jpg","").replace(".JPG","")
                    #print(imagefull)
                    random_op(op,imagefull,imagename,tags[i][0],SaveImagePath)
                    #print(counts)
                    counts=counts+1
    cv2.waitKey(0)