import os
import random

trainval_percent = 0.1
train_percent = 0.9
Root='/media/nizhengqi/0007912600089656/fpn.pytorch-master/data/VOCdevkit2007/VOC2007/'
xmlfilepath = Root+'Annotations'
txtsavepath = Root+'ImageSets/Main'

total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)


tv = int(num * trainval_percent)
tr = int(tv * train_percent)

trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(Root+'ImageSets/Main/trainval.txt', 'w')
ftest = open(Root+'ImageSets/Main/test.txt', 'w')
ftrain = open(Root+'ImageSets/Main/train.txt', 'w')
fval = open(Root+'ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()