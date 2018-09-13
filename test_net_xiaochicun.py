#coding:utf-8
# --------------------------------------------------------
# Pytorch FPN implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, based on code from faster R-CNN
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
from PIL import Image
import pdb
import time
import cv2
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.utils.net_utils import vis_detections

from lib.model.fpn.resnet import resnet

import pdb
#print(torch.cuda.set_device(0))
WIDTH=600
HIGHT=600
#check position is same or near,make box
def check_same_or_near(xmin,xmax,ymin,ymax,all_position,thresh_pixel):
    for key,value in all_position.items():
        s=pow( pow((xmax-xmin-value[2]+value[0]),2)+pow((ymax-ymin-value[3]+value[1]),2),0.5)
        #print("s",s)
        if s<thresh_pixel:
            return  True
    return False
#def MatrixToImage(data):
#    data = data.numpy()*255
#    new_im = Image.fromarray(data.astype(np.uint8))
 #   return new_im
def interest(im2show,data,fpn,all_position,i,all_boxes,r_w,r_h,rat_w,rat_h):

    for key,value in all_position.items():
        x=int(((value[2]-value[0])/2+value[0])*rat_w)
        y=int(((value[3]-value[1])/2+value[1])*rat_h)
        data_tem = data[0][:, :, y-int(HIGHT/2):y + int(HIGHT/2), x-int(WIDTH/2):x + int(WIDTH/2)]
        #print(data[0].shape())
        w = len(data_tem[0][0][0])
        h = len(data_tem[0][0])
        print("INER",w,h)
        if w <= 0 or h <= 0:
            return  None
        if args.cuda:
            data_tem1 = torch.from_numpy(np.array([[h, w, w / h]])).float().cuda()
            data_tem2 = torch.from_numpy(np.array([[1, 1, 1, 1, 1]])).float().cuda()
            data_tem3 = torch.from_numpy(np.array([1])).long().cuda()
        else:
            data_tem1 = torch.from_numpy(np.array([[h, w, w / h]])).float()
            data_tem2 = torch.from_numpy(np.array([[1, 1, 1, 1, 1]])).float()
            data_tem3 = torch.from_numpy(np.array([1])).long()
        im_data.data.resize_(data_tem.size()).copy_(data_tem)
        im_info.data.resize_(data_tem1.size()).copy_(data_tem1)
        gt_boxes.data.resize_(data_tem2.size()).copy_(data_tem2)
        num_boxes.data.resize_(data_tem3.size()).copy_(data_tem3)
        rois, cls_prob, bbox_pred, \
        _, _, _, _, _ = fpn(im_data, im_info, gt_boxes, num_boxes)
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]  # 忽略掉前面一个数值，后面都是BOX
        if cfg.TEST.BBOX_REG:
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                if args.class_agnostic:
                    if args.cuda:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            pred_boxes = boxes
        pred_boxes /= data_tem1[0][2]
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()


        for j in range(1, imdb.num_classes):  # 遍历每一类
            inds = torch.nonzero(scores[:, j] > 0.6).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)  # 排序分数列表降低序
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                for c in range(len(cls_boxes)):  # 调整，获取小图片在大图片里面的坐标
                    cls_boxes[c][0] = (cls_boxes[c][0] + x-int(WIDTH/2)) / rat_w
                    cls_boxes[c][1] = (cls_boxes[c][1] + y-int(HIGHT/2)) / rat_h
                    cls_boxes[c][2] = (cls_boxes[c][2] + x-int(WIDTH/2)) / rat_w
                    cls_boxes[c][3] = (cls_boxes[c][3] + y-int(HIGHT/2)) / rat_h

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)  # 追加
                cls_dets = cls_dets[order]  # 将torch.tensor 按给定的训练排序
                keep = nms(cls_dets, cfg.TEST.NMS,args.cuda)  # 非极大值抑制,获取要保留的
                cls_dets = cls_dets[keep.view(-1).long()]  # 从tensor里面拿出对应的数据结构

                if all_boxes[j][i]==[]:
                    all_boxes[j][i]=cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] =np.vstack((all_boxes[j][i],  cls_dets.cpu().numpy()))




def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/res101_ls.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="D:/LiuweiWork/models-master/research/fpn.pytorch-master/models",
                      nargs=argparse.REMAINDER)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression 是否执行不可知，未知BOX的回归',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=12, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=1199, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
#torch.cuda.empty_cache()
if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_test"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'fpn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    fpn = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fpn = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fpn = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fpn = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fpn.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fpn.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
  else:
    im_data = im_data.cpu()
    im_info = im_info.cpu()
    num_boxes = num_boxes.cpu()
    gt_boxes = gt_boxes.cpu()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True
  else:
    cfg.CUDA=False

  if args.cuda:
    fpn.cuda()
  else:
    fpn.cpu()

  start = time.time()
  max_per_image = 5

  vis = True#args.vis

  if vis:
    thresh = 0.7
  else:
    thresh = 0.7

  save_name = 'faster_rcnn_10'
  num_images = len(imdb.image_index)
  #print(imdb.num_classes)
  all_boxes = [[[] for _ in range(num_images)]#初始化
               for _ in range(imdb.num_classes)]
  #print(all_boxes)

  output_dir = get_output_dir(imdb, save_name)


  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=False)
  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fpn.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))

  for i in range(num_images):
      misc_tic = time.time()
      data = data_iter.next()
      #print(data[0][0,:,:,:])
      #print(data[1])
      hh = len(data[0][0][0])
      ww = len(data[0][0][0][0])
      # cfg.TRAIN.ACTIVE_LARGE = False
      print(hh,ww)
      center_x = 0
      center_y = 0
      # data[0]=data_tem
      counts=0
      #遍历识别出所有框和分类，并保存到数据结构
      #计算出几个特定的可能的关注位置，移动视角到这些位置，识别出中心位置对象，并记录到数据结构B，直到可能的位置都识别完。
      #数据结构B输出到文本。并完成验证工作


      all_clsd = {}
      all_clss={}
      all_position={}
      im = cv2.imread(imdb.image_path_at(i))
      r_w = im.shape[1]
      r_h = im.shape[0]
      #print("r_w",r_w,"r_h",r_h)
      im2show = np.copy(im)
      #print(all_clsd[2])
      for y in range(0, hh, 500):
          for x in range(0, ww, 500):
              print(x,y)
              #image_np = data[x:x + 800, y:y + 800]
              #print(data[0])
              data_tem = data[0][:, :, y:y+600, x:x+600]
              w = len(data_tem[0][0][0])
              h = len(data_tem[0][0])
              #print(w,h)
              if w <= 0 or h <= 0:
                  continue
              if args.cuda:
                  data_tem1 = torch.from_numpy(np.array([[h, w, w/h]])).float().cuda()
                  # print(data[1],data[2],data[3])
                  data_tem2 = torch.from_numpy(np.array([[1, 1, 1, 1, 1]])).float().cuda()
                  data_tem3 = torch.from_numpy(np.array([1])).long().cuda()
              else:
                  data_tem1 = torch.from_numpy(np.array([[h, w, w / h]])).float()
                  # print(data[1],data[2],data[3])
                  data_tem2 = torch.from_numpy(np.array([[1, 1, 1, 1, 1]])).float()
                  data_tem3 = torch.from_numpy(np.array([1])).long()

              im_data.data.resize_(data_tem.size()).copy_(data_tem)
              im_info.data.resize_(data_tem1.size()).copy_(data_tem1)
              gt_boxes.data.resize_(data_tem2.size()).copy_(data_tem2)
              num_boxes.data.resize_(data_tem3.size()).copy_(data_tem3)
              #print(gt_boxes)

              rois, cls_prob, bbox_pred, \
                  _, _, _, _, _ = fpn(im_data, im_info, gt_boxes, num_boxes)
              scores = cls_prob.data
              #print("rois.data",rois.data)
              boxes = rois.data[:, :, 1:5]#忽略掉前面一个数值，后面都是BOX
              #print("boxes",boxes)
              if cfg.TEST.BBOX_REG:
                  # Apply bounding-box regression deltas
                  box_deltas = bbox_pred.data
                  #print("box_deltas0",box_deltas,box_deltas.size())
                  if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                  # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        if args.cuda:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                       + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                            box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS).cpu() \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cpu()
                            box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        if args.cuda:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                       + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                  #print("box_deltas1", box_deltas, box_deltas.size())
                  pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                  #print("pred_boxes0",pred_boxes)

                  pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                  #print("pred_boxes1", pred_boxes)
              else:
                  # Simply repeat the boxes, once for each class
                  pred_boxes = boxes
              pred_boxes /= data_tem1[0][2]
              scores = scores.squeeze()
              pred_boxes = pred_boxes.squeeze()
              rat_h=hh/r_h
              rat_w=ww/r_w
              for j in range(1, imdb.num_classes):#遍历每一类
                  inds = torch.nonzero(scores[:,j]>thresh).view(-1)
                  if inds.numel() > 0:
                    cls_scores = scores[:,j][inds]
                    _, order = torch.sort(cls_scores,0, True)#排序分数列表降低序
                    if args.class_agnostic:
                      cls_boxes = pred_boxes[inds, :]
                    else:
                      cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                    for c in range(len(cls_boxes)):#调整，获取小图片在大图片里面的坐标
                        cls_boxes[c][0]=(cls_boxes[c][0]+x)/rat_w
                        cls_boxes[c][1]=(cls_boxes[c][1]+y)/rat_h
                        cls_boxes[c][2] = (cls_boxes[c][2] + x)/rat_w
                        cls_boxes[c][3] = (cls_boxes[c][3] +y)/rat_h
                        rb=check_same_or_near(cls_boxes[c][0],cls_boxes[c][2],cls_boxes[c][1],cls_boxes[c][3],all_position,10)
                        if rb==False:
                            all_position[counts]=cls_boxes[c]
                            counts=counts+1
                        #print(rb)
      print("all_postion=",all_position)


      re=interest(None, data, fpn, all_position, i,all_boxes,r_w,r_h,rat_w,rat_h)

      if vis:
          for j in range(1,imdb.num_classes):
              if max_per_image > 0:
                  image_scores = np.hstack([all_boxes[j][i][:, -1]
                                            for j in range(1, imdb.num_classes)])
                  if len(image_scores) > max_per_image:
                      image_thresh = np.sort(image_scores)[-max_per_image]
                      for j in range(1, imdb.num_classes):
                          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                          all_boxes[j][i] = all_boxes[j][i][keep, :]
          #im2show=MatrixToImage(data[0])
          for j in range(1, imdb.num_classes):  # 遍历每一类
            im2show = vis_detections(im2show, imdb.classes[j], all_boxes[j][i], 0.5)  # 在图片对象上画框
          cv2.imwrite('images/result%d.png' % (i), im2show)  # 保存图片文件
      misc_toc = time.time()
      nms_time = misc_toc - misc_tic
      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s   \r' \
          .format(i + 1, num_images, nms_time))
      sys.stdout.flush()
      data_tem = None
  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)#对all_box 序列化保存
  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)#验证，并输出
  end = time.time()
  print("test time: %0.4fs" % (end - start))
  if args.cuda:
    torch.cuda.empty_cache()