# pytorch_FPN_Rnn_windows
pytorch faser-rcnn feature pyramid network(fpn) for windows10 cuda9.0 compiled
bg.
Why I create this project
because pytorch faser-rcnn with fpn not well support actully because hard to compile nvcc and vs link in windows env and I have a project use this frame and slove the problem.
For other tech can fellow the lib/copiler.txt to do recompile, other wise just use compiled.


feature:
1、support cpu
2、support gpu
3、FPN
4、large scale image detect
5、one epoch one save model
6、mul model val generate result.txt and useage:python bestline.py can draw  ap lines

env:
windows 10 64x Ti 1080 nvida cuda 9.0 
