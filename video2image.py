# -*- coding: utf-8 -*-
import os, sys
import datetime
# 所有视频按每秒14帧提取
# 所有数据放在DATASET_ROOT下，每一类放在一个文件夹
# 提取的帧放在OUTPUT_ROOT/frames下，每一类在一个文件夹，每一个视频提取的帧在以视频命令的文件夹下
DATASET_ROOT='/media/nizhengqi/0007912600089656/all/IQIYI_VID_DATA_Part1/IQIYI_VID_TRAIN'
OUTPUT_ROOT='/media/nizhengqi/0007912600089656/all/IQIYI_VID_DATA_Part1/out_image/'
# 提取帧的命令
def get_cmd(file, frames = 14):
    # 获取最短的文件名
    basename = os.path.basename(file)
    basename=basename.replace(".mp4","")
    output_path = OUTPUT_ROOT
    input_path = DATASET_ROOT+"/"+file
    #if not os.path.exists(output_path):
    #    os.makedirs(output_path)# ffmpeg -i ${path} -vf fps=fps=8/1 -q 0 $IMGFOLDER/%06d.jpg
    #cmd='ffmpeg -i '+input_path+' -r '+str(frames)+' '+output_path+'/'+basename+'.%4d.jpg > /dev/null'
    cmd = 'ffmpeg -i ' + input_path + ' -vf ' + 'fps=fps=8/1 -q 0 '  + output_path +  basename + '.%d.jpg > /dev/null'
    return cmd
starttime = datetime.datetime.now()
# 遍历DATASET_ROOT
dirs = os.listdir(DATASET_ROOT)
gou="00005570"
print(len(dirs))
for dir in dirs:
    if gou in dir:
        gou="333"
    elif gou !="333":
        continue
    print(dir)
    #for file in os.listdir(DATASET_ROOT+'/'+dir):
        # 如果不是.mp4后缀，忽略
    if not dir.endswith('mp4'):
        print('ignore ',dir)
    else:
        # 提取帧
        cmd = get_cmd(dir)
        print(cmd)
        os.system(cmd)

endtime = datetime.datetime.now()
print((endtime-starttime).seconds)