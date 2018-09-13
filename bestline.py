# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *
import cv2

if __name__ == '__main__':#支持中文
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    with open("result.txt","r") as f:
        sp=f.readlines()
        names = []
        #best xianjia juya 13  fangzhencui 40   fushusheshi 40
        dic={ 'jyzzb':0,'jyhsh':1,'jyhzf':2,'jyhtl':3,'jyhyw':4,'xjqx':5,'jyhqx':6,'fzchy':7,'fzcpx':8,'fzcsh':9,'fzcxs':10,'fnsssh':11,'bspsh':12,'jueyuanzi':13,'ganta':14}
       # 'jyzzb', 'jyhsh', 'jyhzf', 'jyhtl', 'jyhtl', 'jyhyw', 'xjqx', 'jyhqx', 'fzchy', 'fzcpx', 'fzcsh', 'fzcxs', 'fnsssh', 'bspsh', 'jueyuanzi', 'ganta')
        y = np.zeros((dic.__len__(),len(sp)))
        print(y)
        ks=0
        for item in sp:
            item=item.split(" ")
            part1=item[0]
            part1=part1.replace("[","").replace("]","").replace("'","")
            sub_part1=part1.split(",")
            for sub in sub_part1:
                sub_sub_part1=sub.split(":")
                if sub_sub_part1[0] in dic:
                     #print("dic[sub_sub_part1[0]]",sub_sub_part1[0],dic[sub_sub_part1[0]])
                     y[dic[sub_sub_part1[0]]][ks]= sub_sub_part1[1]
            part2=item[1]
            names=np.append(names,part2)
                #item=item.split(':')
            ks=ks+1
        x = range(len(names))
            #y1=[0.86,0.85,0.853,0.849,0.83]
            #plt.plot(x, y, 'ro-')
            #plt.plot(x, y1, 'bo-')
            #pl.xlim(-1, 11)  # 限定横轴的范围
            #pl.ylim(-1, 110)  # 限定纵轴的范围
        sc=0
        for key,value in dic.items():
            print(key,value)
            plt.plot(x, y[value], marker='o', mec='r', mfc='w',label=key)
            sc=sc+1
        plt.legend()  # 让图例生效
        #print(names)
        plt.xticks(x, names, rotation=45)
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel("gen") #X轴标签
        plt.ylabel("AP") #Y轴标签
        plt.title(u"ap line") #标题

        plt.show()
        cv2.waitKey(0)