#coding:utf-8
import os

# dirname1='/media/nizhengqi/0007912600089656/jyz/Annotations'
# filename1='/media/nizhengqi/0007912600089656/jyz/ImageSets/Main/trainval.txt'
# filename2='/media/nizhengqi/0007912600089656/jyz/ImageSets/Main/test.txt'
# PhotoData=open(filename1,'w+')
# PhotoData.close()
# PhotoData2=open(filename2,'w+')
# PhotoData2.close()
# for root, dirs, files in os.walk(dirname1):
#     print(len(files))
#     count=0
#     for name in files:
#         #if count<1100:
#         PhotoData = open(filename1, 'a+')
#         PhotoData.write(name.replace('.xml','')+'\n')
#         print(name)
#         #else:
#          #   PhotoData2 = open(filename2, 'a+')
#          #   PhotoData2.write(name.replace('.xml', '') + '\n')
#           #  print(name)
#         count=count+1
# PhotoData.close()
# PhotoData2.close()




# for root, dirs, files in os.walk(dirname2):
#     for name in files:
#         #print(name)
#         filelist.append(os.path.join(root, name))
# for root, dirs, files in os.walk(dirname3):
#     for name in files:
#         #print(name)
#         filelist.append(os.path.join(root, name))
# zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
# for tar in filelist:
# arcname = tar[len(dirname):]
# zf.write(tar,arcname)

#rename
import os
CWD_PATH = "/media/nizhengqi/0007912600089656/jyz/JPEGImages"
CWD_PATHGT = "/media/nizhengqi/0007912600089656/jyz/Annotations"
CWD_PATH=CWD_PATH.replace("\\","/")
counts=0
re_name=os.listdir(CWD_PATH)
for tem in re_name:
	name=str(counts)
	if ".JPG" in tem or ".png" in tem:
		new_name=name+".jpg"
		new_name2=name+".xml"
		if os.path.exists(CWD_PATH+"/"+tem):
			os.rename(CWD_PATH+"/"+tem,CWD_PATH+"/"+new_name)

		name111=CWD_PATHGT+"/"+tem.replace(".png",".xml").replace(".JPG",".xml").replace(".jpg",".xml")
		print(tem)
		#print(CWD_PATH+"/JPEGImages/"+new_name2)

		if os.path.exists(name111):
			os.rename(name111,CWD_PATHGT+"/"+new_name2)
		counts=counts+1