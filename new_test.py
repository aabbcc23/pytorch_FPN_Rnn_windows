import os
#print (os.environ["TEMP"])

mydir = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0"
#os.environ["MYDIR"] = mydir
#print (os.environ["MYDIR"])

#pathV = os.environ["CUDA_PATH"]
#print (pathV)
#os.environ["CUDA_PATH"]= mydir
#print (os.environ["CUDA_PATH"])
import torch
print(torch.cuda.set_device(0))


#C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0  CUDA_PATH  CUDA_PATH_V9_0   CUDAHOME

#C:\Program Files (x86)\Intel\iCLS Client\;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp;C:\ProgramData\Oracle\Java\javapath;C:\Program Files\Intel\iCLS Client\;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Users\nizhengqi\.dnx\bin;C:\Program Files\Microsoft DNX\Dnvm\;C:\Program Files\Microsoft SQL Server\130\Tools\Binn\;C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\;C:\Program Files (x86)\GtkSharp\2.12\bin;C:\Users\nizhengqi\Downloads\protoc-3.5.0-win32 (1)\bin\;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;d:\Program Files\Git\cmd;C:\Program Files (x86)\Windows Kits\8.1\Windows Performance Toolkit\;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0;D:\Strawberry\c\bin;D:\Strawberry\perl\site\bin;D:\Strawberry\perl\bin;C:\ProgramData\Anaconda3\envs\python2.7;D:\models-master;D:\models-master\research;D:\models-master\research\slim;C:\Python27\Scripts;D:\LiuweiWork\TestPhoto\model\ffmpeg-20180409-3b2fd96-win64-static\bin;%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SYSTEMROOT%\System32\WindowsPowerShell\v1.0\;%SYSTEMROOT%\System32\OpenSSH\;C:\ProgramData\Anaconda3\envs\python2.7\DLLs;C:\ProgramData\Anaconda3\envs\python2.7\libs;C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\DAL;C:\Program Files\Intel\Intel(R) Management Engine Components\DAL;C:\Program Files (x86)\Intel\Intel(R) Management Engine Components\IPT;C:\Program Files\Intel\Intel(R) Management Engine Components\IPT;C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64;C:\Program Files\Microsoft VS Code\bin;C:\ProgramData\Anaconda3\envs\python2.7\Lib\site-packages\torch\lib;