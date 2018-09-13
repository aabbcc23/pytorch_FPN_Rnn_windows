# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 09:34:51 2018

@author: zhoubo
"""

#!/usr/bin/env python

import numpy as np
import os
# on Windows, we need the original PATH without Anaconda's compiler in it:
PATH = os.environ.get('PATH')+ ';C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\bin'
from distutils.spawn import spawn, find_executable
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
from Cython.Distutils import build_ext
from Cython.Build import cythonize
# CUDA specific config
# nvcc is assumed to be in user's PATH
nvcc_compile_args = ['-O', '--ptxas-options=-v', '-arch=sm_60', '-c', '--compiler-options=-fPIC']
nvcc_compile_args = os.environ.get('NVCCFLAGS', '').split() + nvcc_compile_args
cuda_libs = ['cublas','ATen','tbb_static','cuda','cudart_static','cudart']

torch_include='C:\\ProgramData\\Anaconda3\\envs\\python2.7\\Lib\\site-packages\\torch\\lib\\include'
TH_in='C:\\ProgramData\\Anaconda3\\envs\\python2.7\\lib\\site-packages\\torc\\lib\\include\\TH'
THC_in='C:\\ProgramData\\Anaconda3\\envs\\python2.7\\lib\\site-packages\\torc\\lib\\include\\THC'
# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


num_ext = Extension('_nms',
                        sources=[
                                'model\\nms\\src\\nms_cuda_kernel.cu','model\\nms\\src\\nms_cuda.cpp'
                                ],
                        language='c++',
                        libraries=cuda_libs,
                        extra_compile_args=nvcc_compile_args,
                        include_dirs = [numpy_include, 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\include',torch_include,TH_in,THC_in,])

roi_align = Extension('_roi_align',
                        sources=[
                                'model\\roi_align\\src\\roi_align_kernel.cu','model\\roi_align\\src\\roi_align_cuda.cpp'
                                ],
                        language='c++',
                        libraries=cuda_libs,
                        extra_compile_args=nvcc_compile_args,
                        include_dirs = [numpy_include,TH_in,THC_in, 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\include',torch_include])

roi_crop = Extension('_roi_crop',
                        sources=[
                                'model\\roi_crop\\src\\roi_crop_cuda_kernel.cu','model\\roi_crop\\src\\roi_crop_cuda.cpp'
                                ],
                        language='c++',
                        libraries=cuda_libs,
                        extra_compile_args=nvcc_compile_args,
                        include_dirs = [numpy_include,TH_in,THC_in, 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\include',torch_include])

roi_pooling = Extension('_roi_pooling',
                        sources=[
                                'model\\roi_pooling\\src\\roi_pooling_kernel.cu','model\\roi_pooling\\src\\roi_pooling_cuda.cpp'
                                ],
                        language='c++',
                        libraries=cuda_libs,
                        extra_compile_args=nvcc_compile_args,
                        include_dirs = [numpy_include,TH_in,THC_in, 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\include',torch_include])


ext_modules = [
    Extension(
       "model.utils.cython_bbox",
        ["model/utils/bbox.pyx"],
        extra_compile_args=nvcc_compile_args,
        include_dirs = [numpy_include]
    ),
    Extension(
        'pycocotools._mask',
        sources=['pycocotools/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs=[numpy_include, 'pycocotools'],
        extra_compile_args=nvcc_compile_args,
       
    ),roi_pooling,roi_crop,roi_align,num_ext
   
]

class CUDA_build_ext(build_ext):
    """
    Custom build_ext command that compiles CUDA files.
    Note that all extension source files will be processed with this compiler.
    """
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        self.compiler.set_executable('compiler_so', 'nvcc')
        self.compiler.set_executable('linker_so', 'nvcc --shared')
        if hasattr(self.compiler, '_c_extensions'):
            self.compiler._c_extensions.append('.cu')  # needed for Windows
        self.compiler.spawn = self.spawn
        build_ext.build_extensions(self)

    def spawn(self, cmd, search_path=1, verbose=0, dry_run=0):
        """
        Perform any CUDA specific customizations before actually launching
        compile/link etc. commands.
        """
        if (sys.platform == 'darwin' and len(cmd) >= 2 and cmd[0] == 'nvcc' and
                cmd[1] == '--shared' and cmd.count('-arch') > 0):
            # Versions of distutils on OSX earlier than 2.7.9 inject
            # '-arch x86_64' which we need to strip while using nvcc for
            # linking
            while True:
                try:
                    index = cmd.index('-arch')
                    del cmd[index:index+2]
                except ValueError:
                    break
        elif self.compiler.compiler_type == 'msvc':
            # There are several things we need to do to change the commands
            # issued by MSVCCompiler into one that works with nvcc. In the end,
            # it might have been easier to write our own CCompiler class for
            # nvcc, as we're only interested in creating a shared library to
            # load with ctypes, not in creating an importable Python extension.
            # - First, we replace the cl.exe or link.exe call with an nvcc
            #   call. In case we're running Anaconda, we search cl.exe in the
            #   original search path we captured further above -- Anaconda
            #   inserts a MSVC version into PATH that is too old for nvcc.
            cmd[:1] = ['nvcc', '--compiler-bindir',
                       os.path.dirname(find_executable("cl.exe", PATH))
                       or cmd[0]]
            # for i in range(len(cmd)):
            #     print(i,cmd[i])
            # - Secondly, we fix a bunch of command line arguments.
            for idx, c in enumerate(cmd):
                # create .dll instead of .pyd files
                #if '.pyd' in c: cmd[idx] = c = c.replace('.pyd', '.dll')  #20160601, by MrX
                # replace /c by -c
                if c == '/c': cmd[idx] = '-c'
                # replace /DLL by --shared
                elif c == '/DLL': cmd[idx] = '--shared'
                # remove --compiler-options=-fPIC
                elif '-fPIC' in c: del cmd[idx]
                # replace /Tc... by ...
                elif c.startswith('/Tc'): cmd[idx] = c[3:]
                
                #******replace /Tp by ..
                elif c.startswith('/Tp'):cmd[idx]=c[3:]
                
                #******fix ID=2 error
                elif 'ID=2' in c:cmd[idx]=c[:15]
                
                # replace /Fo... by -o ...
                elif c.startswith('/Fo'): cmd[idx:idx+1] = ['-o', c[3:]]
                # replace /LIBPATH:... by -L...
                elif c.startswith('/LIBPATH:'): cmd[idx] = '-L' + c[9:]
                # replace /OUT:... by -o ...
                elif c.startswith('/OUT:'): cmd[idx:idx+1] = ['-o', c[5:]]
                # remove /EXPORT:initlibcudamat or /EXPORT:initlibcudalearn
                elif c.startswith('/EXPORT:'): del cmd[idx]
                # replace cublas.lib by -lcublas
                elif c == 'cublas.lib': cmd[idx] = '-lcublas'
            # - Finally, we pass on all arguments starting with a '/' to the
            #   compiler or linker, and have nvcc handle all other arguments
            if '--shared' in cmd:
                pass_on = '--linker-options='
                # we only need MSVCRT for a .dll, remove CMT if it sneaks in:
                cmd.append('/NODEFAULTLIB:libcmt.lib')
            else:
                pass_on = '--compiler-options='
            cmd = ([c for c in cmd if c[0] != '/'] +
                   [pass_on + ','.join(c for c in cmd if c[0] == '/')])
            # For the future: Apart from the wrongly set PATH by Anaconda, it
            # would suffice to run the following for compilation on Windows:
            # nvcc -c -O -o <file>.obj <file>.cu
            # And the following for linking:
            # nvcc --shared -o <file>.dll <file1>.obj <file2>.obj -lcublas
            # This could be done by a NVCCCompiler class for all platforms.

        # for i in range(len(cmd)):
        #     print(i,cmd[i])

        if cmd[9]=='build\temp.win-amd64-3.5\Release\bbox.obj':
            cmd[12]=cmd[12].replace('ID=2,','')

        spawn(cmd, search_path, verbose, dry_run)

setup(name="py_fast_rcnn_gpu",
      description="Performs linear algebra computation on the GPU via CUDA",
      ext_modules=ext_modules,
      cmdclass={'build_ext': CUDA_build_ext},
)
setup(name='pycocotools',
      packages=['pycocotools'],
      package_dir = {'pycocotools': 'pycocotools'},
      version='2.0',
      ext_modules=
          cythonize(ext_modules)
      )