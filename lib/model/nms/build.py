import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = []
headers = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src\\nms_cuda.cpp']
    headers += ['src\\nms_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src\\nms_cuda_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
print(extra_objects)

ffi = create_extension(
    '_ext.nms',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()

# from torch.utils.ffi import create_extension
# ffi = create_extension(
# name='_ext.nms_cuda',
# headers=['src/nms_cuda.h','src/nms_cuda_kernel.h'],
# sources=['src/nms_cuda.c'],
# with_cuda=False
# )
# ffi.build()