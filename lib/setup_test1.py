# build.py
from torch.utils.ffi import create_extension
ffi = create_extension(
name='my_lib',
headers='test1.h',
sources=['test1.c'],
with_cuda=False
)
ffi.build()