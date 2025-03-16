"""
Faster save/load for simple numpy arrays

Taken from https://github.com/divideconcept/fastnumpyio under MIT License included below

MIT License, Copyright (c) 2022 Robin Lobel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import numpy as np
import numpy.lib.format
import struct

def save(file, array):
    magic_string=b"\x93NUMPY\x01\x00v\x00"
    header=bytes(("{'descr': '"+array.dtype.descr[0][1]+"', 'fortran_order': False, 'shape': "+str(array.shape)+", }").ljust(127-len(magic_string))+"\n",'utf-8')
    if type(file) == str:
        file=open(file,"wb")
    file.write(magic_string)
    file.write(header)
    file.write(array.data)

def pack(array):
    """
    Convert a numpy array to a storable bytes object

    Parameters
    ----------
    array : np.ndarray
        The numpy array to store.

    Returns
    -------
    output_bytes : bytes
        The bytes

    """
    # Only contiguous arrays can be stored directly as bytes
    contiguous_array = np.ascontiguousarray(array)

    size = len(contiguous_array.shape)
    output_bytes = (
        bytes(
            contiguous_array.dtype.byteorder.replace('=', '<' if sys.byteorder == 'little' else '>')
            + contiguous_array.dtype.kind,'utf-8')
        + contiguous_array.dtype.itemsize.to_bytes(1, byteorder='little')
        + struct.pack(f'<B{size}I', size, *contiguous_array.shape)
        + contiguous_array.data)
    return output_bytes

def load(file):
    if type(file) == str:
        file=open(file,"rb")
    header = file.read(128)
    if not header:
        return None
    descr = str(header[19:25], 'utf-8').replace("'","").replace(" ","")
    shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(',)', ')').replace(', }', '').replace('(', '').replace(')', '').split(','))
    datasize = numpy.lib.format.descr_to_dtype(descr).itemsize
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))

def unpack(data):
    dtype = str(data[:2],'utf-8')
    dtype += str(data[2])
    size = data[3]
    shape = struct.unpack_from(f'<{size}I', data, 4)
    datasize=data[2]
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=dtype, buffer=data[4+size*4:4+size*4+datasize])