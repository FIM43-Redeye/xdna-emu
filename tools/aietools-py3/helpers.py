"""aietools helpers.py -- Python3-compatible version.

Original: aietools/data/aie_ml/lib/python_model/model/helpers.py
Fixes: long() -> int(), w2/w1 -> w2//w1, dtype=long -> dtype=np.int64
"""
import random
import numpy as np

def trnc(a,sgn,bits):
  if isinstance(a,complex):
    re = trnc(a.real,sgn,bits)
    im = trnc(a.imag,sgn,bits)
    return re + 1j*im
  msk = 2**bits-1
  a = int(a) & msk
  if bits == 0:
    return 0
  s = (a >> (bits-1)) & 1
  if (s==1) and sgn:
    a = a | ~msk
  return a

def rnd(bits,sgn):
  um = 2**bits
  sm = 2**(bits-1)
  lb = -sm  if sgn else 0
  ub = sm-1 if sgn else um-1
  return random.randint(lb,ub)

ROW_MAJOR = 0
COL_MAJOR = 1

def rc2i(row,col,channel,rows,cols,channels,row_stride,col_stride,channel_stride,order):
  if order == ROW_MAJOR:
    return ((row_stride*row*cols + col)*col_stride*channels + channel)*channel_stride
  else:
    return ((col_stride*col*rows + row)*row_stride*channels + channel)*channel_stride

def change_width(a,w1,w2):
  if w1 > w2:
    a = np.array([int(v) for v in a])
    assert w1%w2 == 0
    m = 2**w2-1
    b = a[...,np.newaxis]
    s = np.array(list((range(0,w1,w2))))
    T = (b >> s) & m
    T = T.reshape(*(T.shape[:-2] + (-1,)))
    return T
  elif w1 < w2:
    assert w2%w1 == 0
    d = a.shape[:-1] + (-1,w2//w1)
    m = 2**w1-1
    b = a.reshape(*d) & m
    s = np.array(list((range(0,w2,w1))))
    T = np.sum(b << s,axis=-1)
    return T
  else:
    return a

def random_tensor(shape,bits,sgn,complex=False):
  if bits == 64 and complex:
      bits = 53
  if sgn: low,high = -2**(bits-1),2**(bits-1)
  else  : low,high = 0,           2**(bits)
  if not complex:
    return np.random.randint(low,high,size=shape,dtype=np.int64)
  else:
    re = np.random.randint(low,high,size=shape,dtype=np.int64)
    im = np.random.randint(low,high,size=shape,dtype=np.int64)
    return re + 1j*im
