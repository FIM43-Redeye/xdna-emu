import math
import itertools
import collections
import cProfile
import copy
import pprint

from helpers import *

class MultMode(collections.namedtuple("MultMode",("bits_x","bits_y","acc_cmb","bfloat","acc_num"))):
  def __str__(self):
    m = ["","bf"]
    return "%dx(%s%dx%s%d->%dacc)" % (self.acc_num,m[self.bfloat],self.bits_x,m[self.bfloat],self.bits_y,self.acc_cmb)

class PermuteMode(collections.namedtuple("PermuteMode",
                                           ("mmode",
                                            "complex_x", "order_x", "convolve_x",
                                            "complex_y", "order_y", "convolve_y",
                                                         "order_o",

                                            "rows",    "row_stride",
                                            "cols",    "col_stride",
                                            "inner",   "inner_stride_x",  "inner_stride_y",
                                            "channels","channel_stride_x","channel_stride_y",
                                            "sparse",
                                            ))):
  def __str__(self):
    cspec = "%d:%dx"         % (self.channels,self.channel_stride_x                                             )
    dspec = "%d:%dx"         % (self.channels,self.channel_stride_y                                             )
    xspec = "%2d:%dx%2d:%d%s"% (self.rows, self.row_stride,    self.inner,self.inner_stride_x," T"[self.order_x])
    yspec = "%2d:%dx%d:%d%s" % (self.inner,self.inner_stride_y,self.cols, self.col_stride,    " T"[self.order_y])
    ospec = "%2dx%2dx%d%s"   % (self.channels,self.rows,       self.cols,                     " T"[self.order_o])
    uspec = "%s%s"           % ((""," conv")[self.convolve_x],(""," cplx")[self.complex_x],                     )
    vspec = "%s%s"           % ((""," conv")[self.convolve_y],(""," cplx")[self.complex_y],                     )
    wspec = "%s"             % (                              (""," cplx")[self.complex_x or self.complex_y]    )
    sparse = "sparse" if self.sparse else ""
    return "(%5s%10s%5s * %5s%9s%s = %7s%s %6s (mmode=%d))" % (cspec,xspec,uspec,dspec,yspec,vspec,ospec,wspec,sparse,self.mmode)

  def idx_x(self,row,inn,channel):
    cx = self.inner if not self.convolve_x else 1
    i  = rc2i(row,inn,channel,self.rows,cx,self.channels,self.row_stride,self.inner_stride_x,self.channel_stride_x,self.order_x)
    return i

  def idx_y(self,inn,col,channel):
    cy = self.cols if not self.convolve_y else 1
    i  = rc2i(inn,col,channel,self.inner,cy,self.channels,self.inner_stride_y,self.col_stride,self.channel_stride_y,self.order_y)
    return i

  def idx_o(self,row,col,channel):
    i = rc2i(row,col,channel,self.rows,self.cols,self.channels,1,1,1,self.order_o)
    return i

class Constants(object):
  def __init__(self):
    # rounding modes
    self.rnd_floor     = 0
    self.rnd_ceil      = 1
    self.rnd_sym_floor = 2
    self.rnd_sym_ceil  = 3
    self.rnd_neg_inf   = 8
    self.rnd_pos_inf   = 9
    self.rnd_sym_zero  = 10
    self.rnd_sym_inf   = 11
    self.rnd_conv_even = 12
    self.rnd_conv_odd  = 13
    self.rnd_modes     = tuple(range(4)) + tuple(range(8,14))

    # ups/srs modes and sizes
    self.scale_half = 0
    self.scale_full = 1
    self.acc_32     = 0
    self.acc_64     = 1
    self.ups_modes = { (self.scale_half,self.acc_32) : (32, 8,32),
                       (self.scale_full,self.acc_32) : (32,16,32),
                       (self.scale_half,self.acc_64) : (16,16,64),
                       (self.scale_full,self.acc_64) : (16,32,64)
                     }
    self.srs_modes = { (self.scale_half,self.acc_32) : (32,32, 8),
                       (self.scale_full,self.acc_32) : (32,32,16),
                       (self.scale_half,self.acc_64) : (16,64,16),
                       (self.scale_full,self.acc_64) : (16,64,32),
                     }

    # vadd modes
    self.vadd_add                 = 0
    self.vadd_sub                 = 2
    self.vadd_sel                 = 4
    self.vadd_addsub              = 5
    self.vadd_band                = 19
    self.vadd_bor                 = 23
    self.vadd_abs_gtz_dyn_sign    = 6
    self.vadd_abs_gtz_signed      = 7
    self.vadd_maxdiff_lt_dyn_sign = 8
    self.vadd_maxdiff_lt_signed   = 9
    self.vadd_sub_lt_dyn_sign     = 10
    self.vadd_sub_lt_signed       = 11
    self.vadd_sub_ge_dyn_sign     = 12
    self.vadd_sub_ge_signed       = 13
    self.vadd_bneg_ltz            = 14
    self.vadd_eqz                 = 15
    self.vadd_min_ge_dyn_sign     = 16
    self.vadd_min_ge_signed       = 17
    self.vadd_min_ge_bf           = 18
    self.vadd_max_lt_dyn_sign     = 20
    self.vadd_max_lt_signed       = 21
    self.vadd_max_lt_bf           = 22
    self.vadd_lt_dyn_sign         = 24
    self.vadd_lt_signed           = 25
    self.vadd_lt_bf               = 26
    self.vadd_neg_gtz             = 27
    self.vadd_ge_dyn_sign         = 28
    self.vadd_ge_signed           = 29
    self.vadd_ge_bf               = 30

    #                                                 data_out,cmp_out,binary
    self.vadd_modes = { self.vadd_add                 : (True,  False, True ),
                        self.vadd_sub                 : (True,  False, True ),
                        self.vadd_sel                 : (True,  False, True ),
                        self.vadd_addsub              : (True,  False, True ),
                        self.vadd_band                : (True,  False, True ),
                        self.vadd_bor                 : (True,  False, True ),
                        self.vadd_abs_gtz_dyn_sign    : (True,  True,  False),
                        self.vadd_abs_gtz_signed      : (True,  True,  False),
                        self.vadd_maxdiff_lt_dyn_sign : (True,  True,  True ),
                        self.vadd_maxdiff_lt_signed   : (True,  True,  True ),
                        self.vadd_sub_lt_dyn_sign     : (True,  True,  True ),
                        self.vadd_sub_lt_signed       : (True,  True,  True ),
                        self.vadd_sub_ge_dyn_sign     : (True,  True,  True ),
                        self.vadd_sub_ge_signed       : (True,  True,  True ),
                        self.vadd_bneg_ltz            : (True,  True,  False),
                        self.vadd_eqz                 : (False, True,  False),
                        self.vadd_min_ge_dyn_sign     : (True,  True,  True ),
                        self.vadd_min_ge_signed       : (True,  True,  True ),
                        self.vadd_min_ge_bf           : (True,  True,  True ),
                        self.vadd_max_lt_dyn_sign     : (True,  True,  True ),
                        self.vadd_max_lt_signed       : (True,  True,  True ),
                        self.vadd_max_lt_bf           : (True,  True,  True ),
                        self.vadd_lt_dyn_sign         : (False, True,  True ),
                        self.vadd_lt_signed           : (False, True,  True ),
                        self.vadd_lt_bf               : (False, True,  True ),
                        self.vadd_neg_gtz             : (True,  True,  False),
                        self.vadd_ge_dyn_sign         : (False, True,  True ),
                        self.vadd_ge_signed           : (False, True,  True ),
                        self.vadd_ge_bf               : (False, True,  True ),
                      }

    # Core level constants
    self.dm_bank_size = 32 * 1024 // 4
    self.dm_bank_num  = 4
    self.load_bw      = 256
    self.store_bw     = 256
    self.data_gran    = 8

    # Instruction level constants
    self.mult_num      = 512
    self.mult_gran_x   = 8
    self.mult_gran_y   = 4

    self.perm_width_x  = 512
    self.perm_width_y  = 512

    self.float_width   = 23

    self.acc_num       = 32
    self.acc_width     = 32
    self.acc_combine   = 2

    self.lanes_num     = self.acc_num//self.acc_combine
    self.mult_per_lane = self.mult_num//self.lanes_num
    self.psa_levels    = int(math.log(self.mult_num//self.acc_num,2))

    self.__make_mult_modes()
    self.__make_perm_modes()
    self.__make_indices()
    self.__make_permute_settings()
    self.__make_offsets()
    self.__generate_misc_parameters()

  def __generate_mode_mapping(self,modes,access):
    s = set(getattr(m,access) for idx,m in enumerate(modes))
    s = list(sorted(s))
    w = (len(s)-1).bit_length()
    rev = dict([(b,i) for i,b in enumerate(s)])
    setattr(self,"num_"+access,len(s))
    setattr(self,"width_"+access,w)
    setattr(self,"mapping_"+access,rev)

  def __generate_misc_parameters(self):
    self.__generate_mode_mapping(self.mult_modes,"bits_x"          )
    self.__generate_mode_mapping(self.mult_modes,"bits_y"          )
    self.__generate_mode_mapping(self.mult_modes,"acc_cmb"         )
    self.__generate_mode_mapping(self.perm_modes,"rows"            )
    self.__generate_mode_mapping(self.perm_modes,"cols"            )
    self.__generate_mode_mapping(self.perm_modes,"inner"           )
    self.__generate_mode_mapping(self.perm_modes,"channels"        )
    self.__generate_mode_mapping(self.perm_modes,"row_stride"      )
    self.__generate_mode_mapping(self.perm_modes,"col_stride"      )
    self.__generate_mode_mapping(self.perm_modes,"inner_stride_x"  )
    self.__generate_mode_mapping(self.perm_modes,"inner_stride_y"  )
    self.__generate_mode_mapping(self.perm_modes,"channel_stride_x")
    self.__generate_mode_mapping(self.perm_modes,"channel_stride_y")

    self.num_order_x   = 2
    self.width_order_x = 1
    self.mapping_order_x = {"row_major" : ROW_MAJOR, "col_major" : COL_MAJOR}
    self.num_order_x   = 2
    self.width_order_y = 1
    self.mapping_order_y = self.mapping_order_x

    self.mapping_sign = {"signed" : int(True), "unsigned" : int(False)}

  def __make_mult_modes(self):
    self.mult_modes = (MultMode(bits_x= 8, bits_y= 4, acc_cmb=1, bfloat=False, acc_num=32),
                       MultMode(bits_x= 8, bits_y= 8, acc_cmb=1, bfloat=False, acc_num=32),
                       #MultMode(bits_x= 8, bits_y= 8, acc_cmb=2, bfloat=False, acc_num=32),
                       MultMode(bits_x=16, bits_y= 8, acc_cmb=1, bfloat=False, acc_num=32),
                       MultMode(bits_x=16, bits_y=16, acc_cmb=1, bfloat=False, acc_num=32),
                       MultMode(bits_x=16, bits_y= 8, acc_cmb=2, bfloat=False, acc_num=32),
                       MultMode(bits_x=16, bits_y=16, acc_cmb=2, bfloat=False, acc_num=32),
                       MultMode(bits_x=16, bits_y=16, acc_cmb=1, bfloat=True,  acc_num=16),
                       MultMode(bits_x=32, bits_y=16, acc_cmb=2, bfloat=False, acc_num=32),
                      )

  def find_mode(self,bits_x,bits_y,acc_cmb=None,bfloat=False,acc_num=32,
                     rows    =None,row_stride      =1,
                     cols    =None,col_stride      =1,
                     inner   =None,inner_stride_x  =1,inner_stride_y  =1,
                     channels=1,   channel_stride_x=1,channel_stride_y=1,
                     convolve_x=False,
                     convolve_y=False,
                     order_x=ROW_MAJOR,
                     order_y=ROW_MAJOR,
                     order_o=ROW_MAJOR,
                     sparse=False):
    cand = []
    for pmode,pm in enumerate(self.perm_modes):
      mm = self.mult_modes[pm.mmode]

      if mm.bits_x           == bits_x                        and \
         mm.bits_y           == bits_y                        and \
        (mm.acc_cmb          == acc_cmb or (acc_cmb is None)) and \
         mm.bfloat           == bfloat                        and \
         mm.acc_num          == acc_num                       and \
         pm.order_x          == order_x                       and \
         pm.order_y          == order_y                       and \
         pm.order_o          == order_o                       and \
         pm.row_stride       == row_stride                    and \
         pm.col_stride       == col_stride                    and \
         pm.inner_stride_x   == inner_stride_x                and \
         pm.inner_stride_y   == inner_stride_y                and \
         pm.channel_stride_x == channel_stride_x              and \
         pm.channel_stride_y == channel_stride_y              and \
         pm.channels         == channels                      and \
         pm.convolve_x       == convolve_x                    and \
         pm.convolve_y       == convolve_y                    and \
        (pm.rows             == rows  or (rows  is None))     and \
        (pm.cols             == cols  or (cols  is None))     and \
        (pm.inner            == inner or (inner is None))     and \
        (pm.sparse           == sparse):
        cand.append(pmode)
    return cand

  def __make_perm_modes(self):
    self.perm_modes = []

    for bits_x,bits_y,acc_cmb,channels,rows,inner,cols,complex_x,complex_y,bfloat,convolve_x,convolve_y,sparse in \
           (# matrix multiply
            ( 8, 4, 1,  1,  4,16, 8,  False, False,  False, False, False, False),
            ( 8, 8, 1,  1,  4, 8, 8,  False, False,  False, False, False, False),
            (16, 8, 1,  1,  4, 4, 8,  False, False,  False, False, False, False), # red9
            (16,16, 1,  1,  4, 2, 8,  False, False,  False, False, False, False),
            (16, 8, 2,  1,  2, 8, 8,  False, False,  False, False, False, False),
            (16, 8, 2,  1,  4, 8, 4,  False, False,  False, False, False, False),
            (16,16, 2,  1,  2, 4, 8,  False, False,  False, False, False, False),
            (16,16, 2,  1,  4, 4, 4,  False, False,  False, False, False, False),
            (16,16, 1,  1,  4, 8, 4,  False, False,  True , False, False, False), # bfloat16
            # element-wise
            (16, 8, 1,  2,  4, 4, 4,  False, False,  False, False, False, False), # 16x8   -  2 channels, 4x4 mmult red9
            #(16, 8, 1,  1,  8, 4, 4,  False, False,  False, False, False, False), # 16x8   -  1 channels, 8x4 mmult red9
            ( 8, 8, 1, 32,  1, 2, 1,  False, False,  False, False, False, False), #  8x8   - 32 channels, 2 weights,
            (16,16, 1, 32,  1, 1, 1,  False, False,  False, False, False, False), # 16x16  - 32 channels, 1 weights,
            (16,16, 2, 16,  1, 2, 1,  False, False,  False, False, False, False), # 16x16  - 16 channels, 2 weights, 2acc
            # convolutions
            ( 8, 8, 1,  8,  4, 4, 1,  False, False,  False, True,  False, False), #  8x8   -  8 channels, 4 weights,  4 pixels - depth-wise red10
            ( 8, 8, 1,  4,  8, 8, 1,  False, False,  False, True,  False, False), #  8x8   -  4 channels, 8 weights,  8 pixels - depth-wise red10
            #( 8, 8, 1,  8,  4, 8, 1,  False, False,  False, True,  False, False), #  4x8   -  8 channels, 8 weights,  4 pixels - depth-wise red10
            ( 8, 8, 1,  1, 32, 8, 1,  False, False,  False, True,  False, False), #  8x8   -  1 channel,  8 weights, 32 pixels - 2D filter
            (16,16, 2,  1, 16, 4, 1,  False, False,  False, True,  False, False), # 16x16  -  1 channel,  4 weights, 16 pixels - 2D filter
            # addtional modes
            (16,16, 1, 16,  1, 2, 1,  False, False,  True,  False, False, False), # bfloat - 16 channels, 2 weights - element-wise
            # FFT modes
            (32,16, 2,  1,  4, 2, 4,  False, False,  False, False, False, False), # 32x16  -  1 channel,  4x2 x 2x4 matrix multiply
            (32,16, 2,  8,  1, 1, 1,  True,  True,   False, False, False, False), # 32x16  -  8 channels, 1 weight  - element-wise
            (16,16, 2,  8,  1, 2, 1,  True,  True,   False, False, False, False), # 16x16  -  8 channels, 2 weights - element-wise
            # sparse modes
            ( 8, 4, 1,  1,  4,32, 8,  False, False,  False, False, False, True ),
            ( 8, 8, 1,  1,  4,16, 8,  False, False,  False, False, False, True ),
            (16, 8, 2,  1,  2,16, 8,  False, False,  False, False, False, True ),
            (16,16, 2,  1,  2, 8, 8,  False, False,  False, False, False, True ),
            (16,16, 1,  1,  4,16, 4,  False, False,  True , False, False, True ), # bfloat16
            ):
      for mmode,mm in enumerate(self.mult_modes):
        if not ((mm.bits_x == bits_x) and (mm.bits_y == bits_y) and (False or (mm.acc_cmb == acc_cmb)) and (mm.bfloat == bfloat)):
          continue

        # upper bound - check later
        stop_row_stride       = 2 # 2**2
        stop_col_stride       = 0
        stop_inner_stride_x   = 2
        stop_inner_stride_y   = 0
        stop_channel_stride_x = 0
        stop_channel_stride_y = 0

        for row_stride,col_stride,                           \
            inner_stride_x,  inner_stride_y,                 \
            channel_stride_x,channel_stride_y,               \
            order_x,         order_y,          order_o in    \
          itertools.product((2**i for i in range(0,stop_row_stride      +1)),
                            (2**i for i in range(0,stop_col_stride      +1)),
                            (2**i for i in range(0,stop_inner_stride_x  +1)),
                            (2**i for i in range(0,stop_inner_stride_y  +1)),
                            (2**i for i in range(0,stop_channel_stride_x+1)),
                            (2**i for i in range(0,stop_channel_stride_y+1)),
                            (ROW_MAJOR,COL_MAJOR),
                            (ROW_MAJOR,COL_MAJOR),
                            (ROW_MAJOR,COL_MAJOR)):

          sz_x  = mm.bits_x * channels*channel_stride_x
          sz_x *= rows*row_stride * inner*inner_stride_x if not convolve_x else (rows-1)*row_stride + 1 + inner*inner_stride_x - 1

          sz_y  = mm.bits_y * channels*channel_stride_y
          sz_y *= cols*col_stride * inner*inner_stride_y if not convolve_y else (cols-1)*col_stride + 1 + inner*inner_stride_y - 1

          pwx = self.perm_width_x
          if sparse:
            sz_y //= 2
            pwx  *= 2

          if (sz_x > pwx) or (sz_y > self.perm_width_y):
            continue

          # Only allow modes for DCT
          allow = False
          #if row_stride == 4 or inner_stride_x > 1:
          #  if   bits_x == 16 and bits_y == 16 and acc_cmb == 1 and rows == 4 and row_stride == 4 and inner_stride_x == 1 and order_x == ROW_MAJOR:
            #  allow = True
          #  #elif bits_x == 16 and bits_y == 16 and acc_cmb == 1 and rows == 4 and row_stride == 1 and inner_stride_x == 2 and order_x == COL_MAJOR:
          #  #  allow = True
          #  elif bits_x == 16 and bits_y == 16 and acc_cmb == 2 and rows == 2 and row_stride == 1 and inner_stride_x == 4 and order_x == COL_MAJOR:
          #    allow = True
          #  elif bits_x == 16 and bits_y ==  8 and acc_cmb == 1 and rows == 4 and row_stride == 1 and inner_stride_x == 2 and order_x == COL_MAJOR:
          #    allow = True
          #  else:
          #    continue

          #if bits_x == 16 and bits_y ==  8 and acc_cmb == 1 and rows == 4 and row_stride == 1 and inner_stride_x == 1 and order_x == COL_MAJOR and channels == 2:
          #  allow = True

          if order_o == COL_MAJOR or order_y == COL_MAJOR:
            continue
          if order_x == COL_MAJOR:
            #if sparse:
            #  continue
            #elif allow:
            #  pass
            #elif convolve_x         or \
            #     rows == 1          or \
            #     inner == 1         or \
            #     bits_x != bits_y   or \
            #     channels > 1       or \
            #     row_stride > 1     or \
            #     inner_stride_x > 1 or \
            #     bfloat:
              continue

          #if row_stride == 2:
          #  continue

          if row_stride > 1 or col_stride > 1 or inner_stride_x > 1 or inner_stride_y > 1:
            continue

          self.perm_modes.append(PermuteMode(mmode=mmode,
                                             rows    =rows,    row_stride      =row_stride,
                                             cols    =cols,    col_stride      =col_stride,
                                             inner   =inner,   inner_stride_x  =inner_stride_x,   inner_stride_y  =inner_stride_y,
                                             channels=channels,channel_stride_x=channel_stride_x, channel_stride_y=channel_stride_y,
                                             order_x=order_x,
                                             order_y=order_y,
                                             order_o=order_o,
                                             complex_x=complex_x,
                                             complex_y=complex_y,
                                             convolve_x=convolve_x,
                                             convolve_y=convolve_y,
                                             sparse=sparse))
          #print "adding",len(self.perm_modes)-1,self.perm_modes[-1],str(len(self.perm_modes)),sz_x,sz_y

  def __make_permute_settings(self):
    self.permute_x = []
    self.permute_y = []

    for pmode,pm in enumerate(self.perm_modes):
      el = self.__make_perms_helper(pmode,0)
      #start = len(self.permute_x)
      #stop  = start + len(el)-1
      #print pmode, start, stop, len(el)/3
      self.permute_x.extend(el)
      self.permute_y.extend(self.__make_perms_helper(pmode,1))

  def __make_perms_helper(self,pmode,Side):
    pm = self.perm_modes[pmode]
    mm = self.mult_modes[pm.mmode]

    bits    = (mm.bits_x,       mm.bits_y       )[Side]
    mgran   = (self.mult_gran_x,self.mult_gran_y)[Side]
    icplx   = (pm.complex_x,    pm.complex_y    )[Side]
    ocplx   = pm.complex_x or pm.complex_y
    rep     = bits//mgran*(2 if icplx else 1)
    indices = self.mpy_indices[(pm.mmode,pm.complex_x,pm.complex_y)]
    mults   = self.mult_num//self.acc_num*mm.acc_cmb
    mults  *= 2 if ocplx else 1

    permute = []
    #print pm,mm

    # -----------------------------------------
    # Sparse modes on X side only

    if pm.sparse and Side == 0:
      assert pm.channels == 1
      assert ocplx == False

      # For 8b x 4b mode, we share the control signals over two columns
      step_col = 2 if self.mult_modes[pm.mmode].bits_y == 4 else 1

      for dense_inn in range(pm.inner//2):
        for col1 in range(0, pm.cols, step_col):
          for k in range(3):
            permute.append([-2]*self.mult_num)

            for col2 in range(step_col): # For 8b x 4b mode
              col = col1 + col2

              for row in range(pm.rows):
                for j,(inn,(cx,ix),(cy,iy),_) in enumerate(indices):
                  if j >= mults or inn != dense_inn:
                    continue

                  lane = pm.idx_o(row,col,0)
                  dst  = lane*mults

                  assert dense_inn < (pm.inner//2) # sparse only exists when 100% inner dim is used (actually 200%)

                  i = pm.idx_x(row,2*dense_inn+k-(dense_inn%2),0)
                  src = i*rep + ix//mgran

                  permute[-1][dst+j] = src

            #print "\t".join(map(str,permute[-1]))

      return permute

    # -----------------------------------------
    # Non-sparse modes or sparse Y side

    permute.append([-1]*self.mult_num)

    for row in range(pm.rows):
      for col in range(pm.cols):
        for channel in range(pm.channels):
            #print "  ",row,col,channel,lane,mults,"-->",dst
            lane = pm.idx_o(row,col,channel)
            dst  = lane*mults
            for j,(inn,(cx,ix),(cy,iy),_) in enumerate(indices):
              if j >= mults:
                continue

              if (pm.channels == 32 and pm.inner == 2) or \
                 (pm.channels ==  8 and pm.inner == 2 and not pm.complex_x) :
                j2  = j & 0x13
                j2 |= (j & 0x08) >> 1
                j2 |= (j & 0x04) << 1
              else:
                j2 = j

              if inn >= pm.inner: # skip if inner dimension is smaller than possible
                src = -2
              else:
                if Side == 0: i = pm.idx_x(row,inn,channel)
                else:         i = pm.idx_y(inn,col,channel)
                o = (ix,iy)[Side]
                c = (cx,cy)[Side]

                src = i*rep + o//mgran + c*bits//mgran

              #if pm.mmode == 0:
              #  print "    ",j,inn,cx,ix,cy,iy,"-->",i,o,src,dst
              permute[-1][dst+j2] = src

    return permute

  def __make_indices_helper(self,mmode,cplx_x,cplx_y):
    mm = self.mult_modes[mmode]
    bx = mm.bits_x//2 if mm.bfloat else mm.bits_x
    by = mm.bits_y//2 if mm.bfloat else mm.bits_y

    mul_per_acc  = self.mult_num//self.acc_num
    mul_per_op   = (bx//self.mult_gran_x)*(by//self.mult_gran_y) # 1 (8x4) to 16 (32x16)
    mul_per_lane = mul_per_acc * mm.acc_cmb                    # 16 or 32
    dup = mul_per_lane // mul_per_op                            # post-adding depth
    dup //= 2 if (cplx_x or cplx_y) else 1                      # half for complex modes

    idx = [[],[]]
    for         d  in range(dup):
      for       cx in range(2 if cplx_x else 1):
        for     cy in range(2 if cplx_y else 1):
          for   ix in range(0,bx,self.mult_gran_x):
            for iy in range(0,by,self.mult_gran_y):
              idx[cx^cy].append((d,(cx,ix),(cy,iy),cx&cy))
    idx = idx[0] + idx[1]

    ilv_len = mul_per_acc if mm.bits_x != 32 else mul_per_lane

    if mm.bits_x == 32:
      npidx = np.arange(len(idx))
      npidx = npidx.reshape((-1,2,ilv_len//2))
      npidx = np.transpose(npidx,axes=(0,2,1))
      npidx = npidx.reshape(-1)
      idx = [idx[i] for i in npidx]

    elif not mm.bfloat:
      npidx = np.arange(len(idx))
      npidx = npidx.reshape((-1,2,ilv_len//2))
      npidx = np.transpose(npidx,axes=(0,2,1))
      npidx = npidx.reshape(-1)
      idx = [idx[i] for i in npidx]

    if len(idx) == 16:
      idx = idx + idx

    get_offset = lambda t: t[1][1] + t[2][1]
    offsets    = list(map(get_offset,idx))
    if not mm.bfloat:
      signs = [((ia == bx-self.mult_gran_x),(ib == by-self.mult_gran_y)) for _,(_,ia),(_,ib),_ in idx]
    else:
      signs = [(False,False) for _ in idx]

    cneg   = tuple(c for _,_,_,c in idx)
    #print mm,cplx_x,cplx_y
    #for o in idx:
    #  print o

    return idx,signs,offsets,cneg

  def __make_indices(self):
    IDX  = {}
    OFFS = [None]*len(self.mult_modes)
    SGNS = [None]*len(self.mult_modes)
    CNEG = {}

    for pmode,pm in enumerate(self.perm_modes):
      mmode = pm.mmode
      key = mmode,pm.complex_x,pm.complex_y
      idx,sgns,offs,cplx_neg = self.__make_indices_helper(*key)

      IDX[key] = idx

      if cplx_neg in CNEG:
        CNEG[cplx_neg].append(pmode)
      else:
        CNEG[cplx_neg] = [pmode]

      if len(sgns) > 32:
        assert sgns[:32] == sgns[32:]
        sgns = sgns[:32]
      assert SGNS[mmode] is None or SGNS[mmode] == sgns
      SGNS[mmode] = sgns

      if len(offs) > 32:
        assert offs[:32] == offs[32:]
        offs = offs[:32]
      assert OFFS[mmode] is None or OFFS[mmode] == offs
      OFFS[mmode] = offs

    self.mpy_indices = IDX
    self.psa_offsets = [OFFS]
    self.sge_signs   = SGNS

    self.complex_neg = []
    self.cneg_modes  = [None]*len(self.perm_modes)
    for cnegmode,(cneg,pmodes) in enumerate(CNEG.items()):
      self.complex_neg.append(cneg)
      for pmode in pmodes:
        self.cneg_modes[pmode] = cnegmode

    #print self.complex_neg
    #print self.cneg_modes
    #pprint.pprint(self.mpy_indices)
    #pprint.pprint(self.psa_offsets)
    #pprint.pprint(self.sge_signs)

  def __combine_offsets(self,Oi,Wi):
    Oo = []
    Wo = []
    for i in range(len(Oi)//2):
      oa,ob = Oi[2*i:2*i+2]
      wa,wb = Wi[2*i:2*i+2]
      o = min(oa,ob)
      w = 2**(oa-o)*wa + 2**(ob-o)*wb
      Oo.append(o)
      Wo.append(w)
    return Oo,Wo

  def __make_offsets(self):
    def max_val_to_width(mvl):
      mvl = list(map(max,zip(*mvl)))
      return [int(math.ceil(math.log(mv+1,2)))+1 for mv in mvl]

    mw = 2**(self.mult_gran_x + self.mult_gran_y)-1

    W = [[[mw]*self.mult_per_lane for _ in self.mult_modes]]
    self.psa_widths = [max_val_to_width(W[0])]

    for level in range(1,self.psa_levels+1):
      self.psa_offsets.append([])
      W.append([])

      for mmode,mm in enumerate(self.mult_modes):
        oi = self.psa_offsets[level-1][mmode]
        ol,wl = self.__combine_offsets(oi,W[level-1][mmode])
        self.psa_offsets[level].append(ol)
        if level == 1 and mm.bfloat:
          W[level].append([mw]*len(wl)) # avoid that the bfloat psa1 shift influences the width
        else:
          W[level].append(wl)

      self.psa_widths.append(max_val_to_width(W[level]))
    self.max_values = W

  def print_modes(self):
    for n in range(self.mult_per_lane):
      for mmode,_ in enumerate(self.mult_modes):
        sa,sb               = self.sge_signs       [mmode][n]
        k,(ca,ia),(cb,ib),c = self.mpy_indices     [(mmode,False,False)][n]
        o         = self.psa_offsets  [0][mmode][n]
        str = " "*(o//min(self.mult_gran_x,self.mult_gran_y)) + "us"[sa] + "us"[sb]
        print("%-12s" % str, "%2d,%d,%d,%2d,%2d" % (k,ia//self.mult_gran_x,ib//self.mult_gran_y,0,0),"|", end=" ")
      print

  def print_psa(self):
    for n in range(self.mult_per_lane):
      print("#%2d |"%n, end=" ")
      for level in range(self.psa_levels+1):
        idx = n//(2**level)

        for mmode,_ in enumerate(self.mult_modes):
          if (n % 2**level) == 0:
            #print "%2d(%5.4f)" % (self.psa_offsets[level][mmode][idx],math.log(self.max_values[level][mmode][idx],2)),
            print("%2d" % (self.psa_offsets[level][mmode][idx],), end=" ")
          else:
            print("         ", end=" ")
        if (n % 2**level) == 0:
          print("/ %2d |" % self.psa_widths[level][idx], end=" ")
        else:
          print("     |", end=" ")
      print

C = Constants()
#C.print_psa()
#C.print_modes()

#fh = open("prmy_red9.txt","w")
#for row in C.permute_y:
#  for elem in row:
#    print >>fh, "%3d" % elem,
#  print >>fh
#fh.close()
#
#fh = open("prmx_red9.txt","w")
#for row in C.permute_x:
#  for elem in row:
#    print >>fh, "%3d" % elem,
#  print >>fh
#fh.close()
