// AIE2RegisterDecoderTables.inc -- Register decoder lookup tables.
//
// Extracted from llvm-aie/llvm/lib/Target/AIE/Disassembler/AIE2Disassembler.cpp
// lines 53-169.  Source: llvm-aie (Apache-2.0 WITH LLVM-exception).
//
// These tables map encoded register numbers to LLVM register enum values
// (from AIE2GenRegisterEnum.inc).  They are used by the TABLEBASEDDECODER
// macro and by the custom register decoders.
//
// Included after `using namespace llvm::AIE2;` so bare register names
// (r0, p0, wl0, etc.) resolve to enum values.

static const unsigned eRDecoderTable[] = {
    r0,  r1,  r2,  r3,  r4,  r5,  r6,  r7,  r8,  r9,  r10,
    r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21,
    r22, r23, r24, r25, r26, r27, r28, r29, r30, r31};

static const unsigned eRS4DecoderTable[] = {r16, r17, r18, r19};

static const unsigned eRS8DecoderTable[] = {r16, r17, r18, r19,
                                            r20, r21, r22, r23};

static const unsigned eLDecoderTable[] = {l0, l1, l2, l3, l4, l5, l6, l7};

static const unsigned ePDecoderTable[] = {
    p0, p1, p2, p3, p4, p5, p6, p7,
};
static const unsigned eMDecoderTable[] = {
    m0, m1, m2, m3, m4, m5, m6, m7,
};
static const unsigned eDJDecoderTable[] = {
    dj0, dj1, dj2, dj3, dj4, dj5, dj6, dj7,
};
static const unsigned eDDecoderTable[] = {
    d0, d1, d2, d3, d4, d5, d6, d7,
};
static const unsigned eDSDecoderTable[] = {d0_3d, d1_3d, d2_3d, d3_3d};

static const unsigned mCRmDecoderTable[] = {
    crVaddSign, crF2FMask, crF2IMask, crFPMask, crMCDEn,   crPackSign,
    crRnd,      crSCDEn,   crSRSSign, crSat,    crUPSSign, crUnpackSign};

static const unsigned mLdaSclDecoderTable[] = {
    r0,  0, m0,  0, r1,  lr, m1,  0, r2,  0,  m2,  0, r3,  p0, m3,  0,
    r4,  0, m4,  0, r5,  0,  m5,  0, r6,  0,  m6,  0, r7,  p1, m7,  0,
    r8,  0, dn0, 0, r9,  0,  dn1, 0, r10, 0,  dn2, 0, r11, p2, dn3, 0,
    r12, 0, dn4, 0, r13, 0,  dn5, 0, r14, 0,  dn6, 0, r15, p3, dn7, 0,
    r16, 0, dj0, 0, r17, 0,  dj1, 0, r18, 0,  dj2, 0, r19, p4, dj3, 0,
    r20, 0, dj4, 0, r21, 0,  dj5, 0, r22, p5, dj6, 0, r23, p5, dj7, 0,
    r24, 0, dc0, 0, r25, 0,  dc1, 0, r26, 0,  dc2, 0, r27, p6, dc3, 0,
    r28, 0, dc4, 0, r29, 0,  dc5, 0, r30, 0,  dc6, 0, r31, p7, dc7, 0};

static const unsigned mLdaCgDecoderTable[] = {
    r0,  0, m0,  0, r1,  0,  m1,  0, r2,  0,  m2,  0, r3,  p0, m3,  0,
    r4,  0, m4,  0, r5,  LC, m5,  0, r6,  0,  m6,  0, r7,  p1, m7,  0,
    r8,  0, dn0, 0, r9,  0,  dn1, 0, r10, 0,  dn2, 0, r11, p2, dn3, 0,
    r12, 0, dn4, 0, r13, 0,  dn5, 0, r14, 0,  dn6, 0, r15, p3, dn7, 0,
    r16, 0, dj0, 0, r17, 0,  dj1, 0, r18, 0,  dj2, 0, r19, p4, dj3, 0,
    r20, 0, dj4, 0, r21, 0,  dj5, 0, r22, p5, dj6, 0, r23, p5, dj7, 0,
    r24, 0, dc0, 0, r25, 0,  dc1, 0, r26, 0,  dc2, 0, r27, p6, dc3, 0,
    r28, 0, dc4, 0, r29, 0,  dc5, 0, r30, 0,  dc6, 0, r31, p7, dc7, 0};

static const unsigned mMvSclDstDecoderTable[] = {
    r0,  crVaddSign, m0,  p0,          r1,  srCarry,
    m1,  LS,         r2,  crF2FMask,   m2,  s0,
    r3,  srCompr_uf, m3,  0,           r4,  crF2IMask,
    m4,  p1,         r5,  srF2FFlags,  m5,  DP,
    r6,  crFPMask,   m6,  0,           r7,  srF2IFlags,
    m7,  0,          r8,  crMCDEn,     dn0, p2,
    r9,  srFPFlags,  dn1, lr,          r10, crPackSign,
    dn2, s1,         r11, srMS0,       dn3, 0,
    r12, crRnd,      dn4, p3,          r13, srSRS_of,
    dn5, CORE_ID,    r14, crSCDEn,     dn6, 0,
    r15, srSS0,      dn7, 0,           r16, crSRSSign,
    dj0, p4,         r17, srSparse_of, dj1, LE,
    r18, crSat,      dj2, s2,          r19, srUPS_of,
    dj3, 0,          r20, crUPSSign,   dj4, p5,
    r21, 0,          dj5, LC,          r22, crUnpackSign,
    dj6, 0,          r23, 0,           dj7, 0,
    r24, 0,          dc0, p6,          r25, 0,
    dc1, SP,         r26, 0,           dc2, s3,
    r27, 0,          dc3, 0,           r28, 0,
    dc4, p7,         r29, 0,           dc5, 0,
    r30, 0,          dc6, 0,           r31, 0,
    dc7, 0};

static const unsigned mWaDecoderTable[] = {
    wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3, wl4,  wh4,  wl5,  wh5,
    wl6, wh6, wl7, wh7, wl8, wh8, wl9, wh9, wl10, wh10, wl11, wh11,
};

static const unsigned mWm_1DecoderTable[] = {
    wl0, wl2, wl4, wl6, wl8, wl10, 0, 0, wl1, wl3, wl5, wl7, wl9, wl11, 0, 0,
    wh0, wh2, wh4, wh6, wh8, wh10, 0, 0, wh1, wh3, wh5, wh7, wh9, wh11, 0, 0};

static const unsigned mXvDecoderTable[] = {x0, x1, x2, x3, x4,  x5,
                                           x6, x7, x8, x9, x10, x11};

static const unsigned mAMmDecoderTable[] = {
    amll0, amlh0, amhl0, amhh0, amll1, amlh1, amhl1, amhh1, amll2,
    amlh2, amhl2, amhh2, amll3, amlh3, amhl3, amhh3, amll4, amlh4,
    amhl4, amhh4, amll5, amlh5, amhl5, amhh5, amll6, amlh6, amhl6,
    amhh6, amll7, amlh7, amhl7, amhh7, amll8, amlh8, amhl8, amhh8,
};

static const unsigned mBMmDecoderTable[] = {
    bml0, bmh0, bml1, bmh1, bml2, bmh2, bml3, bmh3, bml4,
    bmh4, bml5, bmh5, bml6, bmh6, bml7, bmh7, bml8, bmh8,
};

static const unsigned mBMSmDecoderTable[] = {
    bml0, bml1, bml2, bml3, bml4, bml5, bml6, bml7,
    bmh0, bmh1, bmh2, bmh3, bmh4, bmh5, bmh6, bmh7,
};

static const unsigned eCMDecoderTable[] = {cm0, cm1, cm2, cm3, cm4,
                                           cm5, cm6, cm7, cm8};

static const unsigned mQQmDecoderTable[] = {q0, q1, q2, q3};

static const unsigned mSsDecoderTable[] = {s0, s1, s2, s3};

static const unsigned mQQXwDecoderTable[] = {qx0, qx1, qx2, qx3};

static const unsigned eYsDecoderTable[] = {y2, y3, y4, y5};

static const unsigned mQXHLbDecoderTable[] = {
    qwh0, qwl0, qwh1, qwl1, qwh2, qwl2, qwh3, qwl3,
};
