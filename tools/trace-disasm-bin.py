#!/usr/bin/env python3
"""trace-disasm-bin.py -- disassemble a raw trace_raw.bin (uint32 LE words) into
mode-0 trace commands per tile (Single/Multiple/Repeat/Start). Groups packets by
header (col,row,pkt_type) and dumps the memtile (pkt_type=3) command stream.
Byte-level inspection aid for held-level encoding fidelity (#140/#141)."""
import sys, numpy as np
def load_tile_bytes(path, want_pkt_type=3):
    raw = np.fromfile(path, dtype=np.uint32)
    # trim leading/trailing zeros & 0xffffffff sentinel padding
    data=bytearray()
    i=0; n=len(raw)
    # walk packets of 8 words; resync on plausible header (parity/col/row sane)
    streams={}
    while i+8<=n:
        hdr=int(raw[i])
        if hdr in (0,0xFFFFFFFF): i+=1; continue
        pkt_id=hdr&0x1F; ptype=(hdr>>12)&0x3; row=(hdr>>16)&0x1F; col=(hdr>>21)&0x7F
        if col>10 or row>6:  # not a header, skip
            i+=1; continue
        key=(col,row,ptype)
        buf=streams.setdefault(key,bytearray())
        for w in raw[i+1:i+8]:
            v=int(w); buf += bytes([(v>>24)&0xFF,(v>>16)&0xFF,(v>>8)&0xFF,v&0xFF])
        i+=8
    return streams
def disasm(data):
    i=0; cmds=[]
    def slots(m): return [b for b in range(8) if m&(1<<b)]
    while i<len(data):
        b=data[i]
        if b==0xFE: i+=1; continue
        if b==0xF0 or b==0xF1: cmds.append(("Start",)); i+=8; continue
        if (b&0x80)==0: cmds.append(("S0",b>>4&7,b&0xF)); i+=1; continue
        if (b&0xE0)==0x80: cmds.append(("S1",(b>>2)&7,((b&3)<<8)|data[i+1])); i+=2; continue
        if (b&0xE0)==0xA0: cmds.append(("S2",(b>>2)&7,((b&3)<<16)|(data[i+1]<<8)|data[i+2])); i+=3; continue
        if (b&0xF0)==0xC0: cmds.append(("M0",slots(((b&0xF)<<4)|(data[i+1]>>4)),data[i+1]&0xF)); i+=2; continue
        if (b&0xFC)==0xD0: cmds.append(("M1",slots(((b&3)<<6)|(data[i+1]>>2)),((data[i+1]&3)<<8)|data[i+2])); i+=3; continue
        if (b&0xFC)==0xD4: cmds.append(("M2",slots(((b&3)<<6)|(data[i+1]>>2)),((data[i+1]&3)<<16)|(data[i+2]<<8)|data[i+3])); i+=4; continue
        if (b&0xFC)==0xD8: cmds.append(("R1",((b&3)<<8)|data[i+1])); i+=2; continue
        if (b&0xF0)==0xE0: cmds.append(("R0",b&0xF)); i+=1; continue
        cmds.append(("?",hex(b))); i+=1
    return cmds
path=sys.argv[1]
streams=load_tile_bytes(path)
# find memtile (pkt_type 3)
for key in sorted(streams):
    if key[2]==3:
        cmds=disasm(streams[key])
        print(f"=== {path.split('/')[-3] if '/' in path else path}  memtile{key[:2]} pkt3: {len(cmds)} cmds ===")
        for n,c in enumerate(cmds[:48]): print(f"[{n}] {c}")
        break
