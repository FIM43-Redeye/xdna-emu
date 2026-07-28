#!/usr/bin/env python3
# Cycles where BOTH signals are HIGH (full asserted window, not just edges).
# Handles scalar 1/0 and 1-bit vector b1/b0. Usage: file.vcd subA subB P PH
import sys
vcd,subA,subB=sys.argv[1],sys.argv[2],sys.argv[3]; P=int(sys.argv[4]); PH=int(sys.argv[5])
ida=idb=None; na=nb=''
with open(vcd,errors='replace') as f:
    for line in f:
        if line.startswith('$var'):
            p=line.split(); vid,name=p[3],p[4]
            if subA in name and ida is None: ida=vid; na=name
            if subB in name and idb is None: idb=vid; nb=name
        elif line.startswith('#'): break
def parse(line):
    c=line[0]
    if c in '01': return (c!='0'), line[1:].strip()
    if c in 'bB':
        m=line.split()
        if len(m)==2: return (set(m[0][1:])!={'0'}), m[1]
    return None,None
# collect (time,val) edges per signal
edges={ida:[], idb:[]}
with open(vcd,errors='replace') as f:
    t=0
    for line in f:
        if not line: continue
        if line[0]=='#': t=int(line[1:]); continue
        v,vid=parse(line)
        if vid in edges and v is not None: edges[vid].append((t,v))
def high_cycles(ev,tmax):
    cyc=set(); cur=False; last=0
    for (t,v) in ev:
        if cur:  # was high from last..t
            for c in range((last-PH)//P,(t-PH)//P+1): cyc.add(c)
        cur=v; last=t
    if cur:
        for c in range((last-PH)//P,(tmax-PH)//P+1): cyc.add(c)
    return cyc
tmax=max([e[-1][0] for e in edges.values() if e]+[0])
A=high_cycles(edges[ida],tmax); B=high_cycles(edges[idb],tmax)
inter=A&B
print(f"A={na.split('tile_7_3.')[-1]}({len(A)}) B={nb.split('tile_7_3.')[-1]}({len(B)}) BOTH={len(inter)} sample:{sorted(inter)[:15]}")
