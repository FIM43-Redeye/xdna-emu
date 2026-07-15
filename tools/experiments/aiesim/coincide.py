#!/usr/bin/env python3
# Find timestamps where two named signals are BOTH asserted (=1) simultaneously.
# Usage: coincide.py file.vcd <sigA-substr> <sigB-substr>
import sys
vcd,subA,subB=sys.argv[1],sys.argv[2],sys.argv[3]
ida=idb=None
with open(vcd,errors='replace') as f:
    for line in f:
        if line.startswith('$var'):
            p=line.split(); vid,name=p[3],p[4]
            if subA in name and ida is None: ida=vid; na=name
            if subB in name and idb is None: idb=vid; nb=name
        elif line.startswith('#'): break
print(f"A={na if ida else 'MISS'}\nB={nb if idb else 'MISS'}")
if not(ida and idb): sys.exit()
va=vb='0'; t=0; both=0; a_on=b_on=0; win=[]
with open(vcd,errors='replace') as f:
    for line in f:
        c=line[0] if line else ''
        if c=='#':
            t=int(line[1:])
            if va=='1' and vb=='1':
                both+=1
                if len(win)<20: win.append(t)
            continue
        if c in '01':
            vid=line[1:].strip()
            if vid==ida: va=c
            elif vid==idb: vb=c
print(f"timesteps with BOTH asserted: {both}")
print("first such timestamps:", win)
