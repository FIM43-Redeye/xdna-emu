_Appendix A:_ Comparison of AIE Generations

#### _Appendix A_

## Comparison of AIE Generations

|Table 14: Comparison of AIE Generations|Col2|Col3|Col4|
||**AI Engine**|**AIE-ML**|**AIE-ML v2**|
|**Compute**|**Compute**|**Compute**|**Compute**|
|**Bandwidth to Local Memory**|**Bandwidth to Local Memory**|**Bandwidth to Local Memory**|**Bandwidth to Local Memory**|
|Load from data memory in<br>same Tile|64 B/cycle|64 B/cycle|128 B/cycle|
|Store|32 B/cycle|32 B/cycle|64 B/cycle|
|Cascade bandwidth|48 B/cycle|64 B/cycle|64 B/cycle|
|**On-chip Data Memory**|**On-chip Data Memory**|**On-chip Data Memory**|**On-chip Data Memory**|
|Per Compute Tile|32 KB<br>128 B/cycle<br>(8 x 16B banks)|64 KB<br>128 B/cycle<br>(8 x 16B banks)|64 KB<br>256 B/cycle<br>(8 x 32B banks)|
|Per Memory Tile|No memory tiles|512 KB<br>256 B/cycle<br>(16 x 16B banks)|512 KB<br>256 B/cycle<br>(16 x 32B banks)|
|Compute Tile|2 MM2S<br>2 S2MM|2 MM2S<br>2 S2MM|2 MM2S<br>2 S2MM|
|Memory Tile|No memory tiles|6 MM2S<br>6 S2MM|6 MM2S<br>6 S2MM|
|Interface Tile|2 MM2S<br>2 S2MM|2 MM2S<br>2 S2MM|2 MM2S<br>2 S2MM|

|Table 14: Comparison of AIE Generations (cont'd)|Col2|Col3|Col4|
||**AI Engine**|**AIE-ML**|**AIE-ML v2**|
|**Stream Interconnect**|**Stream Interconnect**|**Stream Interconnect**|**Stream Interconnect**|
|**Includes Control Module?**|**Includes Control Module?**|**Includes Control Module?**|**Includes Control Module?**|