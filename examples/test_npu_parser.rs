//! Test NPU parser byte-by-byte

use std::io::{Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};

fn main() {
    let data = std::fs::read("/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/insts.bin").unwrap();
    println!("File size: {} bytes", data.len());

    let mut cursor = Cursor::new(&data[..]);

    // Read header
    let magic = cursor.read_u32::<LittleEndian>().unwrap();
    let flags = cursor.read_u32::<LittleEndian>().unwrap();
    let num_ops = cursor.read_u32::<LittleEndian>().unwrap();
    let total_size = cursor.read_u32::<LittleEndian>().unwrap();

    println!("Magic: 0x{:08X}", magic);
    println!("Flags: 0x{:08X}", flags);
    println!("Num ops: {}", num_ops);
    println!("Total size: {}", total_size);

    // Parse each instruction manually, tracking positions
    for i in 0..num_ops {
        let start = cursor.position();
        let opcode = cursor.read_u8().unwrap();
        let _p1 = cursor.read_u24::<LittleEndian>().unwrap();

        print!("\n[{}] at 0x{:02X}: opcode=0x{:02X} ", i, start, opcode);

        if opcode >= 128 {
            // Custom op: 4-byte header + 4-byte size + payload
            let size = cursor.read_u32::<LittleEndian>().unwrap();
            print!("(custom, size={}) ", size);
            let remaining = size.saturating_sub(8) as usize;
            let mut buf = vec![0u8; remaining];
            cursor.read_exact(&mut buf).ok();
            println!("end=0x{:02X}", cursor.position());
        } else {
            // Standard op - read padding2
            let padding2 = cursor.read_u32::<LittleEndian>().unwrap();

            match opcode {
                0 => {
                    // Write32: 8 header + 8 reg_off (u64) + 4 value + 4 size = 24 bytes
                    let reg_off = cursor.read_u64::<LittleEndian>().unwrap();
                    let value = cursor.read_u32::<LittleEndian>().unwrap();
                    let _size = cursor.read_u32::<LittleEndian>().unwrap();
                    println!("Write32 reg=0x{:08X} val=0x{:08X} end=0x{:02X}",
                        reg_off as u32, value, cursor.position());
                }
                1 => {
                    // BlockWrite: 8 header + 4 reg_off + 4 size + payload
                    let reg_off = cursor.read_u32::<LittleEndian>().unwrap();
                    let size = cursor.read_u32::<LittleEndian>().unwrap();
                    let header_size = 16u32;
                    let payload_bytes = size.saturating_sub(header_size);
                    let num_words = payload_bytes / 4;
                    for _ in 0..num_words {
                        cursor.read_u32::<LittleEndian>().unwrap();
                    }
                    println!("BlockWrite reg=0x{:08X} size={} words={} end=0x{:02X}",
                        reg_off, size, num_words, cursor.position());
                }
                3 => {
                    // MaskWrite: 8 header + 8 reg_off + 4 value + 4 mask + 4 size = 28 bytes
                    let reg_off = cursor.read_u64::<LittleEndian>().unwrap();
                    let value = cursor.read_u32::<LittleEndian>().unwrap();
                    let mask = cursor.read_u32::<LittleEndian>().unwrap();
                    let _size = cursor.read_u32::<LittleEndian>().unwrap();
                    println!("MaskWrite reg=0x{:08X} val=0x{:08X} mask=0x{:08X} end=0x{:02X}",
                        reg_off as u32, value, mask, cursor.position());
                }
                4 => {
                    // MaskPoll: 8 header + 8 reg_off + 4 value + 4 mask + 4 size = 28 bytes
                    let reg_off = cursor.read_u64::<LittleEndian>().unwrap();
                    let value = cursor.read_u32::<LittleEndian>().unwrap();
                    let mask = cursor.read_u32::<LittleEndian>().unwrap();
                    let _size = cursor.read_u32::<LittleEndian>().unwrap();
                    println!("MaskPoll reg=0x{:08X} val=0x{:08X} mask=0x{:08X} end=0x{:02X}",
                        reg_off as u32, value, mask, cursor.position());
                }
                _ => {
                    println!("Unknown (pad2=0x{:08X})", padding2);
                    // Skip rest based on common sizes
                    let mut buf = [0u8; 16];
                    cursor.read_exact(&mut buf).ok();
                }
            }
        }
    }
    println!("\nFinal position: 0x{:X}", cursor.position());
}
