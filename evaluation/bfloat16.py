import struct
import math

def float_as_u32(f):
    
    return struct.unpack('<I', struct.pack('<f', f))[0]

def u32_as_float(u):
    
    return struct.unpack('<f', struct.pack('<I', u))[0]

def bf16_to_fp32(bf16_val):

    return u32_as_float(bf16_val << 16)

def fp32_to_bf16(f_val):

    u = float_as_u32(f_val)
    
    # Check NaN
    if math.isnan(f_val):
        return ((u >> 16) & 0x8000) | 0x7FC0
    
    # Check Inf
    if math.isinf(f_val):
        return ((u >> 16) & 0x8000) | 0x7F80
        
    lsb = u & 0xFFFF
    

    if (lsb > 0x8000) or (lsb == 0x8000 and (u & 0x10000)):
        u += 0x10000
        
    return (u >> 16) & 0xFFFF



def unpack_reg(reg_val_64):
    
    return [(reg_val_64 >> (i * 16)) & 0xFFFF for i in range(4)]

def pack_reg(bf16_list):
    
    res = 0
    for i in range(4):
        res |= (bf16_list[i] << (i * 16))
    return res


def perform_alu_op(opcode, regs, dest, src1, src2, src3=None):
    
    vals_a = unpack_reg(regs[src1])
    vals_b = unpack_reg(regs[src2])
    vals_c = unpack_reg(regs[src3]) if src3 else [0]*4
    
    results_bf16 = []
    
    for i in range(4):
        
        f1 = bf16_to_fp32(vals_a[i])
        f2 = bf16_to_fp32(vals_b[i])
        f3 = bf16_to_fp32(vals_c[i])
        
        
        res_f32 = 0.0
        
        if opcode == 'fadd':
            res_f32 = f1 + f2
        elif opcode == 'fsub':
            res_f32 = f1 - f2
        elif opcode == 'fmul':
            res_f32 = f1 * f2
        elif opcode == 'fmax':

            res_f32 = f1 if (f1 > f2) else f2
        elif opcode == 'fmadd':

            if hasattr(math, 'fma'):
                res_f32 = math.fma(f1, f2, f3)
            else:
                res_f32 = (f1 * f2) + f3

        res_f32 = u32_as_float(float_as_u32(res_f32))

        
        results_bf16.append(fp32_to_bf16(res_f32))
        
    
    out_64 = pack_reg(results_bf16)
    regs[dest] = out_64
    return out_64

regs = {
    'f6':  0x3e00c00034005640,
    'f7':  0x38004000b400d240,
    'f8':  0x4200c4004940c810,
    'f9':  0x3c004000c100c810,
    'f10': 0x4000be002e665000,
    'f11': 0x3800c2004900b400,
    'f12': 0x4200c8807e000000,
    'f13': 0x4000c9004580bc00,
    'f14': 0x3c00c00042003400,
    'f15': 0x44003800be005640,
    'f16': 0x380049000000b400
}

print(f"{'Op':<10} | {'Input 1 (Hex)':<18} | {'Input 2 (Hex)':<18} | {'Input 3 (Hex)':<18} | {'Result (Hex)':<18}")
print("-" * 90)

# FADD.BF16 f1, f6, f7
res_add = perform_alu_op('fadd', regs, 'f1', 'f6', 'f7')
print(f"FADD       | {regs['f6']:016x} | {regs['f7']:016x} | {'-':<18} | {res_add:016x}")

# FSUB.BF16 f2, f8, f9
res_sub = perform_alu_op('fsub', regs, 'f2', 'f8', 'f9')
print(f"FSUB       | {regs['f8']:016x} | {regs['f9']:016x} | {'-':<18} | {res_sub:016x}")

# FMUL.BF16 f3, f10, f11
res_mul = perform_alu_op('fmul', regs, 'f3', 'f10', 'f11')
print(f"FMUL       | {regs['f10']:016x} | {regs['f11']:016x} | {'-':<18} | {res_mul:016x}")

# FMAX.BF16 f4, f12, f13
res_max = perform_alu_op('fmax', regs, 'f4', 'f12', 'f13')
print(f"FMAX       | {regs['f12']:016x} | {regs['f13']:016x} | {'-':<18} | {res_max:016x}")

# FMADD.BF16 f5, f14, f15, f16
res_madd = perform_alu_op('fmadd', regs, 'f5', 'f14', 'f15', 'f16')
print(f"FMADD      | {regs['f14']:016x} | {regs['f15']:016x} | {regs['f16']:016x} | {res_madd:016x}")
