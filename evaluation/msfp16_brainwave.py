import struct
import math



def msfp16_unpack(reg_val):

    out = [0.0] * 4
    
    
    shared_exp_bits = (reg_val >> 56) & 0xFF
    
    if shared_exp_bits == 0:
        return out 
    
    e_unb = shared_exp_bits - 127
    
    for i in range(4):
        
        lane_bits = (reg_val >> (i * 14)) & 0x3FFF
        
        s = (lane_bits >> 13) & 1
        m = lane_bits & 0x1FFF 
        
        if m == 0:
            out[i] = -0.0 if s else 0.0
            continue
            
        
        frac = m / 8192.0
        
        
        val = math.ldexp(frac, e_unb)
        out[i] = -val if s else val
        
    return out

def msfp16_pack(vals):

    decomposed = []
    e_max = -999999999 
    all_zero = True
    
    for x in vals:
        if x == 0.0:
            decomposed.append({'s': 0, 'e': -999999999, 'frac': 0.0})
        else:
            all_zero = False
            s = 1 if math.copysign(1, x) < 0 else 0
            ax = abs(x)
            
            
            m, ei = math.frexp(ax)
            
            
            frac = m * 2.0
            
            
            e = ei - 1
            
            decomposed.append({'s': s, 'e': e, 'frac': frac})
            if e > e_max:
                e_max = e

    if all_zero:
        return 0
        
    
    if e_max > 127: e_max = 127
    if e_max < -126: e_max = -126
    
    shared_exp_bits = (e_max + 127) & 0xFF
    
    
    lanes = 0
    for i in range(4):
        d = decomposed[i]
        s, e, frac = d['s'], d['e'], d['frac']
        
        lane_val = 0
        if e != -999999999:
            shift = e_max - e
            
            
            scaled = math.ldexp(frac, -shift)
            
            
            f = scaled * 8192.0
            
            
            if f < 0.0: f = 0.0
            elif f > 8191.0: f = 8191.0 
            

            mag = int(round(f))
            
            lane_val = (s << 13) | (mag & 0x1FFF)
        else:
            
            lane_val = (s << 13)
            
        lanes |= (lane_val << (i * 14))
        
    return lanes | (shared_exp_bits << 56)



def perform_alu_op(opcode, regs, dest, src1, src2, src3=None):
    
    vals_a = msfp16_unpack(regs[src1])
    vals_b = msfp16_unpack(regs[src2])
    vals_c = msfp16_unpack(regs[src3]) if src3 else [0.0]*4
    
    vals_r = [0.0] * 4
    
    for i in range(4):
        v1, v2, v3 = vals_a[i], vals_b[i], vals_c[i]
        
        if opcode == 'fadd':
            vals_r[i] = v1 + v2
        elif opcode == 'fsub':
            vals_r[i] = v1 - v2
        elif opcode == 'fmul':
            vals_r[i] = v1 * v2
        elif opcode == 'fmax':
            
            if math.isnan(v1): vals_r[i] = v2
            elif math.isnan(v2): vals_r[i] = v1
            else: vals_r[i] = max(v1, v2)
        elif opcode == 'fmadd':
            
            if hasattr(math, 'fma'):
                vals_r[i] = math.fma(v1, v2, v3)
            else:
                vals_r[i] = (v1 * v2) + v3

    
    out_64 = msfp16_pack(vals_r)
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

print(f"{'Op':<12} | {'Input 1 (Hex)':<18} | {'Input 2 (Hex)':<18} | {'Input 3 (Hex)':<18} | {'Result (Hex)':<18}")
print("-" * 92)

# FADD.MSFP16
res_add = perform_alu_op('fadd', regs, 'f1', 'f6', 'f7')
print(f"FADD_MSFP16  | {regs['f6']:016x} | {regs['f7']:016x} | {'-':<18} | {res_add:016x}")

# FSUB.MSFP16
res_sub = perform_alu_op('fsub', regs, 'f2', 'f8', 'f9')
print(f"FSUB_MSFP16  | {regs['f8']:016x} | {regs['f9']:016x} | {'-':<18} | {res_sub:016x}")

# FMUL.MSFP16
res_mul = perform_alu_op('fmul', regs, 'f3', 'f10', 'f11')
print(f"FMUL_MSFP16  | {regs['f10']:016x} | {regs['f11']:016x} | {'-':<18} | {res_mul:016x}")

# FMAX.MSFP16
res_max = perform_alu_op('fmax', regs, 'f4', 'f12', 'f13')
print(f"FMAX_MSFP16  | {regs['f12']:016x} | {regs['f13']:016x} | {'-':<18} | {res_max:016x}")

# FMADD.MSFP16
res_madd = perform_alu_op('fmadd', regs, 'f5', 'f14', 'f15', 'f16')
print(f"FMADD_MSFP16 | {regs['f14']:016x} | {regs['f15']:016x} | {regs['f16']:016x} | {res_madd:016x}")
