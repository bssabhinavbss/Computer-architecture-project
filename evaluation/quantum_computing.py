import math
import struct



QALU_SCALE = 1 << 29
QALU_SCALE_INV = 1.0 / QALU_SCALE
QALU_MASK = 0x3FFFFFFF
QALU_MAX_VAL = (1 << 29) - 1
QALU_MIN_VAL = -(1 << 29)
SQRT_2_INV = 0.7071067811865476

def cpp_round(n):
    
    if n >= 0: return int(n + 0.5)
    else: return int(n - 0.5)

def double_to_fixed(d):
    scaled = d * QALU_SCALE
    if scaled > QALU_MAX_VAL: scaled = QALU_MAX_VAL
    elif scaled < QALU_MIN_VAL: scaled = QALU_MIN_VAL
    return cpp_round(scaled)

def fixed_to_double(fixed):
    if fixed & (1 << 29): fixed = fixed - (1 << 30)
    return fixed * QALU_SCALE_INV



def pack_amplitude(tag, real, imag):
    fixed_r = double_to_fixed(real)
    fixed_i = double_to_fixed(imag)
    tag_bits = (tag & 0xF) << 60
    real_bits = (fixed_r & QALU_MASK) << 30
    imag_bits = (fixed_i & QALU_MASK)
    return tag_bits | real_bits | imag_bits

def unpack_amplitude(reg_val):
    tag = (reg_val >> 60) & 0xF
    fixed_r = (reg_val >> 30) & QALU_MASK
    real = fixed_to_double(fixed_r)
    fixed_i = reg_val & QALU_MASK
    imag = fixed_to_double(fixed_i)
    return tag, real, imag

def get_norm_squared(r, i):
    return r*r + i*i


def run_instruction(opcode, val_a, val_b):
    tag_a, ra, ia = unpack_amplitude(val_a)
    tag_b, rb, ib = unpack_amplitude(val_b)
    
    
    tag_out = tag_b if val_b != 0 else tag_a
    
    if opcode == 'kQAlloc_A':
        return pack_amplitude(tag_out, ra, ia)
    elif opcode == 'kQAlloc_B':
        return pack_amplitude(tag_out, ra, ia)
    elif opcode == 'kQHA':
        res_r = (ra + rb) * SQRT_2_INV
        res_i = (ia + ib) * SQRT_2_INV
        return pack_amplitude(tag_a, res_r, res_i)
    elif opcode == 'kQHB':
        res_r = (ra - rb) * SQRT_2_INV
        res_i = (ia - ib) * SQRT_2_INV
        return pack_amplitude(tag_a, res_r, res_i)
    elif opcode == 'kQXA':
        return val_b
    elif opcode == 'kQXB':
        return val_a
    elif opcode == 'kQPhase':
        theta = ib 
        res_r = ra * math.cos(theta) - ia * math.sin(theta)
        res_i = ra * math.sin(theta) + ia * math.cos(theta)
        return pack_amplitude(tag_a, res_r, res_i)
    elif opcode == 'kQMeas':
        p0 = get_norm_squared(ra, ia)
        p1 = get_norm_squared(rb, ib)
        total = p0 + p1
        if total < 1e-9: return 0
        prob0 = p0 / total
        return 0 if prob0 > 0.5 else 1
    elif opcode == 'kQNormA':
        norm = math.sqrt(get_norm_squared(ra, ia) + get_norm_squared(rb, ib))
        if norm < 1e-9: return val_a
        return pack_amplitude(tag_a, ra/norm, ia/norm)
    elif opcode == 'kQNormB':
        norm = math.sqrt(get_norm_squared(ra, ia) + get_norm_squared(rb, ib))
        if norm < 1e-9: return val_b
        return pack_amplitude(tag_b, rb/norm, ib/norm)


single_test_cases = [
    
    ('kQAlloc_A', 0, 0.5, 0.5, 2, 0.0, 0.0), 
    
    
    ('kQAlloc_B', 0, 0.9, -0.1, 3, 0.0, 0.0),

    
    ('kQHA', 0, 0.5, 0.0, 0, 0.5, 0.0), 

    
    ('kQHB', 0, 0.5, 0.0, 0, 0.5, 0.0), 

    
    ('kQXA', 0, 0.1, 0.1, 2, 0.9, 0.9),

    
    ('kQXB', 2, 0.9, 0.9, 0, 0.1, 0.1),

    
    ('kQPhase', 0, 1.0, 0.0, 0, 0.0, 1.570796327), 

    
    ('kQMeas', 0, 1.0, 0.0, 0, 0.0, 0.0), 

    
    ('kQNormA', 0, 0.3, 0.0, 0, 0.4, 0.0), 

    
    ('kQNormB', 0, 0.3, 0.0, 0, 0.4, 0.0), 
]

print(f"{'Op':<10} | {'Input A (Hex)':<18} | {'Input B (Hex)':<18} | {'Result (Hex/Int)':<18}")
print("-" * 75)

for op, ta, ra, ia, tb, rb, ib in single_test_cases:
    val_a = pack_amplitude(ta, ra, ia)
    val_b = pack_amplitude(tb, rb, ib)
    res = run_instruction(op, val_a, val_b)
    
    if op == 'kQMeas':
        r_str = f"{res} (Classical)"
    else:
        r_str = f"0x{res:016x}"
        
    print(f"{op:<10} | 0x{val_a:016x} | 0x{val_b:016x} | {r_str}")
