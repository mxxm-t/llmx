import struct

# Independent GGUF v3 spec parser (from llama.cpp gguf.h spec)
ALIGN = 32
GGML_TYPE_Q8_0 = 8

f = open("test_model.gguf", "rb")
def rd(n): return f.read(n)
def u32(): return struct.unpack("<I", rd(4))[0]
def u64(): return struct.unpack("<Q", rd(8))[0]
def f32(): return struct.unpack("<f", rd(4))[0]
def s8():  return struct.unpack("<b", rd(1))[0]
def u8():  return struct.unpack("<B", rd(1))[0]
def s16(): return struct.unpack("<h", rd(2))[0]
def u16(): return struct.unpack("<H", rd(2))[0]
def s32(): return struct.unpack("<i", rd(4))[0]
def s64(): return struct.unpack("<q", rd(8))[0]
def f64(): return struct.unpack("<d", rd(8))[0]
def string():
    n = u64()
    return rd(n).decode()

magic = u32()
ver = u32()
n_tensors = u64()
n_kv = u64()
assert magic == 0x46554747, "bad magic"
assert ver == 3, "bad version"

def read_kv_value(vt):
    if vt == 8:  # STRING
        return string()
    if vt == 4:  # UINT32
        return u32()
    if vt == 10: # UINT64
        return u64()
    raise Exception("unhandled type %d" % vt)

kv = {}
for _ in range(n_kv):
    k = string()
    vt = u32()
    kv[k] = read_kv_value(vt)

def align():
    pos = f.tell()
    pad = (ALIGN - (pos % ALIGN)) % ALIGN
    f.seek(pos + pad)

align()
tensors = []
for _ in range(n_tensors):
    name = string()
    nd = u32()
    dims = [u64() for _ in range(nd)]
    ttype = u32()
    off = u64()
    align()
    tensors.append((name, dims, ttype, off))

align()
data_start = f.tell()
print("header OK: %d tensors, %d kv" % (n_tensors, n_kv))
print("kv:", kv)
print("data_start:", data_start)

# verify each tensor offset points at its data and data is aligned
ok = True
for name, dims, ttype, off in tensors:
    pos = data_start + off
    assert pos % ALIGN == 0, "%s data not aligned at %d" % (name, pos)
    n = 1
    for d in dims: n *= d
    assert n % 32 == 0, "%s not divisible by 32" % name
    assert ttype == GGML_TYPE_Q8_0
    f.seek(pos)
    d = u16()  # f16 scale bytes
    # dequantize block 0
    qs = struct.unpack("<32b", rd(32))
    print("tensor %-22s shape=%s off=%6d aligned" % (name, dims, off), "| scale=%.4f q0=%d" % (d, qs[0]))
print("ALL TENSOR OFFSETS VALID")
