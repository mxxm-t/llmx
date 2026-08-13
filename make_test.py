import json, struct, math, random, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

rng = random.Random(42)
def tensor(rows, cols):
    return [[rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]

t1 = tensor(512, 256)   # 131072 = 32*4096 ok
t2 = [rng.gauss(0, 1) for _ in range(256)]  # 256 = 32*8 ok
t3 = tensor(64, 32)     # 2048 = 32*64 ok

json_doc = {
    "name": "MyModel",
    "tensors": [
        {"name": "tok_embeddings.weight", "shape": [256, 512]},  # ne[0]=256 fastest
        {"name": "norm.weight",           "shape": [256]},
        {"name": "layer.0.weight",        "shape": [32, 64]},
    ],
}
with open("test_model.json", "w") as f:
    json.dump(json_doc, f, indent=2)

with open("test_model.bin", "wb") as f:
    for r in t1:
        f.write(struct.pack("<%df" % len(r), *r))
    f.write(struct.pack("<%df" % len(t2), *t2))
    for r in t3:
        f.write(struct.pack("<%df" % len(r), *r))

# ground truth float arrays for error measurement
flat1 = [x for r in t1 for x in r]
flat2 = t2
flat3 = [x for r in t3 for x in r]
with open("test_orig.bin", "wb") as f:
    f.write(struct.pack("<%df" % len(flat1), *flat1))
    f.write(struct.pack("<%df" % len(flat2), *flat2))
    f.write(struct.pack("<%df" % len(flat3), *flat3))

print("created test_model.json, test_model.bin, test_orig.bin")
print("tensor1 elems:", len(flat1), "tensor2:", len(flat2), "tensor3:", len(flat3))
