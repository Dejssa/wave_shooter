"""Build a simple low-poly pickaxe (pick mattock) as models/pickaxe.glb.

Flat-shaded like axe.glb and shovel.glb: wooden handle with darker grain
streaks, black steel head with a collar around the handle top, a curved
pointed pick on one side and a flat chisel blade on the other, bright tips.
Units are metres; the origin is the grip near the butt, the handle points up
+Y, the pick points +Z and the blade -Z. About 0.64 m long with a 0.5 m head.

Usage: python tools/make_pickaxe.py [OUT.glb]   (needs numpy, pygltflib)
"""
import sys
import numpy as np
from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
                       Accessor, Material, PbrMetallicRoughness, Asset)

OUT = sys.argv[1] if len(sys.argv) > 1 else 'models/pickaxe.glb'
rng = np.random.default_rng(4)

MATS = {  # name: (rgb, metallic, roughness)
    'PickSteel':    ([0.055, 0.058, 0.062], 0.3, 0.5),
    'PickEdge':     ([0.75, 0.76, 0.78], 0.2, 0.35),
    'PickWood':     ([0.64, 0.42, 0.25], 0.0, 0.7),
    'PickWoodDark': ([0.42, 0.25, 0.14], 0.0, 0.75),
}
tris = {k: [] for k in MATS}
V3 = lambda *a: np.array(a, dtype=np.float64)

def tri(m, a, b, c, away):
    """add a triangle whose normal points away from the point `away`"""
    if np.cross(b - a, c - a) @ ((a + b + c) / 3 - away) < 0: b, c = c, b
    tris[m].append((a, b, c))

def loft(rings, centres, pick, cap0=True, cap1=True):
    """join rings of points; pick(band, side) chooses each face's material"""
    n = len(rings[0])
    for k in range(len(rings) - 1):
        mid = (centres[k] + centres[k + 1]) / 2
        for i in range(n):
            j = (i + 1) % n; m = pick(k, i)
            tri(m, rings[k][i], rings[k][j], rings[k + 1][j], mid)
            tri(m, rings[k][i], rings[k + 1][j], rings[k + 1][i], mid)
    for on, r, c, other in ((cap0, rings[0], centres[0], centres[1]), (cap1, rings[-1], centres[-1], centres[-2])):
        if on:
            ctr = sum(r) / n
            for i in range(n):
                tri(pick(0 if r is rings[0] else len(rings) - 2, i), ctr, r[i], r[(i + 1) % n], other)

# ---------------------------------------------------------------- handle
BUTT, TOP = -0.10, 0.54
def oval(y, rx, rz, n=8):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
    return [V3(rx * np.cos(t), y, rz * np.sin(t)) for t in th]
levels = [(BUTT, 0.015, 0.018), (BUTT + 0.012, 0.0195, 0.0225), (BUTT + 0.05, 0.0175, 0.021),
          (0.20, 0.016, 0.020), (0.40, 0.0155, 0.0195), (TOP - 0.012, 0.0155, 0.0195), (TOP, 0.012, 0.015)]
streak = rng.random((len(levels), 8)) < 0.3
streak[:, 1:] |= streak[:, :-1] & (rng.random((len(levels), 7)) < 0.3)   # streaks run along the grain
loft([oval(*l) for l in levels], [V3(0, l[0], 0) for l in levels],
     lambda k, i: 'PickWoodDark' if streak[k, i] else 'PickWood')

# ---------------------------------------------------------------- head
HY = TOP - 0.065                               # head centre height
# collar around the handle
collar = [(HY - 0.05, 0.024, 0.030), (HY - 0.04, 0.026, 0.034), (HY + 0.04, 0.026, 0.034), (HY + 0.05, 0.022, 0.028)]
def box_ring(y, hx, hz):
    return [V3(-hx, y, -hz), V3(hx, y, -hz), V3(hx, y, hz), V3(-hx, y, hz)]
loft([box_ring(*c) for c in collar], [V3(0, c[0], 0) for c in collar], lambda k, i: 'PickSteel')

def arm(direction, length, sweep, half_x, half_y, point, edge_from):
    """an arm of the head along ±Z: rectangular sections that taper and curve back toward the grip"""
    t = np.linspace(0, 1, len(half_x))
    rings, centres = [], []
    for s, hx, hy in zip(t, half_x, half_y):
        c = V3(0, HY - sweep * s ** 1.6, direction * (0.028 + length * s))
        rings.append([c + V3(-hx, -hy, 0), c + V3(hx, -hy, 0), c + V3(hx, hy, 0), c + V3(-hx, hy, 0)])
        centres.append(c)
    last = len(rings) - 1
    pick = lambda k, i: 'PickEdge' if k >= edge_from else 'PickSteel'
    if point:                                   # finish in a sharp point
        tip = centres[-1] + V3(0, -sweep * 0.08, direction * 0.03)
        loft(rings, centres, pick, cap0=False, cap1=False)
        for i in range(4):
            tri('PickEdge', rings[-1][i], rings[-1][(i + 1) % 4], tip, centres[-1])
    else:                                       # finish in a straight chisel edge
        e = centres[-1] + V3(0, -sweep * 0.05, direction * 0.02)
        ea, eb = e + V3(-half_x[-1] * 1.05, 0, 0), e + V3(half_x[-1] * 1.05, 0, 0)
        loft(rings, centres, pick, cap0=False, cap1=False)
        r = rings[-1]
        tri('PickEdge', r[0], r[1], eb, centres[-1]); tri('PickEdge', r[0], eb, ea, centres[-1])   # lower bevel
        tri('PickEdge', r[3], r[2], eb, centres[-1]); tri('PickEdge', r[3], eb, ea, centres[-1])   # upper bevel
        tri('PickEdge', r[0], ea, r[3], centres[-1]); tri('PickEdge', r[1], eb, r[2], centres[-1])  # ends

# pick: long, tapering to a point, curving back toward the grip
arm(+1, 0.235, 0.034, [0.014, 0.013, 0.011, 0.008, 0.005], [0.030, 0.024, 0.018, 0.012, 0.007], True, 4)
# chisel blade: shorter, widening and thinning to a flat edge
arm(-1, 0.20, 0.026, [0.014, 0.016, 0.019, 0.022], [0.030, 0.021, 0.013, 0.006], False, 3)

# ---------------------------------------------------------------- write glTF
g = GLTF2(asset=Asset(generator='tools/make_pickaxe.py'))
blob = bytearray()
def push(arr, target):
    while len(blob) % 4: blob.append(0)
    off = len(blob); blob.extend(arr.tobytes())
    g.bufferViews.append(BufferView(buffer=0, byteOffset=off, byteLength=arr.nbytes, target=target))
    return len(g.bufferViews) - 1
def accessor(arr, ctype, typ, target, mm=False):
    a = Accessor(bufferView=push(arr, target), componentType=ctype, count=len(arr), type=typ)
    if mm: a.min = arr.min(0).tolist(); a.max = arr.max(0).tolist()
    g.accessors.append(a); return len(g.accessors) - 1

prims = []; total = 0
for name, (rgb, met, rough) in MATS.items():
    g.materials.append(Material(name=name, pbrMetallicRoughness=PbrMetallicRoughness(
        baseColorFactor=[*rgb, 1.0], metallicFactor=met, roughnessFactor=rough)))
    T = np.array(tris[name], dtype=np.float64)
    nn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); ln = np.linalg.norm(nn, axis=1)
    T, nn = T[ln > 1e-12], nn[ln > 1e-12] / ln[ln > 1e-12, None]
    pos = T.reshape(-1, 3).astype(np.float32); nor = np.repeat(nn, 3, 0).astype(np.float32)
    prims.append(Primitive(attributes=Attributes(POSITION=accessor(pos, 5126, 'VEC3', 34962, True),
                                                 NORMAL=accessor(nor, 5126, 'VEC3', 34962)),
                           indices=accessor(np.arange(len(pos), dtype=np.uint16), 5123, 'SCALAR', 34963),
                           material=len(g.materials) - 1))
    total += len(T)
g.meshes.append(Mesh(name='Pickaxe', primitives=prims))
g.nodes.append(Node(name='Pickaxe', mesh=0))
g.scenes.append(Scene(name='Scene', nodes=[0])); g.scene = 0
while len(blob) % 4: blob.append(0)
g.buffers.append(Buffer(byteLength=len(blob)))
g.set_binary_blob(bytes(blob))
g.save(OUT)
print(f'{OUT}: {total} triangles')
