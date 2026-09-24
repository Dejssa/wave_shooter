"""Build a low-poly pine tree as models/pine.glb.

Flat-shaded like the other models: a straight reddish-brown trunk with some
bare trunk at the base, and stacked cone tiers of dark green foliage. Each
tier has a jagged, drooping rim (needle clusters), a closed underside, and a
small random twist and tilt. Units are metres; the origin is the base of the
trunk, +Y up. About 9 m tall. A different SEED gives a different tree.

Usage: python tools/make_pine.py [OUT.glb] [SEED]   (needs numpy, pygltflib)
"""
import sys
import numpy as np
from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
                       Accessor, Material, PbrMetallicRoughness, Asset)

OUT = sys.argv[1] if len(sys.argv) > 1 else 'models/pine.glb'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 2
rng = np.random.default_rng(SEED)

HEIGHT = 9.0
TIERS = 6
MATS = {  # name: (rgb, roughness)
    'PineBark':   ([0.36, 0.21, 0.13], 0.9),
    'PineNeedles': ([0.13, 0.30, 0.16], 0.85),
    'PineNeedlesLight': ([0.19, 0.38, 0.19], 0.85),
    'PineNeedlesShade': ([0.08, 0.20, 0.12], 0.9),
}
tris = {k: [] for k in MATS}
V3 = lambda *a: np.array(a, dtype=np.float64)

def tri(m, a, b, c, away):
    """add a triangle whose normal points away from the point `away`"""
    if np.cross(b - a, c - a) @ ((a + b + c) / 3 - away) < 0: b, c = c, b
    tris[m].append((a, b, c))

# ---------------------------------------------------------------- trunk
lean = rng.normal(0, 0.05, 2)
def spine(s):
    return V3(lean[0] * s * HEIGHT * 0.1, s * HEIGHT, lean[1] * s * HEIGHT * 0.1)
ys = [0, 0.04, 0.2, 0.55, 0.9]
radii = [0.30, 0.22, 0.19, 0.12, 0.04]
n = 6
rings = []
for s, r in zip(ys, radii):
    c = spine(s); th = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 1)
    rings.append([c + V3(r * np.cos(t), 0, r * np.sin(t)) for t in th])
for k in range(len(rings) - 1):
    mid = (spine(ys[k]) + spine(ys[k + 1])) / 2
    for i in range(n):
        j = (i + 1) % n
        tri('PineBark', rings[k][i], rings[k][j], rings[k + 1][j], mid)
        tri('PineBark', rings[k][i], rings[k + 1][j], rings[k + 1][i], mid)

# ---------------------------------------------------------------- foliage tiers
def rot(ax, ang):
    ax = ax / np.linalg.norm(ax); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K

BOTTOM = 0.17                                  # bare trunk below this fraction of the height
for k in range(TIERS):
    f = k / (TIERS - 1)                        # 0 bottom tier .. 1 top tier
    base_y = BOTTOM + (0.86 - BOTTOM) * (f ** 0.9)
    radius = 2.0 * (1 - f) ** 0.85 + 0.45
    height = 1.9 - 0.6 * f + (0.6 if k == TIERS - 1 else 0)
    apex = spine(base_y) + V3(0, height, 0)
    centre = spine(base_y)
    points = 8 if radius > 1.0 else 7
    th = np.linspace(0, 2 * np.pi, 2 * points, endpoint=False) + rng.uniform(0, 2 * np.pi)
    rim = []
    for i, t in enumerate(th):
        tip = i % 2 == 0                        # alternate: needle-cluster tips and notches
        r = radius * (1.0 if tip else 0.72) * rng.uniform(0.9, 1.08)
        droop = (-0.28 if tip else 0.05) * radius * 0.5 + rng.normal(0, 0.04)
        rim.append(centre + V3(r * np.cos(t), droop, r * np.sin(t)))
    inner = centre + V3(0, radius * 0.18, 0)   # underside rises back up to the trunk
    R = rot(V3(rng.normal(), 0, rng.normal()), rng.normal(0, 0.05))
    place = lambda p: centre + R @ (p - centre)
    apex, inner = place(apex), place(inner); rim = [place(p) for p in rim]
    shade_mat = 'PineNeedlesShade'
    for i in range(len(rim)):
        a, b = rim[i], rim[(i + 1) % len(rim)]
        m = 'PineNeedlesLight' if rng.random() < 0.3 else 'PineNeedles'
        tri(m, a, b, apex, (centre + apex) / 2 - V3(0, 0.3, 0))
        tri(shade_mat, a, b, inner, inner + V3(0, 1.0, 0))

# ---------------------------------------------------------------- write glTF
g = GLTF2(asset=Asset(generator='tools/make_pine.py'))
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
for name, (rgb, rough) in MATS.items():
    g.materials.append(Material(name=name, pbrMetallicRoughness=PbrMetallicRoughness(
        baseColorFactor=[*rgb, 1.0], metallicFactor=0.0, roughnessFactor=rough)))
    T = np.array(tris[name], dtype=np.float64)
    nn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); ln = np.linalg.norm(nn, axis=1)
    T, nn = T[ln > 1e-12], nn[ln > 1e-12] / ln[ln > 1e-12, None]
    pos = T.reshape(-1, 3).astype(np.float32); nor = np.repeat(nn, 3, 0).astype(np.float32)
    prims.append(Primitive(attributes=Attributes(POSITION=accessor(pos, 5126, 'VEC3', 34962, True),
                                                 NORMAL=accessor(nor, 5126, 'VEC3', 34962)),
                           indices=accessor(np.arange(len(pos), dtype=np.uint16), 5123, 'SCALAR', 34963),
                           material=len(g.materials) - 1))
    total += len(T)
g.meshes.append(Mesh(name='Pine', primitives=prims))
g.nodes.append(Node(name='Pine', mesh=0))
g.scenes.append(Scene(name='Scene', nodes=[0])); g.scene = 0
while len(blob) % 4: blob.append(0)
g.buffers.append(Buffer(byteLength=len(blob)))
g.set_binary_blob(bytes(blob))
g.save(OUT)
print(f'{OUT}: {total} triangles')
