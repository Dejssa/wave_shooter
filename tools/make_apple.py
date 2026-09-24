"""Build a low-poly apple as models/apple.glb.

Flat-shaded like the other models: a slightly lobed apple about 8 cm across
with dimples at the top and bottom, red skin that is darker on one side and
has a yellow-green patch near the stem, a curved brown stem and one leaf.
Origin on the ground under the apple (it rests on its base), +Y up.

Usage: python tools/make_apple.py [OUT.glb] [SEED]   (needs numpy, pygltflib)
"""
import sys
import numpy as np
from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
                       Accessor, Material, PbrMetallicRoughness, Asset)

OUT = sys.argv[1] if len(sys.argv) > 1 else 'models/apple.glb'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rng = np.random.default_rng(SEED)
V3 = lambda *a: np.array(a, dtype=np.float64)

MATS = {  # name: (linear rgb, roughness)
    'AppleRed':    ([0.42, 0.02, 0.02], 0.45),
    'AppleDark':   ([0.24, 0.012, 0.015], 0.45),
    'AppleYellow': ([0.55, 0.36, 0.04], 0.5),
    'AppleStem':   ([0.10, 0.05, 0.02], 0.8),
    'AppleLeaf':   ([0.10, 0.32, 0.04], 0.6),
}
tris = {k: [] for k in MATS}

def tri(m, a, b, c, away):
    if np.cross(b - a, c - a) @ ((a + b + c) / 3 - away) < 0: b, c = c, b
    tris[m].append((a, b, c))

# ---------------------------------------------------------------- body: a lathe profile (radius, height)
PROFILE = [(0.000, 0.009), (0.013, 0.002), (0.031, 0.011), (0.041, 0.031),
           (0.039, 0.054), (0.029, 0.071), (0.013, 0.076), (0.000, 0.067)]
N = 10
th = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 1)
rings = []
for k, (r, y) in enumerate(PROFILE):
    lobes = 1 + 0.05 * np.cos(5 * th) * (1 - y / 0.08)           # five soft lobes, strongest at the base
    wob = 1 + 0.025 * rng.standard_normal(N)
    rings.append([V3(r * l * w * np.cos(t), y, r * l * w * np.sin(t)) for t, l, w in zip(th, lobes, wob)])
sun = V3(np.cos(0.5), 0, np.sin(0.5))                            # the side that caught the sun is brighter
centre = V3(0, 0.04, 0)
for k in range(len(PROFILE) - 1):
    for i in range(N):
        j = (i + 1) % N
        a, b, c, d = rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i]
        mid = (a + b + c + d) / 4
        side = (mid - centre) @ sun / 0.04
        if PROFILE[k][1] > 0.06 and side > -0.2 and rng.random() < 0.7:
            m = 'AppleYellow'                                    # patch around the stem cavity
        elif side < -0.3 and rng.random() < 0.75:
            m = 'AppleDark'
        else:
            m = 'AppleRed'
        away = V3(0, PROFILE[k][1] + PROFILE[k + 1][1], 0) / 2
        if k in (0, len(PROFILE) - 2):                           # dimples: orient against the apple centre
            away = centre
        tri(m, a, b, c, away); tri(m, a, c, d, away)

# ---------------------------------------------------------------- stem
top = V3(0, 0.067, 0)
stem_pts = [top, top + V3(0.002, 0.012, 0.001), top + V3(0.006, 0.021, 0.002)]
for k in range(len(stem_pts) - 1):
    p, q = stem_pts[k], stem_pts[k + 1]
    r1, r2 = 0.0028 - 0.0007 * k, 0.0021 - 0.0007 * k
    ax = (q - p) / np.linalg.norm(q - p); u = np.cross(ax, V3(0, 0, 1)); u /= np.linalg.norm(u); v = np.cross(ax, u)
    ring = lambda c, r: [c + r * (np.cos(t) * u + np.sin(t) * v) for t in np.linspace(0, 2 * np.pi, 5, endpoint=False)]
    A, B = ring(p, r1), ring(q, r2)
    for i in range(5):
        j = (i + 1) % 5
        tri('AppleStem', A[i], A[j], B[j], (p + q) / 2); tri('AppleStem', A[i], B[j], B[i], (p + q) / 2)
    if k == len(stem_pts) - 2:
        for i in range(5): tri('AppleStem', B[i], B[(i + 1) % 5], q, p)

# ---------------------------------------------------------------- leaf (folded along its midrib, two-sided)
base = top + V3(0.002, 0.009, 0.0)
tip = base + V3(-0.030, 0.008, 0.014)
axis = tip - base; mid = base + axis * 0.45
side = np.cross(axis, V3(0, 1, 0)); side /= np.linalg.norm(side)
up = V3(0, 0.004, 0)
L, R = mid + side * 0.011 + up * 0.3, mid - side * 0.011 + up * 0.3
M = mid - up * 0.6                                               # the fold dips along the midrib
for a, b in ((L, M), (M, R)):
    for off in (0.0004, -0.0004):
        o = V3(0, off, 0)
        tri('AppleLeaf', base + o, a + o, b + o, mid - V3(0, off * 50, 0))
        tri('AppleLeaf', a + o, tip + o, b + o, mid - V3(0, off * 50, 0))

# ---------------------------------------------------------------- write glTF
g = GLTF2(asset=Asset(generator='tools/make_apple.py'))
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
    if not tris[name]: continue
    g.materials.append(Material(name=name, pbrMetallicRoughness=PbrMetallicRoughness(
        baseColorFactor=[*rgb, 1.0], metallicFactor=0.0, roughnessFactor=rough)))
    T = np.array(tris[name], dtype=np.float64)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); ln = np.linalg.norm(n, axis=1)
    T, n = T[ln > 1e-14], n[ln > 1e-14] / ln[ln > 1e-14, None]
    pos = T.reshape(-1, 3).astype(np.float32); nor = np.repeat(n, 3, 0).astype(np.float32)
    prims.append(Primitive(attributes=Attributes(POSITION=accessor(pos, 5126, 'VEC3', 34962, True),
                                                 NORMAL=accessor(nor, 5126, 'VEC3', 34962)),
                           indices=accessor(np.arange(len(pos), dtype=np.uint16), 5123, 'SCALAR', 34963),
                           material=len(g.materials) - 1))
    total += len(T)
g.meshes.append(Mesh(name='Apple', primitives=prims))
g.nodes.append(Node(name='Apple', mesh=0))
g.scenes.append(Scene(name='Scene', nodes=[0])); g.scene = 0
while len(blob) % 4: blob.append(0)
g.buffers.append(Buffer(byteLength=len(blob)))
g.set_binary_blob(bytes(blob)); g.save(OUT)
print(f'{OUT}: {total} triangles')
