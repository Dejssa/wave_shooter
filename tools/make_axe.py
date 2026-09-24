"""Build a simple low-poly hatchet as models/axe.glb.

Flat-shaded like human_female.glb. Units are metres; the origin is the middle
of the grip (where a hand holds it), the handle points up +Y and the blade
edge faces +Z. About 0.42 m long overall.

Usage: python tools/make_axe.py [OUT.glb]   (needs numpy, pygltflib)
"""
import sys
import numpy as np
from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
                       Accessor, Material, PbrMetallicRoughness, Asset)

OUT = sys.argv[1] if len(sys.argv) > 1 else 'models/axe.glb'

MATS = {  # name: (rgb, metallic, roughness)
    'AxeSteel': ([0.30, 0.31, 0.32], 0.3, 0.5),
    'AxeEdge':  ([0.78, 0.79, 0.81], 0.2, 0.35),
    'AxeHandle': ([0.27, 0.28, 0.29], 0.1, 0.55),
    'AxeGrip':  ([0.80, 0.07, 0.03], 0.0, 0.60),
}
tris = {k: [] for k in MATS}
CENTER = [np.zeros(3)]   # faces are turned to point away from this (set per part)

def tri(m, a, b, c):
    n = np.cross(b - a, c - a)
    if n @ ((a + b + c) / 3 - CENTER[0]) < 0: b, c = c, b
    tris[m].append((a, b, c))

def quad(m, a, b, c, d):
    tri(m, a, b, c); tri(m, a, c, d)

# ---------------------------------------------------------------- handle
def ring(y, rx, rz, n=8, dz=0.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
    return [np.array([rx * np.sin(t), y, dz + rz * np.cos(t)]) for t in th]

def loft(m, rings):
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(len(a)):
            j = (i + 1) % len(a)
            CENTER[0] = np.array([0, (a[i][1] + b[i][1]) / 2, 0])
            quad(m, a[i], a[j], b[j], b[i])

def cap(m, r, up):
    c = sum(r) / len(r)
    CENTER[0] = c - np.array([0, 1 if up else -1, 0])
    for i in range(len(r)):
        tri(m, c, r[i], r[(i + 1) % len(r)])

grip = [ring(-0.078, 0.014, 0.022, dz=0.004),   # butt, flared toward the blade side
        ring(-0.064, 0.0165, 0.026, dz=0.004),
        ring(-0.040, 0.0145, 0.021),
        ring(0.080, 0.0135, 0.019),
        ring(0.125, 0.0135, 0.019)]
loft('AxeGrip', grip)
cap('AxeGrip', grip[0], up=False)
shaft = [ring(0.125, 0.0100, 0.0150),
         ring(0.290, 0.0095, 0.0150),
         ring(0.352, 0.0095, 0.0150)]
# red collar where the grip ends and the bare shaft starts
CENTER[0] = np.array([0, 0.0, 0])
for i in range(8):
    j = (i + 1) % 8
    quad('AxeGrip', grip[-1][i], grip[-1][j], shaft[0][j], shaft[0][i])
loft('AxeHandle', shaft)
cap('AxeHandle', shaft[-1], up=True)

# ---------------------------------------------------------------- head
# poll (back of the head, around the handle)
def box(m, x0, x1, y0, y1, z0, z1, taper=0.0):
    p = lambda x, y, z: np.array([x, y, z])
    t = taper
    v = [p(x0, y0, z0), p(x1, y0, z0), p(x1, y0, z1), p(x0, y0, z1),
         p(x0 + t, y1, z0), p(x1 - t, y1, z0), p(x1 - t, y1, z1), p(x0 + t, y1, z1)]
    for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        quad(m, *[v[k] for k in f])
CENTER[0] = np.array([0, 0.312, 0.003])
box('AxeSteel', -0.015, 0.015, 0.278, 0.347, -0.030, 0.036)

# blade: columns from the neck to the cutting edge, 5 rows top -> bottom (z, y)
cols = [
    [(0.034, 0.345), (0.034, 0.329), (0.034, 0.312), (0.034, 0.296), (0.034, 0.280)],
    [(0.072, 0.347), (0.072, 0.326), (0.072, 0.304), (0.072, 0.283), (0.072, 0.262)],
    [(0.112, 0.355), (0.120, 0.322), (0.122, 0.287), (0.117, 0.253), (0.104, 0.227)],  # bevel line
    [(0.143, 0.367), (0.153, 0.326), (0.156, 0.286), (0.151, 0.246), (0.137, 0.212)],  # edge
]
thick = [0.026, 0.016, 0.007, 0.0012]
mat_between = ['AxeSteel', 'AxeSteel', 'AxeEdge']
CENTER[0] = np.array([0, 0.29, 0.07])
P = [[(np.array([+thick[c] / 2, y, z]), np.array([-thick[c] / 2, y, z])) for z, y in col]
     for c, col in enumerate(cols)]
for c in range(3):
    m = mat_between[c]
    for r in range(4):
        a, b, cc, d = P[c][r], P[c + 1][r], P[c + 1][r + 1], P[c][r + 1]
        quad(m, a[0], d[0], cc[0], b[0])        # +x cheek
        quad(m, a[1], b[1], cc[1], d[1])        # -x cheek
    # top and bottom rims
    for r, flip in ((0, False), (4, True)):
        a, b = P[c][r], P[c + 1][r]
        if flip: quad(m, a[0], b[0], b[1], a[1])
        else:    quad(m, a[0], a[1], b[1], b[0])
# thin face along the cutting edge
for r in range(4):
    a, b = P[3][r], P[3][r + 1]
    quad('AxeEdge', a[0], b[0], b[1], a[1])

# ---------------------------------------------------------------- write glTF
g = GLTF2(asset=Asset(generator='tools/make_axe.py'))
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
    V = np.array(tris[name], dtype=np.float64)
    n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]); ln = np.linalg.norm(n, axis=1)
    V, n = V[ln > 1e-12], n[ln > 1e-12] / ln[ln > 1e-12, None]
    pos = V.reshape(-1, 3).astype(np.float32); nor = np.repeat(n, 3, 0).astype(np.float32)
    prims.append(Primitive(attributes=Attributes(POSITION=accessor(pos, 5126, 'VEC3', 34962, True),
                                                 NORMAL=accessor(nor, 5126, 'VEC3', 34962)),
                           indices=accessor(np.arange(len(pos), dtype=np.uint16), 5123, 'SCALAR', 34963),
                           material=len(g.materials) - 1))
    total += len(V)
g.meshes.append(Mesh(name='Axe', primitives=prims))
g.nodes.append(Node(name='Axe', mesh=0))
g.scenes.append(Scene(name='Scene', nodes=[0])); g.scene = 0
while len(blob) % 4: blob.append(0)
g.buffers.append(Buffer(byteLength=len(blob)))
g.set_binary_blob(bytes(blob))
g.save(OUT)
print(f'{OUT}: {total} triangles')
