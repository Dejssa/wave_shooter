"""Build a simple low-poly round-point shovel as models/shovel.glb.

Flat-shaded like human_female.glb and axe.glb. Units are metres; the origin
is the middle of the yellow shaft (where a hand carries it), the D-handle
points up +Y, the blade points down and its scoop (concave) side faces +Z.
About 1.05 m long overall.

Usage: python tools/make_shovel.py [OUT.glb]   (needs numpy, pygltflib)
"""
import sys
import numpy as np
from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
                       Accessor, Material, PbrMetallicRoughness, Asset)

OUT = sys.argv[1] if len(sys.argv) > 1 else 'models/shovel.glb'

MATS = {  # name: (rgb, metallic, roughness)
    'ShovelBlade': ([0.055, 0.058, 0.062], 0.3, 0.5),
    'ShovelShaft': ([0.93, 0.62, 0.04], 0.0, 0.55),
    'ShovelGrip':  ([0.80, 0.06, 0.04], 0.0, 0.6),
}
tris = {k: [] for k in MATS}
V3 = lambda *a: np.array(a, dtype=np.float64)

def tri(m, a, b, c, away):
    """add a triangle whose normal points away from the point `away`"""
    if np.cross(b - a, c - a) @ ((a + b + c) / 3 - away) < 0: b, c = c, b
    tris[m].append((a, b, c))

def quad(m, a, b, c, d, away):
    tri(m, a, b, c, away); tri(m, a, c, d, away)

# ---------------------------------------------------------------- round parts
def frame(axis):
    axis = axis / np.linalg.norm(axis)
    ref = V3(0, 0, 1) if abs(axis[2]) < 0.9 else V3(1, 0, 0)
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    return u, np.cross(axis, u)

def tube(m, pts, radii, n=8, caps=(True, True)):
    """rings of (rx, rz) radii around a polyline"""
    pts = [V3(*p) for p in pts]
    rings = []
    for i, (p, (r1, r2)) in enumerate(zip(pts, radii)):
        ax = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        u, v = frame(ax)
        th = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
        rings.append([p + r1 * np.cos(t) * u + r2 * np.sin(t) * v for t in th])
    for k in range(len(rings) - 1):
        mid = (pts[k] + pts[k + 1]) / 2
        for i in range(n):
            j = (i + 1) % n
            quad(m, rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i], mid)
    for on, r, p, q in ((caps[0], rings[0], pts[0], pts[1]), (caps[1], rings[-1], pts[-1], pts[-2])):
        if on:
            c = sum(r) / n
            for i in range(n):
                tri(m, c, r[i], r[(i + 1) % n], q)

# socket around the bottom of the shaft, flattened where it meets the blade
tube('ShovelBlade', [(0, -0.262, -0.014), (0, -0.20, -0.006), (0, -0.07, 0)],
     [(0.034, 0.016), (0.026, 0.018), (0.0185, 0.0185)], caps=(False, True))
# yellow shaft
tube('ShovelShaft', [(0, -0.08, 0), (0, 0.35, 0)], [(0.016, 0.016)] * 2, caps=(False, False))
# D-handle: ferrule, two arms, red crossbar
tube('ShovelBlade', [(0, 0.34, 0), (0, 0.40, 0)], [(0.0185, 0.0185), (0.022, 0.016)])
for s in (-1, 1):
    tube('ShovelBlade', [(0, 0.395, 0), (s * 0.058, 0.512, 0)], [(0.011, 0.011)] * 2, n=6)
    tube('ShovelBlade', [(s * 0.058, 0.500, 0), (s * 0.058, 0.530, 0)], [(0.016, 0.016)] * 2, n=6)
tube('ShovelGrip', [(-0.056, 0.515, 0), (0.056, 0.515, 0)], [(0.0145, 0.0145)] * 2, caps=(False, False))

# ---------------------------------------------------------------- blade
# rows from the top edge down to the point: y, half width
ROWS = [(-0.240, 0.110), (-0.310, 0.111), (-0.380, 0.108), (-0.440, 0.088), (-0.490, 0.050)]
TIP = V3(0, -0.522, -0.004)
COLS = [-1, -0.5, 0, 0.5, 1]
DISH, THICK = 0.020, 0.004

def front(y, hw, xn, row):
    dish = DISH * (1 - xn * xn) * (1 - 0.55 * row / (len(ROWS) - 1))
    return V3(xn * hw, y, -dish)

F = [[front(y, hw, xn, r) for xn in COLS] for r, (y, hw) in enumerate(ROWS)]
B = [[p - V3(0, 0, THICK) for p in row] for row in F]
tipB = TIP - V3(0, 0, THICK)
FAR_BACK, FAR_FRONT = V3(0, -0.38, -1), V3(0, -0.38, 1)
for r in range(len(ROWS) - 1):
    for c in range(len(COLS) - 1):
        quad('ShovelBlade', F[r][c], F[r][c + 1], F[r + 1][c + 1], F[r + 1][c], FAR_BACK)
        quad('ShovelBlade', B[r][c], B[r][c + 1], B[r + 1][c + 1], B[r + 1][c], FAR_FRONT)
for c in range(len(COLS) - 1):
    tri('ShovelBlade', F[-1][c], F[-1][c + 1], TIP, FAR_BACK)
    tri('ShovelBlade', B[-1][c], B[-1][c + 1], tipB, FAR_FRONT)
# rim along the sides and point
BLADE_MID = V3(0, -0.36, -0.01)
edge = [(F[r][0], B[r][0]) for r in range(len(ROWS))] + [(TIP, tipB)] + \
       [(F[r][-1], B[r][-1]) for r in reversed(range(len(ROWS)))]
for (a, b), (c, d) in zip(edge[:-1], edge[1:]):
    quad('ShovelBlade', a, c, d, b, BLADE_MID)
# tread: the top edge is folded back into a flat step for the foot
for c in range(len(COLS) - 1):
    a, b = F[0][c], F[0][c + 1]
    a2, b2 = a + V3(0, 0.008, 0), b + V3(0, 0.008, 0)
    ab, bb = a2 - V3(0, 0, 0.024), b2 - V3(0, 0, 0.024)
    mid = (a + b) / 2 - V3(0, 0.004, 0.012)
    quad('ShovelBlade', a, b, b2, a2, mid)          # front lip
    quad('ShovelBlade', a2, b2, bb, ab, mid)        # step top
    quad('ShovelBlade', ab, bb, B[0][c + 1], B[0][c], mid)
for c in (0, len(COLS) - 1):                        # close the tread ends
    a = F[0][c]; a2 = a + V3(0, 0.008, 0); ab = a2 - V3(0, 0, 0.024)
    tri('ShovelBlade', a, a2, ab, V3(0, a[1], a[2]))
    tri('ShovelBlade', a, ab, B[0][c], V3(0, a[1], a[2]))

# spine: the socket runs on down the scoop side as a ridge that fades out near the point
ys = np.linspace(-0.25, -0.47, 5)
cz = np.interp(ys, [r[0] for r in ROWS[::-1]], [F[r][2][2] for r in reversed(range(len(ROWS)))])
wid = np.interp(ys, [-0.47, -0.25], [0.001, 0.013])
hgt = np.interp(ys, [-0.47, -0.25], [0.0005, 0.010])
L = [V3(-w, y, z) for w, y, z in zip(wid, ys, cz)]
R = [V3(w, y, z) for w, y, z in zip(wid, ys, cz)]
C = [V3(0, y, z + h) for y, z, h in zip(ys, cz, hgt)]
for k in range(len(ys) - 1):
    under = V3(0, (ys[k] + ys[k + 1]) / 2, cz[k] - 0.02)
    quad('ShovelBlade', L[k], C[k], C[k + 1], L[k + 1], under)
    quad('ShovelBlade', C[k], R[k], R[k + 1], C[k + 1], under)

# ---------------------------------------------------------------- write glTF
g = GLTF2(asset=Asset(generator='tools/make_shovel.py'))
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
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); ln = np.linalg.norm(n, axis=1)
    T, n = T[ln > 1e-12], n[ln > 1e-12] / ln[ln > 1e-12, None]
    pos = T.reshape(-1, 3).astype(np.float32); nor = np.repeat(n, 3, 0).astype(np.float32)
    prims.append(Primitive(attributes=Attributes(POSITION=accessor(pos, 5126, 'VEC3', 34962, True),
                                                 NORMAL=accessor(nor, 5126, 'VEC3', 34962)),
                           indices=accessor(np.arange(len(pos), dtype=np.uint16), 5123, 'SCALAR', 34963),
                           material=len(g.materials) - 1))
    total += len(T)
g.meshes.append(Mesh(name='Shovel', primitives=prims))
g.nodes.append(Node(name='Shovel', mesh=0))
g.scenes.append(Scene(name='Scene', nodes=[0])); g.scene = 0
while len(blob) % 4: blob.append(0)
g.buffers.append(Buffer(byteLength=len(blob)))
g.set_binary_blob(bytes(blob))
g.save(OUT)
print(f'{OUT}: {total} triangles')
