"""Build a low-poly birch tree as models/birch.glb.

Flat-shaded like the other models: a slender, gently curving white trunk with
black bark marks and a darker base, a few thin upward branches, and a tall airy
crown of faceted leaf clumps in two greens. Units are metres; the origin is
the base of the trunk, +Y up. About 7.5 m tall. A different SEED gives a
different tree.

Usage: python tools/make_birch.py [OUT.glb] [SEED]   (needs numpy, pygltflib)
"""
import sys
import numpy as np
from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
                       Accessor, Material, PbrMetallicRoughness, Asset)

OUT = sys.argv[1] if len(sys.argv) > 1 else 'models/birch.glb'
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 3
rng = np.random.default_rng(SEED)

HEIGHT = 7.5
MATS = {  # name: (rgb, roughness)
    'BirchBark':   ([0.86, 0.84, 0.78], 0.85),
    'BirchMark':   ([0.10, 0.09, 0.09], 0.9),
    'BirchLeaves': ([0.40, 0.58, 0.18], 0.8),
    'BirchLeavesLight': ([0.62, 0.72, 0.24], 0.8),
}
tris = {k: [] for k in MATS}
V3 = lambda *a: np.array(a, dtype=np.float64)

def tri(m, a, b, c, away):
    """add a triangle whose normal points away from the point `away`"""
    if np.cross(b - a, c - a) @ ((a + b + c) / 3 - away) < 0: b, c = c, b
    tris[m].append((a, b, c))

def frame(axis):
    axis = axis / np.linalg.norm(axis)
    ref = V3(0, 0, 1) if abs(axis[2]) < 0.9 else V3(1, 0, 0)
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    return u, np.cross(axis, u)

def tube(pts, radii, n, pick, tip=True):
    """tapered tube along a polyline; pick(band, side) chooses each face's material"""
    rings = []
    for i, (p, r) in enumerate(zip(pts, radii)):
        u, v = frame(pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rings.append([p + r * (np.cos(t) * u + np.sin(t) * v) for t in th])
    for k in range(len(rings) - 1):
        mid = (pts[k] + pts[k + 1]) / 2
        for i in range(n):
            j = (i + 1) % n; m = pick(k, i)
            a, b, c, d = rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i]
            tri(m, a, b, c, mid); tri(m, a, c, d, mid)
    if tip:                                    # close the end with a point
        end = pts[-1] + (pts[-1] - pts[-2]) * 0.25
        for i in range(n):
            tri(pick(len(rings) - 1, i), rings[-1][i], rings[-1][(i + 1) % n], end, pts[-1])

def bark_marks(bands, sides, dark_bands=(), base_dark=0, rate=0.5):
    """white bark; faces in the thin `dark_bands` become short black dashes"""
    marks = np.zeros((bands, sides), bool)
    for k in dark_bands:
        start = rng.integers(0, sides); run = rng.integers(1, sides // 2 + 2)
        marks[k, [(start + i) % sides for i in range(run)]] = rng.random(run) < 0.9
    marks[:base_dark] |= rng.random((base_dark, sides)) < 0.75
    return lambda k, i: 'BirchMark' if marks[min(k, bands - 1), i] else 'BirchBark'

def with_dashes(levels, count, width):
    """add thin bands (pairs of close rings) between regular ring levels"""
    out = list(levels); thin = []
    for y in np.sort(rng.uniform(levels[1], levels[-3], count)):
        out += [y, y + width]
    out = np.array(sorted(out))
    thin = [k for k in range(len(out) - 1) if out[k + 1] - out[k] <= width * 1.01]
    return out, thin

# ---------------------------------------------------------------- trunk
lean = rng.normal(0, 0.18, 2)
def spine(s):
    return V3(0.10 * np.sin(2.4 * s + 1) * s + lean[0] * s * s, HEIGHT * s * 0.92,
              0.08 * np.sin(1.7 * s + 2) * s + lean[1] * s * s)
levels, thin = with_dashes(np.linspace(0, 1, 13), 12, 0.014)
trunk = [spine(s) for s in levels]
trunk_r = 0.16 * (1 - 0.8 * levels) + 0.06 * np.exp(-levels * 30)   # root flare at the base
tube(trunk, trunk_r, 7, bark_marks(len(levels) - 1, 7, thin, base_dark=1))
def trunk_at(s):
    return spine(s), 0.16 * (1 - 0.8 * s)

# ---------------------------------------------------------------- branches
crown = []                                  # (centre, radius, stretch) of leaf clumps
ang = rng.uniform(0, 2 * np.pi)
for k, h in enumerate(np.sort(rng.uniform(0.40, 0.88, 9))):
    base, r0 = trunk_at(h)
    ang += rng.uniform(1.9, 2.9)
    out = V3(np.cos(ang), 0, np.sin(ang))
    length = rng.uniform(0.9, 1.4) * (1.15 - h * 0.55)
    up = rng.uniform(0.8, 1.3)
    droop = rng.uniform(0.1, 0.35)
    pts = [base, base + (out * 0.5 + V3(0, up * 0.5, 0)) * length * 0.5,
           base + (out * 0.9 + V3(0, up * 0.55, 0)) * length,
           base + (out * 1.15 + V3(0, up * 0.55 - droop, 0)) * length]   # tips hang a little
    tube(pts, [r0 * 0.4, r0 * 0.28, r0 * 0.16, r0 * 0.08], 5, bark_marks(3, 5), tip=True)
    crown.append((pts[3] + V3(0, -0.1, 0), rng.uniform(0.45, 0.6), 1.4))    # hanging tip clump
    if rng.random() < 0.7:
        crown.append((pts[2] + V3(0, 0.3, 0), rng.uniform(0.4, 0.55), 1.25))
# a few clumps hugging the upper trunk and the leader
for s in (0.70, 0.82, 0.93):
    p, _ = trunk_at(s); a = rng.uniform(0, 2 * np.pi)
    crown.append((p + V3(np.cos(a) * 0.3, 0.1, np.sin(a) * 0.3), 0.45, 1.35))
crown.append((trunk[-1] + V3(0, 0.25, 0), 0.4, 1.6))

# ---------------------------------------------------------------- leaf clumps
PHI = (1 + 5 ** 0.5) / 2
ICO_V = np.array([(-1, PHI, 0), (1, PHI, 0), (-1, -PHI, 0), (1, -PHI, 0), (0, -1, PHI), (0, 1, PHI),
                  (0, -1, -PHI), (0, 1, -PHI), (PHI, 0, -1), (PHI, 0, 1), (-PHI, 0, -1), (-PHI, 0, 1)], float)
ICO_V /= np.linalg.norm(ICO_V, axis=1, keepdims=True)
ICO_F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
         (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]

def subdivided():
    """icosphere with one subdivision (80 faces)"""
    verts = list(ICO_V); cache = {}
    def mid(a, b):
        key = tuple(sorted((a, b)))
        if key not in cache:
            m = verts[a] + verts[b]; verts.append(m / np.linalg.norm(m)); cache[key] = len(verts) - 1
        return cache[key]
    faces = []
    for a, b, c in ICO_F:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(verts), faces

ICO2_V, ICO2_F = subdivided()
for n, (c, r, stretch) in enumerate(crown):
    V, F = (ICO2_V, ICO2_F) if r > 0.55 else (ICO_V, ICO_F)
    V = V * (1 + rng.normal(0, 0.12, (len(V), 1)))                  # lumpy
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    V = (V @ rot.T) * V3(1.0, stretch, 1.0) * r + c                  # upright, drooping ovals
    m = 'BirchLeavesLight' if rng.random() < 0.4 else 'BirchLeaves'
    for a, b, cc in F:
        tri(m, V[a], V[b], V[cc], c)

# ---------------------------------------------------------------- write glTF
g = GLTF2(asset=Asset(generator='tools/make_birch.py'))
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
g.meshes.append(Mesh(name='Birch', primitives=prims))
g.nodes.append(Node(name='Birch', mesh=0))
g.scenes.append(Scene(name='Scene', nodes=[0])); g.scene = 0
while len(blob) % 4: blob.append(0)
g.buffers.append(Buffer(byteLength=len(blob)))
g.set_binary_blob(bytes(blob))
g.save(OUT)
print(f'{OUT}: {total} triangles, {len(crown)} leaf clumps')
