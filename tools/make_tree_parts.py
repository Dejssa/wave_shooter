"""Build stumps and felled trunks for the birch and pine trees.

  models/birch_stump.glb   models/birch_trunk.glb
  models/pine_stump.glb    models/pine_trunk.glb

Flat-shaded like birch.glb / pine.glb and using the same bark colours.
Stumps: knee-high, root flare with a few roots running into the ground, and a
slightly slanted axe-cut top that shows the wood rings. Origin on the ground
at the stump centre, +Y up.
Trunks: the felled log (3 m birch, 3.6 m pine) lying on the ground along X, tapering from the cut
base (at -X) to the cut top (at +X), ring faces on both ends and a few sawn
branch stubs. Origin on the ground under the middle of the log.

Usage: python tools/make_tree_parts.py [SEED]   (needs numpy, pygltflib)
"""
import sys
import numpy as np
from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
                       Accessor, Material, PbrMetallicRoughness, Asset)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
V3 = lambda *a: np.array(a, dtype=np.float64)

WOOD = {'WoodSap': [0.78, 0.62, 0.40], 'WoodRing': [0.66, 0.50, 0.30], 'WoodHeart': [0.55, 0.38, 0.21]}
SPECIES = {
    'birch': dict(bark='BirchBark', mark='BirchMark', colours={'BirchBark': [0.86, 0.84, 0.78], 'BirchMark': [0.10, 0.09, 0.09]},
                  sides=8, stump_r=0.17, stump_h=0.42, trunk_len=3.0, trunk_r=(0.16, 0.10), stubs=4),
    'pine':  dict(bark='PineBark', mark='PineBarkDark', colours={'PineBark': [0.36, 0.21, 0.13], 'PineBarkDark': [0.22, 0.12, 0.07]},
                  sides=8, stump_r=0.21, stump_h=0.38, trunk_len=3.6, trunk_r=(0.21, 0.12), stubs=6),
}

class Builder:
    def __init__(self): self.tris = {}
    def tri(self, m, a, b, c, away):
        if np.cross(b - a, c - a) @ ((a + b + c) / 3 - away) < 0: b, c = c, b
        self.tris.setdefault(m, []).append((a, b, c))

def frame(axis):
    axis = axis / np.linalg.norm(axis)
    ref = V3(0, 1, 0) if abs(axis[1]) < 0.9 else V3(1, 0, 0)
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    return u, np.cross(axis, u)

def rings_along(pts, radii, n, jitter, rng):
    rings = []
    wob = 1 + jitter * rng.standard_normal(n)                  # same irregular outline along the length
    for i, (p, r) in enumerate(zip(pts, radii)):
        u, v = frame(pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rings.append([p + r * w * (np.cos(t) * u + np.sin(t) * v) for t, w in zip(th, wob)])
    return rings

def bark(b, rings, pts, pick):
    n = len(rings[0])
    for k in range(len(rings) - 1):
        mid = (pts[k] + pts[k + 1]) / 2
        for i in range(n):
            j = (i + 1) % n; m = pick(k, i)
            b.tri(m, rings[k][i], rings[k][j], rings[k + 1][j], mid)
            b.tri(m, rings[k][i], rings[k + 1][j], rings[k + 1][i], mid)

def cut_face(b, ring, centre, inward, rng, bark_mat):
    """flat end face with bark rim, sapwood, a darker ring and the heart"""
    ring = [np.asarray(p) for p in ring]; n = len(ring)
    levels = [(0.90, bark_mat), (0.66, 'WoodSap'), (0.52, 'WoodRing'), (0.30, 'WoodSap'), (0.0, 'WoodHeart')]
    offc = centre
    prev = ring
    for s, m in levels:
        cur = [offc + (p - centre) * s for p in ring] if s > 0 else None
        for i in range(n):
            j = (i + 1) % n
            if cur is None:
                b.tri(m, prev[i], prev[j], offc, centre + inward)
            else:
                b.tri(m, prev[i], prev[j], cur[j], centre + inward); b.tri(m, prev[i], cur[j], cur[i], centre + inward)
        if cur is None: break
        prev = cur

def write(path, b, colours, name):
    g = GLTF2(asset=Asset(generator='tools/make_tree_parts.py'))
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
    for m, T in b.tris.items():
        g.materials.append(Material(name=m, pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[*colours[m], 1.0], metallicFactor=0.0, roughnessFactor=0.85)))
        T = np.array(T, dtype=np.float64)
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); ln = np.linalg.norm(n, axis=1)
        T, n = T[ln > 1e-12], n[ln > 1e-12] / ln[ln > 1e-12, None]
        pos = T.reshape(-1, 3).astype(np.float32); nor = np.repeat(n, 3, 0).astype(np.float32)
        prims.append(Primitive(attributes=Attributes(POSITION=accessor(pos, 5126, 'VEC3', 34962, True),
                                                     NORMAL=accessor(nor, 5126, 'VEC3', 34962)),
                               indices=accessor(np.arange(len(pos), dtype=np.uint16), 5123, 'SCALAR', 34963),
                               material=len(g.materials) - 1))
        total += len(T)
    g.meshes.append(Mesh(name=name, primitives=prims))
    g.nodes.append(Node(name=name, mesh=0))
    g.scenes.append(Scene(name='Scene', nodes=[0])); g.scene = 0
    while len(blob) % 4: blob.append(0)
    g.buffers.append(Buffer(byteLength=len(blob)))
    g.set_binary_blob(bytes(blob)); g.save(path)
    print(f'{path}: {total} triangles')

def bark_picker(sp, bands, n, rng, dark_base=0, dashes=()):
    """birch: white with thin black dashes; pine: plated bark with darker plates"""
    if sp == 'birch':
        marks = np.zeros((bands, n), bool)
        for k in dashes:
            start = rng.integers(0, n); run = rng.integers(1, n // 2 + 1)
            marks[k, [(start + i) % n for i in range(run)]] = True
        marks[:dark_base] |= rng.random((dark_base, n)) < 0.75
    else:                                                      # darker plates, a few per band
        marks = np.zeros((bands, n), bool)
        for k in range(bands):
            marks[k, rng.choice(n, 2, replace=False)] = True
    S = SPECIES[sp]
    return lambda k, i: S['mark'] if marks[min(k, bands - 1), i] else S['bark']

for si, (sp, S) in enumerate(SPECIES.items()):
    colours = dict(WOOD, **S['colours'])
    n = S['sides']

    # ---------------------------------------------------------------- stump
    rng = np.random.default_rng(SEED * 10 + si)
    b = Builder(); R, Hh = S['stump_r'], S['stump_h']
    if sp == 'birch':
        ys = [-0.05, 0.0, 0.06, 0.14, 0.155, 0.27, 0.285, Hh]
        dashes = [3, 5]
    else:
        ys = [-0.05, 0.0, 0.06, 0.16, 0.28, Hh]
        dashes = []
    rad = [R * (1 + 0.45 * np.exp(-max(y, 0) / 0.06)) for y in ys]   # root flare
    pts = [V3(0, y, 0) for y in ys]
    rings = rings_along(pts, rad, n, 0.06, rng)
    # slanted axe cut: tilt the top ring
    tilt = V3(np.cos(rng.uniform(0, 6.3)), 0, np.sin(rng.uniform(0, 6.3))); tilt[1] = 0
    rings[-1] = [p + V3(0, 0.12 * (p - pts[-1]) @ tilt / R * R, 0) for p in rings[-1]]
    bark(b, rings, pts, bark_picker(sp, len(ys) - 1, n, rng, dark_base=2 if sp == 'birch' else 0, dashes=dashes))
    cut_face(b, rings[-1], sum(rings[-1]) / n, V3(0, -0.3, 0), rng, S['bark'])
    # a few roots running into the ground
    for k in range(4):
        a = 2 * np.pi * k / 4 + rng.uniform(-0.4, 0.4)
        out = V3(np.cos(a), 0, np.sin(a))
        reach = rng.uniform(1.8, 2.3)
        rp = [out * R * 0.7 + V3(0, 0.09, 0), out * R * 1.4 + V3(0, 0.03, 0), out * R * reach + V3(0, -0.05, 0)]
        rr = [R * 0.42, R * 0.3, R * 0.14]
        rring = rings_along(rp, rr, 6, 0.05, rng)
        dark = rng.random((2, 6)) < (0.6 if sp == 'birch' else 0.3)
        bark(b, rring, rp, lambda k_, i_, dark=dark: S['mark'] if dark[k_, i_] else S['bark'])
    write(f'models/{sp}_stump.glb', b, colours, f'{sp.capitalize()}Stump')

    # ---------------------------------------------------------------- trunk (felled log)
    rng = np.random.default_rng(SEED * 10 + si + 5)
    b = Builder(); L = S['trunk_len']; r0, r1 = S['trunk_r']
    ts = np.linspace(0, 1, 9)
    if sp == 'birch':                                          # thin dash bands along the log
        ts = np.sort(np.concatenate([ts, [0.20, 0.215, 0.44, 0.455, 0.66, 0.675, 0.85, 0.865]]))
    rad = r0 + (r1 - r0) * ts ** 0.9
    bend = rng.normal(0, 0.05)
    pts = [V3(-L / 2 + L * t, r, bend * np.sin(np.pi * t)) for t, r in zip(ts, rad)]   # resting on the ground
    rings = rings_along(pts, rad, n, 0.05, rng)
    if sp == 'birch':
        dashes = [k for k in range(len(ts) - 1) if ts[k + 1] - ts[k] < 0.02]
        pick = bark_picker(sp, len(ts) - 1, n, rng, dark_base=1, dashes=dashes)
    else:
        pick = bark_picker(sp, len(ts) - 1, n, rng)
    bark(b, rings, pts, pick)
    cut_face(b, rings[0], pts[0], V3(1, 0, 0), rng, S['bark'])
    cut_face(b, rings[-1], pts[-1], V3(-1, 0, 0), rng, S['bark'])
    # sawn branch stubs (pine: in whorls)
    for k in range(S['stubs']):
        t = rng.uniform(0.3, 0.92) if sp == 'birch' else 0.45 + 0.4 * (k // 3) + rng.uniform(-0.02, 0.02)
        a = rng.uniform(0, 2 * np.pi)
        if a > np.pi * 1.15 and a < np.pi * 1.85: a -= np.pi   # not straight into the ground
        c = V3(-L / 2 + L * t, r0 + (r1 - r0) * t ** 0.9, bend * np.sin(np.pi * t))
        rt = r0 + (r1 - r0) * t ** 0.9
        d = V3(0.35, np.cos(a), np.sin(a)); d /= np.linalg.norm(d)
        sp_pts = [c + d * rt * 0.6, c + d * (rt + 0.07)]
        srad = [rt * 0.32, rt * 0.26]
        sr = rings_along(sp_pts, srad, 5, 0.0, rng)
        bark(b, sr, sp_pts, lambda k_, i_: S['bark'])
        cut_face(b, sr[-1], sp_pts[-1], -d, rng, S['bark'])
    write(f'models/{sp}_trunk.glb', b, colours, f'{sp.capitalize()}Trunk')
