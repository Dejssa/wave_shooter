"""Build five low-poly stones as models/stones/stone_<size>.glb.

Each stone is the convex hull of random points on a lumpy ellipsoid, so it is
made of flat chiselled faces like the other flat-shaded models. The bottom is
cut flat and (except for the tiny pebble) sunk a little into the ground.
Faces get a few stone tones; the big ones get moss on their tops.

  tiny     ~0.15 m  smooth pebble, small enough to pick up
  minor    ~0.75 m  chunky, heavy-looking, half sunk - too heavy to lift
  large    ~1.3 m   solid angular rock
  huge     ~2.4 m   boulder with moss
  largest  ~4 m     rock formation of several merged boulders with moss

Origin: on the ground (y = 0) under the middle of the stone, +Y up.

Usage: python tools/make_stones.py [SEED]   (needs numpy, scipy, pygltflib)
"""
import sys
import numpy as np
from scipy.spatial import ConvexHull
from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
                       Accessor, Material, PbrMetallicRoughness, Asset)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5

MATS = {  # name: (rgb, roughness)
    'StoneLight': ([0.30, 0.29, 0.27], 0.85),   # linear values: they display lighter
    'StoneMid':   ([0.22, 0.215, 0.21], 0.9),
    'StoneDark':  ([0.15, 0.148, 0.145], 0.9),
    'Pebble':     ([0.36, 0.32, 0.27], 0.55),
    'PebbleDark': ([0.27, 0.24, 0.20], 0.55),
    'Moss':       ([0.10, 0.20, 0.05], 1.0),
}

# name: list of parts (centre, size xyz, points, lumpiness), sink, moss, palette
STONES = {
    'tiny':    ([((0, 0, 0), (0.15, 0.07, 0.11), 40, 0.05)], 0.0, False, ['Pebble', 'Pebble', 'PebbleDark']),
    'minor':   ([((0, 0, 0), (0.78, 0.56, 0.66), 30, 0.12)], 0.07, False, ['StoneLight', 'StoneMid', 'StoneMid', 'StoneDark']),
    'large':   ([((0, 0, 0), (1.35, 1.05, 1.15), 34, 0.13)], 0.12, False, ['StoneLight', 'StoneMid', 'StoneDark']),
    'huge':    ([((0, 0, 0), (2.5, 1.9, 2.1), 40, 0.12)], 0.2, True, ['StoneLight', 'StoneMid', 'StoneDark']),
    'largest': ([((0, 0, 0), (3.2, 3.3, 2.8), 44, 0.12),
                 ((1.6, 0, 0.6), (2.2, 2.1, 1.9), 30, 0.14),
                 ((-1.4, 0, 0.8), (1.7, 1.4, 1.5), 26, 0.14),
                 ((0.5, 0, -1.3), (1.5, 1.1, 1.3), 22, 0.15)], 0.3, True, ['StoneLight', 'StoneMid', 'StoneDark']),
}

def rock(rng, centre, size, npts, lump, sink):
    """triangles (n,3,3) of one convex rock with a flat bottom at y = -sink"""
    d = rng.normal(size=(npts, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 1] = np.abs(d[:, 1]) ** 0.6                            # upper half, pushed up into a dome
    r = 1 + lump * rng.standard_normal((npts, 1))
    p = d * r * (np.array(size) * [0.5, 1.0, 0.5])              # size y = height above the cut
    p[:, 1] = np.maximum(p[:, 1], 0)                            # flat bottom at y = 0
    ring = np.linspace(0, 2 * np.pi, 9, endpoint=False) + rng.uniform(0, 1)
    base = np.c_[np.cos(ring) * size[0] * 0.46, np.zeros(9), np.sin(ring) * size[2] * 0.46]
    p = np.vstack([p, base])                                    # a wide footprint so it sits well
    p += np.array(centre, float) + np.array([0, -sink, 0])
    hull = ConvexHull(p)
    T = p[hull.simplices]
    c = p.mean(0)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    flip = (n * (T.mean(1) - c)).sum(1) < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    return T

def build(name, parts, sink, moss, palette, rng):
    by_mat = {}
    top = max(pp[0][1] + pp[1][1] - sink for pp in parts)
    for centre, size, npts, lump in parts:
        T = rock(rng, centre, size, npts, lump, sink)
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True)
        for t, nn in zip(T, n):
            if nn[1] < -0.9:
                continue                                        # the hidden bottom
            m = palette[rng.integers(len(palette))]
            if moss and nn[1] > 0.5 and t[:, 1].mean() > 0.55 * top and rng.random() < 0.8:
                m = 'Moss'
            by_mat.setdefault(m, []).append(t)
    return by_mat

def write(path, by_mat, name):
    g = GLTF2(asset=Asset(generator='tools/make_stones.py'))
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
    for m, T in by_mat.items():
        rgb, rough = MATS[m]
        g.materials.append(Material(name=m, pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[*rgb, 1.0], metallicFactor=0.0, roughnessFactor=rough)))
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
    g.set_binary_blob(bytes(blob))
    g.save(path)
    return total

for i, (name, (parts, sink, moss, palette)) in enumerate(STONES.items()):
    rng = np.random.default_rng(SEED * 100 + i)
    total = write(f'models/stones/stone_{name}.glb', build(name, parts, sink, moss, palette, rng), f'Stone_{name}')
    print(f'models/stones/stone_{name}.glb: {total} triangles')
