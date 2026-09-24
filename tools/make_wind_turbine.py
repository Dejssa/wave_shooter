"""Build a small wind turbine as models/wind_turbine.glb.

Backyard / off-grid style, flat-shaded like the other models: tapered mast on
a three-legged steel base with concrete pads, a streamlined nacelle with a
tail vane, and a three-blade rotor with red tips. Fits a 1.8 x 1.8 m
footprint (X by Z) and 3.5 m height (Y): the rotor is 1.8 m across and the
top blade tip reaches 3.5 m.

Node hierarchy so it can be animated:
  WindTurbine (origin on the ground at the mast base)
    Tower
    Head   - at the top of the mast; rotate about Y to face the wind
      Rotor - at the hub; spin about its local Z axis

Usage: python tools/make_wind_turbine.py [OUT.glb]   (needs numpy, pygltflib)
"""
import sys
import numpy as np
from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
                       Accessor, Material, PbrMetallicRoughness, Asset)

OUT = sys.argv[1] if len(sys.argv) > 1 else 'models/wind_turbine.glb'

HEIGHT, RADIUS = 3.5, 0.9        # overall height, rotor radius (1.8 m across)
HUB_Y = HEIGHT - RADIUS          # hub height, so the top blade tip reaches HEIGHT
HUB_Z = 0.34                     # rotor sits this far in front of the mast axis
MATS = {  # name: (rgb, metallic, roughness)
    'TurbineWhite':  ([0.90, 0.91, 0.92], 0.1, 0.45),
    'TurbineGrey':   ([0.62, 0.64, 0.66], 0.3, 0.5),
    'TurbineSteel':  ([0.24, 0.25, 0.27], 0.4, 0.5),
    'TurbineAccent': ([0.82, 0.18, 0.12], 0.0, 0.5),
    'Concrete':      ([0.55, 0.54, 0.52], 0.0, 0.9),
}
V3 = lambda *a: np.array(a, dtype=np.float64)
meshes = {}                       # mesh name -> {material: [tri, ...]}

def tri(mesh, m, a, b, c, away):
    """add a triangle to `mesh` whose normal points away from the point `away`"""
    if np.cross(b - a, c - a) @ ((a + b + c) / 3 - away) < 0: b, c = c, b
    meshes.setdefault(mesh, {}).setdefault(m, []).append((a, b, c))

def frame(axis):
    axis = axis / np.linalg.norm(axis)
    ref = V3(0, 0, 1) if abs(axis[2]) < 0.9 else V3(1, 0, 0)
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    return u, np.cross(axis, u)

def loft(mesh, rings, centres, pick, caps=(True, True)):
    n = len(rings[0])
    for k in range(len(rings) - 1):
        mid = (centres[k] + centres[k + 1]) / 2
        for i in range(n):
            j = (i + 1) % n; m = pick(k)
            tri(mesh, m, rings[k][i], rings[k][j], rings[k + 1][j], mid)
            tri(mesh, m, rings[k][i], rings[k + 1][j], rings[k + 1][i], mid)
    for on, r, c, other, k in ((caps[0], rings[0], centres[0], centres[1], 0),
                               (caps[1], rings[-1], centres[-1], centres[-2], len(rings) - 2)):
        if on:
            ctr = sum(r) / n
            for i in range(n):
                tri(mesh, pick(k), ctr, r[i], r[(i + 1) % n], other)

def tube(mesh, m, pts, radii, n=8, caps=(True, True)):
    pts = [V3(*p) for p in pts]; rings = []
    for i, (p, r) in enumerate(zip(pts, radii)):
        u, v = frame(pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])
        rx, ry = (r, r) if np.isscalar(r) else r
        th = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
        rings.append([p + rx * np.cos(t) * u + ry * np.sin(t) * v for t in th])
    loft(mesh, rings, pts, lambda k: m, caps)

def box(mesh, m, lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    c = V3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    v = [V3(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    for f in ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)):
        a, b, cc, d = (v[i] for i in f)
        tri(mesh, m, a, b, cc, c); tri(mesh, m, a, cc, d, c)

# ---------------------------------------------------------------- tower (world coordinates)
MAST_TOP = HUB_Y - 0.13
tube('Tower', 'TurbineGrey', [(0, 0.07, 0), (0, MAST_TOP, 0)], [0.065, 0.045], caps=(False, True))
box('Tower', 'TurbineSteel', (-0.16, 0.05, -0.16), (0.16, 0.07, 0.16))          # base plate
box('Tower', 'Concrete', (-0.22, 0.0, -0.22), (0.22, 0.05, 0.22))               # footing
for k in range(3):
    a = np.radians(90 + 120 * k)
    foot = V3(0.72 * np.cos(a), 0.07, 0.72 * np.sin(a))
    tube('Tower', 'TurbineSteel', [(0, 0.95, 0), foot], [0.024, 0.024], n=6)
    box('Tower', 'Concrete', tuple(foot + V3(-0.09, -0.07, -0.09)), tuple(foot + V3(0.09, -0.005, 0.09)))
    # a light brace between each leg and the mast lower down
    tube('Tower', 'TurbineSteel', [(0, 0.35, 0), foot * V3(0.55, 0, 0.55) + V3(0, 0.40, 0)], [0.015, 0.015], n=6)

# ---------------------------------------------------------------- head (local: origin at the mast top)
HY = HUB_Y - MAST_TOP             # hub height above the head origin
tube('Head', 'TurbineGrey', [(0, -0.02, 0), (0, HY - 0.08, 0)], [0.055, 0.055], caps=(False, False))   # yaw bearing
nac = [(-0.26, 0.04, 0.045), (-0.18, 0.085, 0.09), (0.02, 0.11, 0.12), (0.20, 0.105, 0.115), (HUB_Z - 0.06, 0.09, 0.095)]
tube('Head', 'TurbineWhite', [(0, HY, z) for z, _, _ in nac], [(rx, ry) for _, rx, ry in nac], n=8)
# tail boom and vane
tube('Head', 'TurbineGrey', [(0, HY + 0.02, -0.22), (0, HY + 0.06, -0.66)], [0.018, 0.018], n=6)
fin = [V3(0, HY - 0.06, -0.58), V3(0, HY + 0.10, -0.56), V3(0, HY + 0.34, -0.88), V3(0, HY - 0.10, -0.88)]
for sx in (-1, 1):                # thin two-sided plate
    off = V3(sx * 0.006, 0, 0)
    tri('Head', 'TurbineAccent', fin[0] + off, fin[1] + off, fin[2] + off, V3(-sx, HY, -0.7))
    tri('Head', 'TurbineAccent', fin[0] + off, fin[2] + off, fin[3] + off, V3(-sx, HY, -0.7))
for i in range(4):                # fin edges
    a, b = fin[i], fin[(i + 1) % 4]
    q = [a + V3(-0.006, 0, 0), b + V3(-0.006, 0, 0), b + V3(0.006, 0, 0), a + V3(0.006, 0, 0)]
    mid = sum(fin) / 4
    tri('Head', 'TurbineAccent', q[0], q[1], q[2], mid); tri('Head', 'TurbineAccent', q[0], q[2], q[3], mid)

# ---------------------------------------------------------------- rotor (local: origin at the hub, axis +Z)
tube('Rotor', 'TurbineWhite', [(0, 0, -0.06), (0, 0, 0.04), (0, 0, 0.11), (0, 0, 0.16)],
     [0.095, 0.10, 0.07, 0.012], n=8, caps=(True, True))                           # spinner
R_ST = [0.07, 0.16, 0.36, 0.62, RADIUS - 0.01]
CHORD = [0.07, 0.14, 0.115, 0.08, 0.04]
TWIST = [28, 22, 14, 8, 4]                                                         # degrees
THICK = [0.30, 0.18, 0.14, 0.12, 0.10]                                             # of the chord
for b in range(3):
    ang = np.radians(90 + 120 * b)
    radial = V3(np.cos(ang), np.sin(ang), 0); tangent = V3(-np.sin(ang), np.cos(ang), 0)
    rings, centres = [], []
    for r, c, tw, th in zip(R_ST, CHORD, TWIST, THICK):
        t = np.radians(tw)
        chord_dir = np.cos(t) * tangent + np.sin(t) * V3(0, 0, 1)                  # twisted section
        norm_dir = -np.sin(t) * tangent + np.cos(t) * V3(0, 0, 1)
        o = radial * r + V3(0, 0, 0.02)
        rings.append([o + chord_dir * c * 0.35, o + norm_dir * c * th / 2,
                      o - chord_dir * c * 0.65, o - norm_dir * c * th / 2 * 0.6])
        centres.append(o)
    loft('Rotor', rings, centres, lambda k: 'TurbineAccent' if k == len(R_ST) - 2 else 'TurbineWhite',
         caps=(False, True))

# ---------------------------------------------------------------- write glTF
g = GLTF2(asset=Asset(generator='tools/make_wind_turbine.py'))
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

mat_index = {}
for name, (rgb, met, rough) in MATS.items():
    mat_index[name] = len(g.materials)
    g.materials.append(Material(name=name, pbrMetallicRoughness=PbrMetallicRoughness(
        baseColorFactor=[*rgb, 1.0], metallicFactor=met, roughnessFactor=rough)))
total = 0; mesh_index = {}
for mname, by_mat in meshes.items():
    prims = []
    for m, T in by_mat.items():
        T = np.array(T, dtype=np.float64)
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); ln = np.linalg.norm(n, axis=1)
        T, n = T[ln > 1e-12], n[ln > 1e-12] / ln[ln > 1e-12, None]
        pos = T.reshape(-1, 3).astype(np.float32); nor = np.repeat(n, 3, 0).astype(np.float32)
        prims.append(Primitive(attributes=Attributes(POSITION=accessor(pos, 5126, 'VEC3', 34962, True),
                                                     NORMAL=accessor(nor, 5126, 'VEC3', 34962)),
                               indices=accessor(np.arange(len(pos), dtype=np.uint16), 5123, 'SCALAR', 34963),
                               material=mat_index[m]))
        total += len(T)
    mesh_index[mname] = len(g.meshes); g.meshes.append(Mesh(name=mname, primitives=prims))
# nodes: 0 root, 1 tower, 2 head, 3 rotor
g.nodes += [Node(name='WindTurbine', children=[1, 2]),
            Node(name='Tower', mesh=mesh_index['Tower']),
            Node(name='Head', mesh=mesh_index['Head'], translation=[0, MAST_TOP, 0], children=[3]),
            Node(name='Rotor', mesh=mesh_index['Rotor'], translation=[0, HY, HUB_Z])]
g.scenes.append(Scene(name='Scene', nodes=[0])); g.scene = 0
while len(blob) % 4: blob.append(0)
g.buffers.append(Buffer(byteLength=len(blob)))
g.set_binary_blob(bytes(blob))
g.save(OUT)
print(f'{OUT}: {total} triangles')
