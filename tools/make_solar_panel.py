"""Build a folding solar panel as models/solar_panel.glb.

Matches the look of a portable folding panel: 3 x 2 sub-panels in dark
aluminium frames with small hinge brackets where they meet, a textured cell
surface (dark blue cells with silver lines, generated here and embedded in
the .glb), and a two-leg kickstand behind. Flat-shaded like the other models.

The panel face is exactly 1.8 m along X by 0.8 m along its own Y. It stands on
its bottom edge, leaning back by TILT degrees from the ground (90 = upright,
0 = flat); the cells face forward/up (+Z side). The origin is on the ground
under the middle of the bottom edge.

Usage: python tools/make_solar_panel.py [OUT.glb] [TILT]   (needs numpy, pygltflib, pillow)
"""
import io
import sys
import numpy as np
from PIL import Image as PILImage
from pygltflib import (GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView, Accessor,
                       Material, PbrMetallicRoughness, Asset, Image, Texture, Sampler, TextureInfo)

OUT = sys.argv[1] if len(sys.argv) > 1 else 'models/solar_panel.glb'
TILT = float(sys.argv[2]) if len(sys.argv) > 2 else 62.0

W, H = 1.8, 0.8                 # panel face size (X, Y)
COLS, ROWS = 3, 2               # sub-panels
GAP = 0.012                     # hinge gap between sub-panels
BORDER = 0.016                  # frame width around each cell area
THICK = 0.022                   # panel thickness
CELLS = (10, 6)                 # cells per sub-panel (across, up)

MATS = {  # name: (rgb, metallic, roughness)
    'SolarFrame':    ([0.10, 0.11, 0.12], 0.4, 0.5),
    'SolarCells':    ([1.0, 1.0, 1.0], 0.1, 0.25),
    'SolarHardware': ([0.55, 0.56, 0.58], 0.5, 0.4),
}
tris = {k: [] for k in MATS}
uvs = []                         # per-corner UVs for SolarCells
V3 = lambda *a: np.array(a, dtype=np.float64)

# ---------------------------------------------------------------- transforms
th = np.radians(TILT)
ey = V3(0, np.sin(th), -np.cos(th))      # panel "up" leans back
ez = V3(0, np.cos(th), np.sin(th))       # panel face normal
ex = V3(1, 0, 0)
LIFT = THICK / 2 * np.cos(th) + 0.004    # keep the back bottom edge above the ground
def P(x, y, z):
    """panel-local (x across, y up the face, z out of the face) -> world"""
    return x * ex + y * ey + z * ez + V3(0, LIFT, 0)

def tri(m, a, b, c, away, uv=None):
    if np.cross(b - a, c - a) @ ((a + b + c) / 3 - away) < 0:
        b, c = c, b
        if uv: uv = (uv[0], uv[2], uv[1])
    tris[m].append((a, b, c))
    if uv: uvs.append(uv)

def box(m, lo, hi, place=P):
    """axis-aligned box in panel-local coordinates"""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    c = place((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    v = [place(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    for f in ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)):
        a, b, cc, d = (v[i] for i in f)
        tri(m, a, b, cc, c); tri(m, a, cc, d, c)

# ---------------------------------------------------------------- sub-panels
sw = (W - GAP * (COLS - 1)) / COLS
sh = (H - GAP * (ROWS - 1)) / ROWS
for r in range(ROWS):
    for c in range(COLS):
        x0 = -W / 2 + c * (sw + GAP); y0 = r * (sh + GAP)
        box('SolarFrame', (x0, y0, -THICK / 2), (x0 + sw, y0 + sh, THICK / 2))
        # cell surface just in front of the frame face, inset by the border
        z = THICK / 2 + 0.0015
        a, b = P(x0 + BORDER, y0 + BORDER, z), P(x0 + sw - BORDER, y0 + BORDER, z)
        cc, d = P(x0 + sw - BORDER, y0 + sh - BORDER, z), P(x0 + BORDER, y0 + sh - BORDER, z)
        back = P(x0 + sw / 2, y0 + sh / 2, 0)
        tri('SolarCells', a, b, cc, back, ((0, 1), (1, 1), (1, 0)))
        tri('SolarCells', a, cc, d, back, ((0, 1), (1, 0), (0, 0)))

# hinge brackets on the front where sub-panels meet, and corner caps
bx, by = 0.022, 0.03
for r in range(ROWS + 1):
    y = min(max(r * (sh + GAP) - GAP / 2, 0.012), H - 0.012)
    for c in range(COLS + 1):
        x = min(max(-W / 2 + c * (sw + GAP) - GAP / 2, -W / 2 + 0.012), W / 2 - 0.012)
        box('SolarHardware', (x - bx / 2, y - by / 2, THICK / 2 - 0.002), (x + bx / 2, y + by / 2, THICK / 2 + 0.008))

# ---------------------------------------------------------------- kickstand
LEG = 0.014
for sx in (-1, 1):
    top = P(sx * 0.55, 0.62, -THICK / 2)
    foot = V3(sx * 0.62, 0.0, top[2] - 0.42)
    axis = foot - top; L = np.linalg.norm(axis); axis /= L
    side = np.cross(axis, V3(1, 0, 0)); side /= np.linalg.norm(side)
    up = np.cross(side, axis)
    place = lambda u, v, w, top=top, axis=axis, side=side, up=up: top + u * up + v * axis + w * side
    box('SolarFrame', (-LEG / 2, 0, -LEG / 2), (LEG / 2, L, LEG / 2), place)
    # hinge block where the leg meets the panel back
    box('SolarHardware', (sx * 0.55 - 0.03, 0.60, -THICK / 2 - 0.02), (sx * 0.55 + 0.03, 0.64, -THICK / 2))
# cross brace between the legs
m0, m1 = P(-0.55, 0.62, -THICK / 2), P(0.55, 0.62, -THICK / 2)
fl, fr = V3(-0.62, 0.0, m0[2] - 0.42), V3(0.62, 0.0, m1[2] - 0.42)
a, b = m0 + (fl - m0) * 0.55, m1 + (fr - m1) * 0.55
brace = lambda u, v, w: V3(0, 0, 0) + (a + (b - a) * ((u + 0.5))) + V3(0, v, w)
box('SolarFrame', (-0.5, -0.006, -0.006), (0.5, 0.006, 0.006), brace)

# ---------------------------------------------------------------- cell texture
def cell_texture(px=512):
    cx, cy = CELLS
    w, h = px, px // 2               # power-of-two; the UVs stretch it over the cell area
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:h, 0:w].astype(float)
    u, v = x / w * cx, y / h * cy
    fu, fv = u % 1, v % 1
    cell = (u.astype(int) + v.astype(int) * cx)
    base = np.array([0.07, 0.15, 0.36]) * (1 + 0.06 * rng.standard_normal(cx * cy))[cell][..., None]
    base = base * (1 + 0.12 * (1 - v / cy))[..., None]              # faint sheen
    gap = (fu < 0.035) | (fu > 0.965) | (fv < 0.05) | (fv > 0.95)
    bus = (np.abs(fu - 0.33) < 0.01) | (np.abs(fu - 0.67) < 0.01)     # busbars
    finger = (np.abs((fv * 12) % 1 - 0.5) < 0.04)                     # thin fingers
    rgb = base
    rgb = np.where(finger[..., None], rgb * 1.25 + 0.03, rgb)
    rgb = np.where(bus[..., None], np.array([0.62, 0.65, 0.70]), rgb)
    rgb = np.where(gap[..., None], np.array([0.72, 0.75, 0.80]), rgb)
    img = PILImage.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, 'PNG'); return buf.getvalue()

# ---------------------------------------------------------------- write glTF
g = GLTF2(asset=Asset(generator='tools/make_solar_panel.py'))
blob = bytearray()
def push(data, target=None):
    while len(blob) % 4: blob.append(0)
    off = len(blob); blob.extend(data)
    g.bufferViews.append(BufferView(buffer=0, byteOffset=off, byteLength=len(data), target=target))
    return len(g.bufferViews) - 1
def accessor(arr, ctype, typ, target, mm=False):
    a = Accessor(bufferView=push(arr.tobytes(), target), componentType=ctype, count=len(arr), type=typ)
    if mm: a.min = arr.min(0).tolist(); a.max = arr.max(0).tolist()
    g.accessors.append(a); return len(g.accessors) - 1

g.images.append(Image(bufferView=push(cell_texture()), mimeType='image/png', name='SolarCells'))
g.samplers.append(Sampler(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071))
g.textures.append(Texture(source=0, sampler=0))

prims = []; total = 0
for name, (rgb, met, rough) in MATS.items():
    pbr = PbrMetallicRoughness(baseColorFactor=[*rgb, 1.0], metallicFactor=met, roughnessFactor=rough)
    if name == 'SolarCells': pbr.baseColorTexture = TextureInfo(index=0)
    g.materials.append(Material(name=name, pbrMetallicRoughness=pbr))
    T = np.array(tris[name], dtype=np.float64)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True)
    pos = T.reshape(-1, 3).astype(np.float32); nor = np.repeat(n, 3, 0).astype(np.float32)
    at = Attributes(POSITION=accessor(pos, 5126, 'VEC3', 34962, True), NORMAL=accessor(nor, 5126, 'VEC3', 34962))
    if name == 'SolarCells':
        at.TEXCOORD_0 = accessor(np.array(uvs, np.float32).reshape(-1, 2), 5126, 'VEC2', 34962)
    prims.append(Primitive(attributes=at, material=len(g.materials) - 1,
                           indices=accessor(np.arange(len(pos), dtype=np.uint16), 5123, 'SCALAR', 34963)))
    total += len(T)
g.meshes.append(Mesh(name='SolarPanel', primitives=prims))
g.nodes.append(Node(name='SolarPanel', mesh=0))
g.scenes.append(Scene(name='Scene', nodes=[0])); g.scene = 0
while len(blob) % 4: blob.append(0)
g.buffers.append(Buffer(byteLength=len(blob)))
g.set_binary_blob(bytes(blob))
g.save(OUT)
print(f'{OUT}: {total} triangles, tilt {TILT:g} deg')
