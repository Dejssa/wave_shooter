"""Generate a seamless stone-wall texture (colour + normal map).

Stylized fieldstones from a wrapping Voronoi pattern: every stone is a polygon
with flat chiselled bevels along its edges and a slightly tilted top face, so
stones catch the light differently (low-poly look). Stones vary in size and in
grey/brown tint, set in recessed mortar. Tiles in both directions.

Usage: python tools/make_stone_texture.py [SIZE] [SEED]   (needs numpy, pillow)
Writes textures/stone.png and textures/stone_normal.png
"""
import sys
import numpy as np
from PIL import Image

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 512
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 11
rng = np.random.default_rng(SEED)

GRID = 6                     # roughly GRID x GRID stones per tile
JITTER = 0.75                # how irregular the stones are (0 = square grid)
MORTAR = 0.010 * SIZE        # half the mortar gap, px
BEVEL = 0.030 * SIZE         # bevel width, px
TILT = 0.35                  # how much each stone's top face leans
LIGHT = np.array([-0.45, 0.55, 0.70]); LIGHT /= np.linalg.norm(LIGHT)   # top-left light
PALETTE = np.array([[0.58, 0.57, 0.55], [0.52, 0.51, 0.50], [0.60, 0.56, 0.50],
                    [0.47, 0.47, 0.48], [0.55, 0.52, 0.47]])
MORTAR_RGB = np.array([0.28, 0.27, 0.25])

def tile_noise(scale, amp=1.0):
    """smooth noise that wraps at the texture edges (filtered in frequency space)"""
    f = np.fft.fftfreq(SIZE)
    fx, fy = np.meshgrid(f, f)
    spec = np.fft.fft2(rng.standard_normal((SIZE, SIZE))) * np.exp(-(fx**2 + fy**2) * (SIZE / scale)**2)
    n = np.real(np.fft.ifft2(spec))
    return amp * (n - n.mean()) / (n.std() + 1e-9)

# stone centres on a jittered grid, plus wrapped copies so the pattern tiles
cell = SIZE / GRID
gy, gx = np.mgrid[0:GRID, 0:GRID]
pts = np.stack([(gx + 0.5 + JITTER * (rng.random(gx.shape) - 0.5)) * cell,
                (gy + 0.5 + JITTER * (rng.random(gy.shape) - 0.5)) * cell], -1).reshape(-1, 2)
n_st = len(pts)
offs = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]) * SIZE
cand = (pts[None, :, :] + offs[:, None, :]).reshape(-1, 2)        # 9 * n_st points
cand_id = np.tile(np.arange(n_st), 9)

y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
P = np.stack([x, y], -1)
stone = np.zeros((SIZE, SIZE), int)
edge = np.zeros((SIZE, SIZE))
local = np.zeros((SIZE, SIZE, 2))
for r0 in range(0, SIZE, 64):                                     # in bands to keep memory low
    p = P[r0:r0 + 64]
    d2 = ((p[:, :, None, :] - cand[None, None]) ** 2).sum(-1)
    near = d2.argmin(-1)
    c1 = cand[near]
    # exact distance to the cell's polygon edges (each bisector is a straight line)
    diff = cand[None, None] - c1[:, :, None, :]
    ln = np.linalg.norm(diff, axis=-1)
    dist = (ln / 2) - ((p - c1)[:, :, None, :] * diff).sum(-1) / np.maximum(ln, 1e-9)
    dist[ln < 1e-6] = np.inf
    edge[r0:r0 + 64] = dist.min(-1)
    stone[r0:r0 + 64] = cand_id[near]
    local[r0:r0 + 64] = (p - c1) / cell

mortar = edge < MORTAR
bevel = np.clip((edge - MORTAR) / BEVEL, 0, 1)
tilt = rng.normal(0, TILT, (n_st, 2))
top = 1 + (local * tilt[stone]).sum(-1) * 0.5                    # leaning top face
height = np.where(mortar, 0, np.minimum(bevel * 1.0, 1) * top)
height = height * 5.0 + tile_noise(5, 0.06)

gxh = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) / 2
gyh = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) / 2
nrm = np.stack([-gxh, gyh, np.ones_like(gxh)], -1)
nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)

# colour: per-stone tint from the palette, grain, and baked light from the facets
base = PALETTE[rng.integers(0, len(PALETTE), n_st)] * (1 + 0.05 * rng.standard_normal((n_st, 1)))
rgb = base[stone] * (1 + tile_noise(30, 0.04)[..., None] + tile_noise(3, 0.015)[..., None])
light = (nrm @ LIGHT) - LIGHT[2]                                  # 0 on a flat face
rgb = rgb * (1 + 0.9 * light[..., None])
pores = (tile_noise(1.0, 1.0) > 2.4) & (bevel >= 1)
rgb = np.where(pores[..., None], rgb * 0.75, rgb)
mort = MORTAR_RGB[None, None, :] * (1 + tile_noise(8, 0.06)[..., None])
rgb = np.where(mortar[..., None], mort, rgb)
Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save('textures/stone.png')
Image.fromarray(((nrm * 0.5 + 0.5) * 255).astype(np.uint8)).save('textures/stone_normal.png')
print('wrote textures/stone.png and textures/stone_normal.png', SIZE, 'px')
