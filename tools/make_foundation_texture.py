"""Generate a seamless foundation-block texture (colour + normal map).

Stylized concrete blocks in a running bond: flat faces with a slight per-block
tint and grain, chunky bevels lit from the top-left, recessed mortar, pores
and a few chipped corners. Tiles in both directions.

Usage: python tools/make_foundation_texture.py [SIZE] [SEED]   (needs numpy, pillow)
Writes textures/foundation_block.png and textures/foundation_block_normal.png
"""
import sys
import numpy as np
from PIL import Image

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 512
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
rng = np.random.default_rng(SEED)

ROWS, COLS = 4, 2            # blocks per tile; every other row is shifted half a block
MORTAR = 0.012 * SIZE        # half the mortar gap, px
BEVEL = 0.028 * SIZE         # bevel width, px
CONCRETE = np.array([0.60, 0.58, 0.55])
MORTAR_RGB = np.array([0.30, 0.29, 0.27])

def tile_noise(scale, amp=1.0):
    """smooth noise that wraps at the texture edges (filtered in frequency space)"""
    f = np.fft.fftfreq(SIZE)
    fx, fy = np.meshgrid(f, f)
    spec = np.fft.fft2(rng.standard_normal((SIZE, SIZE))) * np.exp(-(fx**2 + fy**2) * (SIZE / scale)**2)
    n = np.real(np.fft.ifft2(spec))
    return amp * (n - n.mean()) / (n.std() + 1e-9)

y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
bh, bw = SIZE / ROWS, SIZE / COLS
row = (y // bh).astype(int)
xs = (x + (row % 2) * bw / 2) % SIZE
col = (xs // bw).astype(int)
u, v = xs - col * bw, y - row * bh                     # position inside the block
block = row * COLS + col
d_left, d_right, d_top, d_bot = u, bw - u, v, bh - v
d = np.minimum.reduce([d_left, d_right, d_top, d_bot])

# chipped corners: some blocks lose a small triangle at one corner
chip = np.zeros_like(d, bool)
for b in range(ROWS * COLS):
    for _ in range(rng.integers(0, 3)):
        cu, cv = rng.integers(0, 2), rng.integers(0, 2)
        size = rng.uniform(0.05, 0.11) * SIZE
        du = u if cu == 0 else bw - u
        dv = v if cv == 0 else bh - v
        chip |= (block == b) & (du + dv < size)

mortar = d < MORTAR
# height: 0 in the mortar, rising over the bevel to 1 on the face
h = np.clip((d - MORTAR) / BEVEL, 0, 1)
h = np.where(chip, 0.25, h)

# per-block tint and grain
tint = 1 + 0.06 * rng.standard_normal(ROWS * COLS)
face = CONCRETE[None, None, :] * tint[block][..., None]
face = face * (1 + tile_noise(40, 0.035)[..., None] + tile_noise(3, 0.012)[..., None])

# bevel lighting: top/left edges catch light, bottom/right fall into shade
on_bevel = (~mortar) & (~chip) & (d < MORTAR + BEVEL)
nearest = np.argmin(np.stack([d_left, d_right, d_top, d_bot]), axis=0)
shade = np.array([0.10, -0.12, 0.18, -0.22])[nearest]
face = face * (1 + np.where(on_bevel, shade, 0)[..., None])

# pores: small dark specks on the faces
pores = (tile_noise(1.0, 1.0) > 2.3) & (~mortar) & (~on_bevel) & (~chip)
face = np.where(pores[..., None], face * 0.72, face)
# chips: broken, slightly darker concrete sitting lower than the face
face = np.where((chip & ~mortar)[..., None], face * (0.80 + tile_noise(2, 0.03))[..., None], face)

mort = MORTAR_RGB[None, None, :] * (1 + tile_noise(8, 0.05)[..., None])
rgb = np.where(mortar[..., None], mort, face)
Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save('textures/foundation_block.png')

# normal map (OpenGL convention: +Y up), from the height with wrap-around gradients
height = h * 6.0 + tile_noise(6, 0.08) + np.where(pores, -0.6, 0)
gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) / 2
gy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) / 2
n = np.stack([-gx, gy, np.ones_like(gx)], -1)
n /= np.linalg.norm(n, axis=-1, keepdims=True)
Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save('textures/foundation_block_normal.png')
print('wrote textures/foundation_block.png and textures/foundation_block_normal.png', SIZE, 'px')
