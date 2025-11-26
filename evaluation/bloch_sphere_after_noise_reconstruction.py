import struct
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

QALU_SCALE = 1 << 29
QALU_MASK = 0x3FFFFFFF

def fixed_to_double(val):
    if val & (1 << 29):
        val |= ~QALU_MASK
    return val / QALU_SCALE

def extract_components(qword):
    tag = (qword >> 60) & 0xF
    real_fixed = (qword >> 30) & QALU_MASK
    imag_fixed = qword & QALU_MASK
    real = fixed_to_double(real_fixed)
    imag = fixed_to_double(imag_fixed)
    mag2 = real*real + imag*imag
    return real, imag, mag2, tag



clean_hex      = 0x5a8279980000000
noisy_hex      = 0x5a827998b504f33
normalized_hex = 0x727c9718e4f92e3

clean = extract_components(clean_hex)
noisy = extract_components(noisy_hex)
norm  = extract_components(normalized_hex)



def bloch_coordinates_from_alpha(real, imag):
    mag2 = real*real + imag*imag
    mag = np.sqrt(mag2)

    mag = min(max(mag, 0.0), 1.0)
    theta = 2 * np.arccos(mag)
    phi = np.arctan2(imag, real)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z


clean_xyz = bloch_coordinates_from_alpha(clean[0], clean[1])
noisy_xyz = bloch_coordinates_from_alpha(noisy[0], noisy[1])
norm_xyz  = bloch_coordinates_from_alpha(norm[0],  norm[1])


fig = plt.figure(figsize=(11,11))
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2*np.pi, 200)
v = np.linspace(0, np.pi, 200)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))

ax.plot_surface(xs, ys, zs,
                rstride=5, cstride=5,
                facecolors=plt.cm.coolwarm((zs+1)/2),
                linewidth=0, antialiased=True, alpha=0.5)

ax.quiver(0,0,0, *clean_xyz, color='green', linewidth=3, label="Clean α", zorder=5)
ax.quiver(0,0,0, *noisy_xyz, color='red', linewidth=3, label="Noisy α", zorder=10)
ax.quiver(0,0,0, *norm_xyz,  color='blue', linewidth=4, label="Normalized α", zorder=20)

ax.set_title("3D Bloch Sphere — Clean → Noisy → Normalized", fontsize=16)
ax.set_xlim([-1.2,1.2])
ax.set_ylim([-1.2,1.2])
ax.set_zlim([-1.2,1.2])
ax.set_box_aspect([1,1,1])

ax.legend(loc='upper left')


text = (
    f"Clean α:      real={clean[0]:.6f}, imag={clean[1]:.6f}, |α|²={clean[2]:.6f}, tag={clean[3]}\n"
    f"Noisy α:      real={noisy[0]:.6f}, imag={noisy[1]:.6f}, |α|²={noisy[2]:.6f}, tag={noisy[3]}\n"
    f"Normalized α: real={norm[0]:.6f},  imag={norm[1]:.6f},  |α|²={norm[2]:.6f}, tag={norm[3]}"
)

fig.text(
    0.5, -0.05,
    text,
    ha='center', va='top', fontsize=10,
    bbox=dict(facecolor='white', edgecolor='black', alpha=0.9)
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.20)


for angle in range(0, 360, 2):
    ax.view_init(elev=25, azim=angle)
    plt.pause(0.01)

plt.show()
