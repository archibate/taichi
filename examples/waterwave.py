# vim: st=4 sts=4 sw=4 et

import taichi as ti
import numpy as np

ti.init(arch=ti.opengl)

light_color = 1
kappa = 2
gamma = 0.6
eta = 1.333
depth = 4
dx = 0.02
dt = 0.01
shape = 512, 512
pixels = ti.var(dt=ti.f32, shape=shape)
background = ti.var(dt=ti.f32, shape=shape)
position = ti.var(dt=ti.f32, shape=shape)
velocity = ti.var(dt=ti.f32, shape=shape)
acceleration = ti.var(dt=ti.f32, shape=shape)


@ti.kernel
def reset():
    for i, j in position:
        t = i // 16 + j // 16
        background[i, j] = (t * 0.5) % 1.0
        position[i, j] = 0
        velocity[i, j] = 0
        acceleration[i, j] = 0


@ti.func
def laplacian(i, j):
    return (-4 * position[i, j] + position[i, j - 1] +
            position[i, j + 1] + position[i + 1, j] +
            position[i - 1, j]) / (4 * dx ** 2)


@ti.func
def gradient(i, j):
    return ti.Vector([
        position[i + 1, j] - position[i - 1, j],
        position[i, j + 1] - position[i, j - 1]
    ]) * (0.5 / dx)


@ti.func
def take_linear(i, j):
    m, n = int(i), int(j)
    i, j = i - m, j - n
    ret = 0.0
    if 0 <= i < shape[0] and 0 <= i < shape[1]:
        ret = (i * j * background[m + 1, n + 1] +
              (1 - i) * j * background[m, n + 1] + i *
              (1 - j) * background[m + 1, n] + (1 - i) *
              (1 - j) * background[m, n])
    return ret


@ti.kernel
def touch_at(hurt: ti.f32, x: ti.f32, y: ti.f32):
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        r2 = ti.sqr(i - x) + ti.sqr(j - y)
        position[i, j] = position[i, j] + hurt * ti.exp(-0.02 * r2)


@ti.kernel
def update():
    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        acceleration[i, j] = kappa * laplacian(i, j) - gamma * velocity[i, j]

    for i, j in ti.ndrange((1, shape[0] - 1), (1, shape[1] - 1)):
        velocity[i, j] = velocity[i, j] + acceleration[i, j] * dt
        position[i, j] = position[i, j] + velocity[i, j] * dt


@ti.kernel
def paint():
    for i, j in pixels:
        g = gradient(i, j)
        # https://www.jianshu.com/p/66a40b06b436
        cos_i = 1 / ti.sqrt(1 + g.norm_sqr())
        cos_o = ti.sqrt(1 - (1 - ti.sqr(cos_i)) * (1 / eta ** 2))
        fr = pow(1 - cos_i, 5)
        coh = cos_o * depth
        g = g * coh
        k, l = g[0], g[1]
        color = take_linear(i + k, j + l)
        pixels[i, j] = (1 - fr) * color + fr * light_color


print("[Hint] click on the window to create wavelet")

reset()
gui = ti.GUI("Water Wave", shape)
for frame in range(100000):
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == 'r':
            reset()
        elif e.key == ti.GUI.LMB:
            x, y = e.pos
            touch_at(1, x * shape[0], y * shape[1])
    if 0:
        from random import randrange, random
        if frame % 8 == 0 and randrange(8) == 0:
            hurt = random() * 2
            x, y = randrange(8, shape[0] - 8), randrange(8, shape[1] - 8)
            touch_at(hurt, x, y)
    update()
    paint()
    gui.set_image(pixels)
    gui.show()
