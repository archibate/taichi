import taichi as ti
import pytest


# Not really testable..
# Just making sure it does not crash
# Metal doesn't support ti.print() or 64-bit data
# So does OpenGL..
@pytest.mark.parametrize('dt', [ti.i32, ti.f32, ti.i64, ti.f64])
@ti.archs_excluding(ti.metal, ti.opengl)
def test_print(dt):
    @ti.kernel
    def func():
        print(ti.cast(1234.5, dt))

    func()


# TODO: As described by @k-ye above, what we want to ensure
#       is that, the content shows on console is *correct*.
@ti.archs_excluding(ti.metal, ti.opengl)
def test_multi_print():
    @ti.kernel
    def func(x: ti.i32, y: ti.f32):
        print(x, 1234.5, y)

    func(666, 233.3)


@ti.archs_excluding(ti.metal, ti.opengl)
def test_print_string():
    @ti.kernel
    def func(x: ti.i32, y: ti.f32):
        print('hello, world! %s %f',
              233)  # make sure `%` doesn't break vprintf
        print('cool', x, 'well', y)

    func(666, 233.3)
