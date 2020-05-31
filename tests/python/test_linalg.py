import taichi as ti
import numpy as np
from taichi import approx


@ti.all_archs
def test_basic_utils():
    a = ti.Vector(3, dt=ti.f32)
    b = ti.Vector(3, dt=ti.f32)
    abT = ti.Matrix(3, 3, dt=ti.f32)
    aNormalized = ti.Vector(3, dt=ti.f32)

    normA = ti.var(ti.f32)
    normSqrA = ti.var(ti.f32)

    @ti.layout
    def place():
        ti.root.place(a, b, abT, aNormalized, normA, normSqrA)

    @ti.kernel
    def init():
        a[None] = ti.Vector([1.0, 2.0, 3.0])
        b[None] = ti.Vector([4.0, 5.0, 6.0])
        abT[None] = ti.Matrix.outer_product(a, b)

        normA[None] = a.norm()
        normSqrA[None] = a.norm_sqr()

        aNormalized[None] = ti.Matrix.normalized(a)

    init()

    for i in range(3):
        for j in range(3):
            assert abT[None][i, j] == a[None][i] * b[None][j]

    sqrt14 = np.sqrt(14.0)
    invSqrt14 = 1.0 / sqrt14
    assert normSqrA[None] == 14.0
    assert normA[None] == approx(sqrt14)
    assert aNormalized[None][0] == approx(1.0 * invSqrt14)
    assert aNormalized[None][1] == approx(2.0 * invSqrt14)
    assert aNormalized[None][2] == approx(3.0 * invSqrt14)


@ti.all_archs
def test_cross():
    a = ti.Vector(3, dt=ti.f32)
    b = ti.Vector(3, dt=ti.f32)
    c = ti.Vector(3, dt=ti.f32)

    a2 = ti.Vector(2, dt=ti.f32)
    b2 = ti.Vector(2, dt=ti.f32)
    c2 = ti.var(dt=ti.f32)

    @ti.layout
    def place():
        ti.root.place(a, b, c, a2, b2, c2)

    @ti.kernel
    def init():
        a[None] = ti.Vector([1.0, 2.0, 3.0])
        b[None] = ti.Vector([4.0, 5.0, 6.0])
        c[None] = ti.Matrix.cross(a, b)

        a2[None] = ti.Vector([1.0, 2.0])
        b2[None] = ti.Vector([4.0, 5.0])
        c2[None] = ti.Matrix.cross(a2, b2)

    init()
    assert c[None][0] == -3.0
    assert c[None][1] == 6.0
    assert c[None][2] == -3.0
    assert c2[None] == -3.0


@ti.all_archs
def test_dot():
    a = ti.Vector(3, dt=ti.f32)
    b = ti.Vector(3, dt=ti.f32)
    c = ti.var(dt=ti.f32)

    a2 = ti.Vector(2, dt=ti.f32)
    b2 = ti.Vector(2, dt=ti.f32)
    c2 = ti.var(dt=ti.f32)

    @ti.layout
    def place():
        ti.root.place(a, b, c, a2, b2, c2)

    @ti.kernel
    def init():
        a[None] = ti.Vector([1.0, 2.0, 3.0])
        b[None] = ti.Vector([4.0, 5.0, 6.0])
        c[None] = a.dot(b)

        a2[None] = ti.Vector([1.0, 2.0])
        b2[None] = ti.Vector([4.0, 5.0])
        c2[None] = a2.dot(b2)

    init()
    assert c[None] == 32.0
    assert c2[None] == 14.0


@ti.all_archs
def test_transpose():
    dim = 3
    m = ti.Matrix(dim, dim, ti.f32)

    @ti.layout
    def place():
        ti.root.place(m)

    @ti.kernel
    def transpose():
        mat = m[None].transpose()
        m[None] = mat

    for i in range(dim):
        for j in range(dim):
            m(i, j)[None] = i * 2 + j * 7

    transpose()

    for i in range(dim):
        for j in range(dim):
            assert m(j, i)[None] == approx(i * 2 + j * 7)


def _test_polar_decomp(dim, dt):
    m = ti.Matrix(dim, dim, dt)
    r = ti.Matrix(dim, dim, dt)
    s = ti.Matrix(dim, dim, dt)
    I = ti.Matrix(dim, dim, dt)
    D = ti.Matrix(dim, dim, dt)

    @ti.layout
    def place():
        ti.root.place(m, r, s, I, D)

    @ti.kernel
    def polar():
        R, S = ti.polar_decompose(m[None], dt)
        r[None] = R
        s[None] = S
        m[None] = R @ S
        I[None] = R @ R.transpose()
        D[None] = S - S.transpose()

    def V(i, j):
        return i * 2 + j * 7 + int(i == j) * 3

    for i in range(dim):
        for j in range(dim):
            m(i, j)[None] = V(i, j)

    polar()

    tol = 5e-5 if dt == ti.f32 else 1e-12

    for i in range(dim):
        for j in range(dim):
            assert m(i, j)[None] == approx(V(i, j), abs=tol)
            assert I(i, j)[None] == approx(int(i == j), abs=tol)
            assert D(i, j)[None] == approx(0, abs=tol)


def test_polar_decomp():
    for dim in [2, 3]:
        for dt in [ti.f32, ti.f64]:

            @ti.all_archs_with(default_fp=dt)
            def wrapped():
                _test_polar_decomp(dim, dt)

            wrapped()


@ti.all_archs
def test_matrix():
    x = ti.Matrix(2, 2, dt=ti.i32)

    @ti.layout
    def xy():
        ti.root.dense(ti.i, 16).place(x)

    @ti.kernel
    def inc():
        for i in x(0, 0):
            delta = ti.Matrix([[3, 0], [0, 0]])
            x[i][1, 1] = x[i][0, 0] + 1
            x[i] = x[i] + delta
            x[i] += delta

    for i in range(10):
        x[i][0, 0] = i

    inc()

    for i in range(10):
        assert x[i][0, 0] == 6 + i
        assert x[i][1, 1] == 1 + i


@ti.all_archs
def _test_mat_inverse_size(n):
    m = ti.Matrix(n, n, dt=ti.f32, shape=())
    M = np.empty(shape=(n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            M[i, j] = i * j + i * 3 + j + 1 + int(i == j) * 4
    assert np.linalg.det(M) != 0

    m.from_numpy(M)

    @ti.kernel
    def invert():
        m[None] = m[None].inverse()

    invert()

    m_np = m.to_numpy(keep_dims=True)
    np.testing.assert_almost_equal(m_np, np.linalg.inv(M))


def test_mat_inverse():
    for n in range(1, 5):
        _test_mat_inverse_size(n)


@ti.all_archs
def test_unit_vectors():
    a = ti.Vector(3, dt=ti.i32, shape=3)

    @ti.kernel
    def fill():
        for i in ti.static(range(3)):
            a[i] = ti.Vector.unit(3, i)

    fill()

    for i in range(3):
        for j in range(3):
            assert a[i][j] == int(i == j)


@ti.all_archs
def test_init_matrix_from_vectors():
    m1 = ti.Matrix(3, 3, dt=ti.f32, shape=(3))
    m2 = ti.Matrix(3, 3, dt=ti.f32, shape=(3))
    m3 = ti.Matrix(3, 3, dt=ti.f32, shape=(3))
    m4 = ti.Matrix(3, 3, dt=ti.f32, shape=(3))

    @ti.kernel
    def fill():
        for i in range(3):
            a = ti.Vector([1.0, 4.0, 7.0])
            b = ti.Vector([2.0, 5.0, 8.0])
            c = ti.Vector([3.0, 6.0, 9.0])
            m1[i] = ti.Matrix(rows=[a, b, c])
            m2[i] = ti.Matrix(cols=[a, b, c])
            m3[i] = ti.Matrix(
                rows=[[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
            m4[i] = ti.Matrix(
                cols=[[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])

    fill()

    for j in range(3):
        for i in range(3):
            assert m1[0][i, j] == int(i + 3 * j + 1)
            assert m2[0][j, i] == int(i + 3 * j + 1)
            assert m3[0][i, j] == int(i + 3 * j + 1)
            assert m4[0][j, i] == int(i + 3 * j + 1)


@ti.all_archs
def test_any_all():
    a = ti.Matrix(2, 2, dt=ti.i32, shape=())
    b = ti.var(dt=ti.i32, shape=())

    @ti.kernel
    def func_any():
        b[None] = any(a[None])

    @ti.kernel
    def func_all():
        b[None] = all(a[None])

    for i in range(2):
        for j in range(2):
            a[None][0, 0] = i
            a[None][1, 0] = j
            a[None][1, 1] = i
            a[None][0, 1] = j

            func_any()
            if i == 1 or j == 1:
                assert b[None] == 1
            else:
                assert b[None] == 0

            func_all()
            if i == 1 and j == 1:
                assert b[None] == 1
            else:
                assert b[None] == 0
