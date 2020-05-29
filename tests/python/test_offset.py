import taichi as ti


@ti.all_archs
def test_accessor():
    a = ti.var(dt=ti.i32)

    ti.root.dense(ti.ijkl, 168).place(a, offset=(10-64, 2048, 2100, 2200))

    a[1029, 2100, 2200, 2300] = 1
    assert a[1029, 2100, 2200, 2300] == 1


@ti.all_archs
def test_struct_for_huge_offsets():
    a = ti.var(dt=ti.i32)

    offset = 10-64, 2048, 2100, 2200
    ti.root.dense(ti.ijkl, 4).place(a, offset=offset)

    @ti.kernel
    def test():
        for i, j, k, l in a:
            a[i, j, k, l] = i + j * 10 + k * 100 + l * 1000

    test()

    for i in range(offset[0], offset[0] + 4):
        for j in range(offset[1], offset[1] + 4):
            for k in range(offset[2], offset[2] + 4):
                for l in range(offset[3], offset[3] + 4):
                    assert a[i, j, k, l] == i + j * 10 + k * 100 + l * 1000


@ti.all_archs
def test_struct_for_negative():
    a = ti.var(dt=ti.i32)

    offset = 16, -16
    ti.root.dense(ti.ij, 32).place(a, offset=offset)

    @ti.kernel
    def test():
        for i, j in a:
            a[i, j] = i + j * 10

    test()

    for i in range(16, 48):
        for j in range(-16, 16):
            assert a[i, j] == i + j * 10

@ti.all_archs
def test_offset_wrapper_for_var():
    # for ti.var
    a = ti.var(dt=ti.i32, shape = 16, offset = -48)
    a = ti.var(dt=ti.i32, shape = (16,), offset = (16,))
    a = ti.var(dt=ti.i32, shape = (16, 64), offset = (-16, -64))
    a = ti.var(dt=ti.i32, shape = (16, 64), offset = None)

    # illegal cases
    try:
        a = ti.var(dt=ti.float32, shape = 3, offset = (3, 4))
        a = ti.var(dt=ti.float32, shape = None, offset = (3, 4))
    except AssertionError as e:
        ti.info("Oops! the offset and shape should keep consistent when ti.var is initialized")
        print(e)

@ti.all_archs
def test_offset_wrapper_for_matrix():
    # for ti.matrix, wrapper for key offset 
    a = ti.Matrix(dt=ti.i32, shape = (32, 16, 8), offset = (-8, -16, -32))
    a = ti.Matrix(dt=ti.i32, shape = (32, 16, 8), offset = None)

    # illegal cases
    try : 
        a = ti.Matrix(dt=ti.i32, shape = (32, 16, 8), offset = (32, 16))
        a = ti.Matrix(dt=ti.i32, shape = None, offset = (32, 16))
    except AssertionError as e:
        ti.info("Oops! the offset and shape should keep consistent when ti.Matrix is initialized")
        print(e)