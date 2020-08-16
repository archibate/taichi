import taichi as ti
import numpy as np


@ti.data_oriented
class MGPCG:
    def __init__(self, N=128, n_mg_levels=4, dim=3):
        # grid parameters
        self.use_multigrid = True

        self.N = N

        self.n_mg_levels = n_mg_levels
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 50
        self.dim = dim

        self.N_ext = self.N // 2  # number of ext cells set so that that total grid size is still power of 2
        self.N_tot = 2 * self.N

        # setup sparse simulation data arrays
        self.r = [ti.field(float) for _ in range(self.n_mg_levels)]  # residual
        self.z = [ti.field(float) for _ in range(self.n_mg_levels)]  # M^-1 self.r
        self.x = ti.field(float)  # solution
        self.p = ti.field(float)  # conjugate gradient
        self.Ap = ti.field(float)  # matrix-vector product
        self.alpha = ti.field(float)  # step size
        self.beta = ti.field(float)  # step size
        self.sum = ti.field(float)  # storage for reductions

        indices = [ti.i, ti.ij, ti.ijk][self.dim - 1]
        self.grid = ti.root.pointer(indices, [self.N_tot // 4]).dense(indices, 4)
        self.grid.place(self.x, self.p, self.Ap)

        for l in range(self.n_mg_levels):
            self.grid = ti.root.pointer(indices, self.N_tot // (4 * 2**l)).dense(indices, 4)
            self.grid.place(self.r[l], self.z[l])

        ti.root.place(self.alpha, self.beta, self.sum)

    @ti.func
    def neighbor_sum(self, x, I):
        ret = 0.0
        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            ret += x[I + offset] + x[I - offset]
        return ret

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            self.Ap[I] = (2 * self.dim) * self.p[I] - self.neighbor_sum(
                self.p, I)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            res = self.r[l][I] - (2.0 * self.dim * self.z[l][I] -
                                  self.neighbor_sum(self.z[l], I))
            self.r[l + 1][I // 2] += res * 0.5

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] = self.z[l + 1][I // 2]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase:
                self.z[l][I] = (self.r[l][I] + self.neighbor_sum(
                    self.z[l], I)) / (2.0 * self.dim)

    def apply_preconditioner(self):
        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing << l):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.restrict(l)

        for i in range(self.bottom_smoothing):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing << l):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def per_step_callback(self, i, rTr):
        pass

    def solve(self, steps):
        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]

        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p
        if self.use_multigrid:
            self.apply_preconditioner()
        else:
            self.z[0].copy_from(self.r[0])

        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        # CG
        for i in range(steps):
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + 1e-12)

            # self.x = self.x + self.alpha self.p
            self.update_x()

            # self.r = self.r - self.alpha self.Ap
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]
            if rTr < initial_rTr * 1e-12:
                break

            # self.z = M^-1 self.r
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + 1e-12)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            self.per_step_callback(i, rTr)


@ti.data_oriented
class MGPCGTest1(MGPCG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_gui = 512  # gui resolution
        self.pixels = ti.field(float,
                             (self.N_gui, self.N_gui))  # image buffer

    def run(self):
        import time
        t = time.time()

        self.gui = ti.GUI("Multigrid Preconditioned Conjugate Gradients",
                     (self.N_gui, self.N_gui))

        self.init()
        self.solve(400)

        ti.kernel_profiler_print()

        print(f'Solver time: {time.time() - t:.3f} s')
        ti.core.print_profile_info()
        ti.core.print_stat()

    def per_step_callback(self, i, rTr):
        print(f'iter {i}, residual={rTr}')
        self.paint()
        self.gui.set_image(self.pixels)
        self.gui.show()

    @ti.kernel
    def init(self):
        for I in ti.grouped(
                ti.ndrange(*((self.N_ext, self.N_tot - self.N_ext), ) * self.dim)):
            self.r[0][I] = 5.0
            for k in ti.static(range(self.dim)):
                self.r[0][I] *= ti.cos(2.0 * np.pi * (I[k] - self.N_ext) *
                                       5.0 / self.N_tot)
            self.z[0][I] = 0.0
            self.Ap[I] = 0.0
            self.p[I] = 0.0
            self.x[I] = 0.0

    @ti.kernel
    def paint(self):
        if ti.static(self.dim == 3):
            kk = self.N_tot * 3 // 8
            for i, j in self.pixels:
                ii = int(i * self.N / self.N_gui) + self.N_ext
                jj = int(j * self.N / self.N_gui) + self.N_ext
                self.pixels[i, j] = self.x[ii, jj, kk] / self.N_tot


@ti.data_oriented
class MGPCGTest2(MGPCG):
    def __init__(self):
        super().__init__(dim=2, N=128)

        self.N_gui = 512  # gui resolution
        self.bound = ti.field(float, (self.N, ) * self.dim)  # boundary conditions
        self.pixels = ti.field(float, (self.N_gui, self.N_gui))  # image buffer

    def run(self):
        self.gui = ti.GUI("Multigrid Preconditioned Conjugate Gradients",
                     (self.N_gui, self.N_gui))

        while self.gui.running and not self.gui.get_event(self.gui.ESCAPE):
            mx, my = self.gui.get_cursor_pos()
            self.touch(mx, my)
            self.init()
            self.solve(5)
            self.paint()
            self.gui.set_image(self.pixels)
            self.gui.show()

    @ti.kernel
    def init(self):
        for I in ti.grouped(ti.ndrange(*((self.N_ext, self.N_tot - self.N_ext), ) * self.dim)):
            self.r[0][I] = self.bound[I - self.N_ext]
            self.z[0][I] = 0.0
            self.Ap[I] = 0.0
            self.p[I] = 0.0
            self.x[I] = 0.0

    @ti.kernel
    def touch(self, x: float, y: float):
        for I in ti.grouped(self.bound):
            self.bound[I] = 4.0 * ti.exp(-400.0 * (I / self.N - ti.Vector([x, y])).norm_sqr())

    @ti.kernel
    def paint(self):
        if ti.static(self.dim == 2):
            for i, j in self.pixels:
                ii = int(i * self.N / self.N_gui) + self.N_ext
                jj = int(j * self.N / self.N_gui) + self.N_ext
                self.pixels[i, j] = self.x[ii, jj] / self.N_tot


if __name__ == '__main__':
    ti.init(default_fp=ti.f32, arch=ti.cpu, kernel_profiler=True)
    solver = MGPCGTest2()
    solver.run()
