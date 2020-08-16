import taichi as ti
from mgpcg_advanced import MGPCG
import numpy as np

ti.init(arch=ti.cpu)


@ti.data_oriented
class MGPCGTest(MGPCG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_gui = 512  # gui resolution
        self.bound = ti.field(float, (self.N, ) * self.dim)
        self.pixels = ti.field(float,
                             (self.N_gui, self.N_gui))  # image buffer

    def run(self):
        import time
        t = time.time()

        self.gui = ti.GUI("Multigrid Preconditioned Conjugate Gradients",
                     (self.N_gui, self.N_gui))
        self.gui.fps_limit = 20

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
        import math
        for I in ti.grouped(self.bound):
            self.bound[I] = 0.3
            for k in ti.static(range(self.dim)):
                #self.bound[I] *= ti.sin(2.0 * math.pi * (I[k] - self.N_ext) * 4.0 / self.N_tot)
                self.bound[I] *= ti.exp(-10.0 * ((I[k]) / self.N_tot)**2)
        for I in ti.grouped(ti.ndrange(*((self.N_ext, self.N_tot - self.N_ext), ) * self.dim)):
            self.r[0][I] = self.bound[I - self.N_ext]
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


if __name__ == '__main__':
    ti.init(default_fp=ti.f32, arch=ti.cpu, kernel_profiler=True)
    solver = MGPCGTest()
    solver.run()
