import numpy as np
import taichi as ti


def imwrite(img, filename):
    img = img.to_numpy()
    assert len(img.shape) in [2, 3]
    resx, resy = img.shape[:2]
    if len(img.shape) == 3:
        comp = img.shape[2]
    else:
        comp = 1
    img = np.ascontiguousarray(img.swapaxes(0, 1)[::-1, :, :])
    ptr = img.ctypes.data
    ti.core.imwrite(filename, ptr, resx, resy, comp)


def imread(filename, channels=0):
    ptr, resx, resy, comp = ti.core.imread(filename, channels)
    img = np.ndarray(shape=(resy, resx, comp), dtype=np.uint8)
    img = np.ascontiguousarray(img)
    # TODO(archibate): Figure out how np.ndarray constructor works and replace:
    ti.core.C_memcpy(img.ctypes.data, ptr, resx * resy * comp)
    # Discussion: https://github.com/taichi-dev/taichi/issues/802
    return img.swapaxes(0, 1)[:, ::-1, :]


def imshow(img, winname='Taichi'):
    img = img.to_numpy()
    assert len(img.shape) in [2, 3]
    gui = ti.GUI(winname, res=img.shape[:2])
    while not gui.get_event(ti.GUI.ESCAPE):
        gui.set_image(img)
        gui.show()
