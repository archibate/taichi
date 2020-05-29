.. _offset:

========
Coordinate offsets
========
- A Taichi var or matrix can be defined with an coordinate offset. The offset will move the bound of a tensor along with specified axes so that users can visit it with an offset index. This feature can be useful especially in physical simulation.
- At this moment, offset feature is only available for ti.var and ti.Matrix.
- For example, A matrix shaped (32, 32) with offset (-16, -32) can be defined like this:

.. code-block:: python

    a = ti.Matrix(dt = ti.float32, shape = (32, 32), offset = (-16, 32))

- then image this matrix is put on a map, the 4 corners of it can be accessed in this way:

.. code-block:: python

    a[-16, 32]  # lower left corner
    a[16, 32]   # lower right corner
    a[-16, 64]  # upper left corner
    a[16, 64]   # upper right corner


.. note:: The dims of shape should **keep consistency** with the offset. Otherwise, a ValueError will be thrown out.

.. code-block:: python

    a = ti.Matrix(dt = ti.float32, shape = (32,), offset = (-16, ))     # true
    a = ti.Matrix(dt = ti.float32, shape = None, offset = (32,))        # wrong
    a = ti.Matrix(dt = ti.float32, shape = (32, 32), offset = (-16, ))  # wrong
    a = ti.var(dt = ti.int32, shape = 16, offset = -16)                 # true
    a = ti.var(dt = ti.int32, shape = None, offset = -16)               # wrong
    a = ti.var(dt = ti.int32, shape = (16, 32), offset = -16)           # wrong
