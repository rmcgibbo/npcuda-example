import numpy as np
import numpy.testing as npt
import IPython as ip
from gpuadder import gpuadder

arr = np.array([1,2,2,2], dtype=np.int32)
adder = gpuadder.GPUAdder(arr)
adder.increment()
adder.retreive()

npt.assert_array_equal(arr, [2,3,3,3])

