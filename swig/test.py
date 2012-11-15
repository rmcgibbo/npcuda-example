import numpy as np
import numpy.testing as npt
import IPython as ip
import gpuadder

def test():
    arr = np.array([1,2,2,2], dtype=np.int32)
    adder = gpuadder.GPUAdder(arr)
    adder.increment()
    adder.retreive()

    results2 = np.empty_like(arr)
    adder.retreive_to(results2)
    
    npt.assert_array_equal(arr, [2,3,3,3])
    npt.assert_array_equal(results2, [2,3,3,3])
