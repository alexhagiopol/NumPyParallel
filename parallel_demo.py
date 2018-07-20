import multiprocessing
import numpy as np
from multiprocessing import sharedctypes
import sys
import time


def worker_function(my_array_ctypes, start_index, end_index, shape, iterations):
    """
    Function that is parallelized to modify a single array in shared memory.
    :param my_array_ctypes: CTypes version of Numpy array
    :param start_index: beginning column of array segment that this worker instance will modify
    :param end_index: ending column of array segment that this worker instance will modify
    :param shape: shape of Numpy array
    :param iterations: number of times to perform the increment operation
    """
    # convert from shared memory ctypes array to numpy array that we can manipulate as normal
    my_array = np.ctypeslib.as_array(my_array_ctypes)
    my_array.shape = shape
    # do work in a for loop on the array
    for i in range(1, iterations):
        my_array[:, start_index:end_index] += 1


def main():
    """
    Main pipeline. Read this code to understand how to parallelize worker.
    """

    # accept user settings
    if not len(sys.argv) == 5:
        print("Incorrect usage. Example:\n"
              "python3 parallel_demo.py #_iterations #_rows #_cols #_CPUs \n"
              "Recommended defaults:\n"
              "python3 parallel_demo.py 5000 4096 4096 4")
        exit()
    iterations = sys.argv[1]
    rows = sys.argv[2]
    cols = sys.argv[3]
    CPUs = sys.argv[4]  # FYI: multiprocessing.cpu_count() returns total number of hyperthreads
    assert(1 <= CPUs <= 128)
    print("Starting multiprocessing."
          "\n# Iterations       =", iterations,
          "\nNumpy Array # Rows =", rows,
          "\nNumpy Array # Cols =", cols,
          "\n# CPUs             =", CPUs)

    # create array
    my_array = np.zeros((rows, cols))
    # convert this array to CTypes version in shared memory
    # reference: http://briansimulator.org/sharing-numpy-arrays-between-processes/
    size = my_array.size
    shape = my_array.shape
    my_array.shape = size
    my_array_ctypes = sharedctypes.RawArray('d', my_array)
    my_array = np.frombuffer(my_array_ctypes, dtype=np.float64, count=size)
    my_array.shape = shape

    # spawn multiple processes each of which will execute the worker function on a segment of my_array
    start_time = time.time()
    processes = []
    segment_size = my_array.shape // CPUs
    for i in range(0, CPUs):
        start_index = i * segment_size
        if i == CPUs - 1:  # ndarray will not divide evenly by # of CPUs. final segment contains remainder
            end_index = my_array.shape[1] + 1
        else:
            end_index = (i + 1) * segment_size
        process = multiprocessing.Process(target=worker_function, args=(my_array_ctypes, start_index, end_index, shape, iterations,))
        processes += [process]
        print("Spawning parallel process #", i, "operating on column", start_index, "thru", end_index)
    print("Processing. Check your CPU utilization ;)")

    # start all processes
    for process in processes:
        process.start()
    # wait for all to end before executing more code
    for process in processes:
        process.join()
    end_time = time.time()
    print("Finished job in", end_time - start_time, "seconds.")


if __name__ == "__main__":
    main()