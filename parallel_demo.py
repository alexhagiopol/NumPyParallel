import multiprocessing
import numpy as np
from multiprocessing import sharedctypes
import sys
import time


def worker_function(my_array_ctypes, shape, iterations):
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
        my_array += 1


def main():
    """
    Main pipeline. Read this code to understand how to parallelize worker function.
    """

    # accept user settings
    if not len(sys.argv) == 5:
        print("Incorrect usage. Example:\n"
              "python3 parallel_demo.py #_iterations #_rows #_cols #_CPUs \n"
              "Recommended defaults:\n"
              "python3 parallel_demo.py 5000 10 10 4")
        exit()
    iterations = int(sys.argv[1])
    rows = int(sys.argv[2])
    cols = int(sys.argv[3])
    CPUs = int(sys.argv[4])  # FYI: multiprocessing.cpu_count() returns total number of hyperthreads
    assert(1 <= CPUs)
    print("Starting multiprocessing."
          "\n# Iterations       =", iterations,
          "\nNumpy Array # Rows =", rows,
          "\nNumpy Array # Cols =", cols,
          "\n# CPUs             =", CPUs)

    # create array
    my_array = np.zeros((rows, cols))
    print("Initial array looks like...\n", my_array)

    # spawn multiple processes each of which will execute the worker function on a segment of my_array
    start_time = time.time()
    processes = []
    segment_arrays = []
    segment_size = my_array.shape[1] // CPUs
    for i in range(0, CPUs):
        start_index = i * segment_size
        if i == CPUs - 1:  # ndarray will not divide evenly by # of CPUs. final segment contains remainder
            end_index = my_array.shape[1] + 1
        else:
            end_index = (i + 1) * segment_size

        # convert this array segment to CTypes version in shared memory
        # reference: http://briansimulator.org/sharing-numpy-arrays-between-processes/
        segment_array = np.copy(my_array[:, start_index: end_index])
        size = segment_array.size
        shape = segment_array.shape
        segment_array.shape = size
        segment_array_ctypes = sharedctypes.RawArray('d', segment_array)
        segment_array = np.frombuffer(segment_array_ctypes, dtype=np.float64, count=size)
        segment_array.shape = shape

        # define inputs of worker function
        process = multiprocessing.Process(target=worker_function, args=(segment_array_ctypes, shape, iterations,))
        processes += [process]
        segment_arrays += [segment_array]
        print("Spawning parallel process #", i, "operating on column", start_index, "thru", end_index)
    print("Processing. Check your CPU utilization ;)")

    # start all processes
    for process in processes:
        process.start()
    # wait for all to end before executing more code
    for process in processes:
        process.join()

    # combine the results into a single array
    for i in range(0, CPUs):
        start_index = i * segment_size
        if i == CPUs - 1:  # ndarray will not divide evenly by # of CPUs. final segment contains remainder
            end_index = my_array.shape[1] + 1
        else:
            end_index = (i + 1) * segment_size
        my_array[:, start_index:end_index] = segment_arrays[i]

    end_time = time.time()
    print("Final array looks like...\n", my_array)
    print("Finished job in", end_time - start_time, "seconds.")


if __name__ == "__main__":
    main()
