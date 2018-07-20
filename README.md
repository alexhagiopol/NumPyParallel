## NumPyParallel

### Introduction
This demo project shows how to use Python multiprocessing to 
parallelize elementwise operations on NumPy arrays. To understand
why this is non-trivial, read about Python's [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock)
(GIL). 

I often get stuck in situations where I have to do a 
complex elementwise procedure on a NumPy array that is
difficult or impossible to express in vectorized form.
In MATLAB, parallelizing a `for` loop that operates in a 
non-vectorizable way was done easily using the `parfor` keyword. 
In this project, I show how to achieve equivalent behavior 
in Python - albeit with more code.

### Instructions
After reading `parallel_demo.py` to understand how to do the parallelization,
you can experiment with different combinations of array dimensions and
operation iterations to see the effect of the speedup. The syntax is
as follows:

    python3 parallel_demo.py #_iterations #_rows #_cols #_CPUs
    
The parameters with which I test single-CPU speed are...

    python3 parallel_demo.py 10000 500 500 1
    
The parameters with which I test multi-CPU speed are...

    python3 parallel_demo.py 10000 500 500 4
    
### Results
On my Intel i5-8350U CPU, doing 10000 elementwise operations on a 500x500
matrix with 1 CPU takes `0.803 seconds`. Increasing the CPU count to 4
reduces the time to `0.238 seconds`. Thus we get an approximately 4X
speedup when using 4X the cores. Victory!

### Note
Thermal throttling can drastically affect timing results especially
on mobile machines with weak cooling. Multicore performance can even be 
slower than single core performance if the CPU runs one core at very
high frequency and multiple cores at low frequency due to thermal
or power limitations.