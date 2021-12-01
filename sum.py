from numpy import arange, sqrt
from numba import jit
from time import perf_counter
import sys
import os

from multiprocessing import Pool

def getNumberProcs():
    print(os.cpu_count())

# @jit(nopython=True)
def getRowSum(row):
    """Returns sum of row.

    row: array of values (numpy array)
    """
    return sum([sqrt(x) for x in row])
    

def getSumAr(nrows, nvals, nprocs=8):
    """Returns list of sums.

    nrows: number of rows (pos int)
    nvals: number of values in each row (pos int)
    """
    rows = arange(nrows*nvals).reshape(nrows, nvals)

    rowSums = []

    with Pool(nprocs) as p:
        ret = p.map(getRowSum, rows)
    #for row in rows:
    #    rowSum = getRowSum(row)
    #    rowSums.append(rowSum)

    return ret


def writeSumAr(
        nrows=2880,
        nvals=99999,
        nprocs=36,
        outfile='rowSums',
        ):
    """Writes list of sums to file."""
    start = perf_counter()
    rowSums = getSumAr(nrows, nvals, nprocs)
    end = perf_counter()
    dt = end - start

    with open(f"{outfile}_r{nrows}_v{nvals}_p{nprocs}", 'w') as f:
#         for rowSum in rowSums:
#             f.write(f"{rowSum}\n")
        f.write(f"time = {dt:.5g} seconds\n")
    

if __name__ == '__main__':
    nprocs = int(sys.argv[1])
    writeSumAr(nprocs=nprocs)
    print(f"python detected {os.cpu_count()} cpu's")

#         nprocs='auto',
#     if nprocs=='auto':
#         nprocs = os.cpu_count()
#     writeSumAr()
