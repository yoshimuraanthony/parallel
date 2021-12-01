from numpy import array, arange, zeros, sqrt, log10
from numpy import newaxis as na
from numba import jit
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import perf_counter, time, ctime
import sys
import os

# include date on outfile file names
from datetime import date
from shutil import copyfile

from multiprocessing import Pool

import logging

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    file_handler = logging.FileHandler('multiarg.log')
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
# file_handler.setFormatter(formatter)
# formatter = logging.Formatter('')

"""Scalability tests for multiprocess parallelization

dt1 = one-call time
optimal efficiency when dt1 = 1e-3 seconds
Efficiency drops off quickly when dt1 < 4e-5 seconds

more rows is more efficient on more procs
36 procs is best for ntot > 1e10
"""

@jit(nopython=True)
def getSum(start, end):
    """Returns sum of row.

    start, end: limits of range function (ints)
    """
    tot = 0
    for x in range(start, end):
        tot += sqrt(x)**2
    return tot
    

def getLims(nrows, nvals):
    """Returns rank 2 array of limits

    nrows: number of lim_a2 (pos int)
    nvals: number of values in each row (pos int)
    """
    lim_a2 = zeros((nrows, 2), dtype=int)
    lim_a2 += arange(nrows)[:,na]
    lim_a2[:,1] += nvals
    return lim_a2


def getSums(nrows, nvals, nprocs=8):
    """Returns array of sums."""
    lim_a2 = getLims(nrows, nvals)
    rowSums = []
    with Pool(nprocs) as p:
        sum_list = p.starmap(getSum, lim_a2)
    return sum_list


def get1Time(
        nvals = 10
        ):
    t0 = perf_counter()
    s = getSum(0, nvals)
    t1 = perf_counter()
    return t1 - t0


def getTime(
        nrows = 10,
        nvals = 10,
        nprocs = 2,
        ):
    """Returns calculation time."""
    t0 = perf_counter()
    sum_list = getSums(nrows, nvals, nprocs)
    t1 = perf_counter()
    return t1 - t0


def cycle(
        ncyc = 4,
        lcush = 2,
        ucush = 5,
        ntot = 1e4,
        nprocs = 4,
        ):
    """Cycles through various (nrows, nvals) pairs with constant nrow*nval."""
    logger.info(f'{ctime(time())}')
    ratio = ntot**(1/(ncyc+lcush+ucush))
    logger.info(f'nprocs={nprocs}, ratio={ratio:.6g}')
    dt1_list = []
    dt_list = []
    for n in range(lcush, ncyc+lcush):
        nrows = round(ratio**(n))
        nvals = round(ratio**(ncyc+lcush+ucush-n))

        # time single getSum call
        dt1 = get1Time(nvals)
        dt1 = get1Time(nvals)  # give jit a chance to compile
        dt1_list.append(dt1)

        # time nrows getSum calls
        dt = getTime(nrows, nvals, nprocs)
        dt_list.append(dt)
        logger.info(
                f'nrows={nrows}, nvals={nvals}, dt1={dt1:.6e}, dt={dt:1.6e}')

    return array(dt1_list), array(dt_list)


class Plot():
    """Class to plot exctitation probabilities."""
    def __init__(self,
            fig = None,
            ax = None,
            figsize = (6,6),
            ):
        """Intializes figure and axes."""
        mpl.style.use('figure')

        if fig==None or ax==None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = fig, ax

        # domain and range boundaries
        self.xmax = 0
        self.ymax = 0

    def plot(self,
            ncyc = 20,
            lcush = 15,
            ucush = 18,
            ntot = 1e9,
            nprocs = 10,
            color = None,
            ):
        """Plot probabilities."""
        x_ar, y_ar = cycle(ncyc, lcush, ucush, ntot, nprocs)
        self.ax.semilogx(x_ar, y_ar, label=f'nprocs={nprocs}')

    def decorate(self,
            xlim = 'auto',
            ylim = 'auto',
            xlabel = 'one-call time (s)',
            ylabel = 'total time (s)',
            title = None,
            grid = True,
            legend = True,
            ):
        """
        adds plot attributes
        ebounds: TEM energy range in keV (list of two floats)
        cbounds: cross section range in barn (list of two floats)
        """
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)

        self.ax.grid(grid)
        if legend:
            self.ax.legend()
        plt.tight_layout()

    def save(self,
            outfile = 'dtvsdt1.png',
            dest = '.',
            dpi = 300,
            transparent = False,
            writedate = True,
            ):
        """
        format and save cross section plot
        """
        if dest == '.':
            outpath = outfile
        else:
            outpath = '{}/{}'.format(dest, outfile)
        plt.savefig(outpath, dpi=dpi, transparent=transparent)

        # make copy with date in name
        if writedate:
            today = date.today()
            year = today.year - 2000
            month = today.month
            day = today.day
        
            name, ext = outpath.split('.')
            outpathCopy = '{}_{:02d}{:02d}{:02d}.{}'.format(name, year, month,
                    day, ext)
        
            copyfile(outpath, outpathCopy)

    def clear(self):
        plt.cla()

    def close(self):
        plt.close(self.fig)

    def copy(self):
        return ScatterPlot(fig=self.fig, ax=self.ax)

    def show(self):
        plt.show()


def plot(ntot=1e9, nprocs_list = [4,12,24,36]):
    maplot = Plot()
    for nprocs in nprocs_list:
        maplot.plot(ntot=ntot, nprocs=nprocs)
    maplot.decorate()
    maplot.save(outfile=f'dtvsdt1_n{ntot:.0e}.pdf', writedate=False)
    maplot.show()


#---------------------- SCRATCH -----------------------------------------

def write(
        nrows = 2880,
        nvals = 88888,
        nprocs = 2,
        outfile='ma',
        ):
    """Writes list of sums to file."""
    dt = getTime(nrows, nvals, nprocs)

    with open(f"{outfile}_r{nrows}_v{nvals}_p{nprocs}", 'w') as f:
        f.write(f"time = {dt:.5g} seconds\n")
    

if __name__ == '__main__':
    nprocs = int(sys.argv[1])
    write(nprocs=nprocs)
    print(f"python detected {os.cpu_count()} cpu's")


#         nprocs='auto',
#     if nprocs=='auto':
#         nprocs = os.cpu_count()
#     writeSumAr()


#         for rowSum in rowSums:
#             f.write(f"{rowSum}\n")

# def getNumberProcs():
#     print(os.cpu_count())

