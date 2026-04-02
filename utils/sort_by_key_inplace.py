"""
Zero-allocation in-place indirect sort using Numba.
Based on Numba's internal quicksort implementation.
"""
import collections
from numba import njit
from numba.core import types
from numba.core.extending import register_jitable

QuicksortImplementation = collections.namedtuple(
    'QuicksortImplementation',
    ('compile', 'partition', 'insertion_sort', 'run_quicksort')
)

Partition = collections.namedtuple('Partition', ('start', 'stop'))
SMALL_QUICKSORT = 15
MAX_STACK = 100

def _make_inplace_argsort_impl(wrap, lt=None):
    intp = types.intp
    zero = intp(0)

    def default_lt(a, b):
        return a < b

    LT = wrap(lt if lt is not None else default_lt)

    @wrap
    def GET(A, idx):
        return A[idx]

    @wrap
    def insertion_sort(A, R, low, high):
        assert low >= 0
        if high <= low:
            return

        for i in range(low + 1, high + 1):
            k = R[i]
            v = GET(A, k)
            j = i
            while j > low and LT(v, GET(A, R[j - 1])):
                R[j] = R[j - 1]
                j -= 1
            R[j] = k

    @wrap
    def partition(A, R, low, high):
        assert low >= 0
        assert high > low

        mid = (low + high) >> 1
        
        if LT(GET(A, R[mid]), GET(A, R[low])):
            R[low], R[mid] = R[mid], R[low]
        if LT(GET(A, R[high]), GET(A, R[mid])):
            R[high], R[mid] = R[mid], R[high]
        if LT(GET(A, R[mid]), GET(A, R[low])):
            R[low], R[mid] = R[mid], R[low]
        pivot = GET(A, R[mid])

        R[high], R[mid] = R[mid], R[high]
        i = low
        j = high - 1
        while True:
            while i < high and LT(GET(A, R[i]), pivot):
                i += 1
            while j >= low and LT(pivot, GET(A, R[j])):
                j -= 1
            if i >= j:
                break
            R[i], R[j] = R[j], R[i]
            i += 1
            j -= 1
            
        R[i], R[high] = R[high], R[i]
        return i

    @wrap
    def run_quicksort(A, R):
        if len(R) < 2:
            return R

        stack = [Partition(zero, zero)] * MAX_STACK
        stack[0] = Partition(zero, len(R) - 1)
        n = 1

        while n > 0:
            n -= 1
            low, high = stack[n]
            while high - low >= SMALL_QUICKSORT:
                assert n < MAX_STACK
                i = partition(A, R, low, high)
                
                if high - i > i - low:
                    if high > i:
                        stack[n] = Partition(i + 1, high)
                        n += 1
                    high = i - 1
                else:
                    if i > low:
                        stack[n] = Partition(low, i - 1)
                        n += 1
                    low = i + 1

            insertion_sort(A, R, low, high)

        return R

    return QuicksortImplementation(wrap, partition, insertion_sort, run_quicksort)

# --- Module-Level Compilation ---
# 1. Generate the implementation (standard Python space)
_impl = _make_inplace_argsort_impl(lambda f: register_jitable(f))

# 2. Extract the jitable function to a bare global variable 
# so Numba doesn't have to do a namedtuple attribute lookup at runtime.
_run_quicksort_jitable = _impl.run_quicksort

@njit
def sort_by_key_inplace(indices, a):
    """
    Sorts the `indices` array in-place using the values in array `a` as keys.
    Operates with strictly zero memory allocation.
    
    Parameters:
    indices (numpy.ndarray): The array of integer indices to sort in-place.
    a (numpy.ndarray): The reference array containing values to sort by.
    """
    # 3. Call the bare function directly! 
    # Notice we removed the "_impl." prefix here.
    _run_quicksort_jitable(a, indices)