
from nbodykit.transform import ConcatenateSources, CartesianToEquatorial

def read_range(cat, amin, amax):
    """ Read a portion of the lightcone between two red shift ranges

        The lightcone from FastPM is sorted in Aemit and an index is built.
        So we make use of that.

        CrowCanyon is z > 0; We paste the mirror image to form a full sky.
    """
    edges = cat.attrs['aemitIndex.edges']
    offsets = cat.attrs['aemitIndex.offset']
    start, end = edges.searchsorted([amin, amax])
    if cat.comm.rank == 0:
        cat.logger.info("Range of index is %d to %d" %(( start + 1, end + 1)))
    start = offsets[start + 1]
    end = offsets[end + 1]
    cat =  cat.query_range(start, end)
    cat1 = cat.copy()
    cat1['Position'] = cat1['Position'] * [1, 1, -1.]
    cat3 = ConcatenateSources(cat, cat1)
    if cat1.csize > 0:
        cat3['RA'], cat3['DEC'] = CartesianToEquatorial(cat3['Position'], frame='galactic')
    else:
        cat3['RA'] = 0
        cat3['DEC'] = 0
    return cat3