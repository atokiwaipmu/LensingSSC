
import gc
import numpy
from nbodykit.utils import DistributedArray

def weighted_map(ipix, npix, weights, localsize, comm):
    ipix, labels = numpy.unique(ipix, return_inverse=True)
    N = numpy.bincount(labels)
    weights = numpy.bincount(labels, weights)

    del labels
 
    pairs = numpy.empty(len(ipix) + 1, dtype=[('ipix', 'i4'), ('N', 'i4'), ('weights', 'f8') ])
    pairs['ipix'][:-1] = ipix
    pairs['weights'][:-1] = weights
    pairs['N'][:-1] = N

    pairs['ipix'][-1] = npix - 1 # trick to make sure the final length is correct.
    pairs['weights'][-1] = 0
    pairs['N'][-1] = 0

    disa = DistributedArray(pairs, comm=comm)
    disa.sort('ipix')

    w = disa['ipix'].bincount(weights=disa['weights'].local, local=False, shared_edges=False)
    N = disa['ipix'].bincount(weights=disa['N'].local, local=False, shared_edges=False)

    del disa
    gc.collect()

    if npix - w.cshape[0] != 0:
        if comm.rank == 0:
            print('padding -- this shouldnt have occured ', npix, w.cshape)
        # pad with zeros, since the last few bins can be empty.
        ipadding = DistributedArray.cempty((npix - w.cshape[0],), dtype='i4', comm=comm)
        fpadding = DistributedArray.cempty((npix - w.cshape[0],), dtype='f8', comm=comm)

        fpadding.local[:] = 0
        ipadding.local[:] = 0

        w = DistributedArray.concat(w, fpadding)
        N = DistributedArray.concat(N, ipadding)

    w = DistributedArray.concat(w, localsize=localsize)
    N = DistributedArray.concat(N, localsize=localsize)

    return w.local, N.local

def modified_weighted_map(local_ipix, local_weights, npix):
    # Get the unique indices and the index each value belongs to
    local_unique_indices, local_index_counts = numpy.unique(local_ipix, return_inverse=True)

    # Count the occurrences of each index using np.bincount
    local_counts = numpy.bincount(local_index_counts) # len(local_counts) = len(local_unique_indices)
    local_weights = numpy.bincount(local_index_counts, weights=local_weights)

    pairs = numpy.empty(len(local_unique_indices) + 1, dtype=[('ipix', 'i4'), ('N', 'i4'), ('weights', 'f8') ])
    pairs['ipix'][:-1] = local_unique_indices
    pairs['weights'][:-1] = local_weights
    pairs['N'][:-1] =local_counts

    pairs['ipix'][-1] = npix - 1 # trick to make sure the final length is correct.
    pairs['weights'][-1] = 0
    pairs['N'][-1] = 0

    w = numpy.bincount(pairs['ipix'],weights=pairs['weights']) # len(w) = npix
    N = numpy.bincount(pairs['ipix'],weights=pairs['N'])

    return w, N