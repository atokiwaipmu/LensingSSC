
import numpy as np

def unique_with_mpi(ipix, num_ranks, comm):
    # Split the array into chunks for each process
    chunks = np.array_split(ipix, num_ranks)
    
    # Each process computes unique elements on its chunk
    local_unique = np.unique(chunks[comm.rank])
    comm.barrier()
    
    # Gather all unique elements to root process
    all_unique = comm.gather(local_unique, root=0)
    
    if comm.rank == 0:
        # Root process computes the unique elements across all chunks
        all_unique = np.unique(np.concatenate(all_unique))
    else:
        all_unique = None
    
    comm.barrier()

    # Broadcast the final unique elements to all processes
    all_unique = comm.bcast(all_unique, root=0)
    
    # Each process finds the labels for its local array from the all_unique array
    local_labels = np.searchsorted(all_unique, chunks[comm.rank])

    comm.barrier()
    
    # Gather all local labels to root process
    global_labels = comm.gather(local_labels, root=0)

    if comm.rank == 0:
        # Root process concatenates the gathered labels to form the global labels array
        global_labels = np.concatenate(global_labels)
    else:
        global_labels = None

    comm.barrier()

    # Broadcast the global labels to all processes
    global_labels = comm.bcast(global_labels, root=0)
    
    return all_unique, global_labels