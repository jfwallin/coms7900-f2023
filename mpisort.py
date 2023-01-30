# from the command line, run with
# mpiexec -n 4 python -m mpi4py mpitest.py

from mpi4py import MPI
import sys
import numpy as np

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

#sys.stdout.write(
#    "Hello, World! I am process %d of %d on %s.\n" % (rank, size, name))


# geneate a list of n random numbers between 0 and 1
n = 2000
localArray = np.random.random(n)

# sort the local array
localArray.sort()

# find the start and end of the array based on the range of 0 to 1
# and the number of processor
arrayBoundaries = []


# loop over the processors
bufferSize = n/2

arecv = np.zeros([size, int(bufferSize)], dtype='f') 
for i in range(0, size):
    #  find the elements with in range
    xmin = float(i)/size
    xmax = float(i+1)/size
    asend = localArray[localArray >= xmin]
    asend = asend[asend < xmax]
    
    # copy this data into the send buffer
    sendbuf = np.zeros(int(bufferSize), dtype='f')
    sendbuf[0:len(asend)] = asend
    MPI.COMM_WORLD.Gather(sendbuf, arecv, root=i )

# compress the data into a local array from the gather 
# ignore the zeros
newLocal = []
for i in range(0, size):
    for j in range(0, int(bufferSize)):
        if arecv[i,j] != 0:
            newLocal.append(arecv[i,j])

newLocal.sort()
#print("rank = ", rank, "local size = ", len(newLocal), "localMin = ", newLocal[0], "localMax = ", newLocal[-1])
localData = np.array( [len(newLocal), newLocal[0], newLocal[-1]] , dtype='f')
globalData = np.zeros([size, 3], dtype='f')

MPI.COMM_WORLD.Gather(localData, globalData, root=0)
if rank == 0:
    #print(globalData)
    for i in range(0, size):
        print("node = ", i, " node data size = ", globalData[i,0], "node Min = ", globalData[i,1], "node Max = ", globalData[i,2])

MPI.COMM_WORLD.Barrier()

    