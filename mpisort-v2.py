# from the command line, run with
# mpiexec -n 4 python -m mpi4py mpitest.py

from mpi4py import MPI
import sys
import numpy as np

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

# geneate a list of n random numbers between 0 and 1
n = 100
localArray = np.random.random(n)

# sort the local array
localArray.sort()

# find all the array boundaries for each node after the sort
arrayBounds = np.zeros([size, 2], dtype='f')
for i in range(0, size):
    #  find the elements with in range
    arrayBounds[i, :] = [float(i)/size, float(i+1)/size]

# We need to allocate some buffer space to receive the messages.
# To do this, we need to find the largest message that will be sent
# on the local node.
maxSize = 0
for i in range(0, size):
    #  find the elements with in range
    asend = localArray[(localArray >= arrayBounds[i, 0]) &
                       (localArray < arrayBounds[i, 1])]
    maxSize = max(maxSize, len(asend))

# Now we need to find the largest message size across all the nodess
sizeBuffer = np.zeros(size, dtype='i')
mySize = np.array(maxSize, dtype='i')
MPI.COMM_WORLD.Allgather(mySize, sizeBuffer)
bufferSize = int(np.max(sizeBuffer))

# allocate space to receive all the data
arecv = np.zeros([size, int(bufferSize)], dtype='f')

# loop over all the nodes and send the data to the correct node
# using gather
for i in range(0, size):
    #  find the elements with in range
    asend = localArray[(localArray >= arrayBounds[i, 0]) &
                       (localArray < arrayBounds[i, 1])]
    print("sending to:", i, "from:", rank,  "len=",
          len(asend), "buffer size=", bufferSize)

    # copy this data into the send buffer
    sendbuf = np.zeros(int(bufferSize), dtype='f')
    sendbuf[0:len(asend)] = asend
    MPI.COMM_WORLD.Gather(sendbuf, arecv, root=i)

# compress the data into a local array from the gather
# ignore the zeros - note - zeros are used as markers
# to indicate the end of the data
newLocal = []
for i in range(0, size):
    for j in range(0, int(bufferSize)):
        if arecv[i, j] != 0:
            newLocal.append(arecv[i, j])

# gather the data from all the processors
# to show the bounds of the data on each rank
newLocal.sort()
localData = np.array([len(newLocal), newLocal[0], newLocal[-1]], dtype='f')
globalData = np.zeros([size, 3], dtype='f')

MPI.COMM_WORLD.Gather(localData, globalData, root=0)
if rank == 0:
    # print(globalData)
    sum = 0
    for i in range(0, size):
        print("node %3d  #=%5d     min theoretical/actual % 8.6f / % 8.6f     max theoretical/actual = % 8.6f / % 8.6f" %
              (i, int(globalData[i, 0]), arrayBounds[i, 0], globalData[i, 1], arrayBounds[i, 1], globalData[i, 2]))
        sum += int(globalData[i, 0])
    print("total number of elements = ", sum)
    print("initial number of particles per node = ", n)
    print("total number of nodes = ", size)
    print("total number initial was ", n*size)
