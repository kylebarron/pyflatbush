# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

# Set this to true to enable profiling hooks
# Use with, e.g.
# ../env/bin/python -m cProfile -s tottime bench.py > stats.txt
# cython: profile=False

import numpy as np

cimport numpy as np
from cpython.array cimport array
from cython cimport cdivision
from libc.math cimport ceil, floor
from numpy.math cimport INFINITY


# serialized format version
cdef unsigned short VERSION = 3


cdef class Flatbush:
    cdef readonly unsigned int numItems
    cdef readonly unsigned int nodeSize
    cdef readonly array _levelBounds
    cdef readonly unsigned int _pos
    cdef readonly double minX
    cdef readonly double minY
    cdef readonly double maxX
    cdef readonly double maxY

    cdef readonly bytearray data
    cdef readonly double [:] _boxes
    cdef readonly unsigned int [:] _indices

    @classmethod
    def from_buffer(cls, bytearray data):
        cdef char versionAndType = data[1]

        if data[0] != 0xfb:
            raise ValueError('Data does not appear to be in a Flatbush format.')

        if versionAndType >> 4 != VERSION:
            raise ValueError(f'Got v{versionAndType >> 4} data when expected v{VERSION}.)')

        nodeSize = np.frombuffer(data, dtype=np.uint16, offset=2, count=1)[0]
        numItems = np.frombuffer(data, dtype=np.uint32, offset=4, count=1)[0]

        # TODO: handle array type
        # ARRAY_TYPES[versionAndType & 0x0f]
        return cls(numItems=numItems, nodeSize=nodeSize, data=data)

    def __init__(
        self,
        unsigned int numItems,
        unsigned int nodeSize = 16,
        bytearray data = None,
    ):
        if numItems <= 0:
            raise ValueError('numItems must be greater than 0')
        if nodeSize < 2 or nodeSize > 65535:
            raise ValueError('nodeSize must be between 2 and 65535')

        # ArrayType was a parameter in JS
        # TODO: make function generic across array types?
        ArrayType = np.float64

        self.numItems = numItems
        self.nodeSize = min(max(nodeSize, 2), 65535)

        cdef unsigned int n, numNodes

        # calculate the total number of nodes in the R-tree to allocate space for
        # and the index of each tree level (used in search later)
        n = numItems
        numNodes = n
        self._levelBounds = array('I')
        self._levelBounds.append(n * 4)

        while n != 1:
            n = int(ceil(n / self.nodeSize))
            numNodes += n
            self._levelBounds.append(numNodes * 4)

        # TODO: support uint16 for index
        IndexArrayType = np.uint32
        # if numNodes < 16384:
        #     IndexArrayType = np.uint16
        # else:
        #     IndexArrayType = np.uint32

        # const arrayTypeIndex = ARRAY_TYPES.indexOf(ArrayType)
        arrayTypeIndex = 8
        nodesByteSize = numNodes * 4 * np.finfo(ArrayType).bits / 8

        # if (arrayTypeIndex < 0) {
        #     throw new Error(f'Unexpected typed array class: {ArrayType}.')
        # }

        if data is not None:
            self.data = data
            self._boxes = np.frombuffer(self.data, dtype=ArrayType, offset=8, count=int(numNodes * 4))
            self._indices = np.frombuffer(self.data, dtype=IndexArrayType, offset=int(8 + nodesByteSize), count=int(numNodes))

            self._pos = numNodes * 4
            self.minX = self._boxes[self._pos - 4]
            self.minY = self._boxes[self._pos - 3]
            self.maxX = self._boxes[self._pos - 2]
            self.maxY = self._boxes[self._pos - 1]

        else:
            self.data = bytearray(int(8 + nodesByteSize + numNodes * np.iinfo(IndexArrayType).bits / 8))
            self._boxes = np.frombuffer(self.data, dtype=ArrayType, offset=8, count=int(numNodes * 4))
            self._indices = np.frombuffer(self.data, dtype=IndexArrayType, offset=int(8 + nodesByteSize), count=int(numNodes))

            self._pos = 0
            self.minX = INFINITY
            self.minY = INFINITY
            self.maxX = -INFINITY
            self.maxY = -INFINITY

            self.data[0] = 0xfb
            self.data[1] = (VERSION << 4) + arrayTypeIndex

            np.frombuffer(self.data, dtype=np.uint16, offset=2, count=1)[0] = nodeSize
            np.frombuffer(self.data, dtype=np.uint32, offset=4, count=1)[0] = numItems

        # a priority queue for k-nearest-neighbors queries
        # self._queue = new FlatQueue()

    cpdef unsigned int [:] add_vectorized(
        self,
        double [:] minX,
        double [:] minY,
        double [:] maxX,
        double [:] maxY,
    ) except *:
        cdef Py_ssize_t i
        cdef unsigned int val, array_len
        cdef unsigned int [:] indexes

        assert len(minX) == len(minY) == len(maxX) == len(maxY), (
            'Input arrays must have equal length'
        )

        array_len = len(minX)
        indexes = np.zeros(array_len, dtype=np.uint32)

        for i in range(array_len):
            val = self.add(minX[i], minY[i], maxX[i], maxY[i])
            indexes[i] = val

        return indexes


    cpdef unsigned int add(self, double minX, double minY, double maxX, double maxY):
        cdef unsigned int index

        index = self._pos >> 2
        self._indices[index] = index

        self._boxes[self._pos] = minX
        self._pos += 1
        self._boxes[self._pos] = minY
        self._pos += 1
        self._boxes[self._pos] = maxX
        self._pos += 1
        self._boxes[self._pos] = maxY
        self._pos += 1

        if minX < self.minX:
            self.minX = minX
        if minY < self.minY:
            self.minY = minY
        if maxX > self.maxX:
            self.maxX = maxX
        if maxY > self.maxY:
            self.maxY = maxY

        return index

    @cdivision(True)
    cpdef void finish(self) except *:
        if self._pos >> 2 != self.numItems:
            raise ValueError(f'Added {self._pos >> 2} items when expected {self.numItems}.')

        if self.numItems <= self.nodeSize:
            # only one node, skip sorting and just fill the root box
            self._boxes[self._pos] = self.minX
            self._pos += 1
            self._boxes[self._pos] = self.minY
            self._pos += 1
            self._boxes[self._pos] = self.maxX
            self._pos += 1
            self._boxes[self._pos] = self.maxY
            self._pos += 1
            return

        cdef double width, height, minX, minY, maxX, maxY
        cdef double nodeMinX, nodeMinY, nodeMaxX, nodeMaxY
        cdef double x, y
        cdef unsigned int [:] hilbertValues
        cdef unsigned int hilbertMax
        cdef Py_ssize_t i
        cdef unsigned int pos, end, nodeIndex

        width = (self.maxX - self.minX) or 1.0
        height = (self.maxY - self.minY) or 1.0
        hilbertValues = np.zeros(self.numItems, dtype=np.uint32)
        hilbertMax = (1 << 16) - 1

        # map item centers into Hilbert coordinate space and calculate Hilbert values
        for i in range(self.numItems):
            pos = 4 * i
            minX = self._boxes[pos]
            pos += 1
            minY = self._boxes[pos]
            pos += 1
            maxX = self._boxes[pos]
            pos += 1
            maxY = self._boxes[pos]
            pos += 1

            x = hilbertMax * ((minX + maxX) / 2 - self.minX) / width
            y = hilbertMax * ((minY + maxY) / 2 - self.minY) / height
            hilbertValues[i] = hilbertXYToIndex(16, int(floor(x)), int(floor(y)))

        # sort items by their Hilbert value (for packing later)
        sort(hilbertValues, self._boxes, self._indices, 0, self.numItems - 1, self.nodeSize)

        # generate nodes at each tree level, bottom-up
        # TODO: double check that I refactored this loop correctly
        pos = 0
        for i in range(len(self._levelBounds) - 1):
            end = self._levelBounds[i]

            # generate a parent node for each block of consecutive <nodeSize> nodes
            while pos < end:
                nodeIndex = pos

                # calculate bbox for the new node
                nodeMinX = INFINITY
                nodeMinY = INFINITY
                nodeMaxX = -INFINITY
                nodeMaxY = -INFINITY

                # TODO: I think I refactored this loop correctly
                for j in range(self.nodeSize):
                    if pos >= end:
                        break

                    nodeMinX = min(nodeMinX, self._boxes[pos])
                    pos += 1
                    nodeMinY = min(nodeMinY, self._boxes[pos])
                    pos += 1
                    nodeMaxX = max(nodeMaxX, self._boxes[pos])
                    pos += 1
                    nodeMaxY = max(nodeMaxY, self._boxes[pos])
                    pos += 1

                # add the new node to the tree data
                self._indices[self._pos >> 2] = nodeIndex
                self._boxes[self._pos] = nodeMinX
                self._pos += 1
                self._boxes[self._pos] = nodeMinY
                self._pos += 1
                self._boxes[self._pos] = nodeMaxX
                self._pos += 1
                self._boxes[self._pos] = nodeMaxY
                self._pos += 1


    cpdef array[unsigned int] search(self, double minX, double minY, double maxX, double maxY):
        if self._pos != len(self._boxes):
            raise ValueError('Data not yet indexed - call index.finish().')

        cdef unsigned int nodeIndex, pos, index

        nodeIndex = len(self._boxes) - 4
        queue = array('I')
        results = array('I')

        # TODO: fix while loop syntax
        while True:
            # find the end index of the node
            end = min(nodeIndex + self.nodeSize * 4, upperBound(nodeIndex, self._levelBounds))

            # search through child nodes
            for pos in range(nodeIndex, end, 4):
                # check if node bbox intersects with query bbox
                if maxX < self._boxes[pos]:
                    # maxX < nodeMinX
                    continue
                if maxY < self._boxes[pos + 1]:
                    # maxY < nodeMinY
                    continue
                if minX > self._boxes[pos + 2]:
                    # minX > nodeMaxX
                    continue
                if minY > self._boxes[pos + 3]:
                    # minY > nodeMaxY
                    continue

                index = self._indices[pos >> 2] | 0

                if nodeIndex >= self.numItems * 4:
                    # node; add it to the search queue
                    queue.append(index)

                else:
                    # leaf item
                    results.append(index)

            if len(queue) == 0:
                break

            nodeIndex = queue.pop()

        return results


    # cdef neighbors(self, x, y, maxResults = INFINITY, maxDistance = INFINITY, filterFn):
    #     if self._pos != len(self._boxes):
    #         raise ValueError('Data not yet indexed - call index.finish().')

    #     let nodeIndex = len(self._boxes) - 4
    #     const q = self._queue
    #     const results = []
    #     const maxDistSquared = maxDistance * maxDistance

    #     while nodeIndex != undefined:
    #         # find the end index of the node
    #         const end = min(nodeIndex + self.nodeSize * 4, upperBound(nodeIndex, self._levelBounds))

    #         # add child nodes to the queue
    #         for (let pos = nodeIndex; pos < end; pos += 4) {
    #             const index = self._indices[pos >> 2] | 0

    #             const dx = axisDist(x, self._boxes[pos], self._boxes[pos + 2])
    #             const dy = axisDist(y, self._boxes[pos + 1], self._boxes[pos + 3])
    #             const dist = dx * dx + dy * dy

    #             if nodeIndex >= self.numItems * 4:
    #                 # node (use even id)
    #                 q.append(index << 1, dist)
    #             elif (filterFn == undefined or filterFn(index)):
    #                 # leaf item (use odd id)
    #                 q.append((index << 1) + 1, dist)
    #         }

    #         # pop items from the queue
    #         while len(q) and (q.peek() & 1):
    #             const dist = q.peekValue()
    #             if dist > maxDistSquared:
    #                 q.clear()
    #                 return results

    #             results.append(q.pop() >> 1)

    #             if len(results) == maxResults:
    #                 q.clear()
    #                 return results

    #         nodeIndex = q.pop() >> 1

    #     q.clear()
    #     return results


cdef axisDist(k, min_val, max_val):
    if k < min_val:
        return min_val - k
    elif k <= max_val:
        return 0
    else:
        return k - max_val


cdef unsigned int upperBound(unsigned int value, array[unsigned int] arr):
    """binary search for the first value in the array bigger than the given"""
    cdef int i, j, m

    i = 0
    j = len(arr) - 1
    while i < j:
        m = (i + j) >> 1
        if arr[m] > value:
            j = m
        else:
            i = m + 1

    return arr[i]


@cdivision(True)
cdef void sort(
        unsigned int [:] values,
        double [:] boxes,
        unsigned int [:] indices,
        unsigned int left,
        unsigned int right,
        unsigned int nodeSize):
    """custom quicksort that partially sorts bbox data alongside the hilbert values"""
    if floor(left / nodeSize) >= floor(right / nodeSize):
        return

    cdef unsigned int pivot
    cdef int i, j

    pivot = values[(left + right) >> 1]
    i = left - 1
    j = right + 1

    while True:
        while True:
            i += 1
            if values[i] >= pivot:
                break

        while True:
            j -= 1
            if values[j] <= pivot:
                break

        if i >= j:
            break

        swap(values, boxes, indices, i, j)

    sort(values, boxes, indices, left, j, nodeSize)
    sort(values, boxes, indices, j + 1, right, nodeSize)


cdef void swap(
        unsigned int [:] values,
        double [:] boxes,
        unsigned int [:] indices,
        int i,
        int j):
    """swap two values and two corresponding boxes"""
    cdef unsigned int temp, e
    cdef int k, m
    cdef double a, b, c, d

    temp = values[i]
    values[i] = values[j]
    values[j] = temp

    k = 4 * i
    m = 4 * j

    a = boxes[k]
    b = boxes[k + 1]
    c = boxes[k + 2]
    d = boxes[k + 3]
    boxes[k] = boxes[m]
    boxes[k + 1] = boxes[m + 1]
    boxes[k + 2] = boxes[m + 2]
    boxes[k + 3] = boxes[m + 3]
    boxes[m] = a
    boxes[m + 1] = b
    boxes[m + 2] = c
    boxes[m + 3] = d

    e = indices[i]
    indices[i] = indices[j]
    indices[j] = e


# The below is copied directly from the original C++ source instead of porting from JS' port.
cdef unsigned int deinterleave(unsigned int x):
    x = x & 0x55555555
    x = (x | (x >> 1)) & 0x33333333
    x = (x | (x >> 2)) & 0x0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF
    return x


cdef unsigned int interleave(unsigned int x):
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555
    return x


cdef unsigned int prefixScan(unsigned int x):
    x = (x >> 8) ^ x
    x = (x >> 4) ^ x
    x = (x >> 2) ^ x
    x = (x >> 1) ^ x
    return x


cdef unsigned int descan(unsigned int x):
    return x ^ (x >> 1)


# cdef void hilbertIndexToXY(unsigned int n, unsigned int i, unsigned int& x, unsigned int& y)
# {
#   i = i << (32 - 2 * n);

#   unsigned int i0 = deinterleave(i);
#   unsigned int i1 = deinterleave(i >> 1);

#   unsigned int t0 = (i0 | i1) ^ 0xFFFF;
#   unsigned int t1 = i0 & i1;

#   unsigned int prefixT0 = prefixScan(t0);
#   unsigned int prefixT1 = prefixScan(t1);

#   unsigned int a = (((i0 ^ 0xFFFF) & prefixT1) | (i0 & prefixT0));

#   x = (a ^ i1) >> (16 - n);
#   y = (a ^ i0 ^ i1) >> (16 - n);
# }

cdef unsigned int hilbertXYToIndex(
    unsigned int n,
    unsigned int x,
    unsigned int y
):
    x = x << (16 - n)
    y = y << (16 - n)

    cdef unsigned int A, B, C, D, a, b, c, d, i0, i1

    # Initial prefix scan round, prime with x and y
    a = x ^ y
    b = 0xFFFF ^ a
    c = 0xFFFF ^ (x | y)
    d = x & (y ^ 0xFFFF)

    A = a | (b >> 1)
    B = (a >> 1) ^ a

    C = ((c >> 1) ^ (b & (d >> 1))) ^ c
    D = ((a & (c >> 1)) ^ (d >> 1)) ^ d

    a = A
    b = B
    c = C
    d = D

    A = ((a & (a >> 2)) ^ (b & (b >> 2)))
    B = ((a & (b >> 2)) ^ (b & ((a ^ b) >> 2)))

    C ^= ((a & (c >> 2)) ^ (b & (d >> 2)))
    D ^= ((b & (c >> 2)) ^ ((a ^ b) & (d >> 2)))

    a = A
    b = B
    c = C
    d = D

    A = ((a & (a >> 4)) ^ (b & (b >> 4)))
    B = ((a & (b >> 4)) ^ (b & ((a ^ b) >> 4)))

    C ^= ((a & (c >> 4)) ^ (b & (d >> 4)))
    D ^= ((b & (c >> 4)) ^ ((a ^ b) & (d >> 4)))

    # Final round and projection
    a = A
    b = B
    c = C
    d = D

    C ^= ((a & (c >> 8)) ^ (b & (d >> 8)))
    D ^= ((b & (c >> 8)) ^ ((a ^ b) & (d >> 8)))

    # Undo transformation prefix scan
    a = C ^ (C >> 1)
    b = D ^ (D >> 1)

    # Recover index bits
    i0 = x ^ y
    i1 = b | (0xFFFF ^ (i0 | a))

    return ((interleave(i1) << 1) | interleave(i0)) >> (32 - 2 * n)
