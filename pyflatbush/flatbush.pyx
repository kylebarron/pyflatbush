import numpy as np
cimport numpy as np

import FlatQueue from 'flatqueue'


# serialized format version
VERSION = 3

cdef class Flatbush:
    cdef readonly unsigned int numItems
    cdef readonly unsigned int nodeSize

    # static from(data) {
    #     if (!(data instanceof ArrayBuffer)) {
    #         throw new Error('Data must be an instance of ArrayBuffer.')
    #     }
    #     const [magic, versionAndType] = new Uint8Array(data, 0, 2)
    #     if (magic !== 0xfb) {
    #         throw new Error('Data does not appear to be in a Flatbush format.')
    #     }
    #     if (versionAndType >> 4 !== VERSION) {
    #         throw new Error(`Got v${versionAndType >> 4} data when expected v${VERSION}.`)
    #     }
    #     const [nodeSize] = new Uint16Array(data, 2, 1)
    #     const [numItems] = new Uint32Array(data, 4, 1)

    #     return new Flatbush(numItems, nodeSize, ARRAY_TYPES[versionAndType & 0x0f], data)
    # }

    cdef constructor(self, unsigned int numItems, unsigned int nodeSize = 16, ArrayType = Float64Array, data):
        if numItems <= 0:
            raise ValueError('numItems must be greater than 0')
        if nodeSize < 2 or nodeSize > 65535:
            raise ValueError('nodeSize must be between 2 and 65535')

        self.numItems = numItems
        self.nodeSize = min(max(nodeSize, 2), 65535)

        # calculate the total number of nodes in the R-tree to allocate space for
        # and the index of each tree level (used in search later)
        let n = numItems
        let numNodes = n
        self._levelBounds = [n * 4]
        do {
            n = Math.ceil(n / self.nodeSize)
            numNodes += n
            self._levelBounds.push(numNodes * 4)
        } while (n !== 1)

        self.ArrayType = ArrayType || Float64Array
        self.IndexArrayType = numNodes < 16384 ? Uint16Array : Uint32Array

        const arrayTypeIndex = ARRAY_TYPES.indexOf(self.ArrayType)
        const nodesByteSize = numNodes * 4 * self.ArrayType.BYTES_PER_ELEMENT

        if (arrayTypeIndex < 0) {
            throw new Error(`Unexpected typed array class: ${ArrayType}.`)
        }

        if (data && (data instanceof ArrayBuffer)) {
            self.data = data
            self._boxes = new self.ArrayType(self.data, 8, numNodes * 4)
            self._indices = new self.IndexArrayType(self.data, 8 + nodesByteSize, numNodes)

            self._pos = numNodes * 4
            self.minX = self._boxes[self._pos - 4]
            self.minY = self._boxes[self._pos - 3]
            self.maxX = self._boxes[self._pos - 2]
            self.maxY = self._boxes[self._pos - 1]

        } else {
            self.data = new ArrayBuffer(8 + nodesByteSize + numNodes * self.IndexArrayType.BYTES_PER_ELEMENT)
            self._boxes = new self.ArrayType(self.data, 8, numNodes * 4)
            self._indices = new self.IndexArrayType(self.data, 8 + nodesByteSize, numNodes)
            self._pos = 0
            self.minX = Infinity
            self.minY = Infinity
            self.maxX = -Infinity
            self.maxY = -Infinity

            new Uint8Array(self.data, 0, 2).set([0xfb, (VERSION << 4) + arrayTypeIndex])
            new Uint16Array(self.data, 2, 1)[0] = nodeSize
            new Uint32Array(self.data, 4, 1)[0] = numItems
        }

        # a priority queue for k-nearest-neighbors queries
        self._queue = new FlatQueue()
    }

    cdef add(self, minX, minY, maxX, maxY):
        const index = self._pos >> 2
        self._indices[index] = index
        self._boxes[self._pos++] = minX
        self._boxes[self._pos++] = minY
        self._boxes[self._pos++] = maxX
        self._boxes[self._pos++] = maxY

        if (minX < self.minX) self.minX = minX
        if (minY < self.minY) self.minY = minY
        if (maxX > self.maxX) self.maxX = maxX
        if (maxY > self.maxY) self.maxY = maxY

        return index
    }

    cdef finish(self):
        if (self._pos >> 2 !== self.numItems) {
            throw new Error(`Added ${self._pos >> 2} items when expected ${self.numItems}.`)
        }

        if (self.numItems <= self.nodeSize) {
            # only one node, skip sorting and just fill the root box
            self._boxes[self._pos++] = self.minX
            self._boxes[self._pos++] = self.minY
            self._boxes[self._pos++] = self.maxX
            self._boxes[self._pos++] = self.maxY
            return
        }

        const width = (self.maxX - self.minX) || 1
        const height = (self.maxY - self.minY) || 1
        const hilbertValues = new Uint32Array(self.numItems)
        const hilbertMax = (1 << 16) - 1

        # map item centers into Hilbert coordinate space and calculate Hilbert values
        for (let i = 0; i < self.numItems; i++) {
            let pos = 4 * i
            const minX = self._boxes[pos++]
            const minY = self._boxes[pos++]
            const maxX = self._boxes[pos++]
            const maxY = self._boxes[pos++]
            const x = Math.floor(hilbertMax * ((minX + maxX) / 2 - self.minX) / width)
            const y = Math.floor(hilbertMax * ((minY + maxY) / 2 - self.minY) / height)
            hilbertValues[i] = hilbert(x, y)
        }

        # sort items by their Hilbert value (for packing later)
        sort(hilbertValues, self._boxes, self._indices, 0, self.numItems - 1, self.nodeSize)

        # generate nodes at each tree level, bottom-up
        for (let i = 0, pos = 0; i < self._levelBounds.length - 1; i++) {
            const end = self._levelBounds[i]

            # generate a parent node for each block of consecutive <nodeSize> nodes
            while (pos < end) {
                const nodeIndex = pos

                # calculate bbox for the new node
                let nodeMinX = Infinity
                let nodeMinY = Infinity
                let nodeMaxX = -Infinity
                let nodeMaxY = -Infinity
                for (let i = 0; i < self.nodeSize && pos < end; i++) {
                    nodeMinX = Math.min(nodeMinX, self._boxes[pos++])
                    nodeMinY = Math.min(nodeMinY, self._boxes[pos++])
                    nodeMaxX = Math.max(nodeMaxX, self._boxes[pos++])
                    nodeMaxY = Math.max(nodeMaxY, self._boxes[pos++])
                }

                # add the new node to the tree data
                self._indices[self._pos >> 2] = nodeIndex
                self._boxes[self._pos++] = nodeMinX
                self._boxes[self._pos++] = nodeMinY
                self._boxes[self._pos++] = nodeMaxX
                self._boxes[self._pos++] = nodeMaxY
            }
        }


    cdef search(self, minX, minY, maxX, maxY, filterFn):
        if (self._pos !== self._boxes.length) {
            throw new Error('Data not yet indexed - call index.finish().')
        }

        let nodeIndex = self._boxes.length - 4
        const queue = []
        const results = []

        while (nodeIndex !== undefined) {
            # find the end index of the node
            const end = Math.min(nodeIndex + self.nodeSize * 4, upperBound(nodeIndex, self._levelBounds))

            # search through child nodes
            for (let pos = nodeIndex; pos < end; pos += 4) {
                # check if node bbox intersects with query bbox
                if (maxX < self._boxes[pos]) continue; # maxX < nodeMinX
                if (maxY < self._boxes[pos + 1]) continue; # maxY < nodeMinY
                if (minX > self._boxes[pos + 2]) continue; # minX > nodeMaxX
                if (minY > self._boxes[pos + 3]) continue; # minY > nodeMaxY

                const index = self._indices[pos >> 2] | 0

                if (nodeIndex >= self.numItems * 4) {
                    queue.push(index); # node; add it to the search queue

                } else if (filterFn === undefined || filterFn(index)) {
                    results.push(index); # leaf item
                }
            }

            nodeIndex = queue.pop()
        }

        return results


    cdef neighbors(self, x, y, maxResults = Infinity, maxDistance = Infinity, filterFn):
        if (self._pos !== self._boxes.length) {
            throw new Error('Data not yet indexed - call index.finish().')
        }

        let nodeIndex = self._boxes.length - 4
        const q = self._queue
        const results = []
        const maxDistSquared = maxDistance * maxDistance

        while (nodeIndex !== undefined) {
            # find the end index of the node
            const end = Math.min(nodeIndex + self.nodeSize * 4, upperBound(nodeIndex, self._levelBounds))

            # add child nodes to the queue
            for (let pos = nodeIndex; pos < end; pos += 4) {
                const index = self._indices[pos >> 2] | 0

                const dx = axisDist(x, self._boxes[pos], self._boxes[pos + 2])
                const dy = axisDist(y, self._boxes[pos + 1], self._boxes[pos + 3])
                const dist = dx * dx + dy * dy

                if (nodeIndex >= self.numItems * 4) {
                    q.push(index << 1, dist); # node (use even id)

                } else if (filterFn === undefined || filterFn(index)) {
                    q.push((index << 1) + 1, dist); # leaf item (use odd id)
                }
            }

            # pop items from the queue
            while (q.length && (q.peek() & 1)) {
                const dist = q.peekValue()
                if (dist > maxDistSquared) {
                    q.clear()
                    return results
                }
                results.push(q.pop() >> 1)

                if (results.length === maxResults) {
                    q.clear()
                    return results
                }
            }

            nodeIndex = q.pop() >> 1
        }

        q.clear()
        return results
    }
}

cdef axisDist(k, min, max):
    return k < min ? min - k : k <= max ? 0 : k - max


cdef upperBound(value, arr):
    """binary search for the first value in the array bigger than the given"""
    let i = 0
    let j = arr.length - 1
    while (i < j) {
        const m = (i + j) >> 1
        if (arr[m] > value) {
            j = m
        } else {
            i = m + 1
        }
    }
    return arr[i]


cdef sort(values, boxes, indices, left, right, nodeSize):
    """custom quicksort that partially sorts bbox data alongside the hilbert values"""
    if (Math.floor(left / nodeSize) >= Math.floor(right / nodeSize)) return

    const pivot = values[(left + right) >> 1]
    let i = left - 1
    let j = right + 1

    while (true) {
        do i++; while (values[i] < pivot)
        do j--; while (values[j] > pivot)
        if (i >= j) break
        swap(values, boxes, indices, i, j)
    }

    sort(values, boxes, indices, left, j, nodeSize)
    sort(values, boxes, indices, j + 1, right, nodeSize)


cdef swap(values, boxes, indices, i, j):
    """swap two values and two corresponding boxes"""
    const temp = values[i]
    values[i] = values[j]
    values[j] = temp

    const k = 4 * i
    const m = 4 * j

    const a = boxes[k]
    const b = boxes[k + 1]
    const c = boxes[k + 2]
    const d = boxes[k + 3]
    boxes[k] = boxes[m]
    boxes[k + 1] = boxes[m + 1]
    boxes[k + 2] = boxes[m + 2]
    boxes[k + 3] = boxes[m + 3]
    boxes[m] = a
    boxes[m + 1] = b
    boxes[m + 2] = c
    boxes[m + 3] = d

    const e = indices[i]
    indices[i] = indices[j]
    indices[j] = e


cdef hilbert(x, y):
    """Fast Hilbert curve algorithm by http://threadlocalmutex.com/

    Ported from C++ https://github.com/rawrunprotected/hilbert_curves (public domain)
    """
    a = x ^ y
    b = 0xFFFF ^ a
    c = 0xFFFF ^ (x | y)
    d = x & (y ^ 0xFFFF)

    A = a | (b >> 1)
    B = (a >> 1) ^ a
    C = ((c >> 1) ^ (b & (d >> 1))) ^ c
    D = ((a & (c >> 1)) ^ (d >> 1)) ^ d

    a = A; b = B; c = C; d = D
    A = ((a & (a >> 2)) ^ (b & (b >> 2)))
    B = ((a & (b >> 2)) ^ (b & ((a ^ b) >> 2)))
    C ^= ((a & (c >> 2)) ^ (b & (d >> 2)))
    D ^= ((b & (c >> 2)) ^ ((a ^ b) & (d >> 2)))

    a = A; b = B; c = C; d = D
    A = ((a & (a >> 4)) ^ (b & (b >> 4)))
    B = ((a & (b >> 4)) ^ (b & ((a ^ b) >> 4)))
    C ^= ((a & (c >> 4)) ^ (b & (d >> 4)))
    D ^= ((b & (c >> 4)) ^ ((a ^ b) & (d >> 4)))

    a = A; b = B; c = C; d = D
    C ^= ((a & (c >> 8)) ^ (b & (d >> 8)))
    D ^= ((b & (c >> 8)) ^ ((a ^ b) & (d >> 8)))

    a = C ^ (C >> 1)
    b = D ^ (D >> 1)

    i0 = x ^ y
    i1 = b | (0xFFFF ^ (i0 | a))

    i0 = (i0 | (i0 << 8)) & 0x00FF00FF
    i0 = (i0 | (i0 << 4)) & 0x0F0F0F0F
    i0 = (i0 | (i0 << 2)) & 0x33333333
    i0 = (i0 | (i0 << 1)) & 0x55555555

    i1 = (i1 | (i1 << 8)) & 0x00FF00FF
    i1 = (i1 | (i1 << 4)) & 0x0F0F0F0F
    i1 = (i1 | (i1 << 2)) & 0x33333333
    i1 = (i1 | (i1 << 1)) & 0x55555555

    return ((i1 << 1) | i0) >>> 0
