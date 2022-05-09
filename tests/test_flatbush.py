import numpy as np
import pytest

from pyflatbush import Flatbush

# fmt: off
data = [
    8, 62, 11, 66, 57, 17, 57, 19, 76, 26, 79, 29, 36, 56, 38, 56, 92, 77, 96, 80, 87, 70, 90, 74,
    43, 41, 47, 43, 0, 58, 2, 62, 76, 86, 80, 89, 27, 13, 27, 15, 71, 63, 75, 67, 25, 2, 27, 2, 87,
    6, 88, 6, 22, 90, 23, 93, 22, 89, 22, 93, 57, 11, 61, 13, 61, 55, 63, 56, 17, 85, 21, 87, 33,
    43, 37, 43, 6, 1, 7, 3, 80, 87, 80, 87, 23, 50, 26, 52, 58, 89, 58, 89, 12, 30, 15, 34, 32, 58,
    36, 61, 41, 84, 44, 87, 44, 18, 44, 19, 13, 63, 15, 67, 52, 70, 54, 74, 57, 59, 58, 59, 17, 90,
    20, 92, 48, 53, 52, 56, 92, 68, 92, 72, 26, 52, 30, 52, 56, 23, 57, 26, 88, 48, 88, 48, 66, 13,
    67, 15, 7, 82, 8, 86, 46, 68, 50, 68, 37, 33, 38, 36, 6, 15, 8, 18, 85, 36, 89, 38, 82, 45, 84,
    48, 12, 2, 16, 3, 26, 15, 26, 16, 55, 23, 59, 26, 76, 37, 79, 39, 86, 74, 90, 77, 16, 75, 18,
    78, 44, 18, 45, 21, 52, 67, 54, 71, 59, 78, 62, 78, 24, 5, 24, 8, 64, 80, 64, 83, 66, 55, 70,
    55, 0, 17, 2, 19, 15, 71, 18, 74, 87, 57, 87, 59, 6, 34, 7, 37, 34, 30, 37, 32, 51, 19, 53, 19,
    72, 51, 73, 55, 29, 45, 30, 45, 94, 94, 96, 95, 7, 22, 11, 24, 86, 45, 87, 48, 33, 62, 34, 65,
    18, 10, 21, 14, 64, 66, 67, 67, 64, 25, 65, 28, 27, 4, 31, 6, 84, 4, 85, 5, 48, 80, 50, 81, 1,
    61, 3, 61, 71, 89, 74, 92, 40, 42, 43, 43, 27, 64, 28, 66, 46, 26, 50, 26, 53, 83, 57, 87, 14,
    75, 15, 79, 31, 45, 34, 45, 89, 84, 92, 88, 84, 51, 85, 53, 67, 87, 67, 89, 39, 26, 43, 27, 47,
    61, 47, 63, 23, 49, 25, 53, 12, 3, 14, 5, 16, 50, 19, 53, 63, 80, 64, 84, 22, 63, 22, 64, 26,
    66, 29, 66, 2, 15, 3, 15, 74, 77, 77, 79, 64, 11, 68, 11, 38, 4, 39, 8, 83, 73, 87, 77, 85, 52,
    89, 56, 74, 60, 76, 63, 62, 66, 65, 67
]
# fmt: on


def create_index():
    index = Flatbush(int(len(data) / 4))

    for i in range(0, len(data), 4):
        index.add(data[i], data[i + 1], data[i + 2], data[i + 3])

    index.finish()

    return index


def create_index_vectorized():
    index = Flatbush(int(len(data) / 4))
    arr = np.array(data).reshape(4, -1).astype(np.float64)
    index.add_vectorized(
        arr[0, :],
        arr[1, :],
        arr[2, :],
        arr[3, :],
    )

    index.finish()

    return index


def create_small_index(numItems, nodeSize):
    index = Flatbush(numItems, nodeSize)

    for i in range(0, 4 * numItems, 4):
        index.add(data[i], data[i + 1], data[i + 2], data[i + 3])

    index.finish()

    return index


def test_indexes_a_bunch_of_rectangles():
    index = create_index()

    boxes_len = len(index._boxes)
    assert len(index._boxes) + len(index._indices) == 540
    assert np.array_equal(index._boxes[boxes_len - 4 : boxes_len], [0, 1, 96, 95])
    assert index._indices[int(boxes_len / 4 - 1)] == 400


def test_indexes_a_bunch_of_rectangles_vectorized():
    index = create_index_vectorized()

    boxes_len = len(index._boxes)
    assert len(index._boxes) + len(index._indices) == 540

    # NOTE: this is different from above because the vectorized impl only accepts double
    # as input coords?
    assert np.array_equal(index._boxes[boxes_len - 4 : boxes_len], [0, 2, 96, 92])
    assert index._indices[int(boxes_len / 4 - 1)] == 400


def test_skips_sorting_less_than_nodeSize_number_of_rectangles():
    numItems = 14
    nodeSize = 16
    index = create_small_index(numItems, nodeSize)

    # compute expected root box extents
    rootXMin = np.inf
    rootYMin = np.inf
    rootXMax = -np.inf
    rootYMax = -np.inf
    for i in range(0, 4 * numItems, 4):
        if data[i] < rootXMin:
            rootXMin = data[i]
        if data[i + 1] < rootYMin:
            rootYMin = data[i + 1]
        if data[i + 2] > rootXMax:
            rootXMax = data[i + 2]
        if data[i + 3] > rootYMax:
            rootYMax = data[i + 3]

    # sort should be skipped, ordered progressing indices expected
    expectedIndices = []
    for i in range(numItems):
        expectedIndices.append(i)

    expectedIndices.append(0)

    assert np.array_equal(np.array(index._indices), np.array(expectedIndices))
    assert len(index._boxes) == (numItems + 1) * 4
    assert np.array_equal(
        np.array(index._boxes[len(index._boxes) - 4 : len(index._boxes)]),
        np.array([rootXMin, rootYMin, rootXMax, rootYMax]),
    )


def test_performs_bbox_search():
    index = create_index()

    ids = index.search(40, 40, 60, 60)

    results = []
    for i in range(len(ids)):
        results.append(data[4 * ids[i]])
        results.append(data[4 * ids[i] + 1])
        results.append(data[4 * ids[i] + 2])
        results.append(data[4 * ids[i] + 3])

    assert sorted(results) == sorted(
        [57, 59, 58, 59, 48, 53, 52, 56, 40, 42, 43, 43, 43, 41, 47, 43]
    )


def test_reconstructs_an_index_from_array_buffer():
    index = create_index()
    index2 = Flatbush.from_buffer(index.data)

    assert index.data == index2.data
    assert np.array_equal(index._boxes, index2._boxes)
    assert np.array_equal(index._indices, index2._indices)
    assert index.minX == index2.minX
    assert index.minY == index2.minY
    assert index.maxX == index2.maxX
    assert index.maxY == index2.maxY
    assert index.numItems == index2.numItems
    assert index.nodeSize == index2.nodeSize


def test_throws_an_error_if_added_less_items_than_the_index_size():
    with pytest.raises(ValueError):
        index = Flatbush(len(data) / 4)
        index.finish()


def test_throws_error_with_arrays_of_different_sizes():
    with pytest.raises(ValueError):
        index = Flatbush(len(data) / 4)
        arr = np.array(data).reshape(4, -1)
        index.add_vectorized(
            arr[0, :],
            arr[1, :],
            arr[2, :],
            arr[3, :-1],
        )


def test_throws_an_error_if_searching_before_indexing():
    with pytest.raises(ValueError):
        index = Flatbush(len(data) / 4)
        index.search(0, 0, 20, 20)


def test_does_not_freeze_on_numItems_0():
    with pytest.raises(ValueError):
        Flatbush(0)


@pytest.mark.skip()
def test_performs_a_k_nearest_neighbors_query():
    index = create_index()
    ids = index.neighbors(50, 50, 3)
    assert sorted(ids) == sorted([31, 6, 75])


@pytest.mark.skip()
def test_k_nearest_neighbors_query_accepts_maxDistance():
    index = create_index()
    ids = index.neighbors(50, 50, np.inf, 12)
    assert sorted(ids) == sorted([6, 29, 31, 75, 85])


def test_returns_index_of_newly_added_rectangle():
    count = 5
    index = Flatbush(count)

    ids = []
    for i in range(count):
        id = index.add(data[i], data[i + 1], data[i + 2], data[i + 3])
        ids.append(id)

    assert ids == list(range(5))
