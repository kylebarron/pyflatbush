import json
from time import time

import flatbush
import numpy as np

import pyflatbush

print("loading data")
with open("bounds.txt") as f:
    lines = [json.loads(line) for line in f]

arr = np.array(lines)


start = time()
index_cy = pyflatbush.Flatbush(arr.shape[0])
out = index_cy.add_vectorized(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
index_cy.finish()
end = time()

print(f"Cython create index: {end - start:.2f}s")

start = time()
index_py = flatbush.FlatBush()
for box in lines:
    index_py.add(*box)

index_py.finish()
end = time()
print(f"Pure Python create index: {end - start:.2f}s")
