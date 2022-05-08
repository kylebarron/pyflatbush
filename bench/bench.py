import json
from time import time

import flatbush

import pyflatbush

print("loading data")
with open("bounds.txt") as f:
    lines = [json.loads(line) for line in f]


start = time()
index_cy = pyflatbush.Flatbush(len(lines))
for box in lines:
    index_cy.add(*box)

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
