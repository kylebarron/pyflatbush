import fs from 'fs';
import Flatbush from 'flatbush';

var text = fs.readFileSync('bounds.txt', 'UTF-8');
var textLines = text.trim().split('\n');
var bounds = textLines.map(JSON.parse);

console.time('Create index');
// initialize Flatbush
const index = new Flatbush(bounds.length);

// fill it with bounds
for (const p of bounds) {
    index.add(p[0], p[1], p[2], p[3]);
}

// perform the indexing
index.finish();
console.timeEnd('Create index');



// // make a bounding box query
// const found = index.search(minX, minY, maxX, maxY).map((i) => items[i]);

// // make a k-nearest-neighbors query
// const neighborIds = index.neighbors(x, y, 5);

// // instantly transfer the index from a worker to the main thread
// postMessage(index.data, [index.data]);

// // reconstruct the index from a raw array buffer
// const index = Flatbush.from(e.data);
