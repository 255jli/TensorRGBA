function createGenerator(weights, noiseDim, numClasses) {
  const sizes = [
    [noiseDim + numClasses, 256],
    [256, 512],
    [512, 1024],
    [1024, 1024],
  ];
  let offset = 0;
  const layers = [];
  for (const [inF, outF] of sizes) {
    const weightCount = inF * outF;
    const biasCount = outF;
    const w = weights.subarray(offset, offset + weightCount);
    offset += weightCount;
    const b = weights.subarray(offset, offset + biasCount);
    offset += biasCount;
    layers.push({ w, b, inF, outF });
  }
  return { layers, noiseDim, numClasses };
}

function matvec(w, b, input, inF, outF) {
  const out = new Float32Array(outF);
  for (let o = 0; o < outF; o++) {
    let sum = b[o];
    const rowOffset = o * inF;
    for (let i = 0; i < inF; i++) {
      sum += w[rowOffset + i] * input[i];
    }
    out[o] = sum;
  }
  return out;
}

function relu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = x[i] > 0 ? x[i] : 0;
  }
  return out;
}

function sigmoid(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = 1 / (1 + Math.exp(-x[i]));
  }
  return out;
}

function generateAnimal(generator, label, noise) {
  const oneHot = new Float32Array(generator.numClasses);
  oneHot[label] = 1;
  let x = new Float32Array(generator.noiseDim + generator.numClasses);
  x.set(noise);
  x.set(oneHot, generator.noiseDim);

  const [l1, l2, l3, l4] = generator.layers;
  x = relu(matvec(l1.w, l1.b, x, l1.inF, l1.outF));
  x = relu(matvec(l2.w, l2.b, x, l2.inF, l2.outF));
  x = relu(matvec(l3.w, l3.b, x, l3.inF, l3.outF));
  x = sigmoid(matvec(l4.w, l4.b, x, l4.inF, l4.outF));
  return x;
}

function drawAnimal(canvas, pixels) {
  const ctx = canvas.getContext("2d");
  const imgData = ctx.createImageData(32, 32);
  for (let i = 0; i < 1024; i++) {
    const v = Math.max(0, Math.min(255, Math.round(pixels[i] * 255)));
    imgData.data[i * 4] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

window.createGenerator = createGenerator;
window.generateAnimal = generateAnimal;
window.drawAnimal = drawAnimal;
