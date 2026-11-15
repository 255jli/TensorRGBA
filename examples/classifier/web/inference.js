function createClassifier(weights) {
  const sizes = [
    [784, 256],
    [256, 128],
    [128, 10],
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
  return { layers };
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

function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - max);
    sum += exps[i];
  }
  for (let i = 0; i < exps.length; i++) {
    exps[i] /= sum;
  }
  return exps;
}

function classify(classifier, pixels) {
  let x = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    x[i] = pixels[i];
  }
  const [l1, l2, l3] = classifier.layers;
  x = relu(matvec(l1.w, l1.b, x, l1.inF, l1.outF));
  x = relu(matvec(l2.w, l2.b, x, l2.inF, l2.outF));
  const logits = matvec(l3.w, l3.b, x, l3.inF, l3.outF);
  return softmax(logits);
}

window.createClassifier = createClassifier;
window.classify = classify;
