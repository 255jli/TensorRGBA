function createMlpGenerator(weights, noiseDim, numClasses) {
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
  return { layers, noiseDim, numClasses, architectureId: 0 };
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

function generateMlpImage(generator, label, noise) {
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

function takeWeights(weights, state, count) {
  const values = weights.subarray(state.offset, state.offset + count);
  state.offset += count;
  return values;
}

function createDcganGenerator(weights, noiseDim, numClasses) {
  const state = { offset: 0 };
  const denseWeights = takeWeights(weights, state, 2048 * (noiseDim + numClasses));
  const denseBias = takeWeights(weights, state, 2048);
  const batchNorm0 = takeBatchNorm(weights, state, 128);
  const transposeWeights1 = takeWeights(weights, state, 128 * 64 * 16);
  const batchNorm1 = takeBatchNorm(weights, state, 64);
  const transposeWeights2 = takeWeights(weights, state, 64 * 32 * 16);
  const batchNorm2 = takeBatchNorm(weights, state, 32);
  const transposeWeights3 = takeWeights(weights, state, 32 * 3 * 16);
  const transposeBias3 = takeWeights(weights, state, 3);

  if (state.offset !== weights.length) {
    throw new Error("Unexpected conditional GAN weight count");
  }

  return {
    architectureId: 1,
    noiseDim,
    numClasses,
    denseWeights,
    denseBias,
    batchNorm0,
    transposeWeights1,
    batchNorm1,
    transposeWeights2,
    batchNorm2,
    transposeWeights3,
    transposeBias3,
  };
}

function takeBatchNorm(weights, state, channels) {
  return {
    scale: takeWeights(weights, state, channels),
    bias: takeWeights(weights, state, channels),
    mean: takeWeights(weights, state, channels),
    variance: takeWeights(weights, state, channels),
  };
}

function createGenerator(weights, noiseDim, numClasses, architectureId = 0) {
  if (architectureId === 1) {
    return createDcganGenerator(weights, noiseDim, numClasses);
  }
  return createMlpGenerator(weights, noiseDim, numClasses);
}

function batchNorm(input, parameters, channels, spatialSize) {
  const output = new Float32Array(input.length);
  const pixelsPerChannel = spatialSize * spatialSize;
  for (let channel = 0; channel < channels; channel++) {
    const multiplier = parameters.scale[channel]
      / Math.sqrt(parameters.variance[channel] + 1e-5);
    const offset = parameters.bias[channel]
      - parameters.mean[channel] * multiplier;
    const channelStart = channel * pixelsPerChannel;
    for (let pixel = 0; pixel < pixelsPerChannel; pixel++) {
      const index = channelStart + pixel;
      output[index] = input[index] * multiplier + offset;
    }
  }
  return output;
}

function transposeConv(input, weights, bias, inChannels, outChannels, inputSize) {
  const outputSize = inputSize * 2;
  const outputArea = outputSize * outputSize;
  const inputArea = inputSize * inputSize;
  const output = new Float32Array(outChannels * outputArea);

  if (bias) {
    for (let channel = 0; channel < outChannels; channel++) {
      const channelStart = channel * outputArea;
      output.fill(bias[channel], channelStart, channelStart + outputArea);
    }
  }

  for (let inChannel = 0; inChannel < inChannels; inChannel++) {
    const inputStart = inChannel * inputArea;
    for (let y = 0; y < inputSize; y++) {
      for (let x = 0; x < inputSize; x++) {
        const value = input[inputStart + y * inputSize + x];
        const outputY = y * 2 - 1;
        const outputX = x * 2 - 1;
        for (let outChannel = 0; outChannel < outChannels; outChannel++) {
          const weightStart = (inChannel * outChannels + outChannel) * 16;
          const outputStart = outChannel * outputArea;
          for (let kernelY = 0; kernelY < 4; kernelY++) {
            const targetY = outputY + kernelY;
            if (targetY < 0 || targetY >= outputSize) continue;
            for (let kernelX = 0; kernelX < 4; kernelX++) {
              const targetX = outputX + kernelX;
              if (targetX < 0 || targetX >= outputSize) continue;
              const weightIndex = weightStart + kernelY * 4 + kernelX;
              const outputIndex = outputStart + targetY * outputSize + targetX;
              output[outputIndex] += value * weights[weightIndex];
            }
          }
        }
      }
    }
  }

  return output;
}

function generateDcganImage(generator, label, noise) {
  const input = new Float32Array(generator.noiseDim + generator.numClasses);
  input.set(noise);
  input[generator.noiseDim + label] = 1;

  let output = relu(matvec(
    generator.denseWeights,
    generator.denseBias,
    input,
    input.length,
    2048,
  ));
  output = relu(batchNorm(output, generator.batchNorm0, 128, 4));
  output = relu(batchNorm(
    transposeConv(output, generator.transposeWeights1, null, 128, 64, 4),
    generator.batchNorm1,
    64,
    8,
  ));
  output = relu(batchNorm(
    transposeConv(output, generator.transposeWeights2, null, 64, 32, 8),
    generator.batchNorm2,
    32,
    16,
  ));
  output = transposeConv(
    output,
    generator.transposeWeights3,
    generator.transposeBias3,
    32,
    3,
    16,
  );

  for (let i = 0; i < output.length; i++) {
    output[i] = Math.tanh(output[i]);
  }
  return output;
}

function generateCifarImage(generator, label, noise) {
  if (generator.architectureId === 1) {
    return generateDcganImage(generator, label, noise);
  }
  return generateMlpImage(generator, label, noise);
}

function drawCifarImage(canvas, pixels) {
  const ctx = canvas.getContext("2d");
  const imgData = ctx.createImageData(32, 32);
  for (let i = 0; i < 1024; i++) {
    if (pixels.length === 3072) {
      imgData.data[i * 4] = Math.round((pixels[i] + 1) * 127.5);
      imgData.data[i * 4 + 1] = Math.round((pixels[i + 1024] + 1) * 127.5);
      imgData.data[i * 4 + 2] = Math.round((pixels[i + 2048] + 1) * 127.5);
    } else {
      const value = Math.max(0, Math.min(255, Math.round(pixels[i] * 255)));
      imgData.data[i * 4] = value;
      imgData.data[i * 4 + 1] = value;
      imgData.data[i * 4 + 2] = value;
    }
    imgData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

window.createGenerator = createGenerator;
window.generateCifarImage = generateCifarImage;
window.drawCifarImage = drawCifarImage;
