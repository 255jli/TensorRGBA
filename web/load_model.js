const MAGIC = "TRGBA001";
const HEADER_BYTES = 64;

function parseHeader(dataView) {
  let magic = "";
  for (let i = 0; i < 8; i++) {
    magic += String.fromCharCode(dataView.getUint8(i));
  }
  if (magic !== MAGIC) {
    throw new Error("Invalid TensorRGBA model file");
  }
  const mode = dataView.getUint8(8);
  const noiseDim = dataView.getUint16(9, true);
  const numClasses = dataView.getUint8(11);
  const totalParams = dataView.getUint32(12, true);
  let minVal = 0;
  let scale = 1;
  if (mode === 1) {
    minVal = dataView.getFloat32(16, true);
    scale = dataView.getFloat32(20, true);
  }
  return { mode, noiseDim, numClasses, totalParams, minVal, scale };
}

function extractWeights(buffer, header) {
  if (header.mode === 0) {
    const floatBuffer = buffer.slice(HEADER_BYTES);
    return new Float32Array(floatBuffer, 0, header.totalParams);
  }
  if (header.mode === 1) {
    const uint8View = new Uint8Array(buffer, HEADER_BYTES, header.totalParams);
    const weights = new Float32Array(header.totalParams);
    for (let i = 0; i < header.totalParams; i++) {
      weights[i] = uint8View[i] * header.scale + header.minVal;
    }
    return weights;
  }
  throw new Error("Unknown mode");
}

async function decodeWithCreateImageBitmap(blob) {
  const bitmap = await createImageBitmap(blob, {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return imageData.data.buffer;
}

async function decodeWithImageDecoder(arrayBuffer) {
  if (typeof ImageDecoder === "undefined") {
    throw new Error("ImageDecoder not supported");
  }
  const decoder = new ImageDecoder({
    data: arrayBuffer,
    type: "image/png",
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const result = await decoder.decode();
  const frame = result.image;
  const width = frame.displayWidth;
  const height = frame.displayHeight;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(frame, 0, 0);
  frame.close();
  decoder.close();
  const imageData = ctx.getImageData(0, 0, width, height);
  return imageData.data.buffer;
}

async function loadModelFromPNG(source) {
  let blob;
  let arrayBuffer;

  if (source instanceof Blob) {
    blob = source;
    arrayBuffer = await source.arrayBuffer();
  } else if (typeof source === "string") {
    const response = await fetch(source);
    if (!response.ok) {
      throw new Error("Failed to fetch model");
    }
    arrayBuffer = await response.arrayBuffer();
    blob = new Blob([arrayBuffer], { type: "image/png" });
  } else {
    throw new Error("Unsupported source type");
  }

  let buffer;
  try {
    buffer = await decodeWithCreateImageBitmap(blob);
  } catch (e) {
    try {
      buffer = await decodeWithImageDecoder(arrayBuffer);
    } catch (e2) {
      throw new Error("Failed to decode PNG without color conversion");
    }
  }

  const dataView = new DataView(buffer);
  const header = parseHeader(dataView);
  const weights = extractWeights(buffer, header);
  return { header, weights };
}

window.loadModelFromPNG = loadModelFromPNG;
