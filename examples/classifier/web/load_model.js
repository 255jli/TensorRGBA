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
  const architectureId = dataView.getUint8(24);
  return { mode, noiseDim, numClasses, totalParams, minVal, scale, architectureId };
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

async function loadModelFromPNG(source) {
  let arrayBuffer;
  if (source instanceof Blob) {
    arrayBuffer = await source.arrayBuffer();
  } else if (typeof source === "string") {
    const response = await fetch(source);
    if (!response.ok) {
      throw new Error("Failed to fetch model");
    }
    arrayBuffer = await response.arrayBuffer();
  } else {
    throw new Error("Unsupported source type");
  }

  const { data } = await decodePng(arrayBuffer);
  const buffer = data.buffer;
  const dataView = new DataView(buffer);
  const header = parseHeader(dataView);
  const weights = extractWeights(buffer, header);
  return { header, weights };
}

window.loadModelFromPNG = loadModelFromPNG;
