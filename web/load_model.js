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

async function loadModelFromPNG(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const buffer = imageData.data.buffer;
        const dataView = new DataView(buffer);
        const header = parseHeader(dataView);
        let weights;
        if (header.mode === 0) {
          const floatBuffer = buffer.slice(HEADER_BYTES);
          weights = new Float32Array(floatBuffer, 0, header.totalParams);
        } else if (header.mode === 1) {
          const uint8View = new Uint8Array(buffer, HEADER_BYTES, header.totalParams);
          weights = new Float32Array(header.totalParams);
          for (let i = 0; i < header.totalParams; i++) {
            weights[i] = uint8View[i] * header.scale + header.minVal;
          }
        } else {
          throw new Error("Unknown mode");
        }
        resolve({ header, weights });
      } catch (e) {
        reject(e);
      }
    };
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = url;
  });
}

window.loadModelFromPNG = loadModelFromPNG;
