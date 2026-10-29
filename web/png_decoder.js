async function inflate(data) {
  if (typeof DecompressionStream !== "undefined") {
    const ds = new DecompressionStream("deflate");
    const stream = new Blob([data]).stream().pipeThrough(ds);
    const ab = await new Response(stream).arrayBuffer();
    return new Uint8Array(ab);
  }
  throw new Error("DecompressionStream not available");
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

function unfilter(filtered, width, height, bytesPerPixel) {
  const stride = width * bytesPerPixel;
  const out = new Uint8Array(height * stride);
  let src = 0;
  for (let y = 0; y < height; y++) {
    const filter = filtered[src++];
    const rowStart = y * stride;
    const prevStart = (y - 1) * stride;
    for (let i = 0; i < stride; i++) {
      const x = filtered[src++];
      const a = i >= bytesPerPixel ? out[rowStart + i - bytesPerPixel] : 0;
      const b = y > 0 ? out[prevStart + i] : 0;
      const c = y > 0 && i >= bytesPerPixel ? out[prevStart + i - bytesPerPixel] : 0;
      let val;
      switch (filter) {
        case 0:
          val = x;
          break;
        case 1:
          val = (x + a) & 255;
          break;
        case 2:
          val = (x + b) & 255;
          break;
        case 3:
          val = (x + ((a + b) >> 1)) & 255;
          break;
        case 4:
          val = (x + paeth(a, b, c)) & 255;
          break;
        default:
          throw new Error("Unsupported PNG filter " + filter);
      }
      out[rowStart + i] = val;
    }
  }
  return out;
}

async function decodePng(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  if (bytes.length < 8 ||
      bytes[0] !== 0x89 || bytes[1] !== 0x50 || bytes[2] !== 0x4e || bytes[3] !== 0x47 ||
      bytes[4] !== 0x0d || bytes[5] !== 0x0a || bytes[6] !== 0x1a || bytes[7] !== 0x0a) {
    throw new Error("Invalid PNG signature");
  }

  let offset = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  const idatParts = [];

  while (offset + 8 <= bytes.length) {
    const length = (bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3];
    const type = String.fromCharCode(bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7]);
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (dataEnd + 4 > bytes.length) throw new Error("Truncated PNG chunk");

    if (type === "IHDR") {
      width = (bytes[dataStart] << 24) | (bytes[dataStart + 1] << 16) | (bytes[dataStart + 2] << 8) | bytes[dataStart + 3];
      height = (bytes[dataStart + 4] << 24) | (bytes[dataStart + 5] << 16) | (bytes[dataStart + 6] << 8) | bytes[dataStart + 7];
      bitDepth = bytes[dataStart + 8];
      colorType = bytes[dataStart + 9];
      const compression = bytes[dataStart + 10];
      const filterMethod = bytes[dataStart + 11];
      const interlace = bytes[dataStart + 12];
      if (bitDepth !== 8) throw new Error("Only 8-bit PNG supported");
      if (colorType !== 6) throw new Error("Only RGBA PNG supported");
      if (compression !== 0 || filterMethod !== 0) throw new Error("Unsupported PNG compression/filter method");
      if (interlace !== 0) throw new Error("Interlaced PNG not supported");
    } else if (type === "IDAT") {
      idatParts.push(bytes.subarray(dataStart, dataEnd));
    } else if (type === "IEND") {
      break;
    }
    offset = dataEnd + 4;
  }

  if (!width || !height || idatParts.length === 0) {
    throw new Error("Incomplete PNG");
  }

  let totalIdat = 0;
  for (const p of idatParts) totalIdat += p.length;
  const compressed = new Uint8Array(totalIdat);
  let pos = 0;
  for (const p of idatParts) {
    compressed.set(p, pos);
    pos += p.length;
  }

  const inflated = await inflate(compressed);
  const bytesPerPixel = 4;
  const expected = height * (1 + width * bytesPerPixel);
  if (inflated.length < expected) {
    throw new Error("Decompressed data too short");
  }

  const pixels = unfilter(inflated, width, height, bytesPerPixel);
  return { width, height, data: pixels };
}

window.decodePng = decodePng;
