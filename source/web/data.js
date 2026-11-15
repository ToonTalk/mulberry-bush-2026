// Unpacks build/web/data.b64 (written by web_export.py): gzip, a JSON manifest,
// then int16 columns (dequantised per column), float32 arrays and uint8 arrays.
async function decodeData(b64) {
  const bin = Uint8Array.from(atob(b64.replace(/\s+/g, "")), (c) => c.charCodeAt(0));
  const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"));
  const buf = await new Response(stream).arrayBuffer();
  const hl = new DataView(buf).getUint32(0, true);
  const man = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
  const base = 4 + hl;
  const out = {};
  for (const name of Object.keys(man)) {
    const m = man[name];
    const off = base + m.off, n = m.rows * m.cols;
    let a;
    if (m.t === "q16") {
      const q = new Int16Array(buf.slice(off, off + n * 2));
      a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = q[i] / m.sc[i % m.cols];
    } else if (m.t === "f32") {
      a = new Float32Array(buf.slice(off, off + n * 4));
    } else {
      a = new Uint8Array(buf.slice(off, off + n));
    }
    out[name] = { rows: m.rows, cols: m.cols, a };
  }
  return out;
}
if (typeof module !== "undefined") module.exports = { decodeData };
