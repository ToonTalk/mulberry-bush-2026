// The picture: a port of render.py to WebGL2.
// The CPU computes the same per-element state as render.py (growth, colour,
// projected shadows) and writes it as instanced primitives (lines, ellipses,
// textured shapes). The GPU draws them with analytic anti-aliasing into layers,
// blurs the shadow masks and glow, and does the per-pixel lighting composite.
const Film = (() => {
  const W = 1080, H = 1080, HW = 540, TEXN = 1800;
  const COL = [[1.00, 0.30, 0.56], [0.22, 0.86, 0.74], [0.50, 0.42, 1.00], [1.00, 0.76, 0.30]];
  const AMB = [0.006, 0.004, 0.016];
  const FALLP = 1.3, EXPO = 4.8, GAM = 1 / 1.55, FS = 1.3;
  const PI = Math.PI;
  const LEAF = [[0.00, 0.00], [0.06, 0.20], [0.22, 0.40], [0.45, 0.46], [0.68, 0.36], [0.86, 0.18],
    [1.00, 0.00], [0.86, -0.18], [0.68, -0.36], [0.45, -0.46], [0.22, -0.40], [0.06, -0.20]];
  const FOOT = [[-10, 0], [-9.3, 2.6], [-7.5, 3.6], [-5, 3.6], [-2, 3.5], [1, 3.8], [4, 4.1], [6.2, 3.4],
    [7, 1.0], [6.8, -1.5], [6, -3.6], [3.5, -4.2], [1, -3.2], [-1.5, -1.6], [-4.5, -1.8], [-7.5, -3.0], [-9.3, -2.4]];
  const TOES = [[9.1, -2.5, 1.75], [9.3, 0.3, 1.15], [8.7, 2.0, 1.0], [7.9, 3.4, 0.9], [7.0, 4.5, 0.8]];
  // texture regions of the shape masks, in each shape's own units
  const LEAF_U = [-0.05, 1.05], LEAF_V = [-0.55, 0.55];
  const FOOT_U = [-11, 12], FOOT_V = [-6, 6.5];
  const STAIN_R = 15;
  const RING_TEXT = "HERE WE GO ROUND THE MULBERRY BUSH  ·  SO EARLY IN THE MORNING  ·  ";
  const CREDITS = [["title", "Mulberry Bush", 111.0], ["it", "a round for four dancers", 111.8],
    ["small", "after the 1976 film and dance by Ken Kahn", 112.8],
    ["small", "images and music written in code by Claude · 2026", 113.4]];
  const SERIF = '"Cormorant Garamond", Garamond, "EB Garamond", Georgia, serif';

  let gl, P_prim, P_blur, P_comp, vao, ibuf, texShape, texStain, texGround, texText, texBlack;
  let fbMarks, fbBush, fbEm, fbMask = [], fbMaskTmp, fbSun, fbSunTmp, fbGlowA, fbGlowB;
  let world, textCanvas, textCtx, ringAdv = null;
  const U = {};

  // ------------------------------------------------------------ primitives
  const PF = 12;
  let prim = new Float32Array(PF * 160000);
  let np = 0;
  function room() {
    if ((np + 1) * PF > prim.length) {
      const n = new Float32Array(prim.length * 2);
      n.set(prim);
      prim = n;
    }
    return np++ * PF;
  }
  function line(x0, y0, x1, y1, w, r, g, b, a) {
    const o = room();
    prim[o] = x0; prim[o + 1] = y0; prim[o + 2] = x1; prim[o + 3] = y1;
    prim[o + 4] = w; prim[o + 5] = 0; prim[o + 6] = 0; prim[o + 7] = 0;
    prim[o + 8] = r; prim[o + 9] = g; prim[o + 10] = b; prim[o + 11] = a;
  }
  function ell(cx, cy, rx, ry, ang, r, g, b, a) {
    const o = room();
    prim[o] = cx; prim[o + 1] = cy; prim[o + 2] = rx; prim[o + 3] = ry;
    prim[o + 4] = ang; prim[o + 5] = 0; prim[o + 6] = 0; prim[o + 7] = 1;
    prim[o + 8] = r; prim[o + 9] = g; prim[o + 10] = b; prim[o + 11] = a;
  }
  function shape(cx, cy, ux, uy, vx, vy, layer, kind, r, g, b, a) {
    const o = room();
    prim[o] = cx; prim[o + 1] = cy; prim[o + 2] = ux; prim[o + 3] = uy;
    prim[o + 4] = vx; prim[o + 5] = vy; prim[o + 6] = layer; prim[o + 7] = kind;
    prim[o + 8] = r; prim[o + 9] = g; prim[o + 10] = b; prim[o + 11] = a;
  }
  // a leaf with its base at screen point (bx, by): fx,fy = unit forward on screen, len = length in px
  function leafPrim(bx, by, fx, fy, len, r, g, b, a) {
    const cu = (LEAF_U[0] + LEAF_U[1]) / 2, hu = (LEAF_U[1] - LEAF_U[0]) / 2, hv = (LEAF_V[1] - LEAF_V[0]) / 2 * 0.78;
    shape(bx + fx * len * cu, by + fy * len * cu, fx * len * hu, fy * len * hu, -fy * len * hv, fx * len * hv, 0, 2, r, g, b, a);
  }

  // ------------------------------------------------------------ shaders
  const VS_PRIM = `#version 300 es
  precision highp float;
  layout(location=0) in vec4 iA; layout(location=1) in vec4 iB; layout(location=2) in vec4 iC;
  uniform vec2 uRes;
  out vec2 vL; out vec4 vP; out vec4 vC; flat out int vK; flat out float vLayer;
  const vec2 Q[6] = vec2[6](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,-1), vec2(1,1), vec2(-1,1));
  void main() {
    vec2 q = Q[gl_VertexID];
    int k = int(iB.w + 0.5);
    vK = k; vC = iC; vLayer = iB.z; vP = vec4(0.0);
    vec2 p;
    if (k == 0) {
      vec2 a = iA.xy, d = iA.zw - iA.xy;
      float L = length(d);
      vec2 t = L > 1e-5 ? d / L : vec2(1.0, 0.0);
      vec2 n = vec2(-t.y, t.x);
      float hw = max(iB.x, 1.0) * 0.5 + 1.0;
      float s = mix(-1.0, L + 1.0, q.x * 0.5 + 0.5);
      float r = q.y * hw;
      p = a + t * s + n * r;
      vL = vec2(s, r); vP = vec4(L, iB.x, 0.0, 0.0);
    } else if (k == 1) {
      vec2 r = max(iA.zw, vec2(0.5)) + 1.0;
      float c = cos(iB.x), sn = sin(iB.x);
      vec2 l = q * r;
      p = iA.xy + vec2(l.x * c - l.y * sn, l.x * sn + l.y * c);
      vL = l; vP = vec4(iA.zw, 0.0, 0.0);
    } else {
      p = iA.xy + iA.zw * q.x + iB.xy * q.y;
      vL = q * 0.5 + 0.5;
    }
    gl_Position = vec4(p.x / uRes.x * 2.0 - 1.0, 1.0 - p.y / uRes.y * 2.0, 0.0, 1.0);
  }`;
  const FS_PRIM = `#version 300 es
  precision highp float; precision highp sampler2DArray;
  in vec2 vL; in vec4 vP; in vec4 vC; flat in int vK; flat in float vLayer;
  uniform sampler2DArray uShape, uStain;
  // Pillow draws on render.py's RGBA layers without blending: each shape overwrites
  // colour and alpha where it covers. Target 0 holds the colour, target 1 the alpha;
  // both are blended by coverage only, which is that overwrite with anti-aliasing.
  layout(location = 0) out vec4 o;
  layout(location = 1) out vec4 o1;
  void main() {
    float cov;
    if (vK == 0) {
      float L = vP.x, w = vP.y, ww = max(w, 1.0);
      float cd = clamp(ww * 0.5 + 0.5 - abs(vL.y), 0.0, 1.0) * min(w, 1.0);
      float cs = clamp(vL.x + 0.5, 0.0, 1.0) * clamp(L - vL.x + 0.5, 0.0, 1.0);
      cov = cd * cs;
    } else if (vK == 1) {
      vec2 r = vP.xy, re = max(r, vec2(0.5));
      float k1 = length(vL / re), k2 = length(vL / (re * re));
      float sd = k2 > 1e-6 ? k1 * (k1 - 1.0) / k2 : -min(re.x, re.y);
      cov = clamp(0.5 - sd, 0.0, 1.0) * (r.x * r.y) / (re.x * re.y);
    } else if (vK == 2) {
      cov = texture(uShape, vec3(vL, vLayer), -0.6).r;
    } else {
      cov = texture(uStain, vec3(vL, vLayer), -0.6).r;
    }
    if (cov <= 0.0) discard;
    o = vec4(vC.rgb, cov);
    o1 = vec4(vC.a, 0.0, 0.0, cov);
  }`;
  const VS_FULL = `#version 300 es
  void main() { vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0); gl_Position = vec4(p, 0.0, 1.0); }`;
  const FS_BLUR = `#version 300 es
  precision highp float;
  uniform sampler2D uTex; uniform vec2 uDir; uniform float uSigma; uniform vec2 uInvRes;
  out vec4 o;
  void main() {
    vec2 uv = gl_FragCoord.xy * uInvRes;
    int R = int(ceil(uSigma * 3.0));
    vec4 acc = vec4(0.0); float ws = 0.0;
    for (int i = -40; i <= 40; i++) {
      if (i < -R || i > R) continue;
      float w = exp(-0.5 * float(i * i) / (uSigma * uSigma));
      acc += w * texture(uTex, uv + uDir * float(i));
      ws += w;
    }
    o = acc / ws;
  }`;
  const FS_COMP = `#version 300 es
  precision highp float;
  uniform sampler2D uGround, uMarks, uMarksA, uBush, uBushA, uEm, uGlow, uText, uSun, uM0, uM1, uM2, uM3;
  uniform float uS, uHl, uFade, uSunI, uSunOn, uTextOn, uSf, uSeedA, uFrame;
  uniform vec2 uP[4]; uniform float uI[4]; uniform vec3 uCol[4]; uniform vec3 uAmb, uSunCol;
  out vec4 o;
  const float EXPO = ${EXPO.toFixed(3)}, GAM = ${GAM.toFixed(6)}, FALLP = ${FALLP.toFixed(3)};
  vec3 tm(vec3 x) { return pow(1.0 - exp(-EXPO * max(x, 0.0)), vec3(GAM)); }
  uint hash(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
  void main() {
    vec2 fc = gl_FragCoord.xy;
    vec2 uv = fc / vec2(${W}.0, ${H}.0);
    vec2 X = vec2(fc.x - ${W / 2}.0, fc.y - ${H / 2}.0) / uS;
    vec3 alb = texture(uGround, vec2(${TEXN / 2}.0 + X.x, ${TEXN / 2}.0 - X.y) / ${TEXN}.0).rgb;
    // layers as render.py composites them: colour * alpha over what lies beneath. Pillow's
    // 2x downsampling averaged colour and alpha separately, darkening soft edges; so does this.
    vec3 mk = texture(uMarks, uv).rgb;
    float ma = texture(uMarksA, uv).r;
    alb = alb * (1.0 - ma) + mk * ma;
    float m[4];
    m[0] = texture(uM0, uv).r; m[1] = texture(uM1, uv).r; m[2] = texture(uM2, uv).r; m[3] = texture(uM3, uv).r;
    vec3 Lm = uAmb;
    float H2 = uHl * uHl;
    for (int k = 0; k < 4; k++) {
      if (uI[k] < 0.01) continue;
      vec2 h = X - uP[k];
      float d2 = dot(h, h) + H2;
      vec2 cdir = -uP[k] / (length(uP[k]) + 1e-6);
      float c = dot(h, cdir) / (length(h) + 40.0);
      float sp = 0.32 + 0.68 * pow((1.0 + c) * 0.5, 1.6) * 1.25;
      float E = uI[k] * pow(H2 / d2, FALLP) * sp;
      E *= 1.0 - 0.93 * m[k];
      Lm += E * uCol[k];
    }
    if (uSunI > 0.0) {
      vec3 sl = uSunI * uSunCol * 0.62;
      float sm = uSunOn > 0.5 ? texture(uSun, uv).r : 0.0;
      Lm += (1.0 - 0.88 * sm) * sl;
      Lm += vec3(0.08, 0.09, 0.15) * 0.28 * uSunI;
    }
    vec3 base = tm(alb * Lm);
    vec3 bu = texture(uBush, uv).rgb;
    float ba = texture(uBushA, uv).r;
    base = base * (1.0 - ba) + bu * ba;
    vec3 em = texture(uEm, uv).rgb + texture(uGlow, uv).rgb * 1.6;
    vec2 pc = fc - 0.5;
    for (int k = 0; k < 4; k++) {
      if (uI[k] < 0.01) continue;
      vec2 c = vec2(${W / 2}.0 + uS * uP[k].x, ${H / 2}.0 + uS * uP[k].y);
      vec2 d = pc - floor(c + 0.5);
      if (abs(d.x) > 140.0 || abs(d.y) > 140.0) continue;
      float rr = length(d) / uSf;
      float core = exp(-(rr / 3.3) * (rr / 3.3)) * 1.25;
      float halo = exp(-(rr / 9.0) * (rr / 9.0)) * 0.75 + exp(-rr / 24.0) * 0.32;
      em += (core * vec3(1.0, 0.97, 0.92) + halo * uCol[k]) * uI[k];
    }
    if (uSeedA > 0.0) {
      vec2 d = pc - vec2(${W / 2}.0, ${H / 2}.0);
      if (abs(d.x) <= 140.0 && abs(d.y) <= 140.0) {
        float rr = length(d) / max(uS, 0.5);
        em += (exp(-(rr / 6.0) * (rr / 6.0)) * 0.8 + exp(-rr / 20.0) * 0.25) * vec3(1.0, 0.75, 0.55) * uSeedA;
      }
    }
    base = 1.0 - (1.0 - base) * (1.0 - clamp(em, 0.0, 1.0));
    float vr = length(pc - vec2(${W / 2}.0, ${H / 2}.0)) / (${W}.0 * 0.72);
    base *= 1.0 - 0.42 * pow(vr, 2.4);
    base *= uFade;
    uvec2 ip = uvec2(fc);
    uint h1 = hash(ip.x + 4099u * hash(ip.y + 7919u * hash(uint(uFrame) + 17u)));
    uint h2 = hash(h1 ^ 0x9e3779b9u);
    float u1 = (float(h1 >> 8) + 0.5) / 16777216.0, u2 = float(h2 >> 8) / 16777216.0;
    base += sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2) * 0.011;
    base = clamp(base, 0.0, 1.0);
    if (uTextOn > 0.5) {
      vec4 tx = texture(uText, vec2(uv.x, 1.0 - uv.y));
      base = base * (1.0 - tx.a) + tx.rgb;
    }
    o = vec4(base, 1.0);
  }`;

  function compile(vs, fs) {
    const mk = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, mk(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(p, i);
      const name = info.name.replace(/\[0\]$/, "");
      u[name] = gl.getUniformLocation(p, info.name);
    }
    return { p, u };
  }

  function makeTex(w, h, filter) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  function makeFB(w, h, two) {
    const tex = makeTex(w, h, gl.LINEAR);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    let tex2 = null;
    if (two) {
      tex2 = makeTex(w, h, gl.LINEAR);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, tex2, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    }
    return { fb, tex, tex2, w, h };
  }

  // shape masks (leaf, foot) and the 216 juice stains, rasterised once with canvas 2D
  function makeShapeTextures() {
    const N = 256;
    const cv = document.createElement("canvas");
    cv.width = cv.height = N;
    const c = cv.getContext("2d", { willReadFrequently: true });
    const layers = [];
    const grab = () => {
      const d = c.getImageData(0, 0, cv.width, cv.height).data;
      const out = new Uint8Array(cv.width * cv.height);
      for (let i = 0; i < out.length; i++) out[i] = d[i * 4 + 3];
      return out;
    };
    // leaf
    c.clearRect(0, 0, N, N);
    c.fillStyle = "#fff";
    c.beginPath();
    LEAF.forEach(([u, v], i) => {
      const x = (u - LEAF_U[0]) / (LEAF_U[1] - LEAF_U[0]) * N, y = (v - LEAF_V[0]) / (LEAF_V[1] - LEAF_V[0]) * N;
      i ? c.lineTo(x, y) : c.moveTo(x, y);
    });
    c.closePath(); c.fill();
    layers.push(grab());
    // foot (sole + toes), drawn for the sgn = +1 foot
    c.clearRect(0, 0, N, N);
    const fx = (u) => (u - FOOT_U[0]) / (FOOT_U[1] - FOOT_U[0]) * N;
    const fy = (v) => (v - FOOT_V[0]) / (FOOT_V[1] - FOOT_V[0]) * N;
    c.beginPath();
    FOOT.forEach(([u, v], i) => (i ? c.lineTo(fx(u), fy(v)) : c.moveTo(fx(u), fy(v))));
    c.closePath(); c.fill();
    for (const [u, v, r] of TOES) {
      c.beginPath();
      c.ellipse(fx(u), fy(v), r / (FOOT_U[1] - FOOT_U[0]) * N, r / (FOOT_V[1] - FOOT_V[0]) * N, 0, 0, 2 * PI);
      c.fill();
    }
    layers.push(grab());
    texShape = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, texShape);
    gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.R8, N, N, 2, 0, gl.RED, gl.UNSIGNED_BYTE, null);
    layers.forEach((d, i) => gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, i, N, N, 1, gl.RED, gl.UNSIGNED_BYTE, d));
    gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    aniso();

    // stains: polygon + drops, in world units around the stain centre (y up)
    const M = 128;
    cv.width = cv.height = M;
    const st = world.stains;
    const nS = st.rows;
    texStain = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, texStain);
    gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.R8, M, M, nS, 0, gl.RED, gl.UNSIGNED_BYTE, null);
    const sx = (x) => (x + STAIN_R) / (2 * STAIN_R) * M, sy = (y) => (STAIN_R - y) / (2 * STAIN_R) * M;
    for (let i = 0; i < nS; i++) {
      const r = st.a.subarray(i * st.cols, (i + 1) * st.cols);
      c.clearRect(0, 0, M, M);
      c.fillStyle = "#fff";
      c.beginPath();
      for (let j = 0; j < 11; j++) (j ? c.lineTo(sx(r[2 * j]), sy(r[2 * j + 1])) : c.moveTo(sx(r[0]), sy(r[1])));
      c.closePath(); c.fill();
      for (let j = 0; j < 6; j++) {
        const dx = r[22 + 3 * j], dy = r[23 + 3 * j], rr = r[24 + 3 * j];
        if (rr <= 0) continue;
        c.beginPath();
        c.arc(sx(dx), sy(dy), rr / (2 * STAIN_R) * M, 0, 2 * PI);
        c.fill();
      }
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, i, M, M, 1, gl.RED, gl.UNSIGNED_BYTE, grab());
    }
    gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    aniso();
  }
  // footprints and leaves are long and thin on screen: sample their masks anisotropically
  function aniso() {
    const ext = gl.getExtension("EXT_texture_filter_anisotropic");
    if (!ext) return;
    const mx = gl.getParameter(ext.MAX_TEXTURE_MAX_ANISOTROPY_EXT);
    gl.texParameterf(gl.TEXTURE_2D_ARRAY, ext.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, mx));
  }

  function init(canvas, worldData) {
    world = worldData;
    gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: false, stencil: false,
      premultipliedAlpha: false, preserveDrawingBuffer: true, powerPreference: "high-performance" });
    if (!gl) throw new Error("This page needs WebGL 2.");
    P_prim = compile(VS_PRIM, FS_PRIM);
    P_blur = compile(VS_FULL, FS_BLUR);
    P_comp = compile(VS_FULL, FS_COMP);
    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    ibuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, ibuf);
    for (let i = 0; i < 3; i++) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribDivisor(i, 1);
    }
    fbMarks = makeFB(W, H, true); fbBush = makeFB(W, H, true); fbEm = makeFB(W, H);
    for (let k = 0; k < 4; k++) fbMask.push(makeFB(HW, HW));
    fbMaskTmp = makeFB(HW, HW); fbSun = makeFB(HW, HW); fbSunTmp = makeFB(HW, HW);
    fbGlowA = makeFB(HW, HW); fbGlowB = makeFB(HW, HW);
    texText = makeTex(W, H, gl.LINEAR);
    texGround = makeTex(TEXN, TEXN, gl.LINEAR);
    makeShapeTextures();
    textCanvas = document.createElement("canvas");
    textCanvas.width = W; textCanvas.height = H;
    textCtx = textCanvas.getContext("2d");
  }

  function setGround(canvasOrImage) {
    gl.bindTexture(gl.TEXTURE_2D, texGround);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, canvasOrImage);
  }

  // ------------------------------------------------------------ state
  const Pp = [[0, 0], [0, 0], [0, 0], [0, 0]];
  const I = [0, 0, 0, 0];
  let Hl = 150, Hl2 = 22500, S = 1, sunI = 0, sunCol = [1, 1, 1];
  const ir = [0, 0, 0], tc = [0, 0, 0];
  const LR = [0, 0, 0, 0], LCX = [0, 0, 0, 0], LCY = [0, 0, 0, 0];

  function irr(x, y) {
    let r = AMB[0], g = AMB[1], b = AMB[2];
    for (let k = 0; k < 4; k++) {
      if (I[k] <= 1e-3) continue;
      const hx = x - Pp[k][0], hy = y - Pp[k][1];
      const d2 = hx * hx + hy * hy + Hl2;
      const c = (hx * LCX[k] + hy * LCY[k]) / (Math.sqrt(hx * hx + hy * hy) + 40.0);
      const sp = 0.32 + 0.68 * Math.pow((1 + c) * 0.5, 1.6) * 1.25;
      const E = I[k] * Math.pow(Hl2 / d2, FALLP) * sp;
      r += E * COL[k][0]; g += E * COL[k][1]; b += E * COL[k][2];
    }
    if (sunI > 0) {
      r += sunI * (sunCol[0] * 0.62 + 0.08 * 0.28);
      g += sunI * (sunCol[1] * 0.62 + 0.09 * 0.28);
      b += sunI * (sunCol[2] * 0.62 + 0.15 * 0.28);
    }
    ir[0] = r; ir[1] = g; ir[2] = b;
  }
  const tm1 = (x) => (x <= 0 ? 0 : Math.pow(1 - Math.exp(-EXPO * x), GAM));
  function tm3(r, g, b) { tc[0] = tm1(r); tc[1] = tm1(g); tc[2] = tm1(b); return tc; }
  const smooth = (u) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));
  const sx = (x) => W / 2 + S * x, sy = (y) => H / 2 - S * y;

  // visible segments, in draw order
  let vis = new Int32Array(8192), vx1 = new Float32Array(8192), vy1 = new Float32Array(8192),
    vw = new Float32Array(8192), vage = new Float32Array(8192), nvis = 0;

  // ------------------------------------------------------------ frame build
  const passes = {};
  function begin(name) { passes[name] = [np, 0]; }
  function end(name) { passes[name][1] = np - passes[name][0]; }

  function build(t) {
    np = 0;
    const b = TL.B_of_t(t);
    S = TL.zoom(b);
    for (let k = 0; k < 4; k++) {
      TL.pos(k, b, Pp[k]);
      const fl = 1 + 0.035 * Math.sin(2 * PI * 6.1 * t + 1.3 * k) + 0.025 * Math.sin(2 * PI * 9.7 * t + 2.1 * k)
        + 0.02 * Math.sin(2 * PI * 2.3 * t + k);
      I[k] = TL.intensity(k, b) * fl;
      LR[k] = Math.hypot(Pp[k][0], Pp[k][1]) + 1e-6;
      LCX[k] = -Pp[k][0] / LR[k]; LCY[k] = -Pp[k][1] / LR[k];
    }
    Hl = TL.light_height(b); Hl2 = Hl * Hl;
    const sun = TL.sun(b);
    sunI = sun.I; sunCol = sun.col;
    const ST = TL.ST;
    const SG = world.seg.a, nSeg = world.seg.rows;
    const LV = world.leaves.a, nL = world.leaves.rows, LC = world.leaves.cols;
    const BV = world.berries.a, nB = world.berries.rows, BC = world.berries.cols;

    // visible segments (render.py seg_state)
    nvis = 0;
    if (t >= ST.grow) {
      for (let i = 0; i < nSeg; i++) {
        const o = i * 9;
        const t0 = SG[o + 4];
        if (t0 > t) continue;
        const frac = Math.min(1, Math.max(0, (t - t0) / Math.max(SG[o + 5] - t0, 1e-3)));
        const age = t - t0;
        vis[nvis] = i;
        vx1[nvis] = SG[o] + (SG[o + 2] - SG[o]) * frac;
        vy1[nvis] = SG[o + 1] + (SG[o + 3] - SG[o + 1]) * frac;
        vw[nvis] = SG[o + 6] * (0.3 + 0.7 * smooth(age / 22.0)) * Math.min(1, 0.45 + age / 1.5);
        vage[nvis] = age;
        nvis++;
      }
    }
    const leafOn = t >= ST.leaf;

    // ---- marks: juice stains and footprints, in the ground's albedo
    begin("marks");
    const ST_ = world.stains;
    for (let i = 0; i < nB; i++) {
      const o = i * BC;
      const ts = BV[o + 9] + 0.35;
      if (t < ts) continue;
      const fresh = Math.exp(-(t - ts) / 6.0);
      const sc = Math.min(1.0, (t - ts) / 0.12);
      const cx = sx(BV[o] + BV[o + 10]), cy = sy(BV[o + 1] + BV[o + 11]);
      const h = STAIN_R * S * sc;
      shape(cx, cy, h, 0, 0, h, i, 3, 0.20 + 0.10 * fresh, 0.025, 0.13 + 0.06 * fresh, 235 / 255);
    }
    const PR = world.prints.a, nP = world.prints.rows;
    const fcu = (FOOT_U[0] + FOOT_U[1]) / 2, fhu = (FOOT_U[1] - FOOT_U[0]) / 2;
    const fcv = (FOOT_V[0] + FOOT_V[1]) / 2, fhv = (FOOT_V[1] - FOOT_V[0]) / 2;
    for (let i = 0; i < nP; i++) {
      const o = i * 7;
      const tp = PR[o];
      if (tp > t) continue;
      const age = t - tp;
      let dew = 0;
      if (age < 7.0) dew = 0.55 * Math.pow(1 - age / 7.0, 1.3) * Math.min(1, age / 0.12);
      const sj = Math.sin(tp * 37.1);
      const aj = 0.9 * PR[o + 5] * Math.min(1, age / 0.12) * (0.62 + 0.38 * Math.exp(-age / 25.0)) * (0.7 + 0.3 * sj * sj);
      if (dew < 0.01 && aj < 0.01) continue;
      const a = PR[o + 3];
      const sgn = PR[o + 4] === 1 ? 1 : -1;
      // forward (cos a, sin a), right (sin a, -cos a) in world; on screen y flips
      const fx = Math.cos(a), fy = Math.sin(a), rx = Math.sin(a), ry = -Math.cos(a);
      const k = FS * S;
      const wx = PR[o + 1] + (fx * fcu + rx * sgn * fcv) * FS, wy = PR[o + 2] + (fy * fcu + ry * sgn * fcv) * FS;
      const cx = sx(wx), cy = sy(wy);
      const ux = fx * fhu * k, uy = -fy * fhu * k, vx = rx * sgn * fhv * k, vy = -ry * sgn * fhv * k;
      if (dew >= 0.01) shape(cx, cy, ux, uy, vx, vy, 1, 2, 160 / 255, 182 / 255, 200 / 255, dew);
      if (aj >= 0.01) shape(cx, cy, ux, uy, vx, vy, 1, 2, 52 / 255, 6 / 255, 38 / 255, aj);
    }
    end("marks");

    // ---- shadow masks, one per lantern (half resolution)
    const maskOn = [false, false, false, false];
    const hs = 0.5 * S;
    const mx = (x) => HW / 2 + hs * x, my = (y) => HW / 2 - hs * y;
    if (nvis > 0 || leafOn) {
      for (let k = 0; k < 4; k++) {
        const name = "mask" + k;
        begin(name);
        if (I[k] > 0.02) {
          maskOn[k] = true;
          const Lx = Pp[k][0], Ly = Pp[k][1];
          for (let j = 0; j < nvis; j++) {
            const o = vis[j] * 9;
            const ze0 = Math.min(SG[o + 7], 0.8 * Hl), ze1 = Math.min(SG[o + 8], 0.8 * Hl);
            const f0 = Hl / (Hl - ze0), f1 = Hl / (Hl - ze1);
            const u0 = mx(Lx + (SG[o] - Lx) * f0), v0 = my(Ly + (SG[o + 1] - Ly) * f0);
            const u1 = mx(Lx + (vx1[j] - Lx) * f1), v1 = my(Ly + (vy1[j] - Ly) * f1);
            const ww = vw[j] * (f0 + f1) * 0.25 * S * 1.5;
            if (ww < 1.0) { const v = Math.max(ww, 0.45); line(u0, v0, u1, v1, 1, v, v, v, 1); }
            else line(u0, v0, u1, v1, ww, 1, 1, 1, 1);
          }
          if (leafOn) {
            for (let j = 0; j < nL; j++) {
              const o = j * LC;
              if (LV[o + 5] > t || t >= LV[o + 7]) continue;
              const g = smooth((t - LV[o + 5]) / 3.5);
              if (g < 0.05) continue;
              const f = Hl / (Hl - Math.min(LV[o + 2], 0.8 * Hl));
              const bx = mx(Lx + (LV[o] - Lx) * f), by = my(Ly + (LV[o + 1] - Ly) * f);
              const a = LV[o + 3];
              leafPrim(bx, by, Math.cos(a), -Math.sin(a), LV[o + 4] * g * f * hs, 125 / 255, 125 / 255, 125 / 255, 1);
            }
          }
          for (let j = 0; j < nB; j++) {
            const o = j * BC;
            if (t < BV[o + 5] || t > BV[o + 9] + 0.35) continue;
            const f = Hl / (Hl - Math.min(BV[o + 2], 0.8 * Hl));
            const rr = hs * f * 3.2 * BV[o + 4] * Math.min(1, (t - BV[o + 5]) / 4);
            ell(mx(Lx + (BV[o] - Lx) * f), my(Ly + (BV[o + 1] - Ly) * f), rr, rr, 0, 230 / 255, 230 / 255, 230 / 255, 1);
          }
        }
        end(name);
      }
    } else {
      for (let k = 0; k < 4; k++) { begin("mask" + k); end("mask" + k); }
    }
    // ---- the sun's shadow
    begin("sun");
    let sunOn = false, sunBlur = 1.2;
    if (sunI > 0.005 && nvis > 0) {
      sunOn = true;
      const Ls = 1.0 / Math.tan(sun.el * PI / 180);
      const dx = Math.cos(TL.SUN_AZ), dy = Math.sin(TL.SUN_AZ);
      for (let j = 0; j < nvis; j++) {
        const o = vis[j] * 9;
        const z0 = SG[o + 7], z1 = SG[o + 8];
        const u0 = mx(SG[o] - dx * z0 * Ls), v0 = my(SG[o + 1] - dy * z0 * Ls);
        const u1 = mx(vx1[j] - dx * z1 * Ls), v1 = my(vy1[j] - dy * z1 * Ls);
        const ww = vw[j] * 0.5 * S;
        if (ww < 1.0) { const v = Math.max(ww, 0.3); line(u0, v0, u1, v1, 1, v, v, v, 1); }
        else line(u0, v0, u1, v1, ww, 1, 1, 1, 1);
      }
      sunBlur = 1.2 + 2.5 * Math.min(1.0, Ls / 30);
    }
    end("sun");

    // ---- the bush layer
    begin("bush");
    if (t < ST.grow + 6) {
      const a = 1 - smooth((t - ST.grow - 2) / 4);
      if (a > 0) {
        irr(0, 0);
        const r0 = ir[0], g0 = ir[1], b0 = ir[2];
        const c = tm3(0.55 * r0, 0.20 * g0, 0.22 * b0);
        const rx = 4.2 * S, ry = 3.0 * S;
        ell(sx(0), sy(0), rx, ry, 0, c[0], c[1], c[2], a);
        const h = tm3(r0 * 0.9, g0 * 0.9, b0 * 0.9);
        ell(sx(0) - rx * 0.275, sy(0) - ry * 0.425, rx * 0.225, ry * 0.275, 0, h[0], h[1], h[2], 160 / 255 * a);
      }
    }
    for (let j = 0; j < nvis; j++) {
      const o = vis[j] * 9;
      const x0 = SG[o], y0 = SG[o + 1], x1 = vx1[j], y1 = vy1[j];
      irr((x0 + x1) / 2, (y0 + y1) / 2);
      const yu = Math.min(1, Math.max(0, 1 - vage[j] / 3.0));
      const c = tm3((0.33 * (1 - yu) + 0.62 * yu) * ir[0] * 1.15, (0.24 * (1 - yu) + 0.80 * yu) * ir[1] * 1.15,
        (0.21 * (1 - yu) + 0.42 * yu) * ir[2] * 1.15);
      const u0 = sx(x0), v0 = sy(y0), u1 = sx(x1), v1 = sy(y1);
      const wi = vw[j] * S * 2;            // width in render.py's 2x supersampled pixels
      if (wi < 1.6) line(u0, v0, u1, v1, wi < 1.2 ? 0.5 : 1.0, c[0], c[1], c[2], Math.min(1, wi / 1.6));
      else {
        const r = c[0], g = c[1], bb = c[2];
        line(u0, v0, u1, v1, (wi + 1.3) / 2, 12 / 255, 6 / 255, 10 / 255, 1);
        line(u0, v0, u1, v1, wi / 2, r, g, bb, 1);
        if (wi >= 3) ell(u1, v1, wi / 4, wi / 4, 0, r, g, bb, 1);
      }
    }
    if (leafOn) {
      const breeze = 0.06 + 0.25 * smooth((t - ST.fall) / 8.0);
      for (let j = 0; j < nL; j++) {
        const o = j * LC;
        const born = LV[o + 5];
        if (born > t || t >= LV[o + 7]) continue;
        const g = smooth((t - born) / 3.5);
        if (g <= 0.03) continue;
        irr(LV[o], LV[o + 1]);
        const u = smooth((t - born) / 9.0);
        const ua = smooth((t - LV[o + 6]) / 3.0);
        const m = 0.78 + 0.26 * LV[o + 13];
        const al = [0.46 * (1 - u) + 0.12 * u, 0.68 * (1 - u) + 0.36 * u, 0.20 * (1 - u) + 0.12 * u];
        for (let q = 0; q < 3; q++) al[q] = al[q] * (1 - ua) + LV[o + 8 + q] * ua;
        const c = tm3(al[0] * ir[0] * m + al[0] * 0.012, al[1] * ir[1] * m + al[1] * 0.012, al[2] * ir[2] * m + al[2] * 0.012);
        const r = c[0], gg = c[1], bb = c[2];
        const wob = breeze * Math.sin(1.7 * t + LV[o + 11] * 3) * (1 + ua);
        const a = LV[o + 3] + wob;
        const fx = Math.cos(a), fy = -Math.sin(a);
        const len = LV[o + 4] * g * S;
        const bx = sx(LV[o]), by = sy(LV[o + 1]);
        leafPrim(bx, by, fx, fy, len, r, gg, bb, 245 / 255);
        if (g > 0.4) line(bx, by, bx + fx * len, by + fy * len, 0.5, r * 0.62, gg * 0.62, bb * 0.62, 200 / 255);
      }
    }
    // berries and flowers
    const DRU = [[-0.55, -0.35], [-0.1, 0.4], [0.35, -0.3], [0.7, 0.25], [-0.15, -0.05]];
    for (let j = 0; j < nB; j++) {
      const o = j * BC;
      const tflow = BV[o + 5], tset = BV[o + 6], tred = BV[o + 7], tblk = BV[o + 8], tdrop = BV[o + 9];
      if (t < tflow || t > tdrop + 0.35) continue;
      irr(BV[o], BV[o + 1]);
      const i0 = ir[0], i1 = ir[1], i2 = ir[2];
      const u = sx(BV[o]), v = sy(BV[o + 1]);
      if (t < tset) {
        const gfl = smooth((t - tflow) / 1.5) * (1 - smooth((t - tset + 0.5) / 1.0));
        const c = tm3(0.85 * i0 * 1.3, 0.88 * i1 * 1.3, 0.70 * i2 * 1.3);
        const rr = 1.3 * S * gfl;
        for (let q = 0; q < 3; q++) {
          const aa = BV[o + 3] + q * 2.1;
          ell(u + Math.cos(aa) * 2.2 * S, v + Math.sin(aa) * 2.2 * S, rr, rr, 0, c[0], c[1], c[2], 230 / 255 * gfl);
        }
        continue;
      }
      // berry colour: green -> white -> red -> black
      let uu = smooth((t - tset) / 4.0);
      const al = [0.66 * (1 - uu) + 0.88 * uu, 0.80 * (1 - uu) + 0.88 * uu, 0.52 * (1 - uu) + 0.80 * uu];
      uu = smooth((t - tred) / 3.0);
      al[0] = al[0] * (1 - uu) + 0.66 * uu; al[1] = al[1] * (1 - uu) + 0.035 * uu; al[2] = al[2] * (1 - uu) + 0.13 * uu;
      uu = smooth((t - tblk) / 3.0);
      al[0] = al[0] * (1 - uu) + 0.085 * uu; al[1] = al[1] * (1 - uu) + 0.010 * uu; al[2] = al[2] * (1 - uu) + 0.055 * uu;
      const ripe = uu;
      let size = BV[o + 4] * Math.min(1.0, 0.35 + 0.65 * (t - tset) / 5.0);
      if (t > tdrop) size *= 1 - 0.55 * (t - tdrop) / 0.35;
      const c = tm3(al[0] * i0 * 1.2, al[1] * i1 * 1.2, al[2] * i2 * 1.2);
      const L = 4.6 * size * S, Wd = 3.0 * size * S;
      const ang = BV[o + 3], ca = Math.cos(ang), sa = Math.sin(ang);
      ell(u, v, L, Wd, ang, c[0], c[1], c[2], 1);
      const hl = tm3(al[0] * i0 * 1.2 + i0 * 0.10 * (0.4 + ripe), al[1] * i1 * 1.2 + i1 * 0.10 * (0.4 + ripe),
        al[2] * i2 * 1.2 + i2 * 0.10 * (0.4 + ripe));
      const h0 = hl[0], h1 = hl[1], h2 = hl[2];
      for (const [du, dv] of DRU) {
        ell(u + du * L * ca - dv * Wd * sa, v + du * L * sa + dv * Wd * ca, Wd * 0.38, Wd * 0.38, 0, h0, h1, h2, 150 / 255);
      }
      const sp = tm3(i0 * (0.5 + 0.8 * ripe), i1 * (0.5 + 0.8 * ripe), i2 * (0.5 + 0.8 * ripe));
      const rr = Math.max(0.5, Wd * 0.22);
      ell(u - L * 0.3, v - Wd * 0.3, rr, rr, 0, sp[0], sp[1], sp[2], 210 / 255);
    }
    // leaves in the whirlwind
    if (leafOn) {
      for (let j = 0; j < nL; j++) {
        const o = j * LC;
        if (LV[o + 5] > t || t < LV[o + 7]) continue;
        const s = t - LV[o + 7];
        const r0 = Math.hypot(LV[o], LV[o + 1]), th0 = Math.atan2(LV[o + 1], LV[o]);
        const r = r0 + 20 * s + 15 * s * s, th = th0 + 1.05 * s + 0.32 * s * s;
        const x = r * Math.cos(th), y = r * Math.sin(th);
        if (Math.abs(x) * S > W * 0.75 || Math.abs(y) * S > H * 0.75) continue;
        const rot = LV[o + 3] + LV[o + 13] * (2.6 * s + 0.5 * Math.sin(3.1 * s + LV[o + 11]));
        const sc = 1 + 0.8 * Math.min(s, 1.6);
        irr(x, y);
        const m = 0.85 + 0.3 * LV[o + 13];
        const c = tm3(LV[o + 8] * ir[0] * m, LV[o + 9] * ir[1] * m, LV[o + 10] * ir[2] * m);
        const rr = c[0], gg = c[1], bb = c[2];
        const fx = Math.cos(rot), fy = -Math.sin(rot);
        const len = LV[o + 4] * sc * S;
        const bx = sx(x), by = sy(y);
        leafPrim(bx, by, fx, fy, len, rr, gg, bb, 225 / 255);
        line(bx, by, bx + fx * len, by + fy * len, 0.5, rr * 0.6, gg * 0.6, bb * 0.6, 200 / 255);
      }
    }
    end("bush");

    // ---- emissive: light trails, the ring of hands, dew glints
    begin("em");
    const Tm = TL.trail_memory(b), gain = TL.trail_gain(b);
    const q = [0, 0];
    if (gain > 0.01 && Tm > 0.06) {
      const n = Math.floor(Math.min(900, Math.max(12, Tm * 60)));
      const tau = Math.max(Tm * 0.42, 0.05);
      for (let k = 0; k < 4; k++) {
        let pu = 0, pv = 0, pa = 0;
        for (let i = 0; i < n; i++) {
          const ts = t - Tm + Tm * i / (n - 1);
          const bs = TL.B_of_t(ts);
          TL.pos(k, bs, q);
          const u = sx(q[0]), v = sy(q[1]);
          if (i > 0 && pa >= 0.02) {
            line(pu, pv, u, v, 1.0, Math.min(1, COL[k][0] * pa), Math.min(1, COL[k][1] * pa), Math.min(1, COL[k][2] * pa), 1);
          }
          pa = gain * Math.exp(-(t - ts) / tau) * TL.intensity(k, bs) * 0.75;
          pu = u; pv = v;
        }
      }
    }
    const J = TL.ring_join(b);
    if (J > 0.01) {
      const th = Pp.map((p) => Math.atan2(p[1], p[0])), rd = Pp.map((p) => Math.hypot(p[0], p[1]));
      const order = [0, 1, 2, 3].sort((x, y) => th[x] - th[y]);
      for (let qq = 0; qq < 4; qq++) {
        const i = order[qq], j = order[(qq + 1) % 4];
        let a0 = th[i], a1 = th[j];
        if (a1 <= a0) a1 += 2 * PI;
        const amp = J * 0.55 * Math.sqrt(Math.max(I[i], 0) * Math.max(I[j], 0));
        let pu = 0, pv = 0;
        for (let s = 0; s < 40; s++) {
          const uu = s / 39;
          const aa = a0 + (a1 - a0) * uu;
          const rr = rd[i] + (rd[j] - rd[i]) * uu - 10 * Math.sin(PI * uu);
          const u = sx(rr * Math.cos(aa)), v = sy(rr * Math.sin(aa));
          if (s > 0) {
            const f = (s - 0.5) / 39;
            line(pu, pv, u, v, 1.0, Math.min(1, (COL[i][0] * (1 - f) + COL[j][0] * f) * amp),
              Math.min(1, (COL[i][1] * (1 - f) + COL[j][1] * f) * amp), Math.min(1, (COL[i][2] * (1 - f) + COL[j][2] * f) * amp), 1);
          }
          pu = u; pv = v;
        }
      }
    }
    if (sunI > 0.02) {
      const SP = world.sparkles.a, nSp = world.sparkles.rows;
      const gk = Math.min(1.0, sunI * 1.3);
      for (let i = 0; i < nSp; i++) {
        const o = i * 5;
        const sv = Math.max(0, Math.sin(SP[o + 3] * t * 2.0 + SP[o + 2]));
        const a = Math.pow(sv, 12) * SP[o + 4] * gk;
        if (a <= 0.04) continue;
        const u = sx(SP[o]), v = sy(SP[o + 1]);
        ell(u, v, 0.8, 0.8, 0, a, 235 / 255 * a, 200 / 255 * a, 1);
        const hl = (5 * a + 2) / 2;
        line(u - hl, v, u + hl, v, 0.5, a, 235 / 255 * a, 200 / 255 * a, 1);
      }
    }
    end("em");

    let seedA = 0;
    if (t < ST.grow + 4) {
      seedA = 0.22 * (0.6 + 0.4 * Math.sin(2 * PI * t / 3.0)) * (1 - smooth((t - ST.grow) / 4)) * smooth((t - 3.5) / 3.0);
    }
    return { b, maskOn, sunOn, sunBlur, seedA, fade: TL.fade(b), sf: Math.pow(Hl / 120.0, 0.6) * Math.sqrt(S) };
  }

  // ------------------------------------------------------------ text
  async function loadFonts() {
    try {
      await Promise.all([document.fonts.load('31px "Cormorant Garamond"'), document.fonts.load('64px "Cormorant Garamond"'),
        document.fonts.load('italic 32px "Cormorant Garamond"'), document.fonts.load('25px "Cormorant Garamond"')]);
    } catch (e) { /* fall back to whatever serif is there */ }
    ringAdv = null;
  }
  function drawText(t, b) {
    const ctx = textCtx;
    ctx.clearRect(0, 0, W, H);
    const endA = smooth((TL.TOTAL - 0.6 - t) / 1.2);
    let any = false;
    const b0 = 101.5, b1 = 107.5;
    if (b >= b0) {
      any = true;
      const t0 = TL.t_of_B(b0), t1 = TL.t_of_B(b1);
      ctx.font = "31px " + SERIF;
      const chars = Array.from(RING_TEXT);
      if (!ringAdv) ringAdv = chars.map((c) => ctx.measureText(c).width + 7.0);
      const total = ringAdv.reduce((x, y) => x + y, 0);
      const R = 470 * S;
      const scale = 2 * PI * R / total;
      const rot = 0.022 * (t - t0);
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      let acc = 0;
      for (let i = 0; i < chars.length; i++) {
        const aMid = (acc + ringAdv[i] / 2) * scale / R;
        acc += ringAdv[i];
        const ti = t0 + (t1 - t0) * i / chars.length;
        const al = smooth((t - ti) / 0.8) * endA;
        if (al <= 0 || chars[i] === " ") continue;
        const th = PI / 2 - aMid + rot;
        ctx.save();
        ctx.translate(W / 2 + R * Math.cos(th), H / 2 - R * Math.sin(th));
        ctx.rotate(PI / 2 - th);
        ctx.fillStyle = `rgba(246,226,236,${al * 0.92})`;
        ctx.fillText(chars[i], 0, 9);
        ctx.restore();
      }
    }
    if (b >= 110.6) {
      any = true;
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      const ySmall = [610, 652];
      let si = 0;
      for (const [kind, text, bb] of CREDITS) {
        const al = smooth((b - bb) / 0.6) * endA;
        let font, y, c;
        if (kind === "title") { font = "64px " + SERIF; y = 430; c = "240,214,230"; }
        else if (kind === "it") { font = "italic 32px " + SERIF; y = 512; c = "205,170,215"; }
        else { font = "25px " + SERIF; y = ySmall[si++]; c = "170,150,175"; }
        if (al <= 0) continue;
        ctx.font = font;
        const asc = ctx.measureText(text).fontBoundingBoxAscent || parseFloat(font) * 0.9;
        // render.py's credits are drawn with Pillow onto the opaque frame, which ignores
        // the ink's alpha: in the film they switch on at full strength rather than fade
        ctx.fillStyle = `rgb(${c})`;
        ctx.fillText(text, W / 2, y + asc);
      }
    }
    return any;
  }

  // ------------------------------------------------------------ GPU passes
  function bindAttribs(start) {
    gl.bindBuffer(gl.ARRAY_BUFFER, ibuf);
    for (let i = 0; i < 3; i++) gl.vertexAttribPointer(i, 4, gl.FLOAT, false, PF * 4, start * PF * 4 + i * 16);
  }
  function drawPass(name, fbo) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.fb);
    gl.viewport(0, 0, fbo.w, fbo.h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const [start, count] = passes[name];
    if (!count) return;
    gl.useProgram(P_prim.p);
    gl.uniform2f(P_prim.u.uRes, fbo.w, fbo.h);
    gl.uniform1i(P_prim.u.uShape, 0);
    gl.uniform1i(P_prim.u.uStain, 1);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D_ARRAY, texShape);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D_ARRAY, texStain);
    gl.bindVertexArray(vao);
    bindAttribs(start);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, count);
    gl.disable(gl.BLEND);
  }
  function blurPass(src, dst, dx, dy, sigma) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb);
    gl.viewport(0, 0, dst.w, dst.h);
    gl.useProgram(P_blur.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, src.tex);
    gl.uniform1i(P_blur.u.uTex, 0);
    gl.uniform2f(P_blur.u.uDir, dx / src.w, dy / src.h);
    gl.uniform1f(P_blur.u.uSigma, sigma);
    gl.uniform2f(P_blur.u.uInvRes, 1 / dst.w, 1 / dst.h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  function blur2(fbo, tmp, sigma) {
    blurPass(fbo, tmp, 1, 0, sigma);
    blurPass(tmp, fbo, 0, 1, sigma);
  }

  let lastFrameMs = 0;
  function render(t) {
    const t0 = performance.now();
    const st = build(t);
    gl.bindBuffer(gl.ARRAY_BUFFER, ibuf);
    gl.bufferData(gl.ARRAY_BUFFER, prim.subarray(0, np * PF), gl.STREAM_DRAW);
    gl.bindVertexArray(vao);
    drawPass("marks", fbMarks);
    for (let k = 0; k < 4; k++) {
      drawPass("mask" + k, fbMask[k]);
      if (passes["mask" + k][1]) blur2(fbMask[k], fbMaskTmp, 1.3);
    }
    drawPass("sun", fbSun);
    if (st.sunOn) blur2(fbSun, fbSunTmp, st.sunBlur);
    drawPass("bush", fbBush);
    drawPass("em", fbEm);
    blurPass(fbEm, fbGlowA, 0, 0, 0.01);          // 2x2 box downsample
    blur2(fbGlowA, fbGlowB, 3.0);                 // sigma 6 at full resolution
    const textOn = drawText(t, st.b);
    if (textOn) {
      gl.bindTexture(gl.TEXTURE_2D, texText);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, textCanvas);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    }
    // composite
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.useProgram(P_comp.p);
    const u = P_comp.u;
    const bind = (unit, tex, name) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(u[name], unit); };
    bind(0, texGround, "uGround"); bind(1, fbMarks.tex, "uMarks"); bind(2, fbBush.tex, "uBush"); bind(3, fbEm.tex, "uEm");
    bind(4, fbGlowA.tex, "uGlow"); bind(5, texText, "uText"); bind(6, fbSun.tex, "uSun");
    for (let k = 0; k < 4; k++) bind(7 + k, fbMask[k].tex, "uM" + k);
    bind(11, fbMarks.tex2, "uMarksA"); bind(12, fbBush.tex2, "uBushA");
    gl.uniform1f(u.uS, S); gl.uniform1f(u.uHl, Hl); gl.uniform1f(u.uFade, st.fade);
    gl.uniform1f(u.uSunI, sunI); gl.uniform1f(u.uSunOn, st.sunOn ? 1 : 0); gl.uniform1f(u.uTextOn, textOn ? 1 : 0);
    gl.uniform1f(u.uSf, st.sf); gl.uniform1f(u.uSeedA, st.seedA); gl.uniform1f(u.uFrame, Math.round(t * TL.FPS));
    gl.uniform2fv(u.uP, [Pp[0][0], Pp[0][1], Pp[1][0], Pp[1][1], Pp[2][0], Pp[2][1], Pp[3][0], Pp[3][1]]);
    gl.uniform1fv(u.uI, I);
    gl.uniform3fv(u.uCol, COL.flat());
    gl.uniform3fv(u.uAmb, AMB);
    gl.uniform3fv(u.uSunCol, sunCol);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    lastFrameMs = performance.now() - t0;
    return st.b;
  }

  return { init, setGround, render, loadFonts, W, H, TEXN, get lastFrameMs() { return lastFrameMs; }, get prims() { return np; } };
})();
