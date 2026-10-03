"use strict";

// Two-dimensional, incompressible magnetohydrodynamics on the GPU, in WebGL2,
// stirred with a finger.
//
// In two dimensions the magnetic field lies in the plane and is the curl of a
// single flux function A(x, y):
//
//   B = (dA/dy, -dA/dx).
//
// The field lines are the contours of A, the field has no divergence by
// construction, and the flow carries A along like dye. With J = -lap A, the
// current out of the plane, the equations are
//
//   dA/dt + u . grad A = eta lap A,
//   du/dt + (u . grad) u = -grad p + J grad A + nu lap u,     div u = 0,
//
// where J grad A is the Lorentz force J x B. The density is one, so the
// Alfven speed is |B|. The fluid is solved as in Stam's "Stable Fluids":
// semi-Lagrangian advection, then a pressure projection, on a periodic grid
// over the unit square.
//
// A is kept as a periodic part a plus a uniform gradient G, A = a + G . x, so
// that a uniform field can thread the periodic grid.

// ---- WebGL2 ----------------------------------------------------------------

const canvas = document.getElementById("view");
const overlay = document.getElementById("overlay");
const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false, stencil: false });

function fail(message) {
  overlay.textContent = message;
  throw new Error(message);
}

if (!gl) fail("This needs WebGL2, which this browser does not offer.");

// Single precision where the GPU can render to it, half precision otherwise.
const FULL = gl.getExtension("EXT_color_buffer_float") !== null;
if (!FULL && gl.getExtension("EXT_color_buffer_half_float") === null) {
  fail("This needs a GPU that can render to floating-point textures, which this one cannot.");
}
const FORMATS = FULL
  ? { 1: [gl.R32F, gl.RED], 2: [gl.RG32F, gl.RG], 4: [gl.RGBA32F, gl.RGBA], type: gl.FLOAT }
  : { 1: [gl.R16F, gl.RED], 2: [gl.RG16F, gl.RG], 4: [gl.RGBA16F, gl.RGBA], type: gl.HALF_FLOAT };

gl.bindVertexArray(gl.createVertexArray());

const VERTEX = `#version 300 es
void main() {
  // One triangle covering the whole viewport.
  vec2 corner = vec2(float((gl_VertexID & 1) << 2), float((gl_VertexID & 2) << 1)) - 1.0;
  gl_Position = vec4(corner, 0.0, 1.0);
}`;

const COMMON = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

// The number of cells along each side of the periodic grid.
uniform int N;
out vec4 result;

const ivec2 X = ivec2(1, 0);
const ivec2 Y = ivec2(0, 1);

ivec2 wrap(ivec2 i) {
  return ivec2(mod(vec2(i), float(N)));
}

float scalar(sampler2D field, ivec2 i) {
  return texelFetch(field, wrap(i), 0).r;
}

vec2 vector(sampler2D field, ivec2 i) {
  return texelFetch(field, wrap(i), 0).rg;
}

// Bilinear interpolation at p, in cells, with cell i centered at i + 0.5.
// Done by hand since not every GPU filters floating-point textures.
float scalarAt(sampler2D field, vec2 p) {
  p -= 0.5;
  ivec2 i = ivec2(floor(p));
  vec2 f = p - floor(p);
  return mix(
    mix(scalar(field, i), scalar(field, i + X), f.x),
    mix(scalar(field, i + Y), scalar(field, i + X + Y), f.x),
    f.y
  );
}

vec2 vectorAt(sampler2D field, vec2 p) {
  p -= 0.5;
  ivec2 i = ivec2(floor(p));
  vec2 f = p - floor(p);
  return mix(
    mix(vector(field, i), vector(field, i + X), f.x),
    mix(vector(field, i + Y), vector(field, i + X + Y), f.x),
    f.y
  );
}

float laplacian(sampler2D field, ivec2 i) {
  float n = float(N);
  return (scalar(field, i + X) + scalar(field, i - X) + scalar(field, i + Y) + scalar(field, i - Y)
    - 4.0 * scalar(field, i)) * n * n;
}

vec2 gradient(sampler2D field, ivec2 i) {
  return 0.5 * float(N) * vec2(
    scalar(field, i + X) - scalar(field, i - X),
    scalar(field, i + Y) - scalar(field, i - Y)
  );
}
`;

const SHADERS = {
  // The flux function, carried along by the flow, and diffused.
  advectFlux: {
    uniforms: { a: "texture", u: "texture", dt: "float", G: "vec2", eta: "float" },
    source: `
uniform sampler2D a;
uniform sampler2D u;
uniform float dt;
uniform vec2 G;
uniform float eta;
void main() {
  ivec2 i = ivec2(gl_FragCoord.xy);
  vec2 velocity = vector(u, i);
  // A = a + G . x is constant along the flow, so a picks up -dt u . G.
  float value = scalarAt(a, gl_FragCoord.xy - dt * float(N) * velocity) - dt * dot(G, velocity);
  result = vec4(value + dt * eta * laplacian(a, i), 0.0, 0.0, 1.0);
}`,
  },

  // The velocity, carried along by itself, diffused, and pushed by the
  // Lorentz force of the updated field.
  advectVelocity: {
    uniforms: { u: "texture", a: "texture", dt: "float", G: "vec2", nu: "float" },
    source: `
uniform sampler2D u;
uniform sampler2D a;
uniform float dt;
uniform vec2 G;
uniform float nu;
void main() {
  ivec2 i = ivec2(gl_FragCoord.xy);
  float n = float(N);
  vec2 velocity = vector(u, i);
  vec2 value = vectorAt(u, gl_FragCoord.xy - dt * n * velocity);
  vec2 diffusion = (vector(u, i + X) + vector(u, i - X) + vector(u, i + Y) + vector(u, i - Y)
    - 4.0 * velocity) * n * n;
  float current = -laplacian(a, i);
  vec2 lorentz = current * (G + gradient(a, i));
  result = vec4(value + dt * (nu * diffusion + lorentz), 0.0, 1.0);
}`,
  },

  divergence: {
    uniforms: { u: "texture" },
    source: `
uniform sampler2D u;
void main() {
  ivec2 i = ivec2(gl_FragCoord.xy);
  float value = 0.5 * float(N) * (vector(u, i + X).x - vector(u, i - X).x + vector(u, i + Y).y - vector(u, i - Y).y);
  result = vec4(value, 0.0, 0.0, 1.0);
}`,
  },

  // One Jacobi iteration of lap p = div u.
  pressure: {
    uniforms: { p: "texture", divergence: "texture" },
    source: `
uniform sampler2D p;
uniform sampler2D divergence;
void main() {
  ivec2 i = ivec2(gl_FragCoord.xy);
  float n = float(N);
  float neighbors = scalar(p, i + X) + scalar(p, i - X) + scalar(p, i + Y) + scalar(p, i - Y);
  result = vec4(0.25 * (neighbors - scalar(divergence, i) / (n * n)), 0.0, 0.0, 1.0);
}`,
  },

  project: {
    uniforms: { u: "texture", p: "texture" },
    source: `
uniform sampler2D u;
uniform sampler2D p;
void main() {
  ivec2 i = ivec2(gl_FragCoord.xy);
  result = vec4(vector(u, i) - gradient(p, i), 0.0, 1.0);
}`,
  },

  // A finger in the fluid: the velocity near it moves toward the finger's.
  stir: {
    uniforms: { u: "texture", point: "vec2", target: "vec2", radius: "float", strength: "float" },
    source: `
uniform sampler2D u;
uniform vec2 point;
uniform vec2 target;
uniform float radius;
uniform float strength;
void main() {
  ivec2 i = ivec2(gl_FragCoord.xy);
  float n = float(N);
  vec2 d = gl_FragCoord.xy - point;
  d -= n * round(d / n);
  float weight = strength * exp(-dot(d, d) / (radius * radius));
  vec2 velocity = vector(u, i);
  result = vec4(velocity + weight * (target - velocity), 0.0, 1.0);
}`,
  },

  scale: {
    uniforms: { a: "texture", factor: "float" },
    source: `
uniform sampler2D a;
uniform float factor;
void main() {
  result = vec4(factor * scalar(a, ivec2(gl_FragCoord.xy)), 0.0, 0.0, 1.0);
}`,
  },

  // The current and the vorticity, for drawing.
  diagnostics: {
    uniforms: { a: "texture", u: "texture" },
    source: `
uniform sampler2D a;
uniform sampler2D u;
void main() {
  ivec2 i = ivec2(gl_FragCoord.xy);
  float n = float(N);
  float vorticity = 0.5 * n * (vector(u, i + X).y - vector(u, i - X).y - vector(u, i + Y).x + vector(u, i - Y).x);
  result = vec4(-laplacian(a, i), vorticity, 0.0, 1.0);
}`,
  },

  // The strongest field, fastest flow, and largest current and vorticity in
  // each four by four block of cells, the first step in finding them
  // anywhere. The first two limit the time step, and the others scale the
  // colors.
  extremes: {
    uniforms: { a: "texture", u: "texture", G: "vec2" },
    source: `
uniform sampler2D a;
uniform sampler2D u;
uniform vec2 G;
void main() {
  ivec2 corner = 4 * ivec2(gl_FragCoord.xy);
  vec4 largest = vec4(0.0);
  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      ivec2 i = corner + ivec2(x, y);
      float vorticity = 0.5 * float(N) * (vector(u, i + X).y - vector(u, i - X).y - vector(u, i + Y).x + vector(u, i - Y).x);
      vec4 here = vec4(length(G + gradient(a, i)), length(vector(u, i)), abs(laplacian(a, i)), abs(vorticity));
      largest = any(isnan(here)) ? vec4(1e30) : max(largest, here);
    }
  }
  result = largest;
}`,
  },

  // The largest values in each four by four block of a smaller texture.
  reduce: {
    uniforms: { source: "texture", size: "int" },
    source: `
uniform sampler2D source;
uniform int size;
void main() {
  ivec2 corner = 4 * ivec2(gl_FragCoord.xy);
  vec4 largest = vec4(0.0);
  for (int y = 0; y < 4; y++) {
    for (int x = 0; x < 4; x++) {
      ivec2 i = corner + ivec2(x, y);
      if (i.x < size && i.y < size) largest = max(largest, texelFetch(source, i, 0));
    }
  }
  result = largest;
}`,
  },

  // The picture: field lines, the contours of A, over the current or the
  // vorticity.
  display: {
    uniforms: {
      a: "texture",
      diagnostics: "texture",
      G: "vec2",
      resolution: "vec2",
      spacing: "float",
      scale: "float",
      channel: "int",
      lineWidth: "float",
    },
    source: `
uniform sampler2D a;
uniform sampler2D diagnostics;
uniform vec2 G;
uniform vec2 resolution;
uniform float spacing;
uniform float scale;
uniform int channel;
uniform float lineWidth;

// Catmull-Rom interpolation, so that the field lines are smooth curves
// rather than kinked at the edges of cells.
float catmullRom(float p0, float p1, float p2, float p3, float t) {
  return p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)));
}

float smoothAt(sampler2D field, vec2 p) {
  p -= 0.5;
  ivec2 i = ivec2(floor(p));
  vec2 f = p - floor(p);
  float rows[4];
  for (int y = -1; y <= 2; y++) {
    rows[y + 1] = catmullRom(
      scalar(field, i + ivec2(-1, y)),
      scalar(field, i + ivec2(0, y)),
      scalar(field, i + ivec2(1, y)),
      scalar(field, i + ivec2(2, y)),
      f.x
    );
  }
  return catmullRom(rows[0], rows[1], rows[2], rows[3], f.y);
}

void main() {
  vec2 position = gl_FragCoord.xy / resolution;
  vec2 cell = position * float(N);

  vec2 values = vectorAt(diagnostics, cell);
  float t = tanh((channel == 0 ? values.x : values.y) / scale);
  vec3 dark = vec3(0.047, 0.078, 0.165);
  vec3 positive = vec3(1.0, 0.45, 0.2);
  vec3 negative = vec3(0.22, 0.56, 1.0);
  vec3 color = dark + max(t, 0.0) * (positive - dark) + max(-t, 0.0) * (negative - dark);

  // Field lines at evenly spaced values of A, offset by half a spacing so
  // that no line runs through a region where A is flat at zero.
  float level = (smoothAt(a, cell) + dot(G, position)) / spacing + 0.5;
  float width = fwidth(level);
  float distance = abs(level - round(level));
  float line = 1.0 - smoothstep(0.5 * lineWidth * width, (0.5 * lineWidth + 1.0) * width, distance);
  // Where the lines crowd closer than a few pixels, fade them out.
  line *= 1.0 - smoothstep(0.15, 0.35, width * lineWidth);
  color = mix(color, vec3(0.94, 0.97, 1.0), 0.85 * line);

  result = vec4(color, 1.0);
}`,
  },
};

function compile(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

const vertexShader = compile(gl.VERTEX_SHADER, VERTEX);
const programs = {};
for (const [name, { uniforms, source }] of Object.entries(SHADERS)) {
  const handle = gl.createProgram();
  gl.attachShader(handle, vertexShader);
  gl.attachShader(handle, compile(gl.FRAGMENT_SHADER, COMMON + source));
  gl.linkProgram(handle);
  if (!gl.getProgramParameter(handle, gl.LINK_STATUS)) {
    throw new Error(`${name}: ${gl.getProgramInfoLog(handle)}`);
  }
  const locations = { N: gl.getUniformLocation(handle, "N") };
  for (const uniform of Object.keys(uniforms)) locations[uniform] = gl.getUniformLocation(handle, uniform);
  programs[name] = { handle, uniforms, locations };
}

// ---- Fields ----------------------------------------------------------------

// A texture to render into, of `size` by `size` cells with `channels`
// channels, filled with `data` if it is given.
function target(size, channels, data = null) {
  const [internal, format] = FORMATS[channels];
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, size, size, 0, format, data ? gl.FLOAT : FORMATS.type, data);
  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    fail("This GPU cannot render to the textures the simulation needs.");
  }
  return { texture, framebuffer, size };
}

function release(field) {
  gl.deleteTexture(field.texture);
  gl.deleteFramebuffer(field.framebuffer);
}

// Two textures, one read while the other is written.
function pair(size, channels, data = null) {
  return {
    read: target(size, channels, data),
    write: target(size, channels, data),
    swap() {
      [this.read, this.write] = [this.write, this.read];
    },
  };
}

// Run a program over `output`, or over the canvas if it is null.
function run(name, output, values) {
  const program = programs[name];
  gl.useProgram(program.handle);
  gl.bindFramebuffer(gl.FRAMEBUFFER, output ? output.framebuffer : null);
  if (output) gl.viewport(0, 0, output.size, output.size);
  else gl.viewport(0, 0, canvas.width, canvas.height);
  gl.uniform1i(program.locations.N, N);
  let unit = 0;
  for (const [uniform, type] of Object.entries(program.uniforms)) {
    const value = values[uniform];
    const location = program.locations[uniform];
    if (type === "texture") {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, value.texture);
      gl.uniform1i(location, unit++);
    } else if (type === "float") {
      gl.uniform1f(location, value);
    } else if (type === "int") {
      gl.uniform1i(location, value);
    } else if (type === "vec2") {
      gl.uniform2f(location, value[0], value[1]);
    }
  }
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

// ---- Starting points -------------------------------------------------------

// A field sampled at the centers of the cells of an n by n grid on the unit
// square, row by row from the bottom.
function sample(n, channels, f) {
  const data = new Float32Array(n * n * channels);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const value = f((i + 0.5) / n, (j + 0.5) / n);
      if (channels === 1) data[j * n + i] = value;
      else data.set(value, (j * n + i) * channels);
    }
  }
  return data;
}

// The shortest displacement between two points on the periodic unit square.
function periodic(d) {
  return d - Math.round(d);
}

function logCosh(z) {
  const x = Math.abs(z);
  return x + Math.log1p(Math.exp(-2 * x)) - Math.LN2;
}

const TAU = 2 * Math.PI;

const PRESETS = [
  {
    name: "Field lines",
    note: "A uniform field. Drag across it to bend the field lines, and let go to watch them spring back as Alfvén waves.",
    lines: 20,
    build: (n) => ({ G: [0, 1], a: sample(n, 1, () => 0) }),
  },
  {
    name: "Flux rope",
    note: "A rope of circular field lines, held together by its own current. Push it around and pull it apart.",
    lines: 12,
    build: (n) => {
      const radius = 0.13;
      return {
        G: [0, 0],
        a: sample(n, 1, (x, y) => {
          const dx = periodic(x - 0.5);
          const dy = periodic(y - 0.5);
          return radius * Math.exp(-(dx * dx + dy * dy) / (radius * radius));
        }),
      };
    },
  },
  {
    name: "Two flux ropes",
    note: "Two ropes carrying current the same way attract, squeeze a current sheet between them, and merge as their field lines reconnect. Pull them apart before they do.",
    lines: 14,
    build: (n) => {
      const radius = 0.11;
      return {
        G: [0, 0],
        a: sample(n, 1, (x, y) => {
          let value = 0;
          for (const center of [0.33, 0.67]) {
            const dx = periodic(x - center);
            const dy = periodic(y - 0.5);
            value += radius * Math.exp(-(dx * dx + dy * dy) / (radius * radius));
          }
          return value;
        }),
      };
    },
  },
  {
    name: "Current sheets",
    note: "Two thin sheets of current, where the field reverses. Squeeze a sheet and it reconnects, tearing into islands and shooting out jets.",
    lines: 24,
    build: (n) => {
      const width = Math.max(0.025, 4 / n);
      return {
        G: [0, 0],
        a: sample(n, 1, (x, y) => {
          const sheets = width * (logCosh((y - 0.25) / width) - logCosh((y - 0.75) / width)) - y + 0.5;
          const bump = Math.exp(-(((y - 0.25) / 0.08) ** 2)) + Math.exp(-(((y - 0.75) / 0.08) ** 2));
          return sheets + 0.004 * Math.cos(TAU * x) * bump;
        }),
      };
    },
  },
  {
    name: "Orszag–Tang vortex",
    note: "A classic test: a smooth vortex and field that tangle into turbulence and thin current sheets on their own.",
    lines: 24,
    build: (n) => ({
      G: [0, 0],
      a: sample(n, 1, (x, y) => Math.cos(TAU * y) / TAU + Math.cos(2 * TAU * x) / (2 * TAU)),
      u: sample(n, 2, (x, y) => [-Math.sin(TAU * y), Math.sin(TAU * x)]),
    }),
  },
];

// ---- State -----------------------------------------------------------------

// Simulated time per second of real time.
const SPEED = 0.3;
// Jacobi iterations for the pressure in a frame, spread over its substeps,
// between which the pressure barely changes and carries over.
const PRESSURE_ITERATIONS = 64;
const MAX_SUBSTEPS = 24;

let N = 256;
let fields = null;
let G = [0, 0];
let spacing = 1;
let scales = { current: 20, currentFloor: 20, vorticity: 20, vorticityFloor: 20 };
let presetIndex = 0;
let time = 0;
let strongest = 1;
let fastest = 0;
let substeps = 1;
let slowed = false;
let running = true;
let frames = 0;

const params = {
  strength: 1,
  eta: 1e-4,
  nu: 1e-4,
  brush: 0.05,
};

// The strongest field and current of a flux function, measured on the CPU
// when a preset is loaded, to scale the time step and the colors.
function measure(a, gradientG, n) {
  let field = 0;
  let current = 0;
  const at = (i, j) => a[((j + n) % n) * n + ((i + n) % n)];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const bx = gradientG[1] + 0.5 * n * (at(i, j + 1) - at(i, j - 1));
      const by = -(gradientG[0] + 0.5 * n * (at(i + 1, j) - at(i - 1, j)));
      field = Math.max(field, Math.hypot(bx, by));
      const lap = (at(i + 1, j) + at(i - 1, j) + at(i, j + 1) + at(i, j - 1) - 4 * at(i, j)) * n * n;
      current = Math.max(current, Math.abs(lap));
    }
  }
  return { field, current };
}

function load(index) {
  presetIndex = index;
  const preset = PRESETS[index];
  const built = preset.build(N);
  let a = built.a;
  G = built.G.slice();

  // Every preset starts with its strongest field at one, times the strength.
  const { field } = measure(a, G, N);
  const factor = params.strength / Math.max(field, 1e-12);
  a = a.map((value) => value * factor);
  G = G.map((value) => value * factor);
  const { current } = measure(a, G, N);

  // Field lines evenly spaced in A, set once so that they crowd where the
  // field grows and thin out where it weakens.
  let low = Infinity;
  let high = -Infinity;
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const value = a[j * N + i] + (G[0] * (i + 0.5)) / N + (G[1] * (j + 0.5)) / N;
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
  }
  spacing = (high - low) / preset.lines;
  const currentFloor = Math.max(0.5 * current, 15 * params.strength);
  scales = { current: currentFloor, currentFloor, vorticity: 15, vorticityFloor: 15 };

  if (fields) {
    for (const field of [fields.u.read, fields.u.write, fields.a.read, fields.a.write, fields.p.read, fields.p.write, fields.divergence, fields.diagnostics, ...fields.reductions]) {
      release(field);
    }
  }
  const reductions = [];
  for (let size = Math.ceil(N / 4); ; size = Math.ceil(size / 4)) {
    reductions.push(target(size, 4));
    if (size === 1) break;
  }
  fields = {
    u: pair(N, 2, built.u || new Float32Array(2 * N * N)),
    a: pair(N, 1, a),
    p: pair(N, 1, new Float32Array(N * N)),
    divergence: target(N, 1),
    diagnostics: target(N, 2),
    reductions,
  };
  strongest = params.strength;
  time = 0;
  project();
  presetNote.textContent = preset.note;
}

// ---- Simulation ------------------------------------------------------------

function project(iterations = 24) {
  run("divergence", fields.divergence, { u: fields.u.read });
  for (let k = 0; k < iterations; k++) {
    run("pressure", fields.p.write, { p: fields.p.read, divergence: fields.divergence });
    fields.p.swap();
  }
  run("project", fields.u.write, { u: fields.u.read, p: fields.p.read });
  fields.u.swap();
}

function substep(dt, iterations) {
  run("advectFlux", fields.a.write, { a: fields.a.read, u: fields.u.read, dt, G, eta: params.eta });
  fields.a.swap();
  run("advectVelocity", fields.u.write, { u: fields.u.read, a: fields.a.read, dt, G, nu: params.nu });
  fields.u.swap();
  project(iterations);
}

const readout = new Float32Array(4);

// The strongest field and fastest flow anywhere, reduced on the GPU and read
// back.
function measureExtremes() {
  run("extremes", fields.reductions[0], { a: fields.a.read, u: fields.u.read, G });
  for (let k = 1; k < fields.reductions.length; k++) {
    const source = fields.reductions[k - 1];
    run("reduce", fields.reductions[k], { source, size: source.size });
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, fields.reductions[fields.reductions.length - 1].framebuffer);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, readout);
  strongest = readout[0];
  fastest = readout[1];
  // The colors saturate at a fraction of the largest current and vorticity,
  // followed slowly so that they do not flicker, and never below a floor
  // set by the starting field.
  const follow = 0.05;
  scales.current += follow * (Math.max(0.3 * readout[2], scales.currentFloor) - scales.current);
  scales.vorticity += follow * (Math.max(0.3 * readout[3], scales.vorticityFloor) - scales.vorticity);
}

// Advance by `duration` of simulated time, in as many substeps as keep it
// stable: Alfven waves may cross under half a cell in a substep, the flow
// may carry the field lines at most a cell, and the diffusion must stay
// stable. If that would take too many substeps, slow down instead.
function advance(duration) {
  const h = 1 / N;
  let limit = Math.min((0.4 * h) / Math.max(strongest, 1e-6), h / Math.max(fastest, 1e-6));
  const diffusion = Math.max(params.eta, params.nu);
  if (diffusion > 0) limit = Math.min(limit, (0.2 * h * h) / diffusion);
  substeps = Math.ceil(duration / limit);
  slowed = substeps > MAX_SUBSTEPS;
  if (slowed) {
    substeps = MAX_SUBSTEPS;
    duration = MAX_SUBSTEPS * limit;
  }
  const dt = duration / substeps;
  const iterations = Math.min(24, Math.max(8, Math.round(PRESSURE_ITERATIONS / substeps)));
  for (let k = 0; k < substeps; k++) substep(dt, iterations);
  time += duration;
}

// ---- Fingers ---------------------------------------------------------------

const pointers = new Map();

function position(event) {
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX - rect.left) / rect.width, 1 - (event.clientY - rect.top) / rect.height];
}

canvas.addEventListener("pointerdown", (event) => {
  try {
    canvas.setPointerCapture(event.pointerId);
  } catch {
    // Synthetic pointers cannot be captured, and need not be.
  }
  const [x, y] = position(event);
  pointers.set(event.pointerId, { x, y, lastX: x, lastY: y, vx: 0, vy: 0 });
  event.preventDefault();
});

canvas.addEventListener("pointermove", (event) => {
  const pointer = pointers.get(event.pointerId);
  if (!pointer) return;
  [pointer.x, pointer.y] = position(event);
});

for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
  canvas.addEventListener(type, (event) => pointers.delete(event.pointerId));
}

// Each finger drags the fluid beneath it along at its own velocity, in a
// series of splats along the path it took since the last frame. A finger
// held still holds the fluid still.
function stir(realDt) {
  const radius = params.brush * N;
  for (const pointer of pointers.values()) {
    const dx = pointer.x - pointer.lastX;
    const dy = pointer.y - pointer.lastY;
    pointer.vx = 0.5 * pointer.vx + (0.5 * dx) / realDt;
    pointer.vy = 0.5 * pointer.vy + (0.5 * dy) / realDt;
    let tx = pointer.vx / SPEED;
    let ty = pointer.vy / SPEED;
    const speed = Math.hypot(tx, ty);
    const fastest = 6;
    if (speed > fastest) {
      tx *= fastest / speed;
      ty *= fastest / speed;
    }
    const count = Math.max(1, Math.ceil((Math.hypot(dx, dy) * N) / (0.4 * radius)));
    for (let k = 1; k <= count; k++) {
      const x = pointer.lastX + (dx * k) / count;
      const y = pointer.lastY + (dy * k) / count;
      run("stir", fields.u.write, {
        u: fields.u.read,
        point: [x * N, y * N],
        target: [tx, ty],
        radius,
        strength: 0.4,
      });
      fields.u.swap();
    }
    pointer.lastX = pointer.x;
    pointer.lastY = pointer.y;
  }
}

// ---- Drawing ---------------------------------------------------------------

const colorSelect = document.getElementById("color");

function draw() {
  run("diagnostics", fields.diagnostics, { a: fields.a.read, u: fields.u.read });
  const channel = colorSelect.value === "current" ? 0 : 1;
  run("display", null, {
    a: fields.a.read,
    diagnostics: fields.diagnostics,
    G,
    resolution: [canvas.width, canvas.height],
    spacing,
    scale: channel === 0 ? scales.current : scales.vorticity,
    channel,
    lineWidth: Math.min(window.devicePixelRatio || 1, 2) * 1.2,
  });
}

function resize() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const size = Math.round(Math.min(canvas.clientWidth * ratio, 1280));
  if (canvas.width !== size || canvas.height !== size) {
    canvas.width = size;
    canvas.height = size;
  }
}
new ResizeObserver(resize).observe(canvas);
resize();

// ---- Controls --------------------------------------------------------------

const playButton = document.getElementById("play");
const resetButton = document.getElementById("reset");
const saveButton = document.getElementById("save");
const presetSelect = document.getElementById("preset");
const presetNote = document.getElementById("preset-note");
const gridSelect = document.getElementById("grid");
const sliders = document.getElementById("sliders");
const status = document.getElementById("status");

const SLIDERS = [
  { key: "strength", symbol: "B", name: "field strength", min: 0.05, max: 3, zero: false },
  { key: "eta", symbol: "η", name: "resistivity", min: 1e-6, max: 5e-4, zero: true },
  { key: "nu", symbol: "ν", name: "viscosity", min: 1e-6, max: 5e-4, zero: true },
  { key: "brush", symbol: "r", name: "size of your finger", min: 0.02, max: 0.15, zero: false },
];
const RESOLUTION = 1000;

function toSlider(slider, value) {
  if (slider.zero && value <= 0) return 0;
  const position = Math.log(value / slider.min) / Math.log(slider.max / slider.min);
  return Math.min(RESOLUTION, Math.max(slider.zero ? 1 : 0, Math.round(position * RESOLUTION)));
}

function fromSlider(slider, position) {
  if (slider.zero && position <= 0) return 0;
  return slider.min * Math.pow(slider.max / slider.min, position / RESOLUTION);
}

function format(value) {
  return String(Number(value.toPrecision(3)));
}

function setParameter(key, value) {
  if (key === "strength") {
    // Rescale the field in place, so a stronger or weaker field takes hold
    // as the fluid moves.
    const factor = value / params.strength;
    run("scale", fields.a.write, { a: fields.a.read, factor });
    fields.a.swap();
    G = G.map((component) => component * factor);
    strongest *= factor;
    scales.current *= factor;
    scales.currentFloor *= factor;
  }
  params[key] = value;
}

for (const slider of SLIDERS) {
  const row = document.createElement("div");
  row.className = "slider";
  const id = `parameter-${slider.key}`;
  row.innerHTML = `
    <label for="${id}">${slider.symbol} <small>${slider.name}</small></label>
    <input type="number" id="${id}" inputmode="decimal" step="any" min="0">
    <input type="range" min="0" max="${RESOLUTION}" step="1" aria-label="${slider.symbol}, ${slider.name}">`;
  const [number, range] = row.querySelectorAll("input");
  number.value = format(params[slider.key]);
  range.value = toSlider(slider, params[slider.key]);
  range.addEventListener("input", () => {
    const value = fromSlider(slider, Number(range.value));
    setParameter(slider.key, value);
    number.value = format(value);
  });
  number.addEventListener("change", () => {
    let value = Number(number.value);
    if (!Number.isFinite(value) || value < 0) value = params[slider.key];
    value = Math.min(slider.max, slider.zero && value === 0 ? 0 : Math.max(slider.min, value));
    setParameter(slider.key, value);
    number.value = format(value);
    range.value = toSlider(slider, value);
  });
  sliders.append(row);
}

PRESETS.forEach((preset, index) => {
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = preset.name;
  presetSelect.append(option);
});
presetSelect.addEventListener("change", () => load(Number(presetSelect.value)));

N = window.matchMedia("(max-width: 700px)").matches ? 256 : 384;
gridSelect.value = String(N);
gridSelect.addEventListener("change", () => {
  N = Number(gridSelect.value);
  load(presetIndex);
});

playButton.addEventListener("click", () => {
  running = !running;
  playButton.textContent = running ? "Pause" : "Play";
});
resetButton.addEventListener("click", () => load(presetIndex));
saveButton.addEventListener("click", () => {
  draw();
  canvas.toBlob((blob) => {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `mhd-t${time.toFixed(2)}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }, "image/png");
});

// ---- Running ---------------------------------------------------------------

let last = performance.now();
let reported = last;
let framesSinceReport = 0;

function frame(now) {
  const realDt = Math.min(Math.max((now - last) / 1000, 1e-3), 1 / 20);
  last = now;
  if (running) {
    stir(realDt);
    // Measured every frame, after the fingers have pushed, since a quick
    // stroke can squeeze the field lines together within a frame.
    measureExtremes();
    if (!Number.isFinite(strongest) || strongest > 1e6) load(presetIndex);
    advance(realDt * SPEED);
    frames++;
  } else {
    for (const pointer of pointers.values()) {
      pointer.lastX = pointer.x;
      pointer.lastY = pointer.y;
    }
  }
  draw();

  framesSinceReport++;
  if (now - reported > 500) {
    const fps = Math.round((1000 * framesSinceReport) / (now - reported));
    const parts = [`t = ${time.toFixed(2)}`, `${fps} fps`, `${substeps} substep${substeps === 1 ? "" : "s"}`];
    if (slowed) parts.push("slowed to keep the strong field stable");
    status.textContent = parts.join(" · ");
    reported = now;
    framesSinceReport = 0;
  }
  requestAnimationFrame(frame);
}

load(0);
requestAnimationFrame(frame);

// For testing from the console: step the simulation by hand and read fields.
window.mhd = {
  get N() {
    return N;
  },
  get time() {
    return time;
  },
  get G() {
    return G.slice();
  },
  pause() {
    running = false;
    playButton.textContent = "Play";
  },
  load(index) {
    load(index);
  },
  advance(duration) {
    advance(duration);
  },
  extremes() {
    measureExtremes();
    return { strongest, fastest };
  },
  // Replace the named field, "u" or "a", with `data`, its channels row by row
  // from the bottom.
  write(name, data) {
    const channels = name === "u" ? 2 : 1;
    gl.bindTexture(gl.TEXTURE_2D, fields[name].read.texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, FORMATS[channels][1], gl.FLOAT, data);
  },
  // The named field, "u", "a", "p", or "diagnostics", as RGBA rows from the
  // bottom.
  read(name) {
    const field = name === "u" || name === "a" || name === "p" ? fields[name].read : fields[name];
    if (name === "diagnostics") run("diagnostics", fields.diagnostics, { a: fields.a.read, u: fields.u.read });
    const data = new Float32Array(4 * field.size * field.size);
    gl.bindFramebuffer(gl.FRAMEBUFFER, field.framebuffer);
    gl.readPixels(0, 0, field.size, field.size, gl.RGBA, gl.FLOAT, data);
    return data;
  },
};
