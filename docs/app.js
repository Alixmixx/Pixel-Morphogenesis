const SCALE = 8;
const SEED_ALPHA_CHANNELS = true; // init alpha + all hidden channels at center

let model = null;
let grid = null;
let running = true;
let stepsPerFrame = 1;
let currentTarget = "skull";
let gridSize = 56;
let nChannels = 32;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const offscreen = document.createElement("canvas");
const offCtx = offscreen.getContext("2d");

async function loadModel(name) {
  const resp = await fetch(`models/${name}.json`);
  const params = await resp.json();
  gridSize = params.grid_size;
  nChannels = params.n_channels;
  model = new NCAModel(params);
  canvas.width = gridSize * SCALE;
  canvas.height = gridSize * SCALE;
  offscreen.width = gridSize;
  offscreen.height = gridSize;
  ctx.imageSmoothingEnabled = false;
  resetGrid();
  document.getElementById("status").textContent = `loaded ${name}`;
}

function resetGrid() {
  const size = nChannels * gridSize * gridSize;
  grid = new Float32Array(size);
  const mid = Math.floor(gridSize / 2);
  // set alpha + all hidden channels at center pixel
  for (let c = 3; c < nChannels; c++) {
    grid[c * gridSize * gridSize + mid * gridSize + mid] = 1.0;
  }
}

function render() {
  const imageData = offCtx.createImageData(gridSize, gridSize);
  const data = imageData.data;
  const HW = gridSize * gridSize;

  for (let i = 0; i < HW; i++) {
    const r = grid[0 * HW + i];
    const g = grid[1 * HW + i];
    const b = grid[2 * HW + i];
    const a = Math.max(0, Math.min(1, grid[3 * HW + i]));

    // composite RGBA onto white background
    const ri = Math.max(0, Math.min(255, ((1 - a + r) * 255) | 0));
    const gi = Math.max(0, Math.min(255, ((1 - a + g) * 255) | 0));
    const bi = Math.max(0, Math.min(255, ((1 - a + b) * 255) | 0));

    data[i * 4]     = ri;
    data[i * 4 + 1] = gi;
    data[i * 4 + 2] = bi;
    data[i * 4 + 3] = 255;
  }

  offCtx.putImageData(imageData, 0, 0);
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
}

function keepSeedAlive() {
  const mid = Math.floor(gridSize / 2);
  grid[3 * gridSize * gridSize + mid * gridSize + mid] = 1.0;
}

function tick() {
  if (running && model) {
    for (let i = 0; i < stepsPerFrame; i++) {
      grid = model.step(grid);
      keepSeedAlive();
    }
  }
  render();
  requestAnimationFrame(tick);
}

// --- damage tool ---
let pointerDown = false;
const BRUSH_RADIUS = 3;

function damageAt(canvasX, canvasY) {
  const gx = Math.floor(canvasX / SCALE);
  const gy = Math.floor(canvasY / SCALE);
  const HW = gridSize * gridSize;

  for (let dy = -BRUSH_RADIUS; dy <= BRUSH_RADIUS; dy++) {
    for (let dx = -BRUSH_RADIUS; dx <= BRUSH_RADIUS; dx++) {
      if (dx * dx + dy * dy > BRUSH_RADIUS * BRUSH_RADIUS) continue;
      const ny = gy + dy, nx = gx + dx;
      if (ny < 0 || ny >= gridSize || nx < 0 || nx >= gridSize) continue;
      for (let c = 0; c < nChannels; c++) {
        grid[c * HW + ny * gridSize + nx] = 0;
      }
    }
  }
}

function getCanvasPos(e) {
  const rect = canvas.getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

canvas.addEventListener("pointerdown", (e) => {
  pointerDown = true;
  canvas.setPointerCapture(e.pointerId);
  damageAt(...getCanvasPos(e));
});

canvas.addEventListener("pointermove", (e) => {
  if (pointerDown) damageAt(...getCanvasPos(e));
});

canvas.addEventListener("pointerup", () => { pointerDown = false; });

// --- controls ---
document.getElementById("reset").addEventListener("click", resetGrid);

document.getElementById("pause").addEventListener("click", function() {
  running = !running;
  this.textContent = running ? "pause" : "play";
});

document.getElementById("speed").addEventListener("input", function() {
  stepsPerFrame = parseInt(this.value);
  document.getElementById("speed-label").textContent = this.value + "x";
});

document.querySelectorAll(".target-btn").forEach(btn => {
  btn.addEventListener("click", async function() {
    document.querySelectorAll(".target-btn").forEach(b => b.classList.remove("active"));
    this.classList.add("active");
    currentTarget = this.dataset.target;
    await loadModel(currentTarget);
    // quick-grow to show the formed image
    for (let i = 0; i < 80; i++) {
      grid = model.step(grid);
      keepSeedAlive();
    }
  });
});

// --- init ---
loadModel("skull").then(() => {
  // grow from seed so user sees the formed image immediately
  for (let i = 0; i < 80; i++) {
    grid = model.step(grid);
    keepSeedAlive();
  }
  requestAnimationFrame(tick);
});
