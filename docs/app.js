const SCALE = 8;
const BRUSH_RADIUS = 3;

let model = null;
let running = true;
let pixelBuf = null;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const offscreen = document.createElement("canvas");
const offCtx = offscreen.getContext("2d");

async function loadModel(name) {
  const resp = await fetch(`models/${name}.json`);
  const params = await resp.json();
  model = new NCAModel(params);
  pixelBuf = new Uint8ClampedArray(model.HW * 4);

  canvas.width = model.H * SCALE;
  canvas.height = model.H * SCALE;
  offscreen.width = model.H;
  offscreen.height = model.H;
  ctx.imageSmoothingEnabled = false;

  model.resetGrid();
  document.getElementById("status").textContent = `loaded ${name}`;
}

function render() {
  model.renderToPixels(pixelBuf);
  const imageData = new ImageData(pixelBuf, model.H, model.H);
  offCtx.putImageData(imageData, 0, 0);
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
}

function tick() {
  if (running && model) {
    model.step();
    model.keepSeedAlive();
  }
  render();
  requestAnimationFrame(tick);
}

// damage tool
let pointerDown = false;

function damageAt(canvasX, canvasY) {
  if (!model) return;
  const gx = Math.floor(canvasX / SCALE);
  const gy = Math.floor(canvasY / SCALE);
  model.damageAt(gx, gy, BRUSH_RADIUS);
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

// controls
document.getElementById("reset").addEventListener("click", () => {
  if (!model) return;
  model.resetGrid();
  for (let i = 0; i < 80; i++) {
    model.step();
    model.keepSeedAlive();
  }
});

document.getElementById("pause").addEventListener("click", function() {
  running = !running;
  this.textContent = running ? "pause" : "play";
});

// init
loadModel("skull").then(() => {
  for (let i = 0; i < 80; i++) {
    model.step();
    model.keepSeedAlive();
  }
  requestAnimationFrame(tick);
});
