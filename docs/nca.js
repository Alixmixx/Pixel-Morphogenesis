class NCAModel {
  constructor(params) {
    this.C = params.n_channels;
    this.H = params.grid_size;
    this.W = params.grid_size;
    this.HW = this.H * this.W;
    this.hidden = params.hidden_size;

    this.fc1W = new Float32Array(params.fc1_weight.data);
    this.fc1B = new Float32Array(params.fc1_bias.data);
    this.fc2W = new Float32Array(params.fc2_weight.data);
    this.fc2B = new Float32Array(params.fc2_bias.data);

    this.identity = new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 0]);
    this.dx = new Float32Array([
      -1/8, 0, 1/8, -2/8, 0, 2/8, -1/8, 0, 1/8,
    ]);
    this.dy = new Float32Array([
      -1/8, -2/8, -1/8, 0, 0, 0, 1/8, 2/8, 1/8,
    ]);

    const size = this.C * this.HW;
    this.perceptionBuf = new Float32Array(3 * this.C * this.HW);
    this.deltaBuf = new Float32Array(size);
    this.gridA = new Float32Array(size);
    this.gridB = new Float32Array(size);
    this.hiddenBuf = new Float32Array(this.hidden);
    this.scratchIn = new Float32Array(3 * this.C);
    this.preAliveBuf = new Uint8Array(this.HW);
    this.postAliveBuf = new Uint8Array(this.HW);
    this.cellMaskBuf = new Uint8Array(this.HW);
    this.useA = true;
  }

  getGrid() {
    return this.useA ? this.gridA : this.gridB;
  }

  conv3x3(input, kernel, cIn, output, cOut) {
    const H = this.H, W = this.W, HW = this.HW;
    const inOff = cIn * HW;
    const outOff = cOut * HW;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let sum = 0;
        const yMin = y > 0 ? -1 : 0, yMax = y < H - 1 ? 1 : 0;
        const xMin = x > 0 ? -1 : 0, xMax = x < W - 1 ? 1 : 0;

        for (let ky = yMin; ky <= yMax; ky++) {
          const rowOff = inOff + (y + ky) * W;
          const kRow = (ky + 1) * 3;
          for (let kx = xMin; kx <= xMax; kx++) {
            sum += input[rowOff + x + kx] * kernel[kRow + kx + 1];
          }
        }
        output[outOff + y * W + x] = sum;
      }
    }
  }

  perceive(state) {
    const C = this.C;
    const out = this.perceptionBuf;
    for (let c = 0; c < C; c++) {
      this.conv3x3(state, this.identity, c, out, c * 3);
      this.conv3x3(state, this.dx, c, out, c * 3 + 1);
      this.conv3x3(state, this.dy, c, out, c * 3 + 2);
    }
  }

  updateRule() {
    const C = this.C, HW = this.HW;
    const inC = 3 * C;
    const hidC = this.hidden;
    const perception = this.perceptionBuf;
    const out = this.deltaBuf;
    const h = this.hiddenBuf;
    const scratch = this.scratchIn;

    for (let px = 0; px < HW; px++) {
      for (let j = 0; j < inC; j++) {
        scratch[j] = perception[j * HW + px];
      }

      for (let i = 0; i < hidC; i++) {
        let sum = this.fc1B[i];
        const row = i * inC;
        for (let j = 0; j < inC; j++) {
          sum += this.fc1W[row + j] * scratch[j];
        }
        h[i] = sum > 0 ? sum : 0;
      }

      for (let i = 0; i < C; i++) {
        let sum = this.fc2B[i];
        const row = i * hidC;
        for (let j = 0; j < hidC; j++) {
          sum += this.fc2W[row + j] * h[j];
        }
        out[i * HW + px] = sum;
      }
    }
  }

  aliveMask(state, mask) {
    const H = this.H, W = this.W;
    const alphaOff = 3 * this.HW;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let maxVal = 0;
        const yMin = y > 0 ? y - 1 : 0, yMax = y < H - 1 ? y + 1 : y;
        const xMin = x > 0 ? x - 1 : 0, xMax = x < W - 1 ? x + 1 : x;

        for (let ny = yMin; ny <= yMax; ny++) {
          for (let nx = xMin; nx <= xMax; nx++) {
            const v = state[alphaOff + ny * W + nx];
            if (v > maxVal) maxVal = v;
          }
        }
        mask[y * W + x] = maxVal > 0.1 ? 1 : 0;
      }
    }
  }

  step() {
    const C = this.C, HW = this.HW;
    const state = this.useA ? this.gridA : this.gridB;
    const next = this.useA ? this.gridB : this.gridA;

    this.aliveMask(state, this.preAliveBuf);
    this.perceive(state);
    this.updateRule();

    const cellMask = this.cellMaskBuf;
    for (let i = 0; i < HW; i++) {
      cellMask[i] = Math.random() > 0.2 ? 1 : 0;
    }

    const delta = this.deltaBuf;
    for (let c = 0; c < C; c++) {
      const off = c * HW;
      for (let i = 0; i < HW; i++) {
        next[off + i] = state[off + i] + cellMask[i] * delta[off + i];
      }
    }

    this.aliveMask(next, this.postAliveBuf);

    for (let c = 0; c < C; c++) {
      const off = c * HW;
      for (let i = 0; i < HW; i++) {
        next[off + i] *= this.preAliveBuf[i] * this.postAliveBuf[i];
      }
    }

    this.useA = !this.useA;
  }

  renderToPixels(pixels) {
    const grid = this.getGrid();
    const HW = this.HW;
    for (let i = 0; i < HW; i++) {
      const r = grid[i];
      const g = grid[HW + i];
      const b = grid[2 * HW + i];
      const a = Math.max(0, Math.min(1, grid[3 * HW + i]));
      pixels[i * 4]     = Math.max(0, Math.min(255, ((1 - a + r) * 255) | 0));
      pixels[i * 4 + 1] = Math.max(0, Math.min(255, ((1 - a + g) * 255) | 0));
      pixels[i * 4 + 2] = Math.max(0, Math.min(255, ((1 - a + b) * 255) | 0));
      pixels[i * 4 + 3] = 255;
    }
  }

  keepSeedAlive() {
    const grid = this.getGrid();
    const mid = Math.floor(this.H / 2);
    grid[3 * this.HW + mid * this.W + mid] = 1.0;
  }

  resetGrid() {
    const grid = this.getGrid();
    grid.fill(0);
    const mid = Math.floor(this.H / 2);
    for (let c = 3; c < this.C; c++) {
      grid[c * this.HW + mid * this.W + mid] = 1.0;
    }
  }

  damageAt(gx, gy, radius) {
    const grid = this.getGrid();
    const size = this.H;
    const HW = this.HW;
    const C = this.C;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy > radius * radius) continue;
        const ny = gy + dy, nx = gx + dx;
        if (ny < 0 || ny >= size || nx < 0 || nx >= size) continue;
        for (let c = 0; c < C; c++) {
          grid[c * HW + ny * size + nx] = 0;
        }
      }
    }
  }
}
