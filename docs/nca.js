class NCAModel {
  constructor(params) {
    this.C = params.n_channels;
    this.H = params.grid_size;
    this.W = params.grid_size;
    this.hidden = params.hidden_size;

    this.fc1W = new Float32Array(params.fc1_weight.data);
    this.fc1B = new Float32Array(params.fc1_bias.data);
    this.fc2W = new Float32Array(params.fc2_weight.data);
    this.fc2B = new Float32Array(params.fc2_bias.data);

    this.buildSobelKernels();
  }

  buildSobelKernels() {
    // identity, dx, dy — same as the Python model
    this.identity = new Float32Array([0,0,0, 0,1,0, 0,0,0]);
    this.dx = new Float32Array([
      -1/8, 0, 1/8,
      -2/8, 0, 2/8,
      -1/8, 0, 1/8,
    ]);
    this.dy = new Float32Array([
      -1/8, -2/8, -1/8,
       0,    0,    0,
       1/8,  2/8,  1/8,
    ]);
  }

  // depthwise 3x3 convolution with one kernel, single channel
  conv3x3(input, kernel, cIn, output, cOut) {
    const H = this.H, W = this.W;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let sum = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const ny = y + ky, nx = x + kx;
            if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
              sum += input[cIn * H * W + ny * W + nx] * kernel[(ky + 1) * 3 + (kx + 1)];
            }
          }
        }
        output[cOut * H * W + y * W + x] = sum;
      }
    }
  }

  perceive(state) {
    const C = this.C, H = this.H, W = this.W;
    const out = new Float32Array(3 * C * H * W);
    for (let c = 0; c < C; c++) {
      this.conv3x3(state, this.identity, c, out, c * 3);
      this.conv3x3(state, this.dx, c, out, c * 3 + 1);
      this.conv3x3(state, this.dy, c, out, c * 3 + 2);
    }
    return out;
  }

  update(perception) {
    const C = this.C, H = this.H, W = this.W;
    const inC = 3 * C;
    const hidC = this.hidden;
    const out = new Float32Array(C * H * W);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const px = y * W + x;

        // fc1: [hidden, 3*C] matmul + bias + relu
        const h = new Float32Array(hidC);
        for (let i = 0; i < hidC; i++) {
          let sum = this.fc1B[i];
          const row = i * inC;
          for (let j = 0; j < inC; j++) {
            sum += this.fc1W[row + j] * perception[j * H * W + px];
          }
          h[i] = sum > 0 ? sum : 0; // relu
        }

        // fc2: [C, hidden] matmul + bias
        for (let i = 0; i < C; i++) {
          let sum = this.fc2B[i];
          const row = i * hidC;
          for (let j = 0; j < hidC; j++) {
            sum += this.fc2W[row + j] * h[j];
          }
          out[i * H * W + px] = sum;
        }
      }
    }
    return out;
  }

  aliveMask(state) {
    const H = this.H, W = this.W;
    const mask = new Uint8Array(H * W);
    // max pool 3x3 on alpha channel (channel 3)
    const alphaOffset = 3 * H * W;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let maxVal = 0;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const ny = y + ky, nx = x + kx;
            if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
              const v = state[alphaOffset + ny * W + nx];
              if (v > maxVal) maxVal = v;
            }
          }
        }
        mask[y * W + x] = maxVal > 0.1 ? 1 : 0;
      }
    }
    return mask;
  }

  step(state) {
    const C = this.C, H = this.H, W = this.W;
    const size = C * H * W;

    const preAlive = this.aliveMask(state);
    const perception = this.perceive(state);
    const delta = this.update(perception);

    const next = new Float32Array(size);

    // stochastic update mask — per cell, shared across channels
    const cellMask = new Uint8Array(H * W);
    for (let i = 0; i < H * W; i++) {
      cellMask[i] = Math.random() > 0.2 ? 1 : 0;
    }

    // residual + masks
    for (let c = 0; c < C; c++) {
      for (let i = 0; i < H * W; i++) {
        const idx = c * H * W + i;
        next[idx] = state[idx] + cellMask[i] * delta[idx];
      }
    }

    const postAlive = this.aliveMask(next);

    // apply alive masks
    for (let c = 0; c < C; c++) {
      for (let i = 0; i < H * W; i++) {
        const idx = c * H * W + i;
        next[idx] *= preAlive[i] * postAlive[i];
      }
    }

    return next;
  }
}
