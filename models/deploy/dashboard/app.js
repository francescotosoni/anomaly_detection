/* ============================================
   UR5 Anomaly Detection — Lab Dashboard
   Application Logic (Web Serial + Charts)
   ============================================ */

// ---- Feature Index Mapping (UR5 RTDE, 24 channels) ----
// 0-5:   actual_q_0..5       (Joint Positions, rad)
// 6-11:  actual_current_0..5 (Joint Currents, A)
// 12-17: actual_qd_0..5      (Joint Velocities, rad/s)
// 18-23: actual_TCP_force_0..5 (TCP Forces: Fx,Fy,Fz,Tx,Ty,Tz)

const FEATURE_GROUPS = {
  positions:  { offset: 0,  count: 6, label: 'Position (rad)' },
  currents:   { offset: 6,  count: 6, label: 'Current (A)' },
  velocities: { offset: 12, count: 6, label: 'Velocity (rad/s)' },
  tcp:        { offset: 18, count: 6, label: 'TCP Force' },
};

const TCP_LABELS = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'];

// ---- State ----
let state = {
  port: null,
  reader: null,
  ws: null,
  connected: false,
  simulating: false,
  simInterval: null,
  threshold: 0.029,
  maxPoints: 200,
  selectedJoint: 0,
  anomalyCount: 0,
  scoreHistory: [],
  // --- Debouncing ---
  // Le fasi normali ma rare (rilascio dell'oggetto al place, decelerazione verso
  // home) produce short spikes of reconstruction error: they are the ~1% of normal
  // windows that the q99 threshold admits by construction, always concentrated at
  // the same points of the cycle. An anomaly is declared only if the score stays
  // above threshold for at least `debounceMs` CONSECUTIVE: transients are filtered,
  // real anomalies (push, altered payload) last much longer and get through.
  debounceMs: 250,
  aboveSince: null,   // timestamp del primo campione sopra soglia, null se sotto
  alarmActive: false,
};

// ---- Chart Setup ----
const chartConfigs = {};
const charts = {};

function createChartConfig(label, color, yMin, yMax, datasetCount = 1, datasetLabels = []) {
  const datasets = [];
  const colors = Array.isArray(color) ? color : [color];

  for (let i = 0; i < datasetCount; i++) {
    datasets.push({
      label: datasetLabels[i] || label,
      data: [],
      borderColor: colors[i % colors.length],
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.3,
      fill: false,
    });
  }

  return {
    type: 'line',
    data: { labels: [], datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      interaction: { mode: 'nearest', intersect: false },
      plugins: {
        legend: {
          display: datasetCount > 1,
          position: 'top',
          labels: {
            color: '#8b949e',
            font: { family: "'Inter', sans-serif", size: 10 },
            boxWidth: 8,
            padding: 8,
          },
        },
        tooltip: {
          backgroundColor: 'rgba(13, 17, 23, 0.9)',
          titleColor: '#e6edf3',
          bodyColor: '#8b949e',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
          bodyFont: { family: "'JetBrains Mono', monospace", size: 10 },
        },
      },
      scales: {
        x: {
          display: false,
        },
        y: {
          min: yMin,
          max: yMax,
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: {
            color: '#484f58',
            font: { family: "'JetBrains Mono', monospace", size: 9 },
            maxTicksLimit: 5,
          },
        },
      },
    },
  };
}

function initCharts() {
  // Score chart (with threshold line)
  const scoreConfig = createChartConfig('Anomaly Score', '#2dd4bf', 0, undefined);
  scoreConfig.options.plugins.annotation = undefined; // we'll draw threshold manually
  // Add threshold dataset
  scoreConfig.data.datasets.push({
    label: 'Threshold',
    data: [],
    borderColor: 'rgba(244, 63, 94, 0.6)',
    borderWidth: 1.5,
    borderDash: [6, 4],
    pointRadius: 0,
    fill: false,
    tension: 0,
  });
  scoreConfig.options.plugins.legend = { display: false };
  charts.score = new Chart(document.getElementById('chartScore'), scoreConfig);

  // Sensor charts (single line each, updated by joint selector)
  charts.positions = new Chart(
    document.getElementById('chartPositions'),
    createChartConfig('Position', '#818cf8', undefined, undefined)
  );

  charts.currents = new Chart(
    document.getElementById('chartCurrents'),
    createChartConfig('Current', '#fb923c', undefined, undefined)
  );

  charts.velocities = new Chart(
    document.getElementById('chartVelocities'),
    createChartConfig('Velocity', '#22d3ee', undefined, undefined)
  );

  // TCP chart: 6 lines (Fx, Fy, Fz, Tx, Ty, Tz)
  const tcpColors = ['#c084fc', '#a78bfa', '#818cf8', '#f472b6', '#fb7185', '#fda4af'];
  charts.tcp = new Chart(
    document.getElementById('chartTCP'),
    createChartConfig('TCP', tcpColors, undefined, undefined, 6, TCP_LABELS)
  );
}

// ---- Data Push ----
function pushData(score, features) {
  const maxPts = state.maxPoints;
  const joint = state.selectedJoint;
  const now = new Date().toLocaleTimeString('it-IT', { hour12: false });

  // --- Score Chart ---
  const scoreChart = charts.score;
  scoreChart.data.labels.push(now);
  scoreChart.data.datasets[0].data.push(score);
  scoreChart.data.datasets[1].data.push(state.threshold);

  if (scoreChart.data.labels.length > maxPts) {
    scoreChart.data.labels.shift();
    scoreChart.data.datasets[0].data.shift();
    scoreChart.data.datasets[1].data.shift();
  }

  // Dynamic Y max for score chart
  const maxScore = Math.max(...scoreChart.data.datasets[0].data, state.threshold * 1.5);
  scoreChart.options.scales.y.max = Math.max(maxScore * 1.2, state.threshold * 2);

  scoreChart.update('none');

  // --- Sensor Charts (Positions, Currents, Velocities) ---
  const sensorMap = {
    positions:  features[FEATURE_GROUPS.positions.offset + joint],
    currents:   features[FEATURE_GROUPS.currents.offset + joint],
    velocities: features[FEATURE_GROUPS.velocities.offset + joint],
  };

  for (const [key, value] of Object.entries(sensorMap)) {
    const chart = charts[key];
    chart.data.labels.push(now);
    chart.data.datasets[0].data.push(value);

    if (chart.data.labels.length > maxPts) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
    }
    chart.update('none');

    // Update live value
    const statId = `stat${key.charAt(0).toUpperCase() + key.slice(1)}Value`;
    const el = document.getElementById(statId.replace('Positions', 'Position').replace('Currents', 'Current').replace('Velocities', 'Velocity'));
    if (el) el.textContent = value.toFixed(4);
  }

  // --- TCP Chart (all 6 lines always visible) ---
  const tcpChart = charts.tcp;
  tcpChart.data.labels.push(now);
  for (let i = 0; i < 6; i++) {
    tcpChart.data.datasets[i].data.push(features[FEATURE_GROUPS.tcp.offset + i]);
  }
  if (tcpChart.data.labels.length > maxPts) {
    tcpChart.data.labels.shift();
    for (let i = 0; i < 6; i++) tcpChart.data.datasets[i].data.shift();
  }
  tcpChart.update('none');

  // TCP live stat: show magnitude of Fx,Fy,Fz
  const fx = features[18], fy = features[19], fz = features[20];
  const forceMag = Math.sqrt(fx * fx + fy * fy + fz * fz);
  document.getElementById('statTCPValue').textContent = `|F| = ${forceMag.toFixed(2)} N`;

  // --- Update Global Status ---
  updateGlobalStatus(score);

  // --- Update score stats ---
  state.scoreHistory.push(score);
  if (state.scoreHistory.length > 1000) state.scoreHistory.shift();

  document.getElementById('statScoreCurrent').textContent = score.toFixed(5);
  const mean = state.scoreHistory.reduce((a, b) => a + b, 0) / state.scoreHistory.length;
  document.getElementById('statScoreMean').textContent = mean.toFixed(5);
  const max = Math.max(...state.scoreHistory);
  document.getElementById('statScoreMax').textContent = max.toFixed(5);
}

function updateGlobalStatus(score) {
  const banner = document.getElementById('globalStatus');
  const label = document.getElementById('statusLabel');
  const scoreEl = document.getElementById('scoreDisplay');

  scoreEl.textContent = score.toFixed(4);

  const now = performance.now();
  const above = score > state.threshold;

  if (above) {
    if (state.aboveSince === null) state.aboveSince = now;
  } else {
    state.aboveSince = null;
  }

  // The alarm fires only after `debounceMs` continuously above threshold, and clears
  // as soon as the score falls back. With debounceMs=0 the behaviour is the previous
  // instantaneous one (useful for comparison).
  const heldFor = state.aboveSince === null ? 0 : now - state.aboveSince;
  const shouldAlarm = above && heldFor >= state.debounceMs;

  if (shouldAlarm && !state.alarmActive) {
    state.anomalyCount++;   // conta gli EVENTI, non le singole finestre
  }
  state.alarmActive = shouldAlarm;

  if (shouldAlarm) {
    banner.className = 'global-status global-status--anomaly';
    label.textContent = '⚠ ANOMALY DETECTED';
    scoreEl.className = 'global-status__score global-status__score--danger';
  } else {
    banner.className = 'global-status global-status--normal';
    // Shows that the score is above threshold but not yet confirmed: makes the
    // debouncing visible instead of looking like the model is not reacting.
    label.textContent = above
      ? `◔ Above threshold ${Math.round(heldFor)}/${state.debounceMs} ms`
      : '✓ System Normal';
    scoreEl.className = 'global-status__score global-status__score--safe';
  }

  document.getElementById('statAnomalyCount').textContent = state.anomalyCount;
}

// ---- Clear Charts ----
function clearCharts() {
  for (const chart of Object.values(charts)) {
    chart.data.labels = [];
    for (const ds of chart.data.datasets) ds.data = [];
    chart.update('none');
  }
  state.scoreHistory = [];
  state.anomalyCount = 0;
  // Also clears the debounce state, otherwise an alarm in progress would stay
  // latent and the above-threshold timer would continue from the previous session.
  state.aboveSince = null;
  state.alarmActive = false;
  document.getElementById('statAnomalyCount').textContent = '0';
  document.getElementById('statScoreCurrent').textContent = '—';
  document.getElementById('statScoreMean').textContent = '—';
  document.getElementById('statScoreMax').textContent = '—';
  document.getElementById('scoreDisplay').textContent = '—';
  document.getElementById('statusLabel').textContent = '⏸ Waiting for data...';
  // Neutral state: without it the banner would stay solid green (= "all good")
  // before a single sample has arrived.
  document.getElementById('globalStatus').className = 'global-status global-status--idle';
  document.getElementById('scoreDisplay').className = 'global-status__score';
}

// ---- Web Serial API ----
async function handleConnect() {
  if (!('serial' in navigator)) {
    alert('Web Serial API not supported. Use Chrome or Edge.');
    return;
  }

  try {
    state.port = await navigator.serial.requestPort();
    await state.port.open({ baudRate: 115200 });

    setConnectionState('connected');
    clearCharts();

    const decoder = new TextDecoderStream();
    const readableStreamClosed = state.port.readable.pipeTo(decoder.writable);
    state.reader = decoder.readable.getReader();

    let lineBuffer = '';

    while (true) {
      const { value, done } = await state.reader.read();
      if (done) break;

      lineBuffer += value;
      const lines = lineBuffer.split('\n');
      lineBuffer = lines.pop(); // keep incomplete last line

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const data = JSON.parse(trimmed);
          if (data.s !== undefined && Array.isArray(data.f) && data.f.length === 24) {
            pushData(data.s, data.f);
          }
        } catch (e) {
          // skip non-JSON lines (e.g. debug prints)
        }
      }
    }
  } catch (err) {
    console.error('Serial connection error:', err);
    setConnectionState('disconnected');
  }
}

async function handleStop() {
  // Stop simulation
  if (state.simInterval) {
    clearInterval(state.simInterval);
    state.simInterval = null;
  }

  // Close WebSocket
  if (state.ws) {
    state.ws.close();
    state.ws = null;
  }

  // Close serial
  if (state.reader) {
    try { await state.reader.cancel(); } catch (e) {}
    state.reader = null;
  }
  if (state.port) {
    try { await state.port.close(); } catch (e) {}
    state.port = null;
  }

  state.simulating = false;
  state.connected = false;
  setConnectionState('disconnected');
}

// ---- Simulation ----
function handleSimulate() {
  if (state.simulating) return;

  clearCharts();
  state.simulating = true;
  setConnectionState('simulating');

  let step = 0;
  const Hz = 30; // simulation update rate

  state.simInterval = setInterval(() => {
    step++;
    const t = step / Hz;

    // Simulate 24 features (UR5-like sinusoidal motion)
    const features = [];

    // Positions (actual_q_0..5): slow sinusoidal motion
    for (let j = 0; j < 6; j++) {
      features.push(Math.sin(t * 0.5 + j * 0.8) * (1.5 + j * 0.3) + (Math.random() - 0.5) * 0.02);
    }

    // Currents (actual_current_0..5): correlated with velocity + noise
    for (let j = 0; j < 6; j++) {
      features.push(Math.cos(t * 0.5 + j * 0.8) * (0.8 + j * 0.2) + (Math.random() - 0.5) * 0.1);
    }

    // Velocities (actual_qd_0..5): derivative of positions + noise
    for (let j = 0; j < 6; j++) {
      features.push(Math.cos(t * 0.5 + j * 0.8) * 0.5 * (1.5 + j * 0.3) + (Math.random() - 0.5) * 0.05);
    }

    // TCP Forces (Fx,Fy,Fz,Tx,Ty,Tz): mostly steady with some dynamics
    for (let j = 0; j < 6; j++) {
      features.push(Math.sin(t * 0.3 + j * 1.2) * (5 + j * 2) + (Math.random() - 0.5) * 1.0);
    }

    // Score: normally very low, with occasional spikes (simulated anomalies)
    let score = Math.abs(Math.random() * 0.005 + 0.001);

    // Inject anomaly every ~10 seconds for ~1 second
    const anomalyPhase = t % 10;
    if (anomalyPhase > 8.5 && anomalyPhase < 9.5) {
      score = 0.15 + Math.random() * 0.3;
      // Also distort some features during anomaly
      features[2] += 2.0; // position spike
      features[8] += 3.0; // current spike
    }

    pushData(score, features);
  }, 1000 / Hz);
}

// ---- WebSocket Bridge ----
function handleBridge() {
  const wsUrl = 'ws://localhost:8765';
  clearCharts();

  const ws = new WebSocket(wsUrl);
  state.ws = ws;

  ws.onopen = () => {
    console.log('WebSocket connected to bridge');
    setConnectionState('bridge');
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.s !== undefined && Array.isArray(data.f) && data.f.length === 24) {
        pushData(data.s, data.f);
      }
    } catch (e) {
      // skip invalid messages
    }
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
    alert('Cannot connect to bridge.py on ws://localhost:8765.\nMake sure bridge.py is running.');
    setConnectionState('disconnected');
  };

  ws.onclose = () => {
    console.log('WebSocket closed');
    if (state.ws === ws) {
      state.ws = null;
      setConnectionState('disconnected');
    }
  };
}

// ---- UI State Management ----
function setConnectionState(mode) {
  const badge = document.getElementById('statusBadge');
  const text = document.getElementById('statusText');
  const btnConnect = document.getElementById('btnConnect');
  const btnBridge = document.getElementById('btnBridge');
  const btnSimulate = document.getElementById('btnSimulate');
  const btnStop = document.getElementById('btnStop');

  badge.className = `status-badge status-badge--${mode}`;

  const allDisabled = () => {
    btnConnect.disabled = true;
    if (btnBridge) btnBridge.disabled = true;
    btnSimulate.disabled = true;
    btnStop.disabled = false;
  };

  switch (mode) {
    case 'connected':
      text.textContent = 'Serial';
      allDisabled();
      break;
    case 'bridge':
      text.textContent = 'Bridge';
      allDisabled();
      break;
    case 'simulating':
      text.textContent = 'Simulating';
      allDisabled();
      break;
    case 'disconnected':
    default:
      text.textContent = 'Disconnected';
      btnConnect.disabled = false;
      if (btnBridge) btnBridge.disabled = false;
      btnSimulate.disabled = false;
      btnStop.disabled = true;
      break;
  }
}

// ---- Event Handlers ----
function handleJointChange() {
  state.selectedJoint = parseInt(document.getElementById('jointSelect').value);
  // Clear sensor charts only (keep score)
  for (const key of ['positions', 'currents', 'velocities']) {
    charts[key].data.labels = [];
    charts[key].data.datasets[0].data = [];
    charts[key].update('none');
  }
}

function handleThresholdChange() {
  state.threshold = parseFloat(document.getElementById('thresholdInput').value) || 0.029;
  // 4 decimals: the thresholds in use are ~0.0246 (float32) and ~0.0294 (int8), with two
  // decimali diventerebbero entrambe "0.03".
  updateThresholdDisplay();
  state.aboveSince = null;   // riparte il conteggio del debounce
}

function handleDebounceChange() {
  const v = parseInt(document.getElementById('debounceInput').value);
  state.debounceMs = Number.isFinite(v) && v >= 0 ? v : 250;
  updateThresholdDisplay();
  state.aboveSince = null;
}

function updateThresholdDisplay() {
  document.getElementById('thresholdDisplay').textContent =
    `Threshold: ${state.threshold.toFixed(4)} · debounce ${state.debounceMs} ms`;
}

function handleBufferChange() {
  state.maxPoints = parseInt(document.getElementById('bufferSelect').value);
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  handleThresholdChange();
});
