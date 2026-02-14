/* simulation.js – EdgeMind CX Simulation Results */

var METRIC_CONFIG = [
  { key: 'total_tasks', label: 'Total Tasks', desc: '' },
  { key: 'completed_tasks', label: 'Completed Tasks', desc: '' },
  { key: 'failed_tasks', label: 'Failed Tasks', desc: '' },
  { key: 'avg_service_time', label: 'Avg Service Time', desc: 's' },
  { key: 'avg_processing_time', label: 'Avg Processing Time', desc: 's' },
  { key: 'avg_network_delay', label: 'Avg Network Delay', desc: 's' },
  { key: 'edge_executed_tasks', label: 'Edge Executed Tasks', desc: '' },
];

document.addEventListener('DOMContentLoaded', function () {
  fetch('data/cx_summary.csv')
    .then(function (res) {
      if (!res.ok) {
        showSimulationDataNotFound();
        return null;
      }
      return res.text();
    })
    .then(function (text) {
      if (text === null) return;
      try {
        var data = parseCsvToObject(text);
        console.log(data);
        renderMetricsGrid(data);
        if (data.edge_executed_tasks !== undefined && data.total_tasks !== undefined) {
          drawEdgeChart(data);
        } else {
          console.warn('drawEdgeChart: edge_executed_tasks veya total_tasks undefined, chart çizilmedi.');
        }
      } catch (e) {
        console.error('Simulation data işlenirken hata:', e);
        showSimulationDataNotFound();
      }
    })
    .catch(function (err) {
      console.error('cx_summary.csv yüklenemedi:', err);
      showSimulationDataNotFound();
    });
});

function showSimulationDataNotFound() {
  try {
    var grid = document.getElementById('metricsGrid');
    if (!grid) return;
    grid.innerHTML = '';
    var msg = document.createElement('div');
    msg.className = 'simulation-data-message';
    msg.textContent = 'Simulation data not found';
    grid.appendChild(msg);
  } catch (e) {
    console.error('showSimulationDataNotFound:', e);
  }
}

function renderMetricsGrid(data) {
  var grid = document.getElementById('metricsGrid');
  if (!grid) return;
  if (!data || typeof data !== 'object') return;
  for (var i = 0; i < METRIC_CONFIG.length; i++) {
    var config = METRIC_CONFIG[i];
    var value = data[config.key];
    var card = createMetricCard(config.label, value, config.desc);
    grid.appendChild(card);
  }
}

function createMetricCard(title, value, subDesc) {
  var card = document.createElement('div');
  card.className = 'card';

  var titleEl = document.createElement('div');
  titleEl.className = 'card-title';
  titleEl.textContent = title;

  var valueEl = document.createElement('div');
  valueEl.className = 'card-value';
  valueEl.textContent = formatMetricValue(value);

  card.appendChild(titleEl);
  card.appendChild(valueEl);

  if (subDesc && subDesc.length > 0) {
    var descEl = document.createElement('div');
    descEl.className = 'card-desc';
    descEl.textContent = subDesc;
    card.appendChild(descEl);
  }

  return card;
}

function formatMetricValue(value) {
  if (value === undefined || value === null) return '—';
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(2);
  }
  return String(value);
}

/**
 * CSV metnini satır satır parse eder; key,value formatında her satır için
 * obj[key] = value (sayı ise number, değilse string) döner.
 * @param {string} csv - Ham CSV metni
 * @returns {Object} { key: value, ... }
 */
function parseCsvToObject(csv) {
  var obj = {};
  var lines = csv.split(/\r?\n/);
  for (var i = 0; i < lines.length; i++) {
    var line = lines[i].trim();
    if (!line) continue;
    var commaIndex = line.indexOf(',');
    if (commaIndex === -1) continue;
    var key = line.slice(0, commaIndex).trim();
    var valueStr = line.slice(commaIndex + 1).trim();
    if (!key) continue;
    var value = valueStr === '' ? '' : (isNumeric(valueStr) ? parseFloat(valueStr) : valueStr);
    obj[key] = value;
  }
  return obj;
}

function isNumeric(str) {
  return str !== '' && !isNaN(Number(str));
}

var NS = 'http://www.w3.org/2000/svg';

/**
 * total_tasks ve edge_executed_tasks ile #edgeChart içinde SVG bar chart çizer.
 * @param {Object} metrics - { total_tasks, edge_executed_tasks, ... }
 */
function drawEdgeChart(metrics) {
  if (metrics == null || typeof metrics !== 'object') return;
  if (metrics.edge_executed_tasks === undefined || metrics.total_tasks === undefined) {
    console.warn('drawEdgeChart: edge_executed_tasks veya total_tasks undefined, chart çizilmedi.');
    return;
  }
  var container = document.getElementById('edgeChart');
  if (!container) return;

  var total = Number(metrics.total_tasks) || 0;
  var edge = Number(metrics.edge_executed_tasks) || 0;
  var maxVal = Math.max(total, 1);

  container.innerHTML = '';

  var vbW = 500;
  var vbH = 100;
  var pad = { left: 140, right: 60, top: 20, bottom: 20 };
  var barH = 28;
  var gap = 18;
  var barAreaW = vbW - pad.left - pad.right;

  var svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 ' + vbW + ' ' + vbH);
  svg.setAttribute('width', '100%');
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.setAttribute('aria-label', 'Edge vs Total Tasks');

  var totalBarW = barAreaW;
  var edgeBarW = maxVal > 0 ? (edge / maxVal) * barAreaW : 0;

  var y1 = pad.top;
  var y2 = pad.top + barH + gap;

  var barBg = document.createElementNS(NS, 'rect');
  barBg.setAttribute('x', pad.left);
  barBg.setAttribute('y', 0);
  barBg.setAttribute('width', vbW - pad.left);
  barBg.setAttribute('height', vbH);
  barBg.setAttribute('fill', 'transparent');
  svg.appendChild(barBg);

  var r1 = document.createElementNS(NS, 'rect');
  r1.setAttribute('x', pad.left);
  r1.setAttribute('y', y1);
  r1.setAttribute('width', totalBarW);
  r1.setAttribute('height', barH);
  r1.setAttribute('rx', 6);
  r1.setAttribute('fill', '#334155');
  svg.appendChild(r1);

  var r2 = document.createElementNS(NS, 'rect');
  r2.setAttribute('x', pad.left);
  r2.setAttribute('y', y2);
  r2.setAttribute('width', Math.max(0, edgeBarW));
  r2.setAttribute('height', barH);
  r2.setAttribute('rx', 6);
  r2.setAttribute('fill', '#3b82f6');
  svg.appendChild(r2);

  var labelColor = '#94a3b8';
  var valueColor = '#f1f5f9';

  var t1 = document.createElementNS(NS, 'text');
  t1.setAttribute('x', pad.left - 10);
  t1.setAttribute('y', y1 + barH / 2 + 5);
  t1.setAttribute('text-anchor', 'end');
  t1.setAttribute('fill', labelColor);
  t1.setAttribute('font-size', '13');
  t1.setAttribute('font-weight', '500');
  t1.textContent = 'Total Tasks';
  svg.appendChild(t1);

  var v1 = document.createElementNS(NS, 'text');
  v1.setAttribute('x', pad.left + totalBarW + 8);
  v1.setAttribute('y', y1 + barH / 2 + 5);
  v1.setAttribute('text-anchor', 'start');
  v1.setAttribute('fill', valueColor);
  v1.setAttribute('font-size', '13');
  v1.setAttribute('font-weight', '600');
  v1.textContent = formatMetricValue(total);
  svg.appendChild(v1);

  var t2 = document.createElementNS(NS, 'text');
  t2.setAttribute('x', pad.left - 10);
  t2.setAttribute('y', y2 + barH / 2 + 5);
  t2.setAttribute('text-anchor', 'end');
  t2.setAttribute('fill', labelColor);
  t2.setAttribute('font-size', '13');
  t2.setAttribute('font-weight', '500');
  t2.textContent = 'Edge Executed Tasks';
  svg.appendChild(t2);

  var v2 = document.createElementNS(NS, 'text');
  v2.setAttribute('x', pad.left + edgeBarW + 8);
  v2.setAttribute('y', y2 + barH / 2 + 5);
  v2.setAttribute('text-anchor', 'start');
  v2.setAttribute('fill', valueColor);
  v2.setAttribute('font-size', '13');
  v2.setAttribute('font-weight', '600');
  v2.textContent = formatMetricValue(edge);
  svg.appendChild(v2);

  container.appendChild(svg);
}
