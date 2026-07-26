/* VoxGuard dashboard.
 *
 * No framework and no CDN: charts are hand-built SVG so the page works on a
 * locked-down box with no outbound network. The token lives in
 * sessionStorage and travels as an Authorization header, so there is no
 * cookie for a cross-site request to ride on.
 */

const API = {
  token: sessionStorage.getItem('vg_token') || '',
  async get(path) {
    const res = await fetch(path, { headers: { Authorization: `Bearer ${this.token}` } });
    if (res.status === 401) { signOut(); throw new Error('unauthorized'); }
    if (!res.ok) throw new Error(`${path} -> ${res.status}`);
    return res.json();
  },
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const PALETTE = {
  ban: '#F43F5E', kick: '#F59E0B', timeout: '#38BDF8',
  warn: '#8B5CF6', joins: '#10B981', automod: '#22D3EE', growth: '#6366F1',
};

const state = { days: 30, view: 'overview', guilds: [], guildId: null };

/* ---- helpers ---------------------------------------------------------- */

const fmt = (n) => (n ?? 0).toLocaleString();
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function ago(ts) {
  const secs = Math.floor(Date.now() / 1000 - ts);
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

function duration(secs) {
  const d = Math.floor(secs / 86400), h = Math.floor((secs % 86400) / 3600), m = Math.floor((secs % 3600) / 60);
  return [d && `${d}d`, h && `${h}h`, `${m}m`].filter(Boolean).join(' ');
}

function initials(name) {
  return name.split(/\s+/).slice(0, 2).map((w) => w[0] || '').join('').toUpperCase() || '?';
}

function iconHtml(icon, name, cls = 'server-icon') {
  return icon
    ? `<img class="${cls}" src="${esc(icon)}" alt="">`
    : `<div class="${cls}">${esc(initials(name))}</div>`;
}

/* ---- charts ----------------------------------------------------------- */

/* Grouped bar chart. `series` is [{key,label,color,points:[{day,value}]}]. */
function barChart(el, series, { height = 210 } = {}) {
  const days = series[0]?.points.map((p) => p.day) ?? [];
  if (!days.length) { el.innerHTML = '<div class="empty">No data yet</div>'; return; }

  const W = 760, H = height, pad = { t: 12, r: 10, b: 26, l: 38 };
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const max = Math.max(1, ...series.flatMap((s) => s.points.map((p) => p.value)));
  const niceMax = Math.ceil(max / 4) * 4 || 4;

  const slot = plotW / days.length;
  const groupW = Math.min(slot * 0.72, 34);
  const barW = Math.max(1.5, groupW / series.length - 1);

  const y = (v) => pad.t + plotH - (v / niceMax) * plotH;
  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">`;

  // horizontal grid + y labels
  for (let i = 0; i <= 4; i++) {
    const val = (niceMax / 4) * i, yy = y(val);
    svg += `<line class="grid-line" x1="${pad.l}" y1="${yy}" x2="${W - pad.r}" y2="${yy}"/>`;
    svg += `<text class="axis-text" x="${pad.l - 7}" y="${yy + 3.5}" text-anchor="end">${val}</text>`;
  }

  // bars
  days.forEach((day, di) => {
    const gx = pad.l + slot * di + (slot - groupW) / 2;
    series.forEach((s, si) => {
      const v = s.points[di]?.value ?? 0;
      if (!v) return;
      const h = Math.max(2, plotH - (y(v) - pad.t));
      const x = gx + si * (barW + 1);
      svg += `<rect class="bar" x="${x.toFixed(1)}" y="${y(v).toFixed(1)}" width="${barW.toFixed(1)}" `
           + `height="${h.toFixed(1)}" rx="${Math.min(2.5, barW / 2).toFixed(1)}" fill="${s.color}">`
           + `<title>${esc(day)} · ${esc(s.label)}: ${v}</title></rect>`;
    });
  });

  // x labels, thinned to avoid overlap
  const step = Math.ceil(days.length / 8);
  days.forEach((day, di) => {
    if (di % step) return;
    const label = day.slice(5);
    svg += `<text class="axis-text" x="${(pad.l + slot * di + slot / 2).toFixed(1)}" `
         + `y="${H - 8}" text-anchor="middle">${label}</text>`;
  });

  svg += '</svg>';
  el.innerHTML = svg;
}

/* Filled area/line chart for a single cumulative series. */
function areaChart(el, points, color, { height = 210, label = 'value' } = {}) {
  if (!points?.length) { el.innerHTML = '<div class="empty">No data yet</div>'; return; }

  const W = 760, H = height, pad = { t: 12, r: 10, b: 26, l: 38 };
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const values = points.map((p) => p.value);
  const max = Math.max(1, ...values);
  const min = Math.min(...values, 0);
  const span = Math.max(1, max - min);
  const niceMax = max + Math.ceil(span * 0.15);

  const x = (i) => pad.l + (points.length === 1 ? plotW / 2 : (plotW * i) / (points.length - 1));
  const y = (v) => pad.t + plotH - ((v - min) / Math.max(1, niceMax - min)) * plotH;

  const id = `grad${Math.random().toString(36).slice(2, 8)}`;
  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">`;
  svg += `<defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1">`
       + `<stop offset="0%" stop-color="${color}" stop-opacity="0.34"/>`
       + `<stop offset="100%" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>`;

  for (let i = 0; i <= 4; i++) {
    const val = min + ((niceMax - min) / 4) * i, yy = y(val);
    svg += `<line class="grid-line" x1="${pad.l}" y1="${yy}" x2="${W - pad.r}" y2="${yy}"/>`;
    svg += `<text class="axis-text" x="${pad.l - 7}" y="${yy + 3.5}" text-anchor="end">${Math.round(val)}</text>`;
  }

  const line = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(' ');
  svg += `<path d="${line} L${x(points.length - 1).toFixed(1)},${(pad.t + plotH).toFixed(1)} `
       + `L${x(0).toFixed(1)},${(pad.t + plotH).toFixed(1)} Z" fill="url(#${id})"/>`;
  svg += `<path d="${line}" fill="none" stroke="${color}" stroke-width="2.2" `
       + `stroke-linejoin="round" stroke-linecap="round"/>`;

  points.forEach((p, i) => {
    svg += `<circle cx="${x(i).toFixed(1)}" cy="${y(p.value).toFixed(1)}" r="7" fill="transparent">`
         + `<title>${esc(p.day)} · ${esc(label)}: ${p.value}</title></circle>`;
  });

  const step = Math.ceil(points.length / 8);
  points.forEach((p, i) => {
    if (i % step) return;
    svg += `<text class="axis-text" x="${x(i).toFixed(1)}" y="${H - 8}" text-anchor="middle">${p.day.slice(5)}</text>`;
  });

  svg += '</svg>';
  el.innerHTML = svg;
}

function legend(el, series) {
  el.innerHTML = series.map((s) =>
    `<span class="legend-item"><span class="legend-swatch" style="background:${s.color}"></span>${esc(s.label)}</span>`
  ).join('');
}

function statTile({ label, value, sub, danger }) {
  return `<div class="stat${danger ? ' is-danger' : ''}">
    <div class="stat-label">${esc(label)}</div>
    <div class="stat-value">${esc(value)}</div>
    ${sub ? `<div class="stat-sub">${esc(sub)}</div>` : ''}
  </div>`;
}

/* ---- views ------------------------------------------------------------ */

async function loadOverview() {
  const data = await API.get(`/api/overview?days=${state.days}`);
  const t = data.totals;

  $('#stat-grid').innerHTML = [
    statTile({ label: 'Servers', value: fmt(t.guilds), sub: `${fmt(t.members)} members reached` }),
    statTile({ label: 'Bans', value: fmt(t.bans) }),
    statTile({ label: 'Kicks', value: fmt(t.kicks) }),
    statTile({ label: 'Timeouts', value: fmt(t.timeouts) }),
    statTile({ label: 'Warnings', value: fmt(t.warns) }),
    statTile({ label: 'Automod hits', value: fmt(t.automod_triggers) }),
    statTile({ label: 'Level-ups', value: fmt(t.levelups) }),
    statTile({ label: 'Errors (24h)', value: fmt(t.errors_24h), sub: `${fmt(t.errors_total)} all time`, danger: t.errors_24h > 0 }),
  ].join('');

  const actions = [
    { label: 'Bans', color: PALETTE.ban, points: data.charts.bans },
    { label: 'Kicks', color: PALETTE.kick, points: data.charts.kicks },
    { label: 'Timeouts', color: PALETTE.timeout, points: data.charts.timeouts },
    { label: 'Warnings', color: PALETTE.warn, points: data.charts.warns },
  ];
  barChart($('#chart-actions'), actions);
  legend($('#legend-actions'), actions);

  areaChart($('#chart-growth'), data.growth, PALETTE.growth, { label: 'servers' });
  barChart($('#chart-automod'), [{ label: 'Automod', color: PALETTE.automod, points: data.charts.automod }]);

  const bot = data.bot;
  const status = $('#bot-status');
  status.className = `status ${bot.ready ? 'is-online' : 'is-offline'}`;
  $('#bot-status-text').textContent = bot.ready
    ? `${bot.latency_ms}ms · up ${duration(bot.uptime_seconds)}`
    : 'disconnected';
  if (t.errors_24h > 0) {
    $('#err-badge').hidden = false;
    $('#err-badge').textContent = t.errors_24h > 99 ? '99+' : t.errors_24h;
  } else {
    $('#err-badge').hidden = true;
  }

  await loadServers();
  const top = state.guilds.slice(0, 6);
  const maxMembers = Math.max(1, ...top.map((g) => g.members));
  $('#top-servers').innerHTML = top.length ? top.map((g) => `
    <button class="strip-row" data-guild="${esc(g.id)}">
      ${iconHtml(g.icon, g.name)}
      <span style="min-width:130px;font-size:13px">${esc(g.name)}</span>
      <span class="strip-bar"><span class="strip-fill" style="width:${(g.members / maxMembers) * 100}%"></span></span>
      <span class="strip-val">${fmt(g.members)}</span>
    </button>`).join('') : '<div class="empty">Not in any servers yet</div>';
}

async function loadServers() {
  const data = await API.get('/api/guilds');
  state.guilds = data.guilds;
  renderServers();
}

function renderServers() {
  const query = ($('#server-search')?.value || '').toLowerCase();
  const list = state.guilds.filter((g) => g.name.toLowerCase().includes(query));
  $('#server-grid').innerHTML = list.length ? list.map((g) => `
    <button class="server-card" data-guild="${esc(g.id)}">
      <div class="server-top">
        ${iconHtml(g.icon, g.name)}
        <div>
          <div class="server-name">${esc(g.name)}</div>
          <div class="server-meta">${fmt(g.members)} members · ${fmt(g.actions)} actions</div>
        </div>
      </div>
      <div class="chips">
        ${g.features_on.length
          ? g.features_on.slice(0, 4).map((f) => `<span class="chip">${esc(f)}</span>`).join('')
            + (g.features_on.length > 4 ? `<span class="chip">+${g.features_on.length - 4}</span>` : '')
          : '<span class="chip is-empty">Nothing enabled</span>'}
      </div>
    </button>`).join('') : '<div class="empty">No servers match that search</div>';
}

async function loadGuild(id) {
  state.guildId = id;
  const data = await API.get(`/api/guild/${id}?days=${state.days}`);
  const g = data.guild, t = data.totals;

  $('#guild-head').innerHTML = `
    ${iconHtml(g.icon, g.name)}
    <div>
      <h2>${esc(g.name)}</h2>
      <p class="muted">${fmt(g.members)} members · ${fmt(g.channels)} channels · ${fmt(g.roles)} roles · ${fmt(g.boosts)} boosts</p>
    </div>`;

  $('#guild-stats').innerHTML = [
    statTile({ label: 'Bans', value: fmt(t.bans) }),
    statTile({ label: 'Kicks', value: fmt(t.kicks) }),
    statTile({ label: 'Timeouts', value: fmt(t.timeouts) }),
    statTile({ label: 'Warnings', value: fmt(t.warns) }),
    statTile({ label: 'Automod hits', value: fmt(t.automod_triggers) }),
    statTile({ label: 'Voice flags', value: fmt(t.voice_flags) }),
    statTile({ label: 'Tickets', value: fmt(t.tickets_opened) }),
    statTile({ label: 'Errors (24h)', value: fmt(t.errors_24h), danger: t.errors_24h > 0 }),
  ].join('');

  const actions = [
    { label: 'Bans', color: PALETTE.ban, points: data.charts.bans },
    { label: 'Kicks', color: PALETTE.kick, points: data.charts.kicks },
    { label: 'Timeouts', color: PALETTE.timeout, points: data.charts.timeouts },
    { label: 'Warnings', color: PALETTE.warn, points: data.charts.warns },
  ];
  barChart($('#guild-chart-actions'), actions);
  legend($('#guild-legend'), actions);
  barChart($('#guild-chart-joins'), [{ label: 'Joins', color: PALETTE.joins, points: data.charts.joins }]);

  $('#guild-features').innerHTML = data.features.all.map((f) => `
    <div class="feature-row${f.enabled ? ' is-on' : ''}">
      <span class="feature-dot"></span>
      <div>
        <div class="feature-name">${esc(f.name)}</div>
        <div class="feature-desc">${esc(f.description)}</div>
        ${f.enabled && f.detail ? `<div class="feature-detail">${esc(f.detail)}</div>` : ''}
      </div>
    </div>`).join('');

  $('#guild-cases').innerHTML = data.recent_cases.length ? `
    <table><thead><tr><th>Case</th><th>Action</th><th>User</th><th>Reason</th><th>When</th></tr></thead>
    <tbody>${data.recent_cases.map((c) => `
      <tr>
        <td>#${c.case}</td>
        <td><span class="pill pill-${esc(c.action)}">${esc(c.action)}</span></td>
        <td>${esc(c.user_id)}</td>
        <td>${esc((c.reason || '—').slice(0, 60))}</td>
        <td>${esc(ago(c.at))}</td>
      </tr>`).join('')}</tbody></table>` : '<div class="empty">No moderation cases yet</div>';

  $('#guild-leaderboard').innerHTML = data.leaderboard.length ? `
    <table><thead><tr><th>#</th><th>User</th><th>XP</th><th>Messages</th><th>Voice</th></tr></thead>
    <tbody>${data.leaderboard.map((m, i) => `
      <tr>
        <td>${i + 1}</td>
        <td>${esc(m.user_id)}</td>
        <td>${fmt(m.xp)}</td>
        <td>${fmt(m.messages)}</td>
        <td>${duration(m.voice_seconds)}</td>
      </tr>`).join('')}</tbody></table>` : '<div class="empty">No XP recorded yet</div>';

  showView('guild', g.name, 'Server detail');
}

async function loadErrors() {
  const data = await API.get('/api/errors?limit=150');
  const c = data.counts;
  $('#error-stats').innerHTML = [
    statTile({ label: 'Last hour', value: fmt(c.last_hour), danger: c.last_hour > 0 }),
    statTile({ label: 'Last 24 hours', value: fmt(c.last_24h), danger: c.last_24h > 0 }),
    statTile({ label: 'Last 7 days', value: fmt(c.last_7d) }),
    statTile({ label: 'All time', value: fmt(c.total) }),
  ].join('');

  $('#error-list').innerHTML = data.errors.length ? data.errors.map((e) => `
    <div class="error-row${e.level === 'warning' ? ' is-warning' : ''}">
      <div class="error-source">${esc(e.source)}</div>
      <div>
        <div class="error-msg">${esc(e.message)}</div>
        ${e.guild_name && e.guild_name !== '—' ? `<div class="error-meta">${esc(e.guild_name)}</div>` : ''}
      </div>
      <div class="error-meta">${esc(ago(e.at))}</div>
      ${e.detail ? `<pre class="error-detail">${esc(e.detail)}</pre>` : ''}
    </div>`).join('') : '<div class="empty">No errors recorded — all clear</div>';
}

/* ---- routing ---------------------------------------------------------- */

const TITLES = {
  overview: ['Overview', 'Live across every server'],
  servers: ['Servers', 'Every server VoxGuard is in'],
  errors: ['Error Log', 'Runtime failures, newest first'],
};

function showView(name, title, sub) {
  state.view = name;
  $$('.view').forEach((v) => (v.hidden = true));
  $(`#view-${name}`).hidden = false;
  const [t, s] = TITLES[name] || [title, sub];
  $('#view-title').textContent = title || t;
  $('#view-sub').textContent = sub || s;
  $$('.nav-item').forEach((b) => b.classList.toggle('is-active', b.dataset.view === name));
}

async function refresh() {
  try {
    if (state.view === 'overview') await loadOverview();
    else if (state.view === 'servers') await loadServers();
    else if (state.view === 'errors') await loadErrors();
    else if (state.view === 'guild' && state.guildId) await loadGuild(state.guildId);
  } catch (err) {
    if (err.message !== 'unauthorized') console.error(err);
  }
}

function signOut() {
  sessionStorage.removeItem('vg_token');
  API.token = '';
  $('#app').hidden = true;
  $('#gate').style.display = 'grid';
}

async function boot() {
  $('#gate').style.display = 'none';
  $('#app').hidden = false;
  await refresh();
}

/* ---- events ----------------------------------------------------------- */

$('#gate-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const token = $('#token').value.trim();
  const error = $('#gate-error');
  error.hidden = true;
  try {
    const res = await fetch('/api/auth', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    });
    if (!res.ok) throw new Error('Invalid token');
    sessionStorage.setItem('vg_token', token);
    API.token = token;
    await boot();
  } catch {
    error.textContent = 'That token was not accepted.';
    error.hidden = false;
  }
});

$('#signout').addEventListener('click', signOut);
$('#refresh').addEventListener('click', refresh);
$('#range').addEventListener('change', (e) => { state.days = Number(e.target.value); refresh(); });
$('#server-search').addEventListener('input', renderServers);
$('#back-to-servers').addEventListener('click', () => { showView('servers'); loadServers(); });

$$('.nav-item').forEach((btn) => btn.addEventListener('click', () => {
  showView(btn.dataset.view);
  refresh();
}));

// Server cards are rendered dynamically, so delegate from the document.
document.addEventListener('click', (event) => {
  const target = event.target.closest('[data-guild]');
  if (target) loadGuild(target.dataset.guild);
});

// Poll while the tab is visible so the numbers stay live without hammering.
setInterval(() => { if (!document.hidden && API.token) refresh(); }, 30000);

if (API.token) boot();
