/* VoxGuard dashboard.
 *
 * No framework and no CDN: charts are hand-built SVG so the page works on a
 * locked-down box with no outbound network. The token lives in
 * sessionStorage and travels as an Authorization header, so there is no
 * cookie for a cross-site request to ride on.
 *
 * Series are separated by luminance rather than hue to match the monochrome
 * UI — the ramp runs white -> dark grey, so ordering stays readable when the
 * legend scrolls out of view.
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

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const RAMP = ['#FFFFFF', '#B8B8BE', '#7C7C84', '#4E4E56'];

const NAV = [
  { view: 'overview', icon: 'grid',   label: 'Overview' },
  { view: 'servers',  icon: 'server', label: 'Servers' },
  { view: 'errors',   icon: 'alert',  label: 'Error Log', badge: 'err-badge' },
];

const TITLES = {
  overview: ['Overview', 'Live across every server'],
  servers:  ['Servers', 'Every server running VoxGuard'],
  errors:   ['Error Log', 'Runtime failures, newest first'],
};

const state = { days: 30, view: 'overview', guilds: [], guildId: null };

/* ---- helpers ----------------------------------------------------------- */

const fmt = (n) => (n ?? 0).toLocaleString();
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function ago(ts) {
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function duration(secs) {
  const d = Math.floor(secs / 86400), h = Math.floor((secs % 86400) / 3600), m = Math.floor((secs % 3600) / 60);
  return [d && `${d}d`, h && `${h}h`, `${m}m`].filter(Boolean).join(' ') || '0m';
}

const initials = (name) =>
  name.split(/\s+/).slice(0, 2).map((w) => w[0] || '').join('').toUpperCase() || '?';

const iconHtml = (url, name, cls = 'server-icon') =>
  url ? `<img class="${cls}" src="${esc(url)}" alt="">`
      : `<div class="${cls}">${esc(initials(name))}</div>`;

const emptyState = (text, ico = 'dot') => `<div class="empty">${icon(ico, 22)}<span>${esc(text)}</span></div>`;

/* ---- charts ------------------------------------------------------------ */

function barChart(el, series, { height = 200 } = {}) {
  const days = series[0]?.points.map((p) => p.day) ?? [];
  const total = series.reduce((sum, s) => sum + s.points.reduce((a, p) => a + p.value, 0), 0);
  if (!days.length || !total) { el.innerHTML = emptyState('No activity in this range'); return; }

  const W = 760, H = height, pad = { t: 10, r: 8, b: 24, l: 36 };
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const max = Math.max(1, ...series.flatMap((s) => s.points.map((p) => p.value)));
  const niceMax = Math.ceil(max / 4) * 4 || 4;

  const slot = plotW / days.length;
  const groupW = Math.min(slot * 0.74, 32);
  const barW = Math.max(1.4, groupW / series.length - 1);
  const y = (v) => pad.t + plotH - (v / niceMax) * plotH;

  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">`;
  for (let i = 0; i <= 4; i++) {
    const val = (niceMax / 4) * i, yy = y(val);
    svg += `<line class="grid-line" x1="${pad.l}" y1="${yy}" x2="${W - pad.r}" y2="${yy}"/>`
         + `<text class="axis-text" x="${pad.l - 7}" y="${yy + 3.5}" text-anchor="end">${val}</text>`;
  }

  days.forEach((day, di) => {
    const gx = pad.l + slot * di + (slot - groupW) / 2;
    series.forEach((s, si) => {
      const v = s.points[di]?.value ?? 0;
      if (!v) return;
      const h = Math.max(2, plotH - (y(v) - pad.t));
      svg += `<rect class="bar" x="${(gx + si * (barW + 1)).toFixed(1)}" y="${y(v).toFixed(1)}" `
           + `width="${barW.toFixed(1)}" height="${h.toFixed(1)}" rx="${Math.min(2, barW / 2).toFixed(1)}" `
           + `fill="${s.color}"><title>${esc(day)} · ${esc(s.label)}: ${v}</title></rect>`;
    });
  });

  const step = Math.ceil(days.length / 8);
  days.forEach((day, di) => {
    if (di % step) return;
    svg += `<text class="axis-text" x="${(pad.l + slot * di + slot / 2).toFixed(1)}" y="${H - 7}" `
         + `text-anchor="middle">${day.slice(5)}</text>`;
  });

  el.innerHTML = svg + '</svg>';
}

function areaChart(el, points, { height = 200, label = 'value' } = {}) {
  if (!points?.length) { el.innerHTML = emptyState('No data yet'); return; }

  const W = 760, H = height, pad = { t: 10, r: 8, b: 24, l: 36 };
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const values = points.map((p) => p.value);
  const max = Math.max(1, ...values), min = Math.min(...values, 0);
  const niceMax = max + Math.ceil(Math.max(1, max - min) * 0.15);

  const x = (i) => pad.l + (points.length === 1 ? plotW / 2 : (plotW * i) / (points.length - 1));
  const y = (v) => pad.t + plotH - ((v - min) / Math.max(1, niceMax - min)) * plotH;

  const gid = `g${Math.random().toString(36).slice(2, 8)}`;
  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">`
    + `<defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">`
    + `<stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.20"/>`
    + `<stop offset="100%" stop-color="#FFFFFF" stop-opacity="0"/></linearGradient></defs>`;

  for (let i = 0; i <= 4; i++) {
    const val = min + ((niceMax - min) / 4) * i, yy = y(val);
    svg += `<line class="grid-line" x1="${pad.l}" y1="${yy}" x2="${W - pad.r}" y2="${yy}"/>`
         + `<text class="axis-text" x="${pad.l - 7}" y="${yy + 3.5}" text-anchor="end">${Math.round(val)}</text>`;
  }

  const line = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(' ');
  svg += `<path d="${line} L${x(points.length - 1).toFixed(1)},${(pad.t + plotH).toFixed(1)} `
       + `L${x(0).toFixed(1)},${(pad.t + plotH).toFixed(1)} Z" fill="url(#${gid})"/>`
       + `<path d="${line}" fill="none" stroke="#FFFFFF" stroke-width="1.9" `
       + `stroke-linejoin="round" stroke-linecap="round"/>`;

  points.forEach((p, i) => {
    svg += `<circle cx="${x(i).toFixed(1)}" cy="${y(p.value).toFixed(1)}" r="7" fill="transparent">`
         + `<title>${esc(p.day)} · ${esc(label)}: ${p.value}</title></circle>`;
  });

  const step = Math.ceil(points.length / 8);
  points.forEach((p, i) => {
    if (i % step) return;
    svg += `<text class="axis-text" x="${x(i).toFixed(1)}" y="${H - 7}" text-anchor="middle">${p.day.slice(5)}</text>`;
  });

  el.innerHTML = svg + '</svg>';
}

const legend = (el, series) => {
  el.innerHTML = series.map((s) =>
    `<span class="legend-item"><span class="legend-swatch" style="background:${s.color}"></span>${esc(s.label)}</span>`
  ).join('');
};

const statTile = ({ label, value, sub, alert: isAlert, ico }) => `
  <div class="stat glass${isAlert ? ' is-alert' : ''}">
    <div class="stat-label">${ico ? icon(ico, 13) : ''}${esc(label)}</div>
    <div class="stat-value">${esc(value)}</div>
    ${sub ? `<div class="stat-sub">${esc(sub)}</div>` : ''}
  </div>`;

/* ---- views ------------------------------------------------------------- */

async function loadOverview() {
  const data = await API.get(`/api/overview?days=${state.days}`);
  const t = data.totals;

  $('#stat-grid').innerHTML = [
    statTile({ ico: 'server', label: 'Servers', value: fmt(t.guilds), sub: `${fmt(t.members)} members reached` }),
    statTile({ ico: 'ban', label: 'Bans', value: fmt(t.bans) }),
    statTile({ ico: 'boot', label: 'Kicks', value: fmt(t.kicks) }),
    statTile({ ico: 'clock', label: 'Timeouts', value: fmt(t.timeouts) }),
    statTile({ ico: 'flag', label: 'Warnings', value: fmt(t.warns) }),
    statTile({ ico: 'shield', label: 'Automod', value: fmt(t.automod_triggers) }),
    statTile({ ico: 'brain', label: 'AI moderation', value: fmt(t.ai_moderation_hits) }),
    statTile({ ico: 'mic', label: 'Voice flags', value: fmt(t.voice_flags) }),
    statTile({ ico: 'star', label: 'Level-ups', value: fmt(t.levelups) }),
    statTile({ ico: 'alert', label: 'Errors (24h)', value: fmt(t.errors_24h),
               sub: `${fmt(t.errors_total)} all time`, alert: t.errors_24h > 0 }),
  ].join('');

  const actions = [
    { label: 'Bans', color: RAMP[0], points: data.charts.bans },
    { label: 'Kicks', color: RAMP[1], points: data.charts.kicks },
    { label: 'Timeouts', color: RAMP[2], points: data.charts.timeouts },
    { label: 'Warnings', color: RAMP[3], points: data.charts.warns },
  ];
  barChart($('#chart-actions'), actions);
  legend($('#legend-actions'), actions);

  areaChart($('#chart-growth'), data.growth, { label: 'servers' });

  const filters = [
    { label: 'Automod', color: RAMP[0], points: data.charts.automod },
    { label: 'AI moderation', color: RAMP[2], points: data.charts.ai_moderation },
  ];
  barChart($('#chart-automod'), filters);
  legend($('#legend-filters'), filters);

  const bot = data.bot;
  $('#bot-status').className = `status ${bot.ready ? 'is-online' : 'is-offline'}`;
  $('#bot-status-text').textContent = bot.ready
    ? `${bot.latency_ms}ms · up ${duration(bot.uptime_seconds)}` : 'disconnected';

  const badge = $('#err-badge');
  if (badge) {
    badge.hidden = !t.errors_24h;
    badge.textContent = t.errors_24h > 99 ? '99+' : t.errors_24h;
  }

  await loadServers();
  const top = state.guilds.slice(0, 6);
  const peak = Math.max(1, ...top.map((g) => g.members));
  $('#top-servers').innerHTML = top.length ? top.map((g) => `
    <button class="strip-row" data-guild="${esc(g.id)}">
      ${iconHtml(g.icon, g.name)}
      <span class="strip-name">${esc(g.name)}</span>
      <span class="strip-bar"><span class="strip-fill" style="width:${(g.members / peak) * 100}%"></span></span>
      <span class="strip-val">${fmt(g.members)}</span>
    </button>`).join('') : emptyState('Not in any servers yet', 'server');
}

async function loadServers() {
  state.guilds = (await API.get('/api/guilds')).guilds;
  renderServers();
}

function renderServers() {
  const q = ($('#server-search')?.value || '').toLowerCase();
  const list = state.guilds.filter((g) => g.name.toLowerCase().includes(q));
  $('#server-grid').innerHTML = list.length ? list.map((g) => `
    <button class="server-card glass" data-guild="${esc(g.id)}">
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
          : '<span class="chip is-empty">No features enabled</span>'}
      </div>
    </button>`).join('')
    : emptyState(state.guilds.length ? 'No servers match that search' : 'Not in any servers yet', 'search');
}

async function loadGuild(id) {
  state.guildId = id;
  const data = await API.get(`/api/guild/${id}?days=${state.days}`);
  const g = data.guild, t = data.totals;

  $('#guild-head').innerHTML = `${iconHtml(g.icon, g.name)}
    <div><h2>${esc(g.name)}</h2>
      <p class="muted">${fmt(g.members)} members · ${fmt(g.channels)} channels · ${fmt(g.roles)} roles · ${fmt(g.boosts)} boosts</p>
    </div>`;

  $('#guild-stats').innerHTML = [
    statTile({ ico: 'ban', label: 'Bans', value: fmt(t.bans) }),
    statTile({ ico: 'boot', label: 'Kicks', value: fmt(t.kicks) }),
    statTile({ ico: 'clock', label: 'Timeouts', value: fmt(t.timeouts) }),
    statTile({ ico: 'flag', label: 'Warnings', value: fmt(t.warns) }),
    statTile({ ico: 'shield', label: 'Automod', value: fmt(t.automod_triggers) }),
    statTile({ ico: 'brain', label: 'AI moderation', value: fmt(t.ai_moderation_hits) }),
    statTile({ ico: 'mic', label: 'Voice flags', value: fmt(t.voice_flags) }),
    statTile({ ico: 'alert', label: 'Errors (24h)', value: fmt(t.errors_24h), alert: t.errors_24h > 0 }),
  ].join('');

  const actions = [
    { label: 'Bans', color: RAMP[0], points: data.charts.bans },
    { label: 'Kicks', color: RAMP[1], points: data.charts.kicks },
    { label: 'Timeouts', color: RAMP[2], points: data.charts.timeouts },
    { label: 'Warnings', color: RAMP[3], points: data.charts.warns },
  ];
  barChart($('#guild-chart-actions'), actions);
  legend($('#guild-legend'), actions);
  barChart($('#guild-chart-joins'), [{ label: 'Joins', color: RAMP[0], points: data.charts.joins }]);

  $('#guild-features').innerHTML = data.features.all.map((f) => `
    <div class="feature-row${f.enabled ? ' is-on' : ''}">
      <span class="feature-state">${icon(f.enabled ? 'check' : 'dot', 14)}</span>
      <div>
        <div class="feature-name">${esc(f.name)}</div>
        <div class="feature-desc">${esc(f.description)}</div>
        ${f.enabled && f.detail ? `<div class="feature-detail">${esc(f.detail)}</div>` : ''}
      </div>
    </div>`).join('');

  $('#guild-cases').innerHTML = data.recent_cases.length ? `
    <table><thead><tr><th>Case</th><th>Action</th><th>User</th><th>Reason</th><th>When</th></tr></thead>
    <tbody>${data.recent_cases.map((c) => `<tr>
      <td>#${c.case}</td>
      <td><span class="pill pill-${esc(c.action)}">${esc(c.action)}</span></td>
      <td>${esc(c.user_name || c.user_id)}</td>
      <td>${esc((c.reason || '—').slice(0, 56))}</td>
      <td>${esc(ago(c.at))}</td></tr>`).join('')}</tbody></table>`
    : emptyState('No moderation cases yet', 'file');

  $('#guild-leaderboard').innerHTML = data.leaderboard.length ? `
    <table><thead><tr><th>#</th><th>Member</th><th>XP</th><th>Messages</th><th>Voice</th></tr></thead>
    <tbody>${data.leaderboard.map((m, i) => `<tr>
      <td>${i + 1}</td>
      <td>${esc(m.name || m.user_id)}</td>
      <td>${fmt(m.xp)}</td>
      <td>${fmt(m.messages)}</td>
      <td>${duration(m.voice_seconds)}</td></tr>`).join('')}</tbody></table>`
    : emptyState('No experience recorded yet', 'star');

  showView('guild', g.name, 'Server detail');
}

async function loadErrors() {
  const data = await API.get('/api/errors?limit=150');
  const c = data.counts;
  $('#error-stats').innerHTML = [
    statTile({ ico: 'clock', label: 'Last hour', value: fmt(c.last_hour), alert: c.last_hour > 0 }),
    statTile({ ico: 'clock', label: 'Last 24 hours', value: fmt(c.last_24h), alert: c.last_24h > 0 }),
    statTile({ ico: 'clock', label: 'Last 7 days', value: fmt(c.last_7d) }),
    statTile({ ico: 'file', label: 'All time', value: fmt(c.total) }),
  ].join('');

  $('#error-list').innerHTML = data.errors.length ? data.errors.map((e) => `
    <div class="error-row">
      <div class="error-source">${esc(e.source)}</div>
      <div>
        <div class="error-msg">${esc(e.message)}</div>
        ${e.guild_name && e.guild_name !== '—' ? `<div class="error-meta">${esc(e.guild_name)}</div>` : ''}
      </div>
      <div class="error-meta">${esc(ago(e.at))}</div>
      ${e.detail ? `<pre class="error-detail">${esc(e.detail)}</pre>` : ''}
    </div>`).join('') : emptyState('No errors recorded', 'check');
}

/* ---- routing ----------------------------------------------------------- */

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
  $('#gate').hidden = false;
}

async function boot() {
  $('#gate').hidden = true;
  $('#app').hidden = false;
  await refresh();
}

/* ---- chrome ------------------------------------------------------------ */

function paintChrome() {
  $('#nav').innerHTML = NAV.map((n) => `
    <button class="nav-item${n.view === state.view ? ' is-active' : ''}" data-view="${n.view}">
      ${icon(n.icon, 17)}<span>${n.label}</span>
      ${n.badge ? `<span class="badge" id="${n.badge}" hidden>0</span>` : ''}
    </button>`).join('');

  $('#refresh').innerHTML = icon('refresh', 15);
  $('#signout').innerHTML = `${icon('logout', 14)}<span>Sign out</span>`;
  $('#back-to-servers').innerHTML = `${icon('back', 14)}<span>All servers</span>`;
  $('#search-wrap').insertAdjacentHTML('afterbegin', icon('search', 15));

  $$('.panel-head[data-icon]').forEach((h) =>
    h.insertAdjacentHTML('afterbegin', icon(h.dataset.icon, 16)));

  $$('.nav-item').forEach((btn) => btn.addEventListener('click', () => {
    showView(btn.dataset.view);
    refresh();
  }));
}

/* ---- events ------------------------------------------------------------ */

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
    if (!res.ok) throw new Error('rejected');
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

document.addEventListener('click', (event) => {
  const target = event.target.closest('[data-guild]');
  if (target) loadGuild(target.dataset.guild);
});

setInterval(() => { if (!document.hidden && API.token) refresh(); }, 30000);

paintChrome();
if (API.token) boot();
