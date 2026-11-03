(function () {
  'use strict';
  const root = document.getElementById('emulation-app');
  if (!root) return;
  const byId = id => document.getElementById(id);
  const form = byId('emulation-form');
  let current = null, selectedId = new URLSearchParams(window.location.search).get('session_id') || '', polling = false, dashboardUrl = '';
  if (selectedId) byId('emulation-history').value = selectedId;
  const active = new Set(['starting', 'running', 'pausing', 'paused', 'stopping', 'draining']);
  const labels = {starting: 'Preparazione', running: 'Invio in corso', pausing: 'Pausa richiesta', paused: 'In pausa', stopping: 'Arresto richiesto', draining: 'Calcoli in coda', completed: 'Completata', completed_with_errors: 'Completata con errori nei calcoli', stopped: 'Fermata', failed: 'Errore', interrupted: 'Interrotta'};
  const seconds = value => {
    if (value == null) return '—';
    const scale = value > 0 && value < 0.001 ? 1000000 : value > 0 && value < 1 ? 1000 : 1;
    return Number(value * scale).toLocaleString(undefined, {minimumFractionDigits: 3, maximumFractionDigits: 3}) + (scale === 1000000 ? ' µs' : scale === 1000 ? ' ms' : ' s');
  };
  const modes = {
    baseline: 'Invia i dati a FIWARE e Grafana. Non calcola macromodelli e non avvia SUMO.',
    macromodeling: 'Invia i dati e applica il modello macroscopico ai flussi trasmessi, stimando densità e velocità. Non avvia SUMO.',
    simulation: 'Invia i dati, calcola il macromodello e avvia una simulazione SUMO per ogni fascia oraria completata, senza aprire la GUI.'
  };
  function showError(message) { const element = byId('emulation-errors'); element.textContent = message || ''; element.hidden = !message; }
  async function request(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || Object.entries(payload.errors || {}).map(([name, errors]) => name + ': ' + errors.map(error => error.message).join(', ')).join('\n') || 'Richiesta fallita');
    return payload;
  }
  function cells(target, rows) {
    const body = byId(target);
    body.replaceChildren();
    rows.forEach(values => { const row = document.createElement('tr'); values.forEach(value => { const cell = document.createElement('td'); cell.textContent = value; row.appendChild(cell); }); body.appendChild(row); });
  }
  function display(payload) {
    current = payload.session;
    if (!current) return;
    const progress = current.progress, metrics = payload.timings;
    const busy = active.has(current.status);
    byId('emulation-mode').textContent = current.config.pipeline === 'simulation' && current.config.sumo_gui
      ? 'Invii e simulazioni in parallelo. SUMO apre la GUI sul PC del backend.' : modes[current.config.pipeline];
    byId('emulation-state').textContent = labels[current.status] || current.status;
    byId('emulation-state').className = 'badge fs-6 ' + (current.status === 'failed' ? 'bg-danger' : busy ? 'bg-primary' : 'bg-secondary');
    byId('emulation-pause').disabled = !busy || ['stopping', 'pausing'].includes(current.status);
    byId('emulation-pause').textContent = current.command === 'pause' ? 'Riprendi' : 'Pausa';
    byId('emulation-stop').disabled = !busy || current.status === 'stopping';
    if (byId('emulation-fields')) byId('emulation-fields').disabled = busy;
    if (byId('emulation-start')) byId('emulation-start').disabled = busy;
    const completed = Number(progress.attempted || 0), total = Number(progress.total || 0);
    const percent = total ? 100 * completed / total : 0;
    byId('emulation-progress').firstElementChild.style.width = percent.toFixed(1) + '%';
    byId('emulation-progress').setAttribute('aria-valuenow', percent.toFixed(1));
    byId('emulation-position').textContent = progress.day ? `${progress.day} · ${String(progress.hour).padStart(2, '0')}:00 · ${progress.sensor || ''}` : progress.stage;
    let detail = `${completed} / ${total} invii · ${progress.succeeded || 0} riusciti · ${progress.failed || 0} falliti`;
    if (progress.active_seconds != null) detail += ` · tempo attivo ${seconds(progress.active_seconds)}`;
    if (progress.schedule_lag_seconds > 1) detail += ` · ritardo sul ritmo obiettivo ${seconds(progress.schedule_lag_seconds)}`;
    if (current.config.missing_days.length) detail += ` · ${current.config.missing_days.length} giorni senza dati saltati`;
    byId('emulation-counts').textContent = detail;
    byId('baseline-mean').textContent = seconds(metrics.baseline.mean_seconds);
    byId('baseline-coverage').textContent = `${metrics.baseline.count} visualizzati / ${metrics.baseline.sent_count} trasmessi`;
    byId('macro-mean').textContent = seconds(metrics.macromodeling.mean_compute_seconds_per_sample);
    byId('macro-coverage').textContent = `Solo calcolo / campione (ammortizzato), ${metrics.macromodeling.samples} campioni. Invio → modello disponibile: ${seconds(metrics.macromodeling.mean_end_to_end_seconds)} (include raccolta e coda).`;
    byId('simulation-mean').textContent = seconds(metrics.simulation.mean_end_to_end_seconds);
    byId('simulation-coverage').textContent = `Invio → SUMO completato · ${metrics.simulation.samples} campioni; calcolo medio ${seconds(metrics.simulation.mean_compute_seconds_per_sample)} / campione (ammortizzato)`;
    cells('emulation-sensors', Object.entries(metrics.sensors).map(([name, values]) => [name.replace('urn:ngsi-ld:Device:', ''), values.sent, values.rendered, seconds(values.mean_seconds)]));
    cells('emulation-jobs', metrics.jobs.map(job => [job.day + ' ' + String(job.hour).padStart(2, '0') + ':00', job.error ? job.status + ': ' + job.error : job.status, job.sample_count, seconds(job.macro_seconds), seconds(job.simulation_seconds)]));
    byId('emulation-export').hidden = false;
    byId('emulation-export').href = root.dataset.exportUrl.replace('SESSION', current.id);
    byId('emulation-dashboard-container').hidden = false;
    byId('emulation-dashboard-link').hidden = false;
    byId('emulation-dashboard-link').href = payload.dashboard_url;
    if (dashboardUrl !== payload.dashboard_url) {
      dashboardUrl = payload.dashboard_url;
      byId('emulation-dashboard').src = dashboardUrl;
    }
    if (current.error) showError(current.error);
  }
  async function poll() {
    if (polling || document.visibilityState !== 'visible') return;
    polling = true;
    try { display(await request(root.dataset.statusUrl + (selectedId ? '?session_id=' + encodeURIComponent(selectedId) : ''), {cache: 'no-store'})); }
    catch (error) { showError(error.message); }
    finally { polling = false; }
  }
  if (form) {
    form.addEventListener('submit', async event => {
      event.preventDefault(); showError('');
      const data = new FormData(form);
      byId('emulation-start').disabled = true;
      try {
        const payload = await request(root.dataset.startUrl, {method: 'POST', body: data});
        selectedId = payload.session.id;
        const option = document.createElement('option'); option.value = selectedId; option.textContent = 'Sessione appena avviata · ' + selectedId.slice(0, 8);
        byId('emulation-history').prepend(option); byId('emulation-history').value = selectedId;
        display(payload);
      } catch (error) { showError(error.message); }
      finally { byId('emulation-start').disabled = !!(current && active.has(current.status)); }
    });
    function pacingFields() {
      const mode = form.elements.pacing.value;
      root.querySelector('[data-field="day_seconds"]').hidden = mode !== 'day';
      root.querySelector('[data-field="interval_seconds"]').hidden = mode !== 'interval';
    }
    form.elements.pacing.addEventListener('change', pacingFields); pacingFields();
    function explainMode() {
      byId('emulation-mode-description').textContent = modes[form.elements.pipeline.value];
      root.querySelector('[data-field="sumo_gui"]').hidden = form.elements.pipeline.value !== 'simulation';
      if (form.elements.pipeline.value === 'simulation' && form.elements.sumo_gui.checked) {
        byId('emulation-mode-description').textContent = 'Invia i dati, calcola il macromodello e avvia SUMO con GUI per ogni fascia oraria. Gli invii continuano mentre SUMO lavora.';
      }
      byId('emulation-start').textContent = form.elements.pipeline.value === 'simulation' ? 'Avvia emulazione + SUMO' : form.elements.pipeline.value === 'macromodeling' ? 'Avvia emulazione + macromodello' : 'Avvia solo emulazione';
    }
    form.elements.pipeline.addEventListener('change', explainMode); explainMode();
    form.elements.sumo_gui.addEventListener('change', explainMode);
  }
  async function action(name) {
    if (!current) return;
    showError('');
    const data = new FormData(); data.set('action', name);
    data.set('csrfmiddlewaretoken', form.elements.csrfmiddlewaretoken.value);
    try { display(await request(root.dataset.actionUrl.replace('SESSION', current.id), {method: 'POST', body: data})); }
    catch (error) { showError(error.message); }
  }
  byId('emulation-pause').addEventListener('click', () => action(current.command === 'pause' ? 'resume' : 'pause'));
  byId('emulation-stop').addEventListener('click', () => action('stop'));
  byId('emulation-history').addEventListener('change', event => { selectedId = event.target.value; showError(''); poll(); });
  const timer = setInterval(poll, 2000);
  window.addEventListener('pagehide', () => clearInterval(timer), {once: true});
  poll();
})();
