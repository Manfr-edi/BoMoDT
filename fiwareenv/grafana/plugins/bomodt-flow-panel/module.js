define(['react', '@grafana/data'], function (reactModule, dataModule) {
  'use strict';
  var React = reactModule.default || reactModule;
  var PanelPlugin = dataModule.PanelPlugin || dataModule.default.PanelPlugin;
  function valueAt(field, index) {
    return typeof field.values.get === 'function' ? field.values.get(index) : field.values[index];
  }
  function FlowPanel(props) {
    var rows = [];
    (props.data.series || []).forEach(function (frame) {
      var time = frame.fields.find(function (f) { return f.type === 'time'; });
      var flow = frame.fields.find(function (f) { return f.name === 'trafficflow'; });
      var marker = frame.fields.find(function (f) { return f.name === 'timing_id'; });
      var entity = frame.fields.find(function (f) { return f.name === 'entity_id'; });
      if (!time || !flow) return;
      for (var i = 0; i < frame.length; i++) {
        var x = valueAt(time, i), y = valueAt(flow, i);
        if (x != null && y != null && Number.isFinite(Number(y))) {
          rows.push({time: Number(x), flow: Number(y), id: marker ? valueAt(marker, i) : null,
            entity: entity ? valueAt(entity, i) : 'Traffic flow'});
        }
      }
    });
    rows.sort(function (a, b) { return a.time - b.time; });
    var acknowledged = React.useRef(new Set());
    var pending = React.useRef(new Set());
    var chart = React.useRef(null);
    var signature = rows.map(function (r) { return r.id || ''; }).join(',');
    React.useEffect(function () {
      var cancelled = false, firstFrame, secondFrame;
      var intersecting = false;
      function acknowledge() {
        if (cancelled || document.visibilityState !== 'visible' || props.data.state !== 'Done') return;
        if (!chart.current) return;
        if (!intersecting) return;
        var bounds = chart.current.getBoundingClientRect();
        if (bounds.bottom <= 0 || bounds.top >= window.innerHeight || bounds.right <= 0 || bounds.left >= window.innerWidth) return;
        var ids = Array.from(new Set(rows.map(function (r) { return r.id; }).filter(function (id) {
          return id && !acknowledged.current.has(id) && !pending.current.has(id);
        })));
        if (!ids.length) return;
        var renderedAt = performance.timeOrigin + performance.now();
        for (var offset = 0; offset < ids.length; offset += 500) {
          let batch = ids.slice(offset, offset + 500);
          batch.forEach(function (id) { pending.current.add(id); });
          // Browser and emulator share the local host clock. No cookies or credentials.
          var endpoint = 'http://' + window.location.hostname + ':8000/udtApp/telemetry/rendered';
          fetch(endpoint, {method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({measurement_ids: batch, rendered_at_ms: renderedAt})})
            .then(function (response) { if (!response.ok) throw new Error('Timing acknowledgement failed'); return response.json(); })
            .then(function (payload) { (payload.accepted || []).forEach(function (id) { acknowledged.current.add(id); }); })
            .catch(function (error) { console.warn('BoMoDT telemetry:', error.message); })
            .finally(function () { batch.forEach(function (id) { pending.current.delete(id); }); });
        }
      }
      function schedule() {
        firstFrame = requestAnimationFrame(function () { secondFrame = requestAnimationFrame(acknowledge); });
      }
      schedule();
      var observer = new IntersectionObserver(function (entries) {
        intersecting = entries.some(function (entry) { return entry.isIntersecting; });
        if (intersecting) schedule();
      });
      if (chart.current) observer.observe(chart.current);
      document.addEventListener('visibilitychange', schedule);
      window.addEventListener('scroll', schedule, true);
      return function () {
        cancelled = true;
        cancelAnimationFrame(firstFrame); cancelAnimationFrame(secondFrame);
        document.removeEventListener('visibilitychange', schedule);
        observer.disconnect();
        window.removeEventListener('scroll', schedule, true);
      };
    }, [signature, props.data.state]);
    if (!rows.length) return React.createElement('div', {style: {padding: 16}}, 'No measurements in this interval');
    var width = Math.max(props.width, 150), height = Math.max(props.height, 100);
    var left = 55, top = 15, bottom = height - 35, right = width - 15;
    var minTime = rows[0].time, maxTime = rows[rows.length - 1].time;
    var maxFlow = Math.max.apply(null, rows.map(function (r) { return r.flow; }).concat([1]));
    function x(row) { return left + (row.time - minTime) / (maxTime - minTime || 1) * (right - left); }
    function y(row) { return bottom - row.flow / maxFlow * (bottom - top); }
    var groups = {};
    rows.forEach(function (row) { (groups[row.entity] || (groups[row.entity] = [])).push(row); });
    var children = [
      React.createElement('path', {key: 'axes', d: 'M' + left + ' ' + top + 'V' + bottom + 'H' + right, fill: 'none', stroke: '#888'}),
      React.createElement('text', {key: 'max', x: left - 8, y: top + 8, fill: 'currentColor', textAnchor: 'end', fontSize: 11}, String(maxFlow)),
      React.createElement('text', {key: 'zero', x: left - 8, y: bottom, fill: 'currentColor', textAnchor: 'end', fontSize: 11}, '0'),
      React.createElement('text', {key: 'start', x: left, y: height - 10, fill: 'currentColor', fontSize: 10}, new Date(minTime).toLocaleString()),
      React.createElement('text', {key: 'end', x: right, y: height - 10, textAnchor: 'end', fill: 'currentColor', fontSize: 10}, new Date(maxTime).toLocaleString()),
      React.createElement('text', {key: 'sensors', x: right, y: top + 8, textAnchor: 'end', fill: 'currentColor', fontSize: 11}, Object.keys(groups).length + ' sensor series')
    ];
    var colors = ['#5794f2', '#73bf69', '#ff9830', '#b877d9', '#f2495c', '#56c2d6'];
    Object.keys(groups).forEach(function (name, index) {
      var points = groups[name].map(function (row) { return x(row) + ',' + y(row); }).join(' ');
      children.push(React.createElement('polyline', {key: 'line-' + name, points: points, fill: 'none', stroke: colors[index % colors.length], strokeWidth: 2}, React.createElement('title', null, name)));
    });
    rows.forEach(function (row, index) {
      children.push(React.createElement('circle', {key: index, cx: x(row), cy: y(row), r: 3, fill: '#5794f2'},
        React.createElement('title', null, row.entity + ' · ' + new Date(row.time).toLocaleString() + ': ' + row.flow)));
    });
    return React.createElement('svg', {ref: chart, width: width, height: height, role: 'img', 'aria-label': 'Measured traffic flow'}, children);
  }
  return {plugin: new PanelPlugin(FlowPanel)};
});
