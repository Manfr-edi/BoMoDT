# Measured traffic flow panel

This local Grafana 8.5 panel draws the traffic-flow series and acknowledges each
correlated measurement after a committed SVG render and two animation frames.
IntersectionObserver and page visibility prevent background/offscreen panels
from creating latency samples. The earliest rendering timestamp wins; dashboard refreshes
do not replace earlier samples. This measures a browser frame, not physical display
hardware scanout. The browser and the emulator must share a synchronized clock.

The plugin is loaded from the Compose bind mount. Only this plugin ID is allowed
as an unsigned local development plugin; no build or npm install is needed.
The acknowledgement endpoint is the Django development backend on port 8000,
using the same hostname as Grafana. A different deployment needs that URL and
the endpoint's allowed origins configured accordingly.

## Collecting measurements

1. Restart Django and reload the dashboard after installing this change.
2. Keep Monitor or an entity's dashboard visible. The dashboard refreshes every
   five seconds; choose a Device or TrafficFlowObserved entity that is being sent
   and an interval covering its historical observation date.
3. Restart `main.py`, option 2, so existing IoT device registrations gain `timingId`.
   Each dispatch has a UUID carried through IoT Agent / Orion / QuantumLeap.
4. Run a calibration/simulation for the same observation date and hours, then open
   Results. The times are stored in that run's `execution_times.json`.

Baseline is the mean of individually correlated send-to-first-frame durations in
the most recent emulator session available when the SUMO run finished, restricted
to the selected observation date/hours. The Results card refreshes acknowledgements
for that same session every five seconds, including renders occurring after SUMO
finishes. Missing/unrendered samples are shown as coverage, never converted to zero.
Closing a dashboard, leaving it outside the viewport or selecting a different
entity means those measurements have no baseline sample until they are displayed.

Baseline markers are stored in `sumoenv/telemetry/baseline.sqlite3`. Historical
observation timestamps are only for filtering and are never used as send times.
The `timing_id` Grafana field is read dynamically from the Timescale row, so old
measurements without this field still plot and produce no invented latency.

Macromodeling is the sum of `getMacroscopicModel()` durations for the selected
hours. CSV/network loading is excluded. Simulation includes vType/scenario file
preparation, route generation, SUMO startup and its entire execution. Artificial
10-second waits and postprocessing are excluded. Baseline is a per-measurement
mean, while the two compute durations are run totals; the bars are not additive.

Old result folders without `execution_times.json` display timings as unavailable.
