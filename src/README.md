# qcom_aihub_explorer — source

Generates offline, self-contained HTML snapshots that map Qualcomm AI Hub models
across platforms, tasks, runtimes and precisions (efficiency map: on-device FPS
vs. quantized on-device accuracy), one snapshot per AI Hub release.

> Status: `normalize` and `build` are implemented and tested on v0.64.0.

## Architecture

```
tools/fetch_aihub_repo.sh <tag>          upstream checkout (qualcomm/ai-hub-models @ tag)
        │
        ▼
src/fetch.py                  locate/validate the checkout for a release tag
src/normalize.py              manifest.yaml + perf.yaml + numerics.yaml
                              + devices_and_chipsets.yaml  ->  snapshot dict
src/render.py                 snapshot -> results/aihub-<tag>/ (index.html + assets) + results/index.html
src/qcom_aihub_explorer.py    command line entry point
```

`src/` holds all the code of the tool (`templates/` and `static/` hold the page
template and the assets it inlines). `tools/` holds self-contained helper scripts
and the upstream checkouts that the tool consumes.

### Data sources (per model, `src/qai_hub_models/models/<model_id>/`)

| File | Used for |
|---|---|
| `manifest.yaml` | name, domain, use case, tags, license, parameters, resolution, links, status |
| `perf.yaml` | latency, memory, compute-unit layer split, QAIRT/ORT versions, job id; LLM tokens/s, TTFT |
| `numerics.yaml` | accuracy metric, float reference, on-device accuracy per precision × runtime |

Global: `devices_and_chipsets.yaml` (chipset → marketing name, platform world),
`similar_devices.yaml` (devices whose numbers are borrowed from another chipset).

### Snapshot model (`data/snapshot.json`)

`normalize.build_snapshot()` returns one dict per release, written with sorted keys
so two releases can be diffed directly.

| Key | Content |
|---|---|
| `release` | tag, commit, commit date, release URL, generation time, QAIRT / ONNX Runtime versions seen, skipped (non-published) models |
| `devices` | device name → chipset, marketing name, HTP version, world (Mobile, Compute, Automotive, IoT, XR), OS, `borrowed_from` (set when the device reuses another chipset's numbers) |
| `models` | model id → name, headline, description, domain, use case, tags, license, parsed `params` and `size_mb` per precision, raw technical details, links (AI Hub, GitHub at the tag), `accuracy` |
| `perf` | one row per model × precision × component × device × runtime: latency (ms), peak memory range (MB), layer split (npu/gpu/cpu/total), primary compute unit, QAIRT / ORT version, job id |
| `llm_perf` | same key plus context length: tokens/s, prefill tokens/s, time-to-first-token range |

`accuracy` holds the metric (name, unit, description, `higher_is_better`, range),
the published benchmark, the PyTorch float reference (`torch`), and the on-device
values per device × precision × runtime. Upstream measures accuracy on a single
device per model (Galaxy S25, or X Elite CRD for compute models), so it is not
per-platform.

Multi-component models (e.g. encoder/decoder) keep one perf row per component;
aggregation is a rendering decision.

## Output

```
results/aihub-<tag>/
├── index.html        entry page (opens from file://, no network needed)
├── assets/
│   ├── data.js       compact page data (window.AIHUB_DATA), loaded by <script>, not fetch()
│   ├── explorer.js   page logic (copied from src/static/)
│   ├── explorer.css
│   └── vendor/       Chart.js 4.4.1, chartjs-plugin-zoom 2.0.1, Hammer.js 2.0.8 (MIT, pinned copies)
└── data/snapshot.json normalized data, for diffs across releases (committed)
results/index.html    landing page listing every snapshot
```

The page needs no network access: open `index.html` directly from disk.

## The page

- **Models / LLMs tabs.** Models: X = throughput (FPS) or latency, log scale; Y = accuracy
  metric of the selected task. LLMs: X = tokens/s; Y = Grace2 (or another metric).
- **Filters:** platform (Mobile, Compute, Automotive, IoT, XR) → device, task, metric
  (tasks can mix metrics; the most common is the default; "Parameters" plots every model
  even without accuracy), runtime, precision, context length (LLMs), model checklist.
- **One point per model.** Among the results for the selected device/runtime/precision, the
  page prefers ones whose accuracy was measured on the same precision and runtime, then any
  accuracy for that precision, then picks the fastest. "Float and quantized" draws both points
  linked by a dashed line. Lower-is-better metrics invert the Y axis.
- **Bubbles:** size = parameter count (log scale), colour = primary compute unit (NPU/GPU/CPU).
- **Popup** on hover, pinned on click, with AI Hub, GitHub, paper and source links. The URL
  hash stores all filters and the pinned model (`…#…&pin=yolov8_det`), so views can be shared.
- **Table** of all models in the selection on the device, sortable by any column.
- Devices that reuse another chipset's numbers are marked `*` with a note.

## Usage

```bash
pip install -r requirements.txt                 # Python 3.8+, PyYAML, Jinja2
tools/fetch_aihub_repo.sh v0.64.0              # optional: normalize clones a missing tag itself
python3 src/qcom_aihub_explorer.py build --tag v0.64.0      # normalize + render + landing page
python3 src/qcom_aihub_explorer.py normalize --tag v0.64.0  # data/snapshot.json only
python3 src/qcom_aihub_explorer.py landing                  # rebuild results/index.html only
```

## Conventions

- Only `status: published` models are included.
- Default accuracy axis: quantized on-device accuracy; float available via the Precision filter.
- LLM/VLM models get a dedicated view (tokens/s vs. accuracy, TTFT in details).
