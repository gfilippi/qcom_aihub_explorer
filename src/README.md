# qcom_aihub_explorer — source

Generates offline, self-contained HTML snapshots that map Qualcomm AI Hub models
across platforms, tasks, runtimes and precisions (efficiency map: on-device FPS
vs. quantized on-device accuracy), one snapshot per AI Hub release.

> Status: `normalize` is implemented; page rendering (`build`) is next.

## Architecture

```
tools/fetch_aihub_repo.sh <tag>          upstream checkout (qualcomm/ai-hub-models @ tag)
        │
        ▼
src/fetch.py                  locate/validate the checkout for a release tag
src/normalize.py              manifest.yaml + perf.yaml + numerics.yaml
                              + devices_and_chipsets.yaml  ->  snapshot dict
src/render.py                 snapshot -> results/aihub-<tag>/ (index.html + assets)  (planned)
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
├── assets/           pinned JS/CSS + data.js (embedded data)
├── img/              model thumbnails
└── data/snapshot.json normalized data, for diffs across releases
```

## Usage

```bash
pip install -r requirements.txt                 # Python 3.8+, PyYAML, Jinja2
tools/fetch_aihub_repo.sh v0.64.0              # optional: normalize clones a missing tag itself
python3 src/qcom_aihub_explorer.py normalize --tag v0.64.0
python3 src/qcom_aihub_explorer.py build --tag v0.64.0   # (planned)
```

## Conventions

- Only `status: published` models are included.
- Default accuracy axis: quantized on-device accuracy; float reference available as toggle.
- LLM/VLM models get a dedicated view (tokens/s vs. accuracy, TTFT in details).
