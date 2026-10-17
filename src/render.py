"""Render a normalized snapshot into a self-contained results/aihub-<tag>/ folder.

The page loads everything through relative <script>/<link> tags (no fetch()), so it
works when opened straight from disk (file://) with no network access.
"""

import glob
import json
import os
import shutil

import jinja2

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(SRC_DIR, "templates")
STATIC_DIR = os.path.join(SRC_DIR, "static")

PERF_COLUMNS = ["model", "device", "precision", "component", "runtime", "latency_ms",
                "memory_min_mb", "memory_max_mb", "npu", "gpu", "cpu", "total",
                "compute_unit", "qairt", "onnx_runtime", "job_id"]
LLM_COLUMNS = ["model", "device", "precision", "component", "runtime", "context_length",
               "tokens_per_s", "prefill_tokens_per_s", "ttft_min_ms", "ttft_max_ms", "compute_unit"]


def _env():
    return jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
                              autoescape=jinja2.select_autoescape(["html", "j2"]),
                              keep_trailing_newline=True)


def page_data(snapshot):
    """Compact form of the snapshot for the page: lists plus row arrays indexed by position."""
    model_ids = sorted(snapshot["models"])
    device_names = sorted(snapshot["devices"])
    mi = {m: i for i, m in enumerate(model_ids)}
    di = {d: i for i, d in enumerate(device_names)}

    models = []
    for model_id in model_ids:
        m = dict(snapshot["models"][model_id])
        m.pop("technical_details", None)
        models.append(m)
    devices = []
    for name in device_names:
        d = dict(snapshot["devices"][name])
        d["name"] = name
        devices.append(d)

    perf = []
    for r in snapshot["perf"]:
        layers = r["layers"]
        perf.append([mi[r["model"]], di[r["device"]], r["precision"], r["component"], r["runtime"],
                     r["latency_ms"], r["memory_mb"][0], r["memory_mb"][1],
                     layers["npu"], layers["gpu"], layers["cpu"], layers["total"],
                     r["compute_unit"], r["qairt"], r["onnx_runtime"], r["job_id"]])
    llm = []
    for r in snapshot["llm_perf"]:
        llm.append([mi[r["model"]], di[r["device"]], r["precision"], r["component"], r["runtime"],
                    r["context_length"], r["tokens_per_s"], r["prefill_tokens_per_s"],
                    r["ttft_ms"][0], r["ttft_ms"][1], r["compute_unit"]])

    release = dict(snapshot["release"])
    release.pop("skipped_models", None)
    return {
        "schema": snapshot["schema"],
        "release": release,
        "models": models,
        "devices": devices,
        "perf": {"columns": PERF_COLUMNS, "rows": perf},
        "llm_perf": {"columns": LLM_COLUMNS, "rows": llm},
    }


def render_snapshot(snapshot, out_dir):
    """Write index.html and assets/ into out_dir (data/snapshot.json is written by the caller)."""
    assets = os.path.join(out_dir, "assets")
    if os.path.isdir(assets):
        shutil.rmtree(assets)
    shutil.copytree(STATIC_DIR, assets)

    data = page_data(snapshot)
    with open(os.path.join(assets, "data.js"), "w") as f:
        f.write("window.AIHUB_DATA = ")
        json.dump(data, f, separators=(",", ":"), sort_keys=True)
        f.write(";\n")

    html = _env().get_template("index.html.j2").render(
        release=snapshot["release"],
        n_models=len(snapshot["models"]),
        n_devices=len(snapshot["devices"]),
    )
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)


def render_landing(results_dir):
    """Write results/index.html listing every snapshot folder found under results_dir."""
    snapshots = []
    for path in glob.glob(os.path.join(results_dir, "aihub-*", "data", "snapshot.json")):
        with open(path) as f:
            snap = json.load(f)
        folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        snapshots.append({
            "folder": folder,
            "release": snap["release"],
            "n_models": len(snap["models"]),
            "n_devices": len(snap["devices"]),
            "n_perf": len(snap["perf"]) + len(snap["llm_perf"]),
            "qairt": sorted(set(".".join(v.split(".")[:2]) for v in snap["release"]["tool_versions"].get("qairt", []))),
        })
    snapshots.sort(key=lambda s: s["release"]["commit_date"], reverse=True)
    html = _env().get_template("landing.html.j2").render(snapshots=snapshots)
    with open(os.path.join(results_dir, "index.html"), "w") as f:
        f.write(html)
    return snapshots
