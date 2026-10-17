"""Turn an upstream checkout into a normalized snapshot dict.

Snapshot layout (see src/README.md for field details):
    release   tag, commit, dates, tool versions seen
    devices   device name -> chipset, world, os, borrowed numbers
    models    model id -> metadata, links, accuracy
    perf      one row per model x precision x component x device x runtime
    llm_perf  same key, for LLM token-generation metrics
"""

import datetime
import glob
import os
import re

import yaml

SCHEMA_VERSION = 1
AIHUB_MODEL_URL = "https://aihub.qualcomm.com/models/{id}"
GITHUB_MODEL_URL = "https://github.com/qualcomm/ai-hub-models/tree/{tag}/src/qai_hub_models/models/{id}"
GITHUB_RELEASE_URL = "https://github.com/qualcomm/ai-hub-models/releases/tag/{tag}"

_SCALE = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9}
_SIZE_MB = {"B": 1.0 / 2 ** 20, "KB": 1.0 / 1024, "MB": 1.0, "GB": 1024.0}


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_count(text):
    """'3.49M' -> 3490000.0, '~600M' -> 6e8, '4B (effective)' -> 4e9; None if unparsable."""
    m = re.match(r"^\s*~?\s*([0-9.]+)\s*([KMBG]?)\b", str(text))
    if not m:
        return None
    return float(m.group(1)) * _SCALE[m.group(2)]


def parse_size_mb(text):
    """'13.3 MB' -> 13.3, '~2.4 GB' -> 2457.6; None if unparsable."""
    m = re.match(r"^\s*~?\s*([0-9.]+)\s*(B|KB|MB|GB)\b", str(text))
    if not m:
        return None
    return float(m.group(1)) * _SIZE_MB[m.group(2)]


def _technical(details):
    """Extract numeric parameter count and per-precision model sizes from technical_details."""
    params = None
    sizes = {}
    for key, value in (details or {}).items():
        if key.startswith("Number of parameters"):
            n = parse_count(value)
            if n is not None:
                params = (params or 0.0) + n
        m = re.match(r"^Model size(?: \(([^)]*)\))*", key)
        if m and key.startswith("Model size"):
            mb = parse_size_mb(value)
            if mb is None:
                continue
            # "Model size (encoder) (float)" -> precision is the last parenthesised group.
            groups = re.findall(r"\(([^)]*)\)", key)
            precision = groups[-1] if groups else "float"
            sizes[precision] = sizes.get(precision, 0.0) + mb
    return params, sizes


def load_devices(package_dir):
    registry = _load(os.path.join(package_dir, "devices_and_chipsets.yaml")) or {}
    similar = (_load(os.path.join(package_dir, "similar_devices.yaml")) or {}).get("devices") or {}
    chipsets = registry.get("chipsets") or {}
    form_factors = registry.get("form_factors") or {}
    # Reference device per chipset, used to name the device a borrowed number comes from.
    reference_device = {cid: c.get("reference_device") for cid, c in chipsets.items()}

    devices = {}
    for name, dev in list((registry.get("devices") or {}).items()) + list(similar.items()):
        entry = devices.setdefault(name, {})
        for key, value in dev.items():
            entry.setdefault(key, value)

    out = {}
    for name, dev in devices.items():
        chipset_id = dev.get("chipset")
        chipset = chipsets.get(chipset_id) or {}
        form_factor = dev.get("form_factor")
        world = chipset.get("world") or (form_factors.get(form_factor) or {}).get("world")
        ref = dev.get("reference_chipset")
        borrowed_from = None
        if ref and ref != chipset_id:
            borrowed_from = {"chipset": ref, "device": reference_device.get(ref)}
        os_info = dev.get("os") or {}
        out[name] = {
            "chipset": chipset_id,
            "marketing_name": chipset.get("marketing_name"),
            "htp_version": chipset.get("htp_version"),
            "world": world,
            "form_factor": form_factor,
            "os": " ".join(str(v) for v in (os_info.get("ostype"), os_info.get("version")) if v) or None,
            "npu_count": dev.get("npu_count"),
            "borrowed_from": borrowed_from,
        }
    return out


def _accuracy(numerics, manifest):
    if not numerics or not numerics.get("metrics"):
        return None
    metric = numerics["metrics"][0]
    values = []
    devices = set()
    for device, by_precision in (metric.get("device_metric") or {}).items():
        devices.add(device)
        for precision, by_runtime in (by_precision or {}).items():
            for runtime, v in (by_runtime or {}).items():
                value = (v or {}).get("partial_metric")
                if value is not None:
                    values.append({"device": device, "precision": precision, "runtime": runtime, "value": value})
    rng = metric.get("metric_range") or {}
    return {
        "metric": metric.get("metric_name"),
        "unit": metric.get("metric_unit"),
        "description": metric.get("metric_description"),
        "higher_is_better": metric.get("higher_is_better", True),
        "range": [rng.get("min"), rng.get("max")],
        "dataset": metric.get("dataset_name"),
        "dataset_link": metric.get("dataset_link"),
        "dataset_split": metric.get("dataset_split_description"),
        "num_samples": metric.get("num_partial_samples"),
        "benchmark": metric.get("benchmark_value"),
        "benchmark_source": (manifest.get("numerics_benchmark") or {}).get("source"),
        "torch": metric.get("partial_torch_metric"),
        "devices": sorted(devices),
        "values": sorted(values, key=lambda r: (r["device"], r["precision"], r["runtime"])),
    }


def _model(model_id, manifest, numerics, tag):
    details = manifest.get("technical_details") or {}
    params, sizes = _technical(details)
    return {
        "id": model_id,
        "name": manifest.get("name") or model_id,
        "headline": manifest.get("headline"),
        "description": manifest.get("description"),
        "domain": manifest.get("domain"),
        "use_case": manifest.get("use_case"),
        "tags": manifest.get("tags") or [],
        "form_factors": manifest.get("form_factors") or [],
        "license_type": manifest.get("license_type"),
        "license": manifest.get("license"),
        "research_paper": manifest.get("research_paper"),
        "research_paper_title": manifest.get("research_paper_title"),
        "source_repo": manifest.get("source_repo"),
        "is_llm": bool(manifest.get("model_type_llm")),
        "resolution": details.get("Input resolution"),
        "params": params,
        "size_mb": sizes,
        "technical_details": {k: str(v) for k, v in details.items()},
        "links": {
            "aihub": AIHUB_MODEL_URL.format(id=model_id),
            "github": GITHUB_MODEL_URL.format(tag=tag, id=model_id),
        },
        "accuracy": _accuracy(numerics, manifest),
    }


def _perf_rows(model_id, perf, tool_versions):
    rows, llm_rows = [], []
    for precision, pv in sorted((perf.get("precisions") or {}).items()):
        for component, cv in sorted((pv.get("components") or {}).items()):
            for device, by_runtime in sorted((cv.get("performance_metrics") or {}).items()):
                for runtime, r in sorted((by_runtime or {}).items()):
                    key = {"model": model_id, "precision": precision, "component": component,
                           "device": device, "runtime": runtime}
                    for llm in r.get("llm_metrics") or []:
                        ttft = llm.get("time_to_first_token_range_milliseconds") or {}
                        row = dict(key)
                        row.update({
                            "context_length": llm.get("context_length"),
                            "tokens_per_s": llm.get("tokens_per_second"),
                            "prefill_tokens_per_s": llm.get("prefill_tokens_per_second"),
                            "ttft_ms": [ttft.get("min"), ttft.get("max")],
                            "compute_unit": llm.get("desired_compute_unit"),
                        })
                        llm_rows.append(row)
                    latency = r.get("inference_time_milliseconds")
                    if latency is None or r.get("job_status", "Passed") != "Passed":
                        continue
                    mem = r.get("estimated_peak_memory_range_mb") or {}
                    layers = r.get("layer_counts") or {}
                    tools = r.get("tool_versions") or {}
                    for name, version in tools.items():
                        tool_versions.setdefault(name, set()).add(str(version))
                    row = dict(key)
                    row.update({
                        "latency_ms": latency,
                        "memory_mb": [mem.get("min"), mem.get("max")],
                        "layers": {u: layers.get(u, 0) for u in ("npu", "gpu", "cpu", "total")},
                        "compute_unit": r.get("primary_compute_unit"),
                        "qairt": tools.get("qairt"),
                        "onnx_runtime": tools.get("onnx_runtime"),
                        "job_id": r.get("job_id"),
                    })
                    rows.append(row)
    return rows, llm_rows


def build_snapshot(checkout, include_status=("published",)):
    """Read every model under the checkout and return the normalized snapshot dict."""
    models, perf, llm_perf, skipped = {}, [], [], {}
    tool_versions = {}
    for model_dir in sorted(glob.glob(os.path.join(checkout.models_dir, "*", ""))):
        model_id = os.path.basename(os.path.dirname(model_dir))
        manifest = _load(os.path.join(model_dir, "manifest.yaml"))
        if manifest is None:
            continue
        status = manifest.get("status")
        if status not in include_status:
            skipped[model_id] = status
            continue
        numerics = _load(os.path.join(model_dir, "numerics.yaml"))
        models[model_id] = _model(model_id, manifest, numerics, checkout.tag)
        perf_yaml = _load(os.path.join(model_dir, "perf.yaml"))
        if perf_yaml:
            rows, llm_rows = _perf_rows(model_id, perf_yaml, tool_versions)
            perf.extend(rows)
            llm_perf.extend(llm_rows)

    all_devices = load_devices(checkout.package_dir)
    used = set(r["device"] for r in perf) | set(r["device"] for r in llm_perf)
    for m in models.values():
        if m["accuracy"]:
            used.update(m["accuracy"]["devices"])
    devices = {}
    for name in sorted(used):
        devices[name] = all_devices.get(name) or {"chipset": None, "world": None, "borrowed_from": None}

    return {
        "schema": SCHEMA_VERSION,
        "release": {
            "tag": checkout.tag,
            "commit": checkout.commit,
            "commit_date": checkout.commit_date,
            "release_url": GITHUB_RELEASE_URL.format(tag=checkout.tag),
            "generated_at": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
            "tool_versions": {k: sorted(v) for k, v in sorted(tool_versions.items())},
            "skipped_models": skipped,
        },
        "devices": devices,
        "models": models,
        "perf": perf,
        "llm_perf": llm_perf,
    }
