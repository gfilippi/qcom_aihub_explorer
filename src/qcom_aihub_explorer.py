"""Command line entry point: python3 src/qcom_aihub_explorer.py <command> --tag <tag>."""

import argparse
import collections
import json
import os
import sys

import fetch
import normalize
import render

__version__ = "0.1.0"

RESULTS_DIR = os.path.join(fetch.PROJECT_ROOT, "results")
README_PATH = os.path.join(fetch.PROJECT_ROOT, "README.md")


def snapshot_dir(tag):
    return os.path.join(RESULTS_DIR, "aihub-" + tag)


def write_snapshot(args):
    checkout = fetch.get_checkout(args.tag, fetch=not args.no_fetch)
    snapshot = normalize.build_snapshot(checkout)
    out_dir = os.path.join(snapshot_dir(args.tag), "data")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, "snapshot.json")
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=1, sort_keys=True)
        f.write("\n")

    models = snapshot["models"]
    worlds = collections.Counter(d.get("world") for d in snapshot["devices"].values())
    print("Release      %s (%s)" % (snapshot["release"]["tag"], snapshot["release"]["commit_date"]))
    print("Models       %d published, %d skipped, %d with accuracy, %d LLM"
          % (len(models), len(snapshot["release"]["skipped_models"]),
             sum(1 for m in models.values() if m["accuracy"]),
             sum(1 for m in models.values() if m["is_llm"])))
    print("Devices      %d  %s" % (len(snapshot["devices"]), dict(worlds)))
    print("Perf rows    %d (+%d LLM)" % (len(snapshot["perf"]), len(snapshot["llm_perf"])))
    for tool, versions in snapshot["release"]["tool_versions"].items():
        print("%-12s %s" % (tool, ", ".join(versions)))
    print("Wrote        %s (%.1f MB)" % (out_path, os.path.getsize(out_path) / 1e6))
    return snapshot


def cmd_normalize(args):
    write_snapshot(args)


def cmd_build(args):
    snapshot = write_snapshot(args)
    out_dir = snapshot_dir(args.tag)
    render.render_snapshot(snapshot, out_dir)
    print("Wrote        %s" % os.path.join(out_dir, "index.html"))
    write_landing()


def write_landing():
    snapshots = render.render_landing(RESULTS_DIR)
    print("Wrote        %s (%d snapshots)" % (os.path.join(RESULTS_DIR, "index.html"), len(snapshots)))
    if render.update_readme_table(README_PATH, RESULTS_DIR, snapshots):
        print("Updated      %s (snapshot table)" % README_PATH)


def cmd_landing(args):
    write_landing()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qcom_aihub_explorer")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    for name, func, help_text in (
            ("normalize", cmd_normalize, "read the upstream checkout and write data/snapshot.json"),
            ("build", cmd_build, "normalize, then render index.html + assets and refresh results/index.html")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--tag", required=True, help="upstream release tag, e.g. v0.64.0")
        p.add_argument("--no-fetch", action="store_true", help="fail instead of cloning a missing checkout")
        p.set_defaults(func=func)
    p = sub.add_parser("landing", help="regenerate results/index.html and the README table from the existing snapshots")
    p.set_defaults(func=cmd_landing)
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
