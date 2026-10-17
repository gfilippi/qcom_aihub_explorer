"""Locate (and optionally fetch) the upstream qualcomm/ai-hub-models checkout for a tag."""

import os
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools")
FETCH_SCRIPT = os.path.join(TOOLS_DIR, "fetch_aihub_repo.sh")


class Checkout(object):
    """An upstream checkout at a release tag."""

    def __init__(self, tag, path):
        self.tag = tag
        self.path = path
        self.package_dir = os.path.join(path, "src", "qai_hub_models")
        self.models_dir = os.path.join(self.package_dir, "models")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", self.path] + list(args)).decode().strip()

    @property
    def commit(self):
        return self.git("rev-parse", "HEAD")

    @property
    def commit_date(self):
        return self.git("log", "-1", "--format=%cI")


def checkout_path(tag):
    return os.path.join(TOOLS_DIR, "ai-hub-models-" + tag)


def get_checkout(tag, fetch=True):
    """Return the Checkout for `tag`, cloning it via tools/fetch_aihub_repo.sh if missing."""
    path = checkout_path(tag)
    if not os.path.isdir(os.path.join(path, ".git")):
        if not fetch:
            raise FileNotFoundError("No checkout for %s at %s" % (tag, path))
        subprocess.check_call([FETCH_SCRIPT, tag])
    checkout = Checkout(tag, path)
    if not os.path.isdir(checkout.models_dir):
        # Older releases kept the package at the repo root.
        checkout.package_dir = os.path.join(path, "qai_hub_models")
        checkout.models_dir = os.path.join(checkout.package_dir, "models")
    if not os.path.isdir(checkout.models_dir):
        raise FileNotFoundError("No models directory found in %s" % path)
    return checkout
