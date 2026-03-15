import json
import os
import re
from typing import Any

PACKAGES = {
    "nativeres": {
        "tag": "nativeres",
        "package": "nativeres",
        "workspace_package": "nativeres",
        "path": ".",
        "environment": "pypi",
        "url": "https://pypi.org/p/nativeres",
    },
    "vsview-nativeres": {
        "tag": "vsview-nativeres",
        "package": "vsview-nativeres",
        "workspace_package": "vsview-nativeres",
        "path": "src/plugin",
        "environment": "pypi-vsview",
        "url": "https://pypi.org/p/vsview-nativeres",
    },
}


def _write_output(name: str, value: str) -> None:
    output_file = os.getenv("GITHUB_OUTPUT")

    if not output_file:
        print(f"{name}={value}")
        return

    with open(output_file, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def _resolve_target() -> dict[str, Any]:
    event = os.getenv("GITHUB_EVENT_NAME", "")
    ref = os.getenv("GITHUB_REF", "")
    release_tag = os.getenv("RELEASE_TAG", "")
    dispatch_package = os.getenv("INPUT_PACKAGE", "")
    dispatch_tag = os.getenv("INPUT_TAG_NAME", "")

    target = ""
    tag = ""

    if event == "workflow_dispatch":
        target = dispatch_package
        tag = dispatch_tag
    elif event == "release":
        tag = release_tag
        match = re.match(r"^(.+)/v", tag)
        if match:
            target = match.group(1)
    elif ref.startswith("refs/tags/"):
        tag = ref.removeprefix("refs/tags/")
        match = re.match(r"^(.+)/v", tag)
        if match:
            target = match.group(1)

    selected = PACKAGES.get(target)

    if not selected:
        raise SystemExit(
            f"Unsupported publish target '{target or tag or ref}'. "
            "Expected a release tag like 'nativeres/v*' or 'vsview-nativeres/v*'."
        )

    return {**selected, "tag": tag}


def main() -> None:
    target = _resolve_target()
    _write_output("matrix", json.dumps([target]))


if __name__ == "__main__":
    main()
