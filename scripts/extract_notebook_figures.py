"""Extract embedded notebook output images into the figures directory.

This script does not execute notebooks. It only reads image outputs that are
already stored in the committed .ipynb files, writes them as normal image files,
and creates a small index plus a CSV manifest.
"""

from __future__ import annotations

import base64
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = [
    {
        "path": Path("notebooks/01_two_step_nucleation_turnbull_fisher_Fe.ipynb"),
        "out_dir": Path("figures/01_nucleation_engine"),
        "group": "Nucleation engine",
    },
    {
        "path": Path("notebooks/02_phase_field_coupling_Fe.ipynb"),
        "out_dir": Path("figures/02_phase_field_coupling"),
        "group": "Phase-field coupling",
    },
]


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def nearest_heading(cells: list[dict], cell_index: int) -> str:
    for index in range(cell_index, -1, -1):
        cell = cells[index]
        if cell.get("cell_type") != "markdown":
            continue
        lines = [line.strip() for line in cell_source(cell).splitlines()]
        for line in reversed(lines):
            if line.startswith("#"):
                return line.lstrip("#").strip().replace("`", "")
    return "Notebook output"


def extension_for_mime(mime: str) -> str:
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/svg+xml": "svg",
    }.get(mime, mime.split("/", 1)[-1].replace("+xml", ""))


def png_dimensions(data: bytes) -> str:
    if len(data) >= 24 and data[12:16] == b"IHDR":
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return f"{width}x{height}"
    return ""


def decode_image(mime: str, value: str | list[str]) -> bytes:
    text = "".join(value) if isinstance(value, list) else str(value)
    if mime == "image/svg+xml":
        return text.encode("utf-8")
    return base64.b64decode("".join(text.split()))


def main() -> None:
    manifest: list[dict[str, str | int]] = []

    for config in NOTEBOOKS:
        notebook_path = REPO_ROOT / config["path"]
        out_dir = REPO_ROOT / config["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = notebook["cells"]
        figure_number = 0

        for cell_index, cell in enumerate(cells):
            for output_index, output in enumerate(cell.get("outputs", []), start=1):
                for mime, value in output.get("data", {}).items():
                    if not mime.startswith("image/"):
                        continue

                    data = decode_image(mime, value)
                    figure_number += 1
                    extension = extension_for_mime(mime)
                    filename = f"cell_{cell_index:02d}_fig_{figure_number:02d}.{extension}"
                    figure_path = out_dir / filename
                    figure_path.write_bytes(data)

                    manifest.append(
                        {
                            "file": figure_path.relative_to(REPO_ROOT).as_posix(),
                            "source_notebook": config["path"].as_posix(),
                            "source_cell": cell_index,
                            "output_index": output_index,
                            "mime": mime,
                            "dimensions": png_dimensions(data) if mime == "image/png" else "",
                            "section": nearest_heading(cells, cell_index),
                            "group": config["group"],
                        }
                    )

    write_manifest(manifest)
    write_index(manifest)


def write_manifest(manifest: list[dict[str, str | int]]) -> None:
    fieldnames = [
        "file",
        "source_notebook",
        "source_cell",
        "output_index",
        "mime",
        "dimensions",
        "section",
        "group",
    ]
    manifest_path = REPO_ROOT / "figures/manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def write_index(manifest: list[dict[str, str | int]]) -> None:
    lines = [
        "# Figure Index",
        "",
        "Figures in this directory were extracted from saved outputs embedded in the two main notebooks.",
        "The notebooks were not re-executed during extraction.",
        "",
    ]

    for group in [config["group"] for config in NOTEBOOKS]:
        lines.extend([f"## {group}", ""])
        for row in [item for item in manifest if item["group"] == group]:
            image_path = str(row["file"]).removeprefix("figures/")
            dimension_text = f", {row['dimensions']}" if row["dimensions"] else ""
            lines.extend(
                [
                    f"### {row['section']} - cell {row['source_cell']}",
                    "",
                    f"Source: `{row['source_notebook']}`, output {row['output_index']}{dimension_text}.",
                    "",
                    f"![{row['section']}]({image_path})",
                    "",
                ]
            )

    (REPO_ROOT / "figures/README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
