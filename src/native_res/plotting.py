import csv
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from logging import getLogger
from os import PathLike
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event
from matplotlib.figure import figaspect
from matplotlib.rcsetup import cycler
from matplotlib.widgets import Button

logger = getLogger(__name__)


def make_plot(dims: Sequence[float], errors: Sequence[float], dimension_mode: str, window_title: str) -> None:
    with plt.style.context(get_plotting_style()):
        fig, ax = plt.subplots(figsize=figaspect(1 / 2))

        assert fig.canvas.manager

        fig.canvas.manager.set_window_title(window_title)

        ax.set_title(f"Error vs {dimension_mode}", fontsize=12, pad=10)
        ax.plot(dims, errors, marker="o", markersize=3, markeredgewidth=0.4, linestyle="-", linewidth=1, alpha=0.9)
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.3)
        ax.set(xlabel=dimension_mode, ylabel="Error", yscale="log")
        ax.tick_params(axis="both", which="major", labelsize=9)

        fig.tight_layout(rect=(0, 0.06, 1, 1))

        def on_save_image(event: Event) -> Any:
            image_filetypes = [("PNG image", "*.png"), ("SVG vector", "*.svg"), ("All files", "*.*")]

            path = ask_save_path(".png", "Save plot as...", image_filetypes)

            if not path:
                return
            try:
                with hide_axes(ax_save, ax_json):
                    fig.savefig(path, dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())

                logger.debug(f"Saved figure to {path}")
            except Exception:
                logger.exception("Failed to save figure")

        def on_export_data(event: Event) -> Any:
            data_filetypes = [("CSV file", "*.csv"), ("JSON file", "*.json"), ("All files", "*.*")]

            path = ask_save_path(".json", "Export data as...", data_filetypes)

            if not path:
                return
            try:
                write_data(path, dims, errors, dimension_mode)
            except Exception:
                logger.exception("Failed to export data:")

        btn_w = 0.18
        btn_h = 0.045
        gap = 0.02
        left = 0.02

        ax_save = fig.add_axes((left, 0.01, btn_w, btn_h))
        btn_save = Button(ax_save, "Save Image")
        btn_save.on_clicked(on_save_image)

        ax_json = fig.add_axes((left + btn_w + gap, 0.01, btn_w, btn_h))
        btn_json = Button(ax_json, "Export Data")
        btn_json.on_clicked(on_export_data)

        style_button(ax_save, btn_save)
        style_button(ax_json, btn_json)

        plt.show()
        plt.close(fig)


def ask_save_path(default_ext: str, title: str, filetypes: list[tuple[str, str]]) -> str:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        return filedialog.asksaveasfilename(defaultextension=default_ext, filetypes=filetypes, title=title)
    finally:
        root.destroy()


def write_data(path: str | PathLike[str], dims: Sequence[float], errors: Sequence[float], dimension_mode: str) -> None:
    path = Path(path)

    if path.suffix == ".csv":
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["dimension", "error"])

            for d, e in zip(dims, errors):
                writer.writerow([d, e])

        logger.debug("Exported CSV to %s", path)

    else:
        payload = {"dimension_mode": dimension_mode, "dims": list(dims), "errors": list(errors)}

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        logger.debug("Exported JSON to %s", path)


@contextmanager
def hide_axes(*axes: Axes) -> Iterator[None]:
    old_visibility = (ax.get_visible() for ax in axes)
    try:
        for ax in axes:
            ax.set_visible(False)
        yield
    finally:
        for ax, old in zip(axes, old_visibility):
            ax.set_visible(old)


BUTTON_BG = "#2A3440"
BUTTON_HOVER = "#232C36"
BUTTON_TEXT = "#FFFFFFD9"
BUTTON_BORDER = "#FFFFFF3D"


def style_button(ax: Axes, btn: Button) -> None:
    ax.set_facecolor(BUTTON_BG)
    ax.patch.set_edgecolor(BUTTON_BORDER)
    ax.patch.set_linewidth(1)

    btn.color = BUTTON_BG
    btn.hovercolor = BUTTON_HOVER

    btn.label.set_color(BUTTON_TEXT)
    btn.label.set_fontsize(9)


def get_plotting_style() -> dict[str, Any]:
    """vspreview style"""
    return {
        "axes.edgecolor": "#FFFFFF3D",
        "axes.facecolor": "#FFFFFF07",
        "axes.labelcolor": "#FFFFFFD9",
        "axes.prop_cycle": cycler(
            "color", ["#FF6200", "#696969", "#525199", "#60A6DA", "#D0D93C", "#A8A8A9", "#FF0000", "#349651", "#AB0066"]
        ),
        "figure.facecolor": "#19232D",
        "legend.edgecolor": "#FFFFFFD9",
        "legend.facecolor": "inherit",
        "legend.framealpha": 0.12,
        "markers.fillstyle": "full",
        "savefig.facecolor": "#19232D",
        "text.color": "white",
        "xtick.color": "#FFFFFFD9",
        "ytick.color": "#FFFFFFD9",
    }
