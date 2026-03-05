"""CLI module"""

import warnings
from itertools import zip_longest
from logging import DEBUG, INFO, basicConfig, captureWarnings, getLogger
from typing import Annotated, Any, Literal, assert_never, cast

from jetpytools import SPath
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import pretty_repr
from rich.progress import BarColumn, Progress, TextColumn
from rich.style import Style
from rich.table import Table
from vskernels import ComplexKernel
from vsmasktools import EdgeDetect
from vssource import Indexer

from ..funcs import getfnative, getfscaler
from .components import (
    NativeCommand,
    ScalerCommand,
    base_dim_opt,
    crop_opt,
    debug_opt,
    dim_mode_opt,
    dim_opt,
    frame_opt,
    global_debug_opt,
    indexer_opt,
    input_file_arg,
    kernel_opt,
    mask_opt,
    metric_mode_opt,
    native_app,
    range_dim_opt,
    scaler_app,
    show_default_kernels_opt,
    show_vskernels_opt,
    step_opt,
)
from .helpers import get_progress, get_videonode_from_input
from .kernels import default_kernels

console = Console(stderr=True)

warnings.filterwarnings("always")
captureWarnings(True)
basicConfig(
    level=INFO,
    handlers=[RichHandler(level=DEBUG, console=console)],
    format="{name}: {message}",
    style="{",
)

logger = getLogger(__name__)


@native_app.command(
    cls=NativeCommand,
    help="Determine the best (fractional) native resolution for an upscaled frame.",
    no_args_is_help=True,
)
def getfnative_cli(
    input_file: Annotated[SPath, input_file_arg],
    range_dim: Annotated[tuple[int, int] | None, range_dim_opt] = None,
    dim_mode: Annotated[Literal["height", "width"], dim_mode_opt] = "height",
    kernel: Annotated[ComplexKernel, kernel_opt] = cast(ComplexKernel, "bilinear"),
    frame: Annotated[int, frame_opt] = 0,
    step: Annotated[float, step_opt] = 1,
    crop: Annotated[tuple[int, int, int, int] | None, crop_opt] = None,
    metric_mode: Annotated[Literal["MAE", "MSE", "RMSE"], metric_mode_opt] = "MAE",
    indexer: Annotated[Indexer, indexer_opt] = cast(Indexer, "bs"),
    _show_vskernels: Annotated[bool, show_vskernels_opt] = False,
    _debug: Annotated[bool, debug_opt] = False,
    _global_debug: Annotated[bool, global_debug_opt] = False,
) -> None:
    import numpy as np

    from ..plotting import make_plot

    clip = get_videonode_from_input(input_file, indexer, frame, console)

    # Resolve dimension and the range of dimensions to check
    if range_dim:
        start, stop = range_dim
    else:
        match dim_mode:
            case "height":
                dim = clip.height
            case "width":
                dim = clip.width
            case _:
                assert_never(dim_mode)

        start, stop = int(dim * 0.465), int(dim * 0.925)

    # Build the list of dims (int or fractional)
    step_f = float(step)
    if step_f.is_integer():
        dims = range(start, stop + 1, int(step_f))
    else:
        num = int((stop - start) / step_f) + 1
        dims = np.linspace(start, start + step_f * (num - 1), num).tolist()

    # Pair with the fixed dimension
    match dim_mode:
        case "height":
            dimensions = zip_longest([clip.width], dims, fillvalue=clip.width)
        case "width":
            dimensions = zip_longest(dims, [clip.height], fillvalue=clip.height)
        case _:
            assert_never(dim_mode)

    # Pretty progress
    progress = get_progress(console)
    gtask_id = progress.add_task("Gathering data...", total=None)

    logger.debug(kernel)

    with progress:
        results = getfnative(
            clip,
            dimensions,
            kernel,
            crop,
            metric_mode=metric_mode,
            progress_cb=lambda curr, total: progress.update(
                gtask_id,
                completed=curr,
                total=total,
                visible=True,
                refresh=True,
            ),
        )
        progress.update(gtask_id, total=100, completed=100, refresh=True)

    dims, errors = zip(*results)

    # Pop out a matplot
    make_plot([getattr(d, dim_mode) for d in dims], errors, dim_mode, f"Error plot - {kernel} on {dim_mode}")


@scaler_app.command(
    cls=ScalerCommand,
    help="Find the best inverse scaler for a given frame.",
    epilog="""
[dim]Notes:

 - getfscaler gives heuristic results; it's not infallible.

 - Always visually verify the suggested scaler and parameters on multiple frames before trusting them.[/dim]
""",
    no_args_is_help=True,
)
def getfscaler_cli(
    input_file: Annotated[SPath, input_file_arg],
    dim: Annotated[float, dim_opt],
    dim_mode: Annotated[Literal["height", "width"], dim_mode_opt] = "height",
    base_dim_opt: Annotated[int | None, base_dim_opt] = None,
    kernels: Annotated[list[ComplexKernel], kernel_opt] = [],
    frame: Annotated[int, frame_opt] = 0,
    crop: Annotated[tuple[int, int, int, int] | None, crop_opt] = None,
    metric_mode: Annotated[Literal["MAE", "MSE", "RMSE"], metric_mode_opt] = "MAE",
    mask: Annotated[type[EdgeDetect] | None, mask_opt] = None,
    indexer: Annotated[Indexer, indexer_opt] = cast(Indexer, "bs"),
    _show_vskernels: Annotated[bool, show_vskernels_opt] = False,
    _show_default_kernels: Annotated[bool, show_default_kernels_opt] = False,
    _debug: Annotated[bool, debug_opt] = False,
    _global_debug: Annotated[bool, global_debug_opt] = False,
) -> None:
    clip = get_videonode_from_input(input_file, indexer, frame, console)

    # Resolve dimension to check
    scaler_args: dict[str, Any] = {"width": clip.width, "height": clip.height} | {
        dim_mode: dim,
        f"base_{dim_mode}": base_dim_opt,
    }

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
        transient=True,
    )
    task = progress.add_task("Gathering data...", total=None)

    with progress:
        ress = getfscaler(
            clip,
            kernels=(*default_kernels, *kernels),
            crop=crop,
            metric_mode=metric_mode,
            mask=mask,
            **scaler_args,
        )
    progress.update(task, completed=100, total=100, visible=False, refresh=True)

    # Results are sorted and displayed to the CLI for the user
    sorted_ress = sorted(ress, key=lambda r: r.error)
    best = sorted_ress[0]

    logger.debug("%s", pretty_repr(sorted_ress, max_width=200, indent_size=2))

    width, height = scaler_args["width"], scaler_args["height"]

    dwidth = f"{width:.0f}" if float(width).is_integer() else f"{width:.3f}"
    dheight = f"{height:.0f}" if float(height).is_integer() else f"{height:.3f}"

    table = Table(
        title=f"Results for frame {frame} — Resolution: {dwidth}x{dheight}",
        title_style=Style(bold=True),
        caption=f"Smallest error archieved by {best.kernel.pretty_string}: {best.error:.13f}",
        caption_style=Style(bold=True, dim=True),
        caption_justify="left",
        min_width=80,
    )
    table.add_column("Kernel")
    table.add_column("Error %", justify="center")
    table.add_column(metric_mode, justify="right")

    for res in sorted_ress:
        table.add_row(
            res.kernel.pretty_string, f"{res.error * 100 / best.error if best.error else 0:.2f} %", f"{res.error:.13f}"
        )

    console.rule()
    console.print(table, new_line_start=True)
    console.rule()
    console.print(
        "Getfscaler is not infallible!\n"
        "Always visually verify the suggested scaler and parameters on multiple frames before trusting them.",
        style=Style(color="yellow", dim=True),
    )
