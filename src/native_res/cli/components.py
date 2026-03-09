from click import Context, HelpFormatter
from jetpytools import SPath
from typer import Argument, Option, Typer
from typer.core import TyperCommand
from vsmasktools import EdgeDetect

from .helpers import (
    resolve_dimension,
    resolve_dimension_mode,
    resolve_idx,
    resolve_kernel,
    set_debug,
    set_global_debug,
    show_default_kernels,
    show_vskernels,
)


class NativeCommand(TyperCommand):
    def format_usage(self, ctx: Context, formatter: HelpFormatter) -> None:
        formatter.write_usage(ctx.command_path, "INPUT [OPTIONS]")


class ScalerCommand(TyperCommand):
    def format_usage(self, ctx: Context, formatter: HelpFormatter) -> None:
        formatter.write_usage(ctx.command_path, "INPUT DIM [OPTIONS]")


# DEBUG
debug_opt = Option(
    "--debug",
    help="Enable debug output.",
    hidden=True,
    is_eager=True,
    callback=set_debug,
)
global_debug_opt = Option(
    "--global-debug",
    help="Enable global debug output.",
    hidden=True,
    is_eager=True,
    callback=set_global_debug,
)


# App
app = Typer(
    name="native-res",
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
    add_completion=False,
    no_args_is_help=True,
)

# Commons
input_file_arg = Argument(
    help="Path to the source material to analyze."
    "Supports videos, images, or VapourSynth scripts. For scripts, the first output is used.",
    metavar="INPUT",
    resolve_path=True,
    parser=SPath,
)
frame_opt = Option(
    "--frame",
    "-f",
    help="The specific frame number to extract and analyze from video inputs. Ignored for images.",
    metavar="INTEGER",
)
kernel_opt = Option(
    "--kernel",
    "-k",
    help="The kernel(s) to use for inverse scaling.\n\n"
    "Can be a kernel name or a class call with parameters (e.g., 'Bicubic(b=0, c=0.5)').\n\n"
    "Use --show-kernels for a list of available kernels.",
    metavar="Kernel|Kernel(arg0=..., arg1=...)",
    parser=resolve_kernel,
)

dim_mode_opt = Option(
    "--dim-mode",
    "-dm",
    help="Specifies whether to analyze based on the [bold]height[/] or [bold]width[/] of the frame.",
    metavar="height|width|h|w",
    parser=resolve_dimension_mode,
)
crop_opt = Option(
    "--crop",
    "-c",
    help="Crop the input frame before analysis to remove black bars.\n\n"
    "Format: [bold]LEFT RIGHT TOP BOTTOM[/] (e.g., '0 0 240 240').",
    metavar="L R T B",
)
metric_mode_opt = Option(
    "--metric-mode",
    "-mm",
    help="The mathematical metric used to compare scaling results.\n\n"
    "- [bold]MAE[/] (Mean Absolute Error)\n\n"
    "- [bold]MSE[/] (Mean Squared Error)\n\n"
    "- [bold]RMSE[/] (Root Mean Squared Error)",
    metavar="MAE|MSE|RMSE",
    parser=lambda value: value.upper(),
)
indexer_opt = Option(
    "--indexer",
    "-idx",
    help="The VapourSynth indexer used to load files.\n\nSpecifying the plugin namespace is also allowed (e.g., 'bs').",
    metavar="STRING",
    parser=resolve_idx,
    show_default="BestSource",
)

# Helpers
show_default_kernels_opt = Option(
    "--show-kernels",
    help="Show the default kernels and exit.",
    is_eager=True,
    show_default=False,
    callback=show_default_kernels,
)
show_vskernels_opt = Option(
    "--show-vskernels",
    help="Show the builtin supported kernels from vskernels and exit.",
    is_eager=True,
    show_default=False,
    callback=show_vskernels,
)

# getfnative exclusive
range_dim_opt = Option(
    "--range-dim",
    "-rd",
    help="The inclusive range of resolutions to test.\n\nSpecify as [bold]START END[/] (e.g., '500 1080').",
    metavar="INTEGER INTEGER",
    show_default=False,
)
step_opt = Option("--step", "-s", help="The increment step between resolutions in the tested range.", metavar="NUMBER")


# getfscaler exclusive
dim_opt = Argument(
    help="The suspected native resolution to verify. "
    "Use an integer for exact pixels (e.g., 720) or a float for sub-pixel dimensions (e.g., 719.8).",
    metavar="NUMBER",
    parser=resolve_dimension,
)
base_dim_opt = Option(
    "--base-dim",
    "-b",
    help="Base integer dimension if checking for fractional resolution.",
)
mask_opt = Option(
    "--mask",
    "-m",
    help="Edge-detection mask to reduce noise influence on the metric. "
    "Pass a mask name (e.g., 'Prewitt') or a class name from vsmasktools.",
    metavar="EDGEDETECT",
    parser=EdgeDetect.from_param,
)
