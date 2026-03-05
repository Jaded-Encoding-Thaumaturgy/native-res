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
        formatter.write_usage("getfnative", "INPUT [OPTIONS]")


class ScalerCommand(TyperCommand):
    def format_usage(self, ctx: Context, formatter: HelpFormatter) -> None:
        formatter.write_usage("getfscaler", "INPUT WIDTH HEIGHT [OPTIONS]")


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


# Apps
native_app = Typer(
    name="getfnative",
    help="Determine the native (fractional) resolution of upscaled material (primarily anime).",
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    add_completion=False,
)
scaler_app = Typer(
    name="getfscaler",
    help="Identify the best inverse scaler for a single frame.",
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    add_completion=False,
)


# Commons
input_file_arg = Argument(
    help="Path to input file; video, image or script (output 0). ",
    metavar="INPUT",
    resolve_path=True,
    parser=SPath,
)
frame_opt = Option("--frame", "-f", help="Frame number to analyze (for video inputs).", metavar="INTEGER")
kernel_opt = Option(
    "--kernel",
    "-k",
    help="Select or add a custom kernel.",
    metavar="Kernel|KernelName(arg0=..., arg1=...)",
    parser=resolve_kernel,
)

dim_mode_opt = Option(
    "--dim-mode",
    "-dm",
    help="The dimension to check.",
    metavar="height|width|h|w",
    parser=resolve_dimension_mode,
)
crop_opt = Option(
    "--crop",
    "-c",
    help="Crop as (left right top bottom) to remove borders before analysis.",
    metavar="L R T B",
)
metric_mode_opt = Option(
    "--metric-mode",
    "-mm",
    help="The metric mode to use",
    metavar="MAE|MSE|RMSE",
    parser=lambda value: value.upper(),
)
indexer_opt = Option(
    "--indexer",
    "-idx",
    help="Indexer backend to use.",
    metavar="STRING",
    parser=resolve_idx,
    show_default="BestSource",
)

# Helpers
show_default_kernels_opt = Option(
    "--show-default-kernels",
    help="Show the default kernels and exit.",
    is_eager=True,
    show_default=False,
    callback=show_default_kernels,
)
show_vskernels_opt = Option(
    "--show-kernels",
    help="Show the builtin kernels and exit.",
    is_eager=True,
    show_default=False,
    callback=show_builtin_kernels,
)

# getfnative exclusive
range_dim_opt = Option(
    "--range-dim",
    "-rd",
    help="Inclusive range of dimension to check.",
    metavar="INTEGER INTEGER",
    show_default=False,
)
step_opt = Option("--step", "-s", help="Positive number that specifies the increment.", metavar="NUMBER")


# getfscaler exclusive
dim_opt = Argument(
    help="Approximate native dimension value. "
    "Use integer for exact pixels or fractional number for sub-pixel dimensions.",
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
    help="Edge-detection mask to reduce noise influence on the metric. Pass a mask name or class.",
    metavar="EDGEDETECT",
    parser=EdgeDetect.from_param,
)
