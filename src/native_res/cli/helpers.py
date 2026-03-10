import ast
import re
from logging import DEBUG, getLogger
from typing import Any

from jetpytools import CustomValueError, SPath, get_subclasses
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from typer import BadParameter, Exit
from vskernels import ComplexKernel
from vssource import BestSource, CacheIndexer, Indexer
from vstools import Matrix, vs


# Callbacks
def resolve_dimension(value: str) -> float:
    nb = float(value)

    if nb.is_integer():
        return int(nb)

    return nb


def resolve_idx(idx: str) -> Indexer:
    indexer = Indexer.from_param(idx, BadParameter)

    args = dict[str, Any]()

    if issubclass(indexer, CacheIndexer):
        # Set cache path to None
        args[indexer._cache_arg_name] = None

        if issubclass(indexer, BestSource):
            args["show_pretty_progress"] = True

    return indexer(**args)


def resolve_dimension_mode(mode: str) -> str:
    match mode:
        case "height" | "h":
            return "height"
        case "width" | "w":
            return "width"
        case _:
            raise BadParameter("Unknown dimension passed")


def resolve_kernel(value: str) -> ComplexKernel:
    matched = re.match(r"^([A-Za-z_]\w*)(?:\((.*)\))?$", value.strip(), re.DOTALL)

    if not matched:
        raise BadParameter("expected KernelName(...) or KernelName")

    name, args_text = matched.group(1), (matched.group(2) or "").strip()

    kernel = ComplexKernel.from_param(name, BadParameter)

    if not args_text:
        return kernel()

    expr = f"{name}({args_text})"

    node = ast.parse(expr, mode="eval")
    call = node.body

    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name):
        raise ValueError

    return kernel(
        *(ast.literal_eval(a) for a in call.args),
        **{kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg},
    )


def set_debug(value: bool) -> None:
    if value:
        getLogger((__package__ or "").split(".")[0]).setLevel(DEBUG)


def set_global_debug(value: bool) -> None:
    if value:
        getLogger().setLevel(DEBUG)


def show_default_kernels(value: bool) -> None:
    if value:
        from ..kernels import default_kernels

        console = Console(stderr=True)

        for kernel in default_kernels:
            console.print(str(kernel))

        raise Exit(0)


def show_vskernels(value: bool) -> None:
    if value:
        all_kernels = {k for k in get_subclasses(ComplexKernel) if not k.is_abstract}

        console = Console(stderr=True)

        for kernel in sorted(all_kernels, key=lambda k: k.__name__):
            console.print(kernel.__name__)

        raise Exit(0)


# Helpers
def get_videonode_from_input(path: SPath, indexer: Indexer, frame: int, console: Console) -> vs.VideoNode:
    if not path.exists():
        raise BadParameter(f"{path.to_str()!r} doesn't exist.")

    if path.suffix in (".py", ".vpy"):
        from vsengine import load_script

        load_script(path, module="__native_res__").result()
        out = next(iter(vs.get_outputs().values()))

        if not isinstance(out, vs.VideoOutputTuple):
            raise CustomValueError("Unknown VapourSynth output", get_videonode_from_input, type(out))

        return out.clip

    if isinstance(indexer, BestSource):
        from signal import SIG_DFL, SIGINT, signal

        signal(SIGINT, SIG_DFL)

    clip = indexer.source(path, 32, idx_props=False)[frame]
    return clip.resize.Bilinear(format=vs.GRAYS, matrix=Matrix.BT709, matrix_in=Matrix.from_video(clip))


def get_progress(console: Console) -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
