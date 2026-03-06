"""Python API"""

from collections.abc import Callable, Iterable, Sequence
from logging import getLogger
from typing import Annotated, Any, Literal, NamedTuple

from jetpytools import CustomOverflowError, FuncExcept, to_arr
from vsexprtools import ExprOp, norm_expr
from vskernels import ComplexKernel, ComplexKernelLike, Kernel, LeftShift, Point, TopShift
from vsmasktools import MaskLike, normalize_mask
from vsscale import Rescale
from vsscale.helpers import BottomCrop, CropRel, LeftCrop, RightCrop, TopCrop
from vstools import clip_data_gather, core, depth, get_prop, get_y, vs

logger = getLogger(__name__)

_point_resize = Point()


class ResolutionFrac(NamedTuple):
    """Fractional resolution expressed as floating-point width and height."""

    width: float
    """Horizontal size (may be fractional)."""

    height: float
    """Vertical size (may be fractional)."""


class GetNativeResult(NamedTuple):
    """Result for `getfnative`, pairing a fractional resolution with an error metric."""

    dim: ResolutionFrac
    """The tested fractional resolution as an `ResolutionFrac`."""

    error: float
    """Computed error for this resolution (higher means worse)."""


def getfnative(
    clip: Annotated[vs.VideoNode, "OneFrameClip"],
    dimensions: Iterable[tuple[float, float]],
    kernel: ComplexKernelLike,
    crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | None = None,
    shift: tuple[TopShift, LeftShift] = (0, 0),
    metric_mode: Literal["MSE", "MAE", "RMSE"] = "MAE",
    borders_aware: bool | int | tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] = 8,
    progress_cb: Callable[[int, int], None] | None = None,
    *,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> list[GetNativeResult]:
    """
    Determine the best (fractional) native resolution for an upscaled frame.

    This checks a list of candidate resolutions by descaling the single frame using `kernel`
    and comparing the result to the source frame using `metric_mode`.

    Only the first (and only) frame of `clip` is evaluated.

    Args:
        clip: Source clip. Must contain exactly one frame.
        dimensions: Iterable of candidate resolutions to test. Each item may be a `(width, height)` tuple.
        kernel: Kernel used to perform each descale attempt.
        crop: Optional crop to apply before descaling. Aspect ratio is preserved.
        shift: Optional pixel shift applied during descaling: `(top_shift, left_shift)`.
        metric_mode: Error metric to use: `"MSE"`, `"MAE"`, or `"RMSE"`.
        borders_aware: Amount (or crop tuple) to ignore at image borders to avoid edge noise when computing the metric.
        progress_cb: Optional progress callback called as `progress_cb(current, total)`.
        func: Function returned for custom error handling.
        kwargs: Additional arguments passed to the Rescale class.

    Raises:
        CustomOverflowError: If `clip` does not contain exactly one frame.

    Returns:
        A list of `GetNativeResult` items.
        Each item contains the tested fractional resolution (`dim`) and its associated `error`.
    """
    func = func or getfnative

    if clip.num_frames != 1:
        raise CustomOverflowError("Clip must have only one frame", func)

    clip = depth(get_y(clip), 32)

    kernel = ComplexKernel.ensure_obj(kernel, func)
    crops = _norm_border_crops(borders_aware, kernel)

    dimensions = list(dimensions)

    rescale_list = [
        Rescale(clip, res[1], kernel, _point_resize, width=res[0], crop=crop, shift=shift, **kwargs)
        for res in dimensions
    ]

    rescaled = clip.std.BlankClip(length=len(dimensions)).std.FrameEval(lambda n: rescale_list[n].rescale)
    rescaled = (
        norm_expr([rescaled, clip], getattr(ExprOp, metric_mode.lower())(clip), func=func)
        .std.CropRel(*crops)
        .std.PlaneStats()
    )

    errors = clip_data_gather(rescaled, progress_cb, lambda _, f: get_prop(f, "PlaneStatsAverage", float))

    try:
        return [GetNativeResult(ResolutionFrac(*d), e) for d, e in zip(dimensions, errors)]
    finally:
        for r in rescale_list:
            r.__vs_del__(-1)


class GetScalerResult(NamedTuple):
    """Result for `getfscaler`, pairing a kernel with its error."""

    kernel: ComplexKernel
    """The `ComplexKernel` instance that was evaluated."""

    error: float
    """Computed error for that kernel (lower is better)."""


def getfscaler(
    clip: Annotated[vs.VideoNode, "OneFrameClip"],
    width: float,
    height: float,
    kernels: ComplexKernelLike | Sequence[ComplexKernelLike],
    base_width: int | None = None,
    base_height: int | None = None,
    crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | None = None,
    shift: tuple[TopShift, LeftShift] = (0, 0),
    metric_mode: Literal["MSE", "MAE", "RMSE"] = "MAE",
    mask: MaskLike | None = None,
    borders_aware: bool | int | tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] = 8,
    *,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> list[GetScalerResult]:
    """
    Find the best inverse scaler (kernel) for a given single-frame clip.

    Each supplied kernel is tested by descaling the single frame to the provided
    `(width, height)` and computing an error metric.

    If `width`/`height` are floats, fractional descaling is performed.

    Args:
        clip: Source clip. Must contain exactly one frame.
        width: Width to be descaled to. If passed as a float, a fractional descale is performed.
        height: Height to be descaled to. If passed as a float, a fractional descale is performed.
        kernels: A single kernel or a sequence of kernels to evaluate.
        base_width: Optional integer height to contain the clip within.
        base_height: Optional integer width to contain the clip within.
        crop: Optional crop to apply before descaling. Aspect ratio is preserved.
        shift: Pixel shifts applied during descaling: `(top_shift, left_shift)`.
        metric_mode: Error metric to use: `"MSE"`, `"MAE"`, or `"RMSE"`.
        mask: Optional edge-detection mask to reduce noise influence on the metric.
        borders_aware: Amount (or crop tuple) to ignore at image borders to avoid edge noise when computing the metric.
        func: Function returned for custom error handling.
        kwargs: Additional arguments passed to the Rescale class.

    Raises:
        CustomOverflowError: If `clip` does not contain exactly one frame.

    Returns:
        A list of `GetScalerResult` items. Each item contains the evaluated `kernel` and its associated `error`.
    """
    func = func or getfscaler

    if clip.num_frames != 1:
        raise CustomOverflowError("Clip must have only one frame", func)

    resolved_kernels = {ComplexKernel.ensure_obj(k, func) for k in to_arr(kernels)}  # type:ignore[arg-type]

    return [
        GetScalerResult(
            kernel,
            get_descale_error(
                clip,
                width,
                height,
                kernel,
                base_width,
                base_height,
                crop,
                shift,
                metric_mode,
                mask,
                borders_aware,
                **kwargs,
            ),
        )
        for kernel in resolved_kernels
    ]


def get_descale_error(
    clip: Annotated[vs.VideoNode, "OneFrameClip"],
    width: float,
    height: float,
    kernel: ComplexKernelLike,
    base_width: int | None = None,
    base_height: int | None = None,
    crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | None = None,
    shift: tuple[TopShift, LeftShift] = (0, 0),
    metric_mode: Literal["MSE", "MAE", "RMSE"] = "MAE",
    mask: MaskLike | None = None,
    borders_aware: bool | int | tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] = 8,
    *,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> float:
    """
    Compute the descale error for a single-frame clip using a specific kernel.

    Args:
        clip: Source clip. Must contain exactly one frame.
        width: Width to be descaled to. If passed as a float, a fractional descale is performed.
        height: Height to be descaled to. If passed as a float, a fractional descale is performed.
        kernel: Kernel used for the descale operation.
        base_width: Optional integer height to contain the clip within.
        base_height: Optional integer width to contain the clip within.
        crop: Optional crop to apply before descaling. Aspect ratio is preserved.
        shift: Pixel shifts applied during descaling: `(top_shift, left_shift)`.
        metric_mode: Error metric to use: `"MSE"`, `"MAE"`, or `"RMSE"`.
        mask: Optional edge-detection mask to reduce noise influence on the metric.
        borders_aware: Amount (or crop tuple) to ignore at image borders to avoid edge noise when computing the metric.
        func: Function returned for custom error handling.
        kwargs: Additional arguments passed to the Rescale class.

    Raises:
        CustomOverflowError: If `clip` does not contain exactly one frame.

    Returns:
        The computed error as a `float`. Lower values indicate a better descale match.
    """

    func = func or get_descale_error

    if clip.num_frames != 1:
        raise CustomOverflowError("Clip must have only one frame", func)

    clip = depth(get_y(clip), 32)

    kernel = ComplexKernel.ensure_obj(kernel, func)

    rs = Rescale(
        clip,
        height,
        kernel,
        width=width,
        upscaler=_point_resize,
        base_height=base_height,
        base_width=base_width,
        crop=crop or CropRel(),
        shift=shift,
        **kwargs,
    )

    rescaled = rs.rescale

    if mask:
        mask = normalize_mask(mask, clip, clip, func=func)

        rescaled = core.std.MaskedMerge(clip, rescaled, mask)

    crops = _norm_border_crops(borders_aware, kernel)

    rescaled = (
        norm_expr([rescaled, clip], getattr(ExprOp, metric_mode.lower())(clip), func=func)
        .std.CropRel(*crops)
        .std.PlaneStats()
    )

    try:
        return get_prop(rescaled, "PlaneStatsAverage", float, func=func)
    finally:
        rs.__vs_del__(-1)


def _norm_border_crops(
    value: bool | int | tuple[LeftCrop, RightCrop, TopCrop, BottomCrop], kernel: Kernel
) -> tuple[LeftCrop, RightCrop, TopCrop, BottomCrop]:
    match value:
        case True:
            return (kernel.kernel_radius, kernel.kernel_radius, kernel.kernel_radius, kernel.kernel_radius)
        case False:
            return (0, 0, 0, 0)
        case int():
            return (value, value, value, value)
        case _:
            return value
