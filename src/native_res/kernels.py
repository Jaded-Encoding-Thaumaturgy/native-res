from vskernels import AdobeBicubic, BicubicSharp, Bilinear, ComplexKernel, FFmpegBicubic, Lanczos, Mitchell

default_kernels: tuple[ComplexKernel, ...] = (
    # Bicubic-based
    Mitchell(),  # Bicubic b=0.333, c=0.333
    BicubicSharp(),  # Bicubic b=0.0, c=1.0
    # Bicubic-based but from specific applications
    FFmpegBicubic(),  # Bicubic b=0.0, c=0.6. FFmpeg's swscale
    AdobeBicubic(),  # Bicubic b=0.0, c=0.75. Adobe's "Bicubic" interpolation
    # Bilinear-based
    Bilinear(),
    # Lanczos-based
    Lanczos(taps=2),
    Lanczos(taps=3),
    Lanczos(taps=4),
    Lanczos(taps=5),
)
