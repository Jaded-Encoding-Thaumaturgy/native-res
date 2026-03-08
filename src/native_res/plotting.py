from collections.abc import Sequence
from contextlib import suppress
from importlib.util import find_spec
from logging import getLogger

import pyqtgraph as pg
from pyqtgraph.exporters import Exporter, MatplotlibExporter
from PySide6.QtWidgets import QMainWindow, QStyle, QVBoxLayout, QWidget

logger = getLogger(__name__)

if find_spec("matplotlib") is None:
    # Remove MatplotlibExporter from pyqtgraph's list so it doesn't show in the Export dialog
    with suppress(ValueError):
        Exporter.Exporters.remove(MatplotlibExporter)


class StandalonePlotWindow(QMainWindow):
    def __init__(self, title: str, dims: Sequence[float], errors: Sequence[float], dimension_mode: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        self.resize(900, 500)

        self.dims = dims
        self.errors = errors
        self.dimension_mode = dimension_mode

        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget(self, title=f"Error vs {dimension_mode}", background=None)  # pyright: ignore[reportArgumentType]
        self.plot_widget.setLogMode(y=True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Error")
        self.plot_widget.setLabel("bottom", dimension_mode)
        self.plot_widget.plot(
            list(dims),
            list(errors),
            pen=pg.mkPen(width=1.5),
            symbol="o",
            symbolSize=5,
        )

        main_layout.addWidget(self.plot_widget)
