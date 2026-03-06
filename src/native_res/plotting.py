import csv
import json
from collections.abc import Sequence
from logging import getLogger
from pathlib import Path

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, SVGExporter
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMainWindow, QStyle, QToolBar, QVBoxLayout, QWidget

logger = getLogger(__name__)


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

        self.plot_widget = pg.PlotWidget(self, title=f"Error vs {dimension_mode}")
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

        # Toolbar
        toolbar = QToolBar(
            "Actions",
            self,
            movable=False,
            iconSize=QSize(20, 20),
            toolButtonStyle=Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, toolbar)

        save_image_action = QAction(
            "Save Image",
            self,
            icon=self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
        )
        save_image_action.triggered.connect(self._on_save_image)
        toolbar.addAction(save_image_action)

        export_data_action = QAction(
            "Export Data",
            self,
            icon=self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
        )
        export_data_action.triggered.connect(self._on_export_data)
        toolbar.addAction(export_data_action)

    def _on_save_image(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot as...",
            "",
            "PNG image (*.png);;SVG vector (*.svg);;All files (*.*)",
        )

        if not path_str:
            return

        path = Path(path_str)

        try:
            if path.suffix == ".svg":
                exporter = SVGExporter(self.plot_widget.plotItem)
            else:
                exporter = ImageExporter(self.plot_widget.plotItem)

            exporter.export(path_str)
            logger.info("Saved figure to %s", path_str)
        except Exception:
            logger.exception("Failed to save figure")

    def _on_export_data(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export data as...",
            "",
            "CSV file (*.csv);;JSON file (*.json);;All files (*.*)",
        )

        if path:
            try:
                self._write_data(path)
            except Exception:
                logger.exception("Failed to export data:")

    def _write_data(self, path_str: str) -> None:
        path = Path(path_str)

        if path.suffix == ".csv":
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["dimension", "error"])

                for d, e in zip(self.dims, self.errors):
                    writer.writerow([d, e])

            logger.info("Exported CSV to %s", path)

        elif path.suffix == ".json":
            payload = json.dumps(
                {"dimension_mode": self.dimension_mode, "dims": list(self.dims), "errors": list(self.errors)},
                indent=2,
            )
            path.write_text(payload, encoding="utf-8")
            logger.info("Exported JSON to %s", path)
