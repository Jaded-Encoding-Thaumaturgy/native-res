import csv
import json
from abc import abstractmethod
from collections.abc import Sequence
from logging import getLogger
from typing import Literal

import numpy as np
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QLogValueAxis, QScatterSeries, QValueAxis
from PySide6.QtCore import QMargins, QPointF, Qt
from PySide6.QtGui import (
    QAction,
    QColor,
    QContextMenuEvent,
    QFont,
    QMouseEvent,
    QPainter,
    QPalette,
    QPen,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtSvg import QSvgGenerator
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsTextItem,
    QMenu,
    QWidget,
)
from scipy.signal import argrelextrema

type FloatArray1D = Sequence[float] | np.ndarray[tuple[Literal[1]], np.dtype[np.floating]]

logger = getLogger(__name__)


def get_color_scheme() -> Qt.ColorScheme:
    if not isinstance(app := QApplication.instance(), QApplication):
        raise NotImplementedError("QApplication instance is not available")
    return app.styleHints().colorScheme()


def get_chart_theme() -> QChart.ChartTheme:
    match get_color_scheme():
        case Qt.ColorScheme.Dark:
            return QChart.ChartTheme.ChartThemeDark
        case Qt.ColorScheme.Light:
            return QChart.ChartTheme.ChartThemeLight
        case _:
            raise NotImplementedError("Unsupported color scheme")


class LabelText(QGraphicsTextItem):
    def set_html_text(self, text: str, palette: QPalette) -> None:
        bg = palette.color(QPalette.ColorRole.Window).name()
        fg = palette.color(QPalette.ColorRole.WindowText).name()
        border = palette.color(QPalette.ColorRole.Highlight).name()

        self.setHtml(
            f"<div style='background-color: {bg}; color: {fg}; border: 1px solid {border}; "
            "border-radius: 4px; padding: 4px; font-family: sans-serif;'>"
            f"{text}</div>"
        )


class CustomHorizontalTitle(QGraphicsTextItem):
    def __init__(self, title: str, parent: QGraphicsItem, font: QFont) -> None:
        super().__init__(title, parent)
        self.setZValue(5)
        self.setFont(font)

    def update_from_chart(self, chart: QChart) -> None:
        if not (area := chart.plotArea()).isValid():
            return

        # Position the horizontal Y-axis title in the left margin
        # Center x horizontally in the left margin (area.left())
        # Center y vertically relative to the plot area
        rect = self.boundingRect()
        self.setPos(
            (area.left() - rect.width()) / 2,
            area.top() + (area.height() - rect.height()) / 2,
        )


class BasePlotWidget(QChartView):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        chart = QChart(
            theme=get_chart_theme(),
            title=title,
            margins=QMargins(50, 10, 10, 10),
        )
        super().__init__(chart, parent)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)

        self._is_panning = False
        self._last_mouse_pos = QPointF()

        # Crosshair overlays
        cross_color = self.palette().color(QPalette.ColorRole.ToolTipText)
        cross_color.setAlphaF(0.8)
        cross_pen = QPen(cross_color, 1, Qt.PenStyle.DashLine)
        self.v_line = QGraphicsLineItem(chart)
        self.v_line.setPen(cross_pen)
        self.v_line.setZValue(10)

        self.h_line = QGraphicsLineItem(chart)
        self.h_line.setPen(cross_pen)
        self.h_line.setZValue(10)

        # Tooltip-style hover label
        self.label = LabelText(chart)
        self.label.setZValue(11)

        self.set_overlays_visible(False)

        self.menu = QMenu(self)
        self.reset_action = QAction("Reset Zoom", self)
        self.reset_action.triggered.connect(self.reset_zoom)
        self.menu.addAction(self.reset_action)

        self.menu.addSeparator()
        self.export_menu = self.menu.addMenu("Export")

        self.png_action = QAction("Export as PNG", self)
        self.png_action.triggered.connect(self.export_png)
        self.export_menu.addAction(self.png_action)

        self.svg_action = QAction("Export as SVG", self)
        self.svg_action.triggered.connect(self.export_svg)
        self.export_menu.addAction(self.svg_action)

        self.json_action = QAction("Export as JSON", self)
        self.json_action.triggered.connect(self.export_json)
        self.export_menu.addAction(self.json_action)

        self.csv_action = QAction("Export as CSV", self)
        self.csv_action.triggered.connect(self.export_csv)
        self.export_menu.addAction(self.csv_action)

    @property
    def is_panning(self) -> bool:
        return self._is_panning

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        self.menu.exec(event.globalPos())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = True
            self._last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.set_overlays_visible(False)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            # Re-trigger move event to restore crosshair immediately
            self.mouseMoveEvent(event)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_panning:
            pos = event.position()
            delta = pos - self._last_mouse_pos
            self.chart().scroll(-delta.x(), delta.y())
            self._last_mouse_pos = pos
            event.accept()
            return
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() > 0:
            self.chart().zoom(1.25)
        else:
            self.chart().zoom(1 / 1.25)
        event.accept()

    @abstractmethod
    def reset_zoom(self) -> None: ...

    @abstractmethod
    def export_json(self) -> None: ...

    @abstractmethod
    def export_csv(self) -> None: ...

    def export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG Files (*.png)")
        if path:
            pixmap = self.grab()
            pixmap.save(path, "PNG")
            logger.info("Exported PNG to %s", path)

    def export_svg(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export SVG", "", "SVG Files (*.svg)")
        if path:
            generator = QSvgGenerator(
                size=self.size(),
                viewBox=self.rect().toRectF(),
                title=self.chart().title(),
                fileName=path,
            )
            with QPainter(generator) as painter:
                self.render(painter)
            logger.info("Exported SVG to %s", path)

    def set_overlays_visible(self, visible: bool) -> None:
        self.v_line.setVisible(visible)
        self.h_line.setVisible(visible)
        self.label.setVisible(visible)


class RescalePlotWidget(BasePlotWidget):
    def __init__(
        self,
        title: str,
        dims: FloatArray1D,
        errors: FloatArray1D,
        dimension_mode: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self.chart().legend().hide()

        self.dims = np.asarray(dims, np.float64)
        self.errors = np.asarray(errors, np.float64)
        self.dimension_mode = dimension_mode
        self._last_snap_idx = -1

        self.series = QLineSeries(self)
        self.series.setPen(
            QPen(
                self.palette().color(QPalette.ColorRole.BrightText),
                1.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.SquareCap,
                Qt.PenJoinStyle.MiterJoin,
            )
        )
        self.series.setMarkerSize(2.5)
        self.series.setPointsVisible(True)
        if self.dims.size > 0:
            self.series.appendNp(self.dims, self.errors)  # type: ignore[arg-type]
        self.chart().addSeries(self.series)

        # Setup X Axis (Linear)
        self.axis_x = QValueAxis(self)
        self.axis_x.setTitleText(dimension_mode)
        self.axis_x.setLabelFormat("%.0f")
        self.axis_x.setTickCount(11)
        self.chart().addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.series.attachAxis(self.axis_x)

        if self.dims.size > 0:
            x_min, x_max = self.dims.min(), self.dims.max()
            pad = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
            self.initial_x_range = (x_min - pad, x_max + pad)
            self.axis_x.setRange(*self.initial_x_range)
        else:
            self.initial_x_range = (0.0, 10.0)

        # Setup Y Axis (Logarithmic)
        self.axis_y = QLogValueAxis(self, base=10.0, minorTickCount=4)
        self.axis_y.setTitleVisible(False)
        self.chart().addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(self.axis_y)

        if self.errors.size > 0:
            y_min = max(self.errors.min(), 1e-15)
            y_max = self.errors.max()
            self.initial_y_range = (y_min * 0.5, y_max * 2.0)
            self.axis_y.setRange(*self.initial_y_range)
        else:
            self.initial_y_range = (1e-15, 1.0)

        # Custom horizontal Y title
        self.axis_y_title = CustomHorizontalTitle("Error", self.chart(), self.axis_y.titleFont())

        logger.debug("RescalePlotWidget %r initialized", title)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.is_panning:
            return super().mouseMoveEvent(event)

        pos = event.position()
        area = self.chart().plotArea()

        if not area.contains(pos) or self.dims.size <= 0:
            return self.set_overlays_visible(False)

        val = self.chart().mapToValue(pos)

        # Lookup for nearest data point
        idx = np.clip(self.dims.searchsorted(val.x()), 1, self.dims.size - 1)
        best_idx = int(idx if abs(self.dims[idx] - val.x()) < abs(self.dims[idx - 1] - val.x()) else idx - 1)

        if best_idx == self._last_snap_idx and self.v_line.isVisible():
            return

        self._last_snap_idx = best_idx

        x, y = self.dims[best_idx], self.errors[best_idx]
        point = self.chart().mapToPosition(QPointF(x, y))

        # Update crosshairs to snap to the data point
        self.v_line.setLine(point.x(), area.top(), point.x(), area.bottom())
        self.h_line.setLine(area.left(), point.y(), area.right(), point.y())

        self.label.set_html_text(f"<b>{self.dimension_mode}:</b> {x:g}<br><b>Error:</b> {y:.10f}", self.palette())

        # Avoid tooltip occlusion at chart edges
        lx, ly = point.x() + 15, point.y() - 45
        if lx + 160 > area.right():  # 160 is estimated tooltip width
            lx = point.x() - 165
        if ly < area.top():
            ly = point.y() + 15

        self.label.setPos(lx, ly)
        self.set_overlays_visible(True)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.axis_y_title.update_from_chart(self.chart())

    def reset_zoom(self) -> None:
        self.axis_x.setRange(*self.initial_x_range)
        self.axis_y.setRange(*self.initial_y_range)

    def export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON", "", "JSON Files (*.json)")
        if path:
            data = {
                "dimension_mode": self.dimension_mode,
                "data": [{"x": float(x), "y": float(y)} for x, y in zip(self.dims, self.errors)],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info("Exported JSON to %s", path)

    def export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if path:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([f"{self.dimension_mode}", "Error"])
                writer.writerows(zip(self.dims, self.errors))

            logger.info("Exported CSV to %s", path)


            return

        # Position the horizontal Y-axis title in the left margin
        # Center x horizontally in the left margin (area.left())
        # Center y vertically relative to the plot area
        rect = self.axis_y_title.boundingRect()
        self.axis_y_title.setPos(
            (area.left() - rect.width()) / 2,
            area.top() + (area.height() - rect.height()) / 2,
        )
