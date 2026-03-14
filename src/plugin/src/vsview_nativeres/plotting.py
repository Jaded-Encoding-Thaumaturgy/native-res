from abc import abstractmethod
from typing import Literal

from PySide6.QtCharts import QChart
from PySide6.QtCore import QIODevice, Qt
from PySide6.QtGui import QColor, QImage, QPen
from PySide6.QtWidgets import QWidget

from nativeres.plotting import BasePlotWidget, FrequencyPlotWidget, RescalePlotWidget, get_chart_theme


class CustomBasePlotWidget(BasePlotWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)

        self._theme: Literal[False] | QChart.ChartTheme = False

        self.addActions(self.menu.actions())
        for act in self.menu.actions():
            act.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)

        self.reset_action.setShortcut("Esc")

    def render_to_image(self) -> QImage:
        self.set_background_visible(True)
        image = super().render_to_image()
        self.set_background_visible(False)
        return image

    def render_to_svg(self, file: str | None = None, output: QIODevice | None = None) -> None:
        self.set_background_visible(True)
        super().render_to_svg(file, output)
        self.set_background_visible(False)

    @abstractmethod
    def set_theme(self, theme: Literal[False] | QChart.ChartTheme) -> None: ...

    def set_background_visible(self, visible: bool) -> None:
        if not visible and self._theme is not False:
            return
        self.chart().setBackgroundVisible(visible)
        self.viewport().setStyleSheet("background: transparent; border: none;" if not visible else "")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, not visible)


class CustomRescalePlotWidget(RescalePlotWidget, CustomBasePlotWidget):
    def set_theme(self, theme: Literal[False] | QChart.ChartTheme) -> None:
        self._theme = theme
        # Force changing theme
        self.chart().setTheme(QChart.ChartTheme(~self.chart().theme().value % 8))
        if theme is not False:
            self.set_background_visible(True)
            self.chart().setTheme(theme)
        else:
            self.set_background_visible(False)
            self.chart().setTheme(get_chart_theme())
            self.series.setPen(self.default_series_pen)


class CustomFrequencyPlotWidget(FrequencyPlotWidget, CustomBasePlotWidget):
    def set_theme(
        self,
        theme: Literal[False] | QChart.ChartTheme,
        *,
        hpen_color: QColor | None = None,
        vpen_color: QColor | None = None,
        gray_pen_color: QColor | None = None,
    ) -> None:
        self._theme = theme
        # Force changing theme
        self.chart().setTheme(QChart.ChartTheme(~self.chart().theme().value % 8))
        if theme is not False:
            self.set_background_visible(True)
            self.chart().setTheme(theme)
            self.set_pen(self.series_h.pen(), self.series_v.pen())
        else:
            self.set_background_visible(False)
            self.chart().setTheme(get_chart_theme())
            self.set_pen(
                QPen(hpen_color, 1) if hpen_color is not None else self.H_PEN,
                QPen(vpen_color, 1) if vpen_color is not None else self.V_PEN,
                QPen(gray_pen_color, 1) if gray_pen_color is not None else self.GRAY_PEN,
            )

        self.apply_focus()

    def set_pen(
        self,
        hpen: QPen | None = None,
        vpen: QPen | None = None,
        gray_pen: QPen | None = None,
    ) -> None:
        if hpen is not None:
            self.h_pen = hpen
        if vpen is not None:
            self.v_pen = vpen
        if gray_pen is not None:
            self.gray_pen = gray_pen

        self.apply_focus()
