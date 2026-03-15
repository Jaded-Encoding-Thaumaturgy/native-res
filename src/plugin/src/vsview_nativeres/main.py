from __future__ import annotations

from concurrent.futures import Future
from itertools import zip_longest
from logging import getLogger
from math import ceil, floor
from typing import TYPE_CHECKING, cast

import numpy as np
import vapoursynth as vs
from jetpytools import fallback
from PySide6.QtCore import QSignalBlocker, QTimer
from PySide6.QtGui import QPalette, QResizeEvent, QShowEvent, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from vskernels import ComplexKernel
from vstools import get_h, get_w
from vsview.api import (
    Accordion,
    IconName,
    IconReloadMixin,
    PluginAPI,
    PluginSettings,
    SegmentedControl,
    VideoOutputProxy,
    WidgetPluginBase,
    run_in_background,
    run_in_loop,
)

from nativeres.constants import HIGH_RATE, LOW_RATE
from nativeres.funcs import (
    GetNativeResult,
    MetricMode,
    NpFloatArray1D,
    get_dct_distribution,
    getnative,
    getscaler,
    resolve_kernel,
)
from nativeres.kernels import default_kernels

from .components import GetNativeImportList, ProgressBar, TabContainer
from .settings import GlobalSettings, LocalSettings
from .utils import warmup_plots

if TYPE_CHECKING:
    # Lazy import to avoid long startup times because of QChart
    # Import is done in warmup_plots
    from .plotting import CustomFrequencyPlotWidget, CustomRescalePlotWidget

logger = getLogger(__name__)


class GetNativeTab(TabContainer, IconReloadMixin):
    def __init__(
        self,
        parent: QWidget,
        api: PluginAPI,
        settings: PluginSettings[GlobalSettings, LocalSettings],
    ) -> None:
        super().__init__(parent)

        self.api = api
        self.settings = settings

        self.controls_section = Accordion("Controls", self)
        controls = self.controls_section.add_hlayout()

        self.dimension = SegmentedControl(["Width", "Height"], self.controls_section)
        self.dimension.current_layout.setContentsMargins(0, 0, 0, 0)
        self.dimension.current_layout.setSpacing(0)
        self.dimension.segmentChanged.connect(self.on_segment_changed)

        controls.addLayout(self.make_vgroup("Dimension", self.dimension, parent=self.controls_section), 1)

        self.range_min_spin = QSpinBox(self.controls_section, suffix=" px", minimum=0, maximum=99999, singleStep=1)
        self.range_max_spin = QSpinBox(self.controls_section, suffix=" px", minimum=0, maximum=99999, singleStep=1)
        self.range_min_spin.valueChanged.connect(self._on_range_min_changed)
        self.range_max_spin.valueChanged.connect(self._on_range_max_changed)
        range_layout = self.make_vgroup(
            "Range", self.range_min_spin, self.range_max_spin, parent=self.controls_section, stretch=False
        )

        self.step_spin = QDoubleSpinBox(
            self.controls_section,
            decimals=3,
            minimum=0,
            maximum=1,
            stepType=QDoubleSpinBox.StepType.AdaptiveDecimalStepType,
            value=1.0,
        )
        step_layout = self.make_vgroup("Step", self.step_spin, parent=self.controls_section, stretch=False)

        self.range_step_layout = QVBoxLayout()
        self.range_step_layout.addLayout(range_layout)
        self.range_step_layout.addLayout(step_layout)
        self.range_step_layout.addStretch()

        controls.addLayout(self.range_step_layout, 1)

        self.kernels_cb = QComboBox(self.controls_section)
        for k in self.settings.global_.kernels:
            self.kernels_cb.addItem(k.pretty_string, k)
        kernels_layout = self.make_vgroup("Kernel", self.kernels_cb, parent=self.controls_section, stretch=False)
        self.metrics_cb = QComboBox(self.controls_section)
        self.metrics_cb.addItems(MetricMode.__value__.__args__)
        self.metrics_cb.setCurrentText(fallback(self.settings.local_.getnative.last_metric, "MAE"))
        metrics_layout = self.make_vgroup("Metric", self.metrics_cb, parent=self.controls_section, stretch=False)

        self.kernels_metrics_layout = QVBoxLayout()
        self.kernels_metrics_layout.addLayout(kernels_layout)
        self.kernels_metrics_layout.addSpacing(self.range_max_spin.height() + 4)
        self.kernels_metrics_layout.addLayout(metrics_layout)
        self.kernels_metrics_layout.addStretch()

        controls.addLayout(self.kernels_metrics_layout, 1)

        self.import_btn = QPushButton("Import...", self)
        self.imported_results = GetNativeImportList(self)
        self.imported_results.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        import_layout = self.make_vgroup(
            "History", self.import_btn, self.imported_results, parent=self.controls_section, stretch=False
        )

        controls.addLayout(import_layout, 1)

        self.calculate_btn = QPushButton("Calculate", self)
        self.calculate_btn.clicked.connect(self.on_calculate_clicked)

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.calculate_btn)
        self.btn_layout.addStretch()

        self.add_section(self.controls_section)
        self.add_section(self.btn_layout)

        self.progress_bar = ProgressBar(self)
        self.progress_bar.setFixedHeight(24)

        self.progress_container = QWidget(self)
        progress_layout = QVBoxLayout(self.progress_container)
        progress_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_bar)

        self.plot_stack = QStackedWidget(self)
        self.plot_stack.setFrameShape(QFrame.Shape.StyledPanel)
        self.plot_stack.setFrameShadow(QFrame.Shadow.Sunken)

        self.canvas = QStackedWidget(self)
        self.canvas.setFrameShape(QFrame.Shape.StyledPanel)
        self.canvas.setFrameShadow(QFrame.Shadow.Sunken)
        self.canvas.addWidget(self.progress_container)
        self.canvas.addWidget(self.plot_stack)
        self.canvas.setCurrentWidget(self.progress_container)

        self.progress_container.hide()

        self.add_section(self.canvas, 1)

        self._reload_icons()
        self._setup_default_values()
        self.api.aboutToSaveLocal.connect(self.snapshot_ui_values)
        self.api.globalSettingsChanged.connect(self.on_global_settings_changed)
        self.register_icon_callback(self._reload_icons)

    @property
    def current_dimension(self) -> str:
        return self.dimension.buttons[self.dimension.index].text()

    def _setup_default_values(self) -> None:
        self.dimension.index = fallback(self.settings.local_.getnative.last_dimension, 1)
        self.range_max_spin.setValue(
            fallback(
                self.settings.local_.getnative.last_max_range,
                ceil(self.api.current_voutput.vs_output.clip.height * 0.925),
            )
        )
        self.range_min_spin.setValue(
            fallback(
                self.settings.local_.getnative.last_min_range,
                floor(self.api.current_voutput.vs_output.clip.height * 0.465),
            )
        )
        self.step_spin.setValue(fallback(self.settings.local_.getnative.last_step, 1.0))
        self.kernels_cb.setCurrentText(
            k.pretty_string if (k := self.settings.local_.getnative.last_kernel) else "Bilinear()"
        )
        self.metrics_cb.setCurrentText(fallback(self.settings.local_.getnative.last_metric, "MAE"))

        self.update_limits()

    def snapshot_ui_values(self) -> None:
        self.settings.local_.getnative.last_dimension = self.dimension.index
        self.settings.local_.getnative.last_max_range = self.range_max_spin.value()
        self.settings.local_.getnative.last_min_range = self.range_min_spin.value()
        self.settings.local_.getnative.last_step = self.step_spin.value()
        self.settings.local_.getnative.last_kernel = resolve_kernel(self.kernels_cb.currentText())
        self.settings.local_.getnative.last_metric = self.metrics_cb.currentText()

    def _get_max_dim(self) -> int:
        clip = self.api.current_voutput.vs_output.clip
        return clip.height if self.current_dimension == "Height" else clip.width

    def update_limits(self) -> None:
        max_dim = self._get_max_dim()

        with QSignalBlocker(self.range_min_spin), QSignalBlocker(self.range_max_spin):
            # Absolute limits first
            self.range_min_spin.setMaximum(max_dim - 1)
            self.range_max_spin.setMaximum(max_dim)

            # Then relative limits to maintain min < max
            cur_min = self.range_min_spin.value()
            cur_max = self.range_max_spin.value()

            self.range_min_spin.setMaximum(min(self.range_min_spin.maximum(), cur_max - 1))
            self.range_max_spin.setMinimum(cur_min + 1)

    def _on_range_min_changed(self, value: int) -> None:
        self.range_max_spin.setMinimum(value + 1)

    def _on_range_max_changed(self, value: int) -> None:
        self.range_min_spin.setMaximum(min(value - 1, self._get_max_dim() - 1))

    def on_global_settings_changed(self) -> None:
        kernel = self.kernels_cb.currentText()
        self.kernels_cb.clear()

        for k in self.settings.global_.kernels:
            self.kernels_cb.addItem(k.pretty_string, k)

        self.kernels_cb.setCurrentText(kernel)

        from .plotting import CustomRescalePlotWidget

        for i in range(self.plot_stack.count()):
            if isinstance((plot := self.plot_stack.widget(i)), CustomRescalePlotWidget):
                plot.set_theme(self.settings.global_.get_chart_theme())

    def on_segment_changed(self, index: int) -> None:
        self.settings.local_.getnative.last_dimension = index

        clip = self.api.current_voutput.vs_output.clip

        with QSignalBlocker(self.range_min_spin), QSignalBlocker(self.range_max_spin):
            match self.current_dimension:
                case "Height":
                    self.range_min_spin.setValue(get_h(self.range_min_spin.value(), clip))
                    self.range_max_spin.setValue(get_h(self.range_max_spin.value(), clip))
                    self.update_limits()
                case "Width":
                    self.update_limits()
                    self.range_min_spin.setValue(get_w(self.range_min_spin.value(), clip))
                    self.range_max_spin.setValue(get_w(self.range_max_spin.value(), clip))
                case _:
                    raise ValueError("Invalid dimension")

    def on_calculate_clicked(self) -> None:
        self.calculate_btn.setDisabled(True)

        clip = self.api.current_voutput.vs_output.clip

        start = self.range_min_spin.value()
        stop = self.range_max_spin.value()
        step_f = self.step_spin.value()

        if step_f.is_integer():
            dims = range(start, stop + 1, int(step_f))
            x_label_fmt = "%.0f"
        else:
            num = int((stop - start) / step_f) + 1
            dims = np.linspace(start, start + step_f * (num - 1), num).tolist()
            x_label_fmt = f"%.{str(step_f)[::-1].find('.') + 1}f"

        match dim_mode := self.current_dimension:
            case "Height":
                dimensions = zip_longest([clip.width], dims, fillvalue=clip.width)
            case "Width":
                dimensions = zip_longest(dims, [clip.height], fillvalue=clip.height)
            case _:
                raise ValueError("Invalid dimension")

        frame = self.api.current_frame
        kernel = self.kernels_cb.currentData()
        metric_mode = cast(MetricMode, self.metrics_cb.currentText())

        self.canvas.setCurrentWidget(self.progress_container)
        self.progress_bar.update_progress(fmt="Gathering data %v / %m", value=0)
        self.progress_container.show()

        @run_in_background(name="GetNativeResults")
        def get_results() -> list[GetNativeResult]:
            with self.api.vs_context():
                return getnative(
                    self.api.current_voutput.vs_output.clip,
                    frame,
                    dimensions,
                    kernel,
                    metric_mode=metric_mode,
                    progress_cb=(
                        lambda current, total: self.progress_bar.update_progress(
                            value=current,
                            range=(0, total),
                        )
                    ),
                    func=self.on_calculate_clicked,
                )

        @run_in_loop(return_future=False)
        def on_completed(f: Future[list[GetNativeResult]]) -> None:
            self.progress_bar.reset_progress()
            self.calculate_btn.setEnabled(True)
            if f.exception():
                logger.exception("Failed to get native results")
                return

            results = f.result()

            plot = self.create_rescale_plot(results, kernel, dim_mode, frame, x_label_fmt)
            self.plot_stack.addWidget(plot)
            # TODO: add in history when it's added
            self.plot_stack.setCurrentWidget(plot)
            self.canvas.setCurrentWidget(self.plot_stack)

        future_results = get_results()
        future_results.add_done_callback(on_completed)

    def create_rescale_plot(
        self,
        results: list[GetNativeResult],
        kernel: ComplexKernel,
        dim_mode: str,
        frame: int,
        x_label_fmt: str,
    ) -> CustomRescalePlotWidget:
        from .plotting import CustomRescalePlotWidget

        dims, errors = zip(*results)

        title = f"{kernel.pretty_string} on {dim_mode.lower()} - frame {frame}"
        dims = np.fromiter((getattr(d, dim_mode.lower()) for d in dims), dtype=np.float64)

        plot = CustomRescalePlotWidget(title, dims, errors, dim_mode.title(), self.plot_stack)
        plot.set_theme(self.settings.global_.get_chart_theme())
        plot.axis_x.setLabelFormat(x_label_fmt)

        return plot

    def on_import_btn_clicked(self) -> None:
        # TODO
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Results",
            "",
            "JSON & CSV Files (*.json *.csv)",
        )

        if not files:
            return

    def _reload_icons(self) -> None:
        icon = (IconName.FILE_IMPORT, self.palette().color(QPalette.ColorGroup.Normal, QPalette.ColorRole.ButtonText))
        self.import_btn.setIcon(self.make_icon(icon))


class GetScalerTab(TabContainer):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.section = Accordion("Controls", self)
        controls = self.section.add_hlayout()

        self.target_h = QSpinBox(self.section, suffix=" px")
        self.target_h.setRange(100, 4320)
        self.target_h.setValue(720)
        controls.addLayout(self.make_vgroup("Target Height", self.target_h, parent=self.section))

        self.calculate_btn = QPushButton("Calculate Scaler", self)

        self.table = QTableWidget(self.section)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Kernel", "Error %", "MAE"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(300)

        self.add_section(self.section)
        self.add_section(self.calculate_btn)
        self.add_section(self.table)


class GetFreqTab(TabContainer):
    def __init__(
        self,
        parent: QWidget,
        api: PluginAPI,
        settings: PluginSettings[GlobalSettings, LocalSettings],
    ) -> None:
        super().__init__(parent)

        self.api = api
        self.settings = settings

        self.plot: CustomFrequencyPlotWidget | None = None

        self.calc_timer = QTimer(self, singleShot=True, interval=50)
        self.calc_timer.timeout.connect(lambda: self.on_calculate_clicked() if self.plot else None)

        self.controls_section = Accordion("Controls", self)
        controls = self.controls_section.add_hlayout()

        self.cull_rate_spin = QDoubleSpinBox(
            self.controls_section,
            decimals=2,
            minimum=0.01,
            maximum=10.0,
            value=3.0,
            suffix=" x",
            singleStep=0.1,
            stepType=QDoubleSpinBox.StepType.AdaptiveDecimalStepType,
        )
        self.cull_rate_spin.setToolTip("Cull the sides/top of the frame to focus on the center.")
        self.cull_rate_spin.valueChanged.connect(lambda _: self.calc_timer.start())
        controls.addLayout(self.make_vgroup("Cull Rate", self.cull_rate_spin, parent=self.controls_section))

        self.radius_spin = QSpinBox(self.controls_section, minimum=1, maximum=100, value=50, suffix=" px")
        self.radius_spin.setToolTip("Radius for finding peaks/spikes in the frequency plot")
        self.radius_spin.valueChanged.connect(self.on_radius_changed)
        controls.addLayout(self.make_vgroup("Radius", self.radius_spin, parent=self.controls_section))

        self.calculate_btn = QPushButton("Calculate", self)
        self.calculate_btn.clicked.connect(self.on_calculate_clicked)
        calculate_layout = self.make_vgroup("", parent=self.controls_section)
        calculate_layout.addWidget(self.calculate_btn)
        controls.addLayout(calculate_layout)

        controls.addStretch()

        self.add_section(self.controls_section)

        self.canvas = QStackedWidget(self)
        self.canvas.setFrameShape(QFrame.Shape.StyledPanel)
        self.canvas.setFrameShadow(QFrame.Shadow.Sunken)
        self.canvas.setMinimumHeight(400)

        self._last_request_id = 0

        self.add_section(self.canvas, 1)

        self.api.globalSettingsChanged.connect(self.on_global_settings_changed)
        self.api.register_on_destroy(self.on_destroy)

    def on_global_settings_changed(self) -> None:
        if self.plot:
            self.plot.set_theme(
                self.settings.global_.get_chart_theme(),
                hpen_color=self.settings.global_.frequency_width_color,
                vpen_color=self.settings.global_.frequency_height_color,
            )

    def on_destroy(self) -> None:
        self.calc_timer.stop()
        if self.plot:
            self.canvas.removeWidget(self.plot)
            self.plot.deleteLater()
            self.plot = None

    def on_radius_changed(self, value: int) -> None:
        if self.plot:
            self.plot.check_radius = value
            self.plot.set_spikes_h()
            self.plot.set_spikes_v()

    def on_calculate_clicked(self) -> None:
        clip = self.api.current_voutput.vs_output.clip
        frame = self.api.current_frame
        cull_rate = self.cull_rate_spin.value()

        self._last_request_id += 1
        request_id = self._last_request_id

        @run_in_background(name="GetDCTDistribution")
        def get_results() -> tuple[NpFloatArray1D, NpFloatArray1D]:
            with self.api.vs_context():
                return get_dct_distribution(clip, frame, cull_rate)

        @run_in_loop(return_future=False)
        def on_completed(f: Future[tuple[NpFloatArray1D, NpFloatArray1D]]) -> None:
            if request_id != self._last_request_id:
                return

            if e := f.exception():
                logger.error("Failed to get DCT distribution: %s", e)
                return

            results = f.result()
            new_plot = self.create_freq_plot(results, clip, frame)

            if self.plot:
                self.canvas.removeWidget(self.plot)
                self.plot.deleteLater()

            self.canvas.addWidget(new_plot)
            self.canvas.setCurrentWidget(new_plot)
            self.plot = new_plot

        future_results = get_results()
        future_results.add_done_callback(on_completed)

    def create_freq_plot(
        self,
        results: tuple[NpFloatArray1D, NpFloatArray1D],
        clip: vs.VideoNode,
        frame: int,
    ) -> CustomFrequencyPlotWidget:
        from .plotting import CustomFrequencyPlotWidget

        title = f"DCT Distribution - frame {frame}"

        min_val_h, max_val_h = floor(clip.width * LOW_RATE), ceil(clip.width * HIGH_RATE)
        min_val_v, max_val_v = floor(clip.height * LOW_RATE), ceil(clip.height * HIGH_RATE)

        plot = CustomFrequencyPlotWidget(
            title,
            results[0],
            results[1],
            min_val_h,
            max_val_h,
            min_val_v,
            max_val_v,
            self.radius_spin.value(),
            self.canvas,
        )
        plot.set_theme(
            self.settings.global_.get_chart_theme(),
            hpen_color=self.settings.global_.frequency_width_color,
            vpen_color=self.settings.global_.frequency_height_color,
        )

        return plot


class NativeResPlugin(WidgetPluginBase[GlobalSettings, LocalSettings]):
    identifier = "jet_vsview_nativeres"
    display_name = "Native Resolution"

    def __init__(self, parent: QWidget, api: PluginAPI) -> None:
        super().__init__(parent, api)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.tabs = QTabWidget(self, movable=False, tabsClosable=False)

        self.tab_getnative = GetNativeTab(self.tabs, self.api, self.settings)
        self.tabs.addTab(self.tab_getnative, "Get Native")

        self.tab_getscaler = GetScalerTab(self.tabs)
        self.tabs.addTab(self.tab_getscaler, "Get Scaler")

        self.tab_getfreq = GetFreqTab(self.tabs, self.api, self.settings)
        self.tabs.addTab(self.tab_getfreq, "Get Frequencies")

        main_layout.addWidget(self.tabs)

    def resizeEvent(self, event: QResizeEvent, /) -> None:
        super().resizeEvent(event)

        r1 = self.tab_getnative.range_step_layout.geometry()
        r2 = self.tab_getnative.kernels_metrics_layout.geometry()

        if r1.isValid() and r2.isValid():
            self.tab_getnative.calculate_btn.setFixedWidth((r2.right() - r1.left() + 1) // 2)

    def showEvent(self, event: QShowEvent) -> None:
        warmup_plots()
        return super().showEvent(event)

    # Plugin hooks
    def on_current_voutput_changed(self, voutput: VideoOutputProxy, tab_index: int) -> None:
        self.update_ui(voutput)

    def on_current_frame_changed(self, n: int) -> None:
        self.update_ui(self.api.current_voutput)

    @run_in_loop(return_future=False)
    def update_ui(self, voutput: VideoOutputProxy) -> None:
        if self.api.is_playing:
            return

        # GetNative Tab
        self.tab_getnative.update_limits()

        # # GetScaler Tab
        # self.tab_getscaler.target_h.setMaximum(voutput.vs_output.clip.height)
        # self.tab_getscaler.target_w.setMaximum(voutput.vs_output.clip.width)

        # GetFreq Tab
        if self.tab_getfreq.plot:
            self.tab_getfreq.calc_timer.start()

    def on_playback_stopped(self) -> None:
        self.update_ui(self.api.current_voutput)
