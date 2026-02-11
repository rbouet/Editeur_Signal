import sys
import numpy as np
import pandas as pd
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.signal import butter, filtfilt, iirnotch


# ---------------- ViewBox custom ----------------
class EEGViewBox(pg.ViewBox):
    def __init__(self, editor=None):
        super().__init__(enableMenu=False)
        self.editor = editor

    def wheelEvent(self, ev, axis=None):
        if self.editor is not None:
            self.editor.on_wheel(ev)
        ev.accept()

    def mousePressEvent(self, ev):
        if self.editor is not None:
            self.editor.on_mouse_press(ev)
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self.editor is not None:
            self.editor.on_mouse_move(ev)
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if self.editor is not None:
            self.editor.on_mouse_release(ev)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        ev.ignore()


# ---------------- EEG Editor ----------------
class EEGEditor(QtWidgets.QMainWindow):
    def __init__(self, signals, times, channel_names, markers_df=None):
        super().__init__()

        self.signals_raw = signals.copy()
        self.signals = signals.copy()
        self.times = times
        self.channel_names = channel_names
        self.markers_df = markers_df

        self.n_channels, self.n_times = signals.shape
        self.fs = 1 / np.mean(np.diff(times))

        self.window_sec = 20.0
        self.n_display = 20
        self.current_chan_start = 0
        self.start_idx = 0
        self.gain = 1.0
        self.channel_spacing = np.percentile(np.abs(signals), 95) * 3

        self.rm_mode = False
        self.dragging = False
        self.drag_t0 = None
        self.selection_item = None
        self._undo_stack = []

        self._init_ui()
        self._plot_signals()

    # ---------------- UI ----------------
    def _init_ui(self):
        self.setWindowTitle("EEG Spike Editor")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.view_box = EEGViewBox(editor=self)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box, background='#F0F0F0')
        self.plot_widget.showGrid(x=True, y=False)
        self.plot_widget.setLabel('bottom', 'Temps (s)')
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.getAxis('left').setTicks(self._make_channel_ticks())
        layout.addWidget(self.plot_widget)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.n_times - 1)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider)

        controls = QtWidgets.QGridLayout()
        layout.addLayout(controls)

        self.btn_plus = QtWidgets.QPushButton("+")
        self.btn_minus = QtWidgets.QPushButton("-")
        self.btn_prev = QtWidgets.QPushButton("Chan Prev")
        self.btn_next = QtWidgets.QPushButton("Chan Next")
        self.btn_rm = QtWidgets.QPushButton("rm Spike")
        self.btn_rm.setCheckable(True)
        self.btn_exit = QtWidgets.QPushButton("Exit")
        self.btn_save = QtWidgets.QPushButton("Save mk")
        self.btn_undo = QtWidgets.QPushButton("Undo")

        self.bp_low = QtWidgets.QDoubleSpinBox()
        self.bp_high = QtWidgets.QDoubleSpinBox()
        self.bp_low.setValue(1.0)
        self.bp_high.setValue(50.0)
        self.bp_low.setSuffix(" Hz")
        self.bp_high.setSuffix(" Hz")

        self.notch_freq = QtWidgets.QDoubleSpinBox()
        self.notch_freq.setValue(50.0)
        self.notch_freq.setSuffix(" Hz")

        self.btn_bp = QtWidgets.QPushButton("Apply Band-pass")
        self.btn_notch = QtWidgets.QPushButton("Apply Notch")

        controls.addWidget(self.btn_plus, 0, 0)
        controls.addWidget(self.btn_minus, 0, 1)
        controls.addWidget(self.btn_prev, 0, 3)
        controls.addWidget(self.btn_next, 0, 4)
        controls.addWidget(self.btn_exit, 3, 5)

        controls.addWidget(self.btn_rm, 3, 0)
        controls.addWidget(self.btn_undo, 3, 1)
        controls.addWidget(self.btn_save, 3, 2)

        label = QtWidgets.QLabel("BP low")
        label.setAlignment(QtCore.Qt.AlignRight)  # alignement à droite
        controls.addWidget(label, 1, 0)
        controls.addWidget(self.bp_low, 1, 1)
        label = QtWidgets.QLabel("BP high")
        label.setAlignment(QtCore.Qt.AlignRight)  # alignement à droite
        controls.addWidget(label, 2, 0)
        controls.addWidget(self.bp_high, 2, 1)
        controls.addWidget(self.btn_bp, 1, 2)

#        controls.addWidget(QtWidgets.QLabel("Notch"), 1, 3)
        controls.addWidget(self.notch_freq, 1, 3)
        controls.addWidget(self.btn_notch, 1, 4)

        self.btn_plus.clicked.connect(self._zoom_in)
        self.btn_minus.clicked.connect(self._zoom_out)
        self.btn_prev.clicked.connect(self._prev_channels)
        self.btn_next.clicked.connect(self._next_channels)
        self.btn_rm.clicked.connect(self._toggle_rm_mode)
        self.btn_undo.clicked.connect(self._undo_last_removal)
        self.btn_exit.clicked.connect(self.close)
        self.btn_bp.clicked.connect(self._apply_bandpass)
        self.btn_notch.clicked.connect(self._apply_notch)
        self.btn_save.clicked.connect(self._save_markers)

    # ---------------- Zoom ----------------
    def _zoom_in(self):
        self.gain *= 1.2
        self._plot_signals()

    def _zoom_out(self):
        self.gain /= 1.2
        self._plot_signals()

    # ---------------- Channel navigation ----------------
    def _prev_channels(self):
        self.current_chan_start = max(0, self.current_chan_start - self.n_display)
        self._plot_signals()

    def _next_channels(self):
        self.current_chan_start = min(self.n_channels - self.n_display,
                                      self.current_chan_start + self.n_display)
        self._plot_signals()

    # ---------------- Mouse ----------------
    def on_mouse_press(self, ev):
        if not self.rm_mode:
            return
        pos = ev.scenePos()
        mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
        self.drag_t0 = mouse_point.x()
        self.dragging = True
        self.selection_item = pg.LinearRegionItem(values=(self.drag_t0, self.drag_t0),
                                                 brush=(255, 0, 0, 40))
        self.plot_widget.addItem(self.selection_item)

    def on_mouse_move(self, ev):
        if self.rm_mode and self.dragging:
            pos = ev.scenePos()
            mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
            self.selection_item.setRegion((self.drag_t0, mouse_point.x()))

    def on_mouse_release(self, ev):
        if not self.rm_mode or not self.dragging:
            return
        t0, t1 = self.selection_item.getRegion()
        self._remove_markers_in_window(t0, t1)
        self.plot_widget.removeItem(self.selection_item)
        self.selection_item = None
        self.dragging = False

    # ---------------- rm Spike ----------------
    def _toggle_rm_mode(self):
        self.rm_mode = self.btn_rm.isChecked()
        self.btn_rm.setStyleSheet("background-color: red; color: white;" if self.rm_mode else "")

    def _remove_markers_in_window(self, t0, t1):
        if self.markers_df is None or len(self.markers_df) == 0:
            return

        # Sauvegarde pour undo
        self._undo_stack.append(self.markers_df.copy())

        s0, s1 = sorted([int(t0 * self.fs), int(t1 * self.fs)])

        visible_channels = self.channel_names[
            self.current_chan_start : self.current_chan_start + self.n_display
        ]

        mask_time = (self.markers_df["sample"] >= s0) & \
                    (self.markers_df["sample"] <= s1)

        mask_chan = self.markers_df["channel"].isin(visible_channels)

        mask_delete = mask_time & mask_chan

        self.markers_df = self.markers_df[~mask_delete].reset_index(drop=True)
        self._plot_signals()


    # ---------------- Save ----------------
    def _save_markers(self):
        if self.markers_df is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save markers", "", "Text (*.txt)")
        if path:
            out = self.markers_df.rename(columns={"sample": "sample_index"})
            out.to_csv(path, sep="\t", index=False)

    # ---------------- Filters ----------------
    def _apply_bandpass(self):
        low, high = self.bp_low.value(), self.bp_high.value()
        b, a = butter(4, [low / (self.fs / 2), high / (self.fs / 2)], btype='band')
        self.signals = filtfilt(b, a, self.signals_raw, axis=1)
        self._plot_signals()

    def _apply_notch(self):
        f0 = self.notch_freq.value()
        b, a = iirnotch(f0, 30, self.fs)
        self.signals = filtfilt(b, a, self.signals, axis=1)
        self._plot_signals()

    # ---------------- Navigation ----------------
    def on_wheel(self, ev):
        self.window_sec *= 0.9 if ev.delta() > 0 else 1.1
        self.window_sec = float(np.clip(self.window_sec, 1, 60))
        self._plot_signals()

    def _on_slider(self, value):
        self.start_idx = value
        self._plot_signals()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Right:
            self.slider.setValue(min(self.start_idx + int(self.fs), self.n_times - 1))
        elif event.key() == QtCore.Qt.Key_Left:
            self.slider.setValue(max(self.start_idx - int(self.fs), 0))

    def _undo_last_removal(self):
        if len(self._undo_stack) == 0:
            return
        self.markers_df = self._undo_stack.pop()
        self._plot_signals()

    # ---------------- Plot ----------------
    def _make_channel_ticks(self):
        ticks = []
        offset = 0
        for i in range(self.current_chan_start,
                       min(self.current_chan_start + self.n_display, self.n_channels)):
            ticks.append((offset, self.channel_names[i]))
            offset += self.channel_spacing
        return [ticks]

    def _plot_signals(self):
        self.plot_widget.clear()
        self.plot_item.getAxis('left').setTicks(self._make_channel_ticks())

        win_len = int(self.window_sec * self.fs)
        end_idx = min(self.start_idx + win_len, self.n_times)

        offset = 0
        for ch in range(self.current_chan_start,
                        min(self.current_chan_start + self.n_display, self.n_channels)):
            sig = self.signals[ch, self.start_idx:end_idx] * self.gain
            t = self.times[self.start_idx:end_idx]
            self.plot_widget.plot(t, sig + offset, pen=pg.mkPen('k'))

            if self.markers_df is not None:
                rows = self.markers_df[self.markers_df["channel"] == self.channel_names[ch]]
                for _, r in rows.iterrows():
                    idx = int(r["sample"])
                    if self.start_idx <= idx < end_idx:
                        self.plot_widget.plot([self.times[idx]],
                                              [self.signals[ch, idx] * self.gain + offset],
                                              pen=None, symbol='star',
                                              symbolBrush='r', symbolSize=12)
            offset += self.channel_spacing


