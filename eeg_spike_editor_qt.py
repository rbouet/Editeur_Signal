import sys
import numpy as np
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.signal import butter, filtfilt, iirnotch

import sys
import numpy as np
import pandas as pd
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.signal import butter, filtfilt, iirnotch


class EEGViewBox(pg.ViewBox):
    """ViewBox custom : molette OK (zoom temporel), drag souris interdit."""
    def __init__(self, editor=None):
        super().__init__(enableMenu=False)
        self.editor = editor

    def wheelEvent(self, ev, axis=None):
        if self.editor is not None:
            self.editor.on_wheel(ev)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        # Interdire tout drag souris
        ev.ignore()


class EEGEditor(QtWidgets.QMainWindow):
    def __init__(self, signals, times, channel_names, markers_df=None,
                 window_sec=20, n_display=20):
        super().__init__()
        self.signals_raw = signals.copy()
        self.signals = signals.copy()
        self.times = times
        self.channel_names = channel_names
        self.markers_df = markers_df

        self.n_channels, self.n_times = signals.shape
        self.fs = 1 / np.mean(np.diff(times))

        self.window_sec = float(window_sec)
        self.n_display = int(n_display)
        self.current_chan_start = 0
        self.gain = 1.0
        self.start_idx = 0

        self.channel_spacing = np.percentile(np.abs(self.signals), 95) * 2
        self.channel_spacing = max(self.channel_spacing, 1e-6)

        self._init_ui()
        self._plot_signals()

    # ---------------- UI ----------------
    def _init_ui(self):
        self.setWindowTitle("EEG Editor")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Plot avec ViewBox custom
        self.view_box = EEGViewBox(editor=self)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box, background='w')
        self.plot_widget.showGrid(x=True, y=False)
        self.plot_widget.setLabel('bottom', 'Temps (s)')
        self.plot_widget.getPlotItem().hideAxis('left')
        self.plot_widget.setMenuEnabled(False)
        layout.addWidget(self.plot_widget)

        # Ascenseur temporel
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.n_times - 1)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider)

        # Controls
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.btn_prev = QtWidgets.QPushButton("<< Channels")
        self.btn_next = QtWidgets.QPushButton("Channels >>")
        self.btn_zoom_in = QtWidgets.QPushButton("+")
        self.btn_zoom_out = QtWidgets.QPushButton("-")
        self.btn_exit = QtWidgets.QPushButton("Exit")

        self.bp_low = QtWidgets.QDoubleSpinBox()
        self.bp_low.setRange(0.1, 200)
        self.bp_low.setValue(1.0)
        self.bp_high = QtWidgets.QDoubleSpinBox()
        self.bp_high.setRange(0.1, 200)
        self.bp_high.setValue(50.0)
        self.btn_bp = QtWidgets.QPushButton("Band-pass")

        self.notch_freq = QtWidgets.QDoubleSpinBox()
        self.notch_freq.setRange(1, 200)
        self.notch_freq.setValue(50.0)
        self.btn_notch = QtWidgets.QPushButton("Notch")

        self.btn_reset = QtWidgets.QPushButton("Reset filtres")

        for w in [self.btn_prev, self.btn_next, self.btn_zoom_in,
                  self.btn_zoom_out,
                  QtWidgets.QLabel("BP low (Hz)"), self.bp_low,
                  QtWidgets.QLabel("BP high (Hz)"), self.bp_high, self.btn_bp,
                  QtWidgets.QLabel("Notch (Hz)"), self.notch_freq, self.btn_notch,
                  self.btn_reset, self.btn_exit]:
            controls.addWidget(w)

        self.btn_prev.clicked.connect(self._prev_channels)
        self.btn_next.clicked.connect(self._next_channels)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self.btn_exit.clicked.connect(self.close)
        self.btn_bp.clicked.connect(self._apply_bandpass)
        self.btn_notch.clicked.connect(self._apply_notch)
        self.btn_reset.clicked.connect(self._reset_filters)

    # ---------------- Molette souris ----------------
    def on_wheel(self, ev):
        delta = ev.delta()
        # Zoom temporel (taille de fenêtre)
        self.window_sec *= 0.9 if delta > 0 else 1.1
        self.window_sec = float(np.clip(self.window_sec, 1.0, 60.0))
        self._plot_signals()

    # ---------------- Clavier ----------------
    def keyPressEvent(self, event):
        step = int(self.fs)  # 1 seconde
        if event.key() == QtCore.Qt.Key_Left:
            self.slider.setValue(max(0, self.slider.value() - step))
        elif event.key() == QtCore.Qt.Key_Right:
            self.slider.setValue(min(self.slider.maximum(), self.slider.value() + step))

    # ---------------- Navigation ----------------
    def _on_slider(self, value):
        self.start_idx = int(value)
        self._plot_signals()

    def _zoom_in(self):
        self.gain = min(self.gain * 1.2, 20)
        self._plot_signals()

    def _zoom_out(self):
        self.gain = max(self.gain / 1.2, 0.05)
        self._plot_signals()

    def _prev_channels(self):
        self.current_chan_start = max(0, self.current_chan_start - self.n_display)
        self._plot_signals()

    def _next_channels(self):
        self.current_chan_start = min(self.n_channels - self.n_display,
                                      self.current_chan_start + self.n_display)
        self._plot_signals()

    # ---------------- Filtres ----------------
    def _apply_bandpass(self):
        low = self.bp_low.value()
        high = self.bp_high.value()
        nyq = self.fs / 2
        if low >= high or high >= nyq:
            QtWidgets.QMessageBox.warning(self, "Erreur filtre",
                                          "Fréquences band-pass invalides.")
            return

        b, a = butter(4, [low, high], btype='band', fs=self.fs)
        self.signals = filtfilt(b, a, self.signals_raw, axis=1)
        self._plot_signals()

    def _apply_notch(self):
        f0 = self.notch_freq.value()
        nyq = self.fs / 2
        if f0 <= 0 or f0 >= nyq:
            QtWidgets.QMessageBox.warning(self, "Erreur filtre",
                                          "Fréquence notch invalide.")
            return

        b, a = iirnotch(f0, 30, self.fs)
        self.signals = filtfilt(b, a, self.signals_raw, axis=1)
        self._plot_signals()

    def _reset_filters(self):
        self.signals = self.signals_raw.copy()
        self._plot_signals()


    # ---------------- Plot ----------------
    def _plot_signals(self):
        self.plot_widget.clear()
        win_len = int(self.window_sec * self.fs)
        end_idx = min(self.start_idx + win_len, self.n_times)

        ch_start = self.current_chan_start
        ch_end = min(ch_start + self.n_display, self.n_channels)

        offset = 0.0
        for ch in range(ch_start, ch_end):
            sig = self.signals[ch, self.start_idx:end_idx] * self.gain
            t = self.times[self.start_idx:end_idx]

            self.plot_widget.plot(t, sig + offset, pen=pg.mkPen('k', width=1))
            label = pg.TextItem(self.channel_names[ch], color='k', anchor=(1, 0.5))
            label.setPos(t[0], offset)
            self.plot_widget.addItem(label)

            if self.markers_df is not None:
                rows = self.markers_df[self.markers_df["channel"] == self.channel_names[ch]]
                for _, r in rows.iterrows():
                    idx = int(r["sample"])
                    if self.start_idx <= idx < end_idx:
                        self.plot_widget.plot([self.times[idx]],
                                              [self.signals[ch, idx] * self.gain + offset],
                                              pen=None,
                                              symbol='star',
                                              symbolBrush='r',
                                              symbolSize=12)

            offset += self.channel_spacing


# ---------------- Example main ----------------

def main():
   
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    editor = EEGEditor(signals, times, channel_names, markers_df)
    editor.show()
    editor.resize(1400, 2500)
    app.exec()


if __name__ == "__main__":
    main()
