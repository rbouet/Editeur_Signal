import sys
import numpy as np
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.signal import butter, filtfilt, iirnotch


class EEGViewBox(pg.ViewBox):
    def __init__(self, parent=None):
        super().__init__(enableMenu=False)
        self.editor = parent

    def wheelEvent(self, ev, axis=None):
        if self.editor is not None:
            self.editor.on_wheel(ev)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        if self.editor is not None:
            self.editor.on_drag(ev)
        ev.accept()

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

        self.window_sec = window_sec
        self.n_display = n_display
        self.current_chan_start = 0
        self.gain = 1.0

        self.start_idx = 0
        self.drag_last_x = None

        self.channel_spacing = np.percentile(np.abs(self.signals), 95) * 2
        self.channel_spacing = max(self.channel_spacing, 1e-6)

        self._init_ui()
        self._plot_signals()

    def _init_ui(self):
        self.setWindowTitle("EEG Editor")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.view_box = EEGViewBox(parent=self)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box, background='w')

        self.plot_widget.showGrid(x=True, y=False)
        self.plot_widget.setLabel('bottom', 'Temps (s)')
        layout.addWidget(self.plot_widget)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.btn_prev = QtWidgets.QPushButton("<< Channels")
        self.btn_next = QtWidgets.QPushButton("Channels >>")
        self.btn_zoom_in = QtWidgets.QPushButton("+")
        self.btn_zoom_out = QtWidgets.QPushButton("-")
        self.btn_exit = QtWidgets.QPushButton("Exit")

        self.btn_bp = QtWidgets.QPushButton("Band-pass")
        self.btn_notch = QtWidgets.QPushButton("Notch")
        self.btn_reset = QtWidgets.QPushButton("Reset filtres")

        for b in [self.btn_prev, self.btn_next, self.btn_zoom_in,
                  self.btn_zoom_out, self.btn_bp, self.btn_notch,
                  self.btn_reset, self.btn_exit]:
            controls.addWidget(b)

        self.btn_prev.clicked.connect(self._prev_channels)
        self.btn_next.clicked.connect(self._next_channels)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self.btn_exit.clicked.connect(self.close)
        self.btn_bp.clicked.connect(self._apply_bandpass)
        self.btn_notch.clicked.connect(self._apply_notch)
        self.btn_reset.clicked.connect(self._reset_filters)

       

    # ---------------- Navigation ----------------

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

    def on_wheel(self, ev):
        delta = ev.delta()
        self.window_sec *= 0.9 if delta > 0 else 1.1
        self.window_sec = np.clip(self.window_sec, 1, 60)
        self._plot_signals()

        delta = ev.delta()
        self.window_sec *= 0.9 if delta > 0 else 1.1
        self.window_sec = np.clip(self.window_sec, 1, 60)
        self._plot_signals()

    def on_drag(self, ev):
        if ev.isStart():
            self.drag_last_x = ev.pos().x()
        elif ev.isFinish():
            pass
        else:
            dx = ev.pos().x() - self.drag_last_x
            shift = int(-dx * 5)
            self.start_idx = int(np.clip(self.start_idx + shift, 0, self.n_times - 1))
            self.drag_last_x = ev.pos().x()
            self._plot_signals()


    # ---------------- Filtres ----------------

    def _apply_bandpass(self):
        b, a = butter(4, [1, 40], fs=self.fs, btype='band')
        self.signals = filtfilt(b, a, self.signals_raw, axis=1)
        self._plot_signals()

    def _apply_notch(self):
        b, a = iirnotch(50, 30, self.fs)
        self.signals = filtfilt(b, a, self.signals_raw, axis=1)
        self._plot_signals()

    def _reset_filters(self):
        self.signals = self.signals_raw.copy()
        self._plot_signals()

    # ---------------- Plot ----------------

    def _plot_signals(self):
        self.plot_widget.clear()
        end_idx = self.start_idx + int(self.window_sec * self.fs)
        end_idx = min(end_idx, self.n_times)

        ch_start = self.current_chan_start
        ch_end = min(ch_start + self.n_display, self.n_channels)

        offset = 0
        for ch in range(ch_start, ch_end):
            sig = self.signals[ch, self.start_idx:end_idx] * self.gain
            t = self.times[self.start_idx:end_idx]

            self.plot_widget.plot(t, sig + offset, pen=pg.mkPen('k', width=1))
            label = pg.TextItem(self.channel_names[ch], color='k', anchor=(1, 0.5))
            label.setPos(t[0], offset)
            self.plot_widget.addItem(label)

            # Markers
            if self.markers_df is not None:
                ch_name = self.channel_names[ch]
                rows = self.markers_df[self.markers_df["channel"] == ch_name]
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
    app.exec()


if __name__ == "__main__":
    main()
