import sys
import numpy as np
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg


class EEGEditor(QtWidgets.QMainWindow):
    def __init__(self, signals, times, channel_names,
                 window_sec=20, n_display=20):
        super().__init__()

        self.signals = signals
        self.times = times
        self.channel_names = channel_names

        self.n_channels, self.n_times = signals.shape
        self.window_sec = window_sec
        self.n_display = n_display

        self.fs = 1 / np.mean(np.diff(times))

        self.current_time = times[0]
        self.current_chan_start = 0
        self.gain = 1.0

        # ✅ AJOUT ICI
        self.channel_spacing = np.percentile(np.abs(self.signals), 95) * 2

        self._init_ui()
        self._plot_signals()


    def _init_ui(self):
        self.setWindowTitle("EEG Signal Editor (Pro)")
        self.resize(1200, 800)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel("bottom", "Temps", units="s")
        layout.addWidget(self.plot_widget)

        # Controls
        ctrl_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl_layout)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.n_times - int(self.window_sec * self.fs) - 1)
        self.slider.valueChanged.connect(self._on_scroll)
        ctrl_layout.addWidget(QtWidgets.QLabel("Temps"))
        ctrl_layout.addWidget(self.slider)

        self.btn_zoom_in = QtWidgets.QPushButton("+")
        self.btn_zoom_out = QtWidgets.QPushButton("-")
        self.btn_next = QtWidgets.QPushButton("Canaux +")
        self.btn_prev = QtWidgets.QPushButton("Canaux -")

        self.btn_zoom_in.clicked.connect(self._zoom_in)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        self.btn_next.clicked.connect(self._next_channels)
        self.btn_prev.clicked.connect(self._prev_channels)

        ctrl_layout.addWidget(self.btn_zoom_in)
        ctrl_layout.addWidget(self.btn_zoom_out)
        ctrl_layout.addWidget(self.btn_prev)
        ctrl_layout.addWidget(self.btn_next)

    def _plot_signals(self):
        self.plot_widget.clear()

        start_idx = self.slider.value()
        end_idx = start_idx + int(self.window_sec * self.fs)
        end_idx = min(end_idx, self.n_times)

        ch_start = self.current_chan_start
        ch_end = min(ch_start + self.n_display, self.n_channels)

        offset = 0

        for ch in range(ch_start, ch_end):
            sig = self.signals[ch, start_idx:end_idx] * self.gain
            t = self.times[start_idx:end_idx]

            self.plot_widget.plot(t, sig + offset, pen=pg.mkPen(width=1))

            label = pg.TextItem(self.channel_names[ch], anchor=(1, 0.5))
            label.setPos(t[0], offset)
            self.plot_widget.addItem(label)

            offset += self.channel_spacing


    def _on_scroll(self, value):
        self._plot_signals()

    def _zoom_in(self):
        self.gain *= 1.2
        self._plot_signals()

    def _zoom_out(self):
        self.gain /= 1.2
        self._plot_signals()

    def _next_channels(self):
        self.current_chan_start += self.n_display
        if self.current_chan_start >= self.n_channels:
            self.current_chan_start = self.n_channels - self.n_display
        self.current_chan_start = max(0, self.current_chan_start)
        self._plot_signals()

    def _prev_channels(self):
        self.current_chan_start -= self.n_display
        if self.current_chan_start < 0:
            self.current_chan_start = 0
        self._plot_signals()


def main():
    fs = 250
    duration = 120
    times = np.arange(0, duration, 1/fs)

    n_channels = 64
    signals = np.array([
        np.sin(2 * np.pi * (i + 1) * times) + 0.1 * np.random.randn(len(times))
        for i in range(n_channels)
    ])

    channel_names = [f"Ch{i+1:02d}" for i in range(n_channels)]

    app = QtWidgets.QApplication(sys.argv)
    editor = EEGEditor(signals, times, channel_names,
                      window_sec=20, n_display=20)
    editor.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
