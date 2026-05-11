import sys

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from scipy.signal import butter, filtfilt, find_peaks, iirnotch


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


class EEGEditor(QtWidgets.QMainWindow):
    def __init__(
        self,
        signals,
        times,
        channel_names,
        markers_df=None,
        window_sec=20,
        n_display=20,
    ):
        super().__init__()

        self.signals_raw = np.asarray(signals).copy()
        self.signals = np.asarray(signals).copy()
        self.times = np.asarray(times)
        self.channel_names = list(channel_names)
        self.markers_df = self._normalize_markers(markers_df)

        self.n_channels, self.n_times = self.signals.shape
        self.fs = 1 / np.mean(np.diff(self.times))

        self.window_sec = window_sec
        self.n_display = min(n_display, self.n_channels)
        self.current_chan_start = 0
        self.start_idx = 0
        self.gain = 1.0
        self.channel_spacing = max(np.percentile(np.abs(self.signals), 95) * 3, 1e-12)

        self.add_mode = False
        self.add_train_mode = False
        self.rm_mode = False
        self.dragging = False
        self.drag_t0 = None
        self.drag_y0 = None
        self.selection_item = None
        self._mouse_down_scene_pos = None
        self._undo_stack = []

        self.curves = {}
        self.spike_items = {}

        self._init_ui()
        self._plot_signals()

    def _init_ui(self):
        self.setWindowTitle("Edit_signal - sEEG Spike Editor")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.view_box = EEGViewBox(editor=self)
        self.plot_widget = pg.PlotWidget(viewBox=self.view_box, background="#F0F0F0")
        self.plot_widget.showGrid(x=True, y=False)
        self.plot_widget.setLabel("bottom", "Temps (s)")
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.getAxis("left").setTicks(self._make_channel_ticks())
        layout.addWidget(self.plot_widget)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.n_times - 1))
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider)

        controls = QtWidgets.QGridLayout()
        layout.addLayout(controls)

        self.btn_plus = QtWidgets.QPushButton("+")
        self.btn_minus = QtWidgets.QPushButton("-")
        self.btn_prev = QtWidgets.QPushButton("Chan Prev")
        self.btn_next = QtWidgets.QPushButton("Chan Next")
        self.btn_add = QtWidgets.QPushButton("Add Spike")
        self.btn_add.setCheckable(True)
        self.btn_rm = QtWidgets.QPushButton("Rm Spike")
        self.btn_rm.setCheckable(True)
        self.btn_exit = QtWidgets.QPushButton("Exit")
        self.btn_save = QtWidgets.QPushButton("Save mk")
        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_add_train = QtWidgets.QPushButton("Add train")
        self.btn_add_train.setCheckable(True)
        self.bp_std_deriv_train = QtWidgets.QDoubleSpinBox()
        self.bp_std_deriv_train.setValue(0.5)

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

        controls.addWidget(self.btn_add, 3, 0)
        controls.addWidget(self.btn_rm, 3, 1)
        controls.addWidget(self.btn_undo, 3, 2)
        controls.addWidget(self.btn_save, 3, 3)

        controls.addWidget(self.btn_add_train, 4, 0)
        controls.addWidget(self.bp_std_deriv_train, 4, 1)

        controls.addWidget(QtWidgets.QLabel("BP low"), 1, 0)
        controls.addWidget(self.bp_low, 1, 1)
        controls.addWidget(QtWidgets.QLabel("BP high"), 2, 0)
        controls.addWidget(self.bp_high, 2, 1)
        controls.addWidget(self.btn_bp, 1, 2)

        controls.addWidget(self.notch_freq, 1, 3)
        controls.addWidget(self.btn_notch, 1, 4)

        self.btn_plus.clicked.connect(self._zoom_in)
        self.btn_minus.clicked.connect(self._zoom_out)
        self.btn_prev.clicked.connect(self._prev_channels)
        self.btn_next.clicked.connect(self._next_channels)
        self.btn_add.clicked.connect(self._toggle_add_mode)
        self.btn_rm.clicked.connect(self._toggle_rm_mode)
        self.btn_undo.clicked.connect(self._undo_last_action)
        self.btn_exit.clicked.connect(self._exit_app)
        self.btn_bp.clicked.connect(self._apply_bandpass)
        self.btn_notch.clicked.connect(self._apply_notch)
        self.btn_save.clicked.connect(self._save_markers)
        self.btn_add_train.clicked.connect(self._toggle_add_train_mode)

    def _zoom_in(self):
        self.gain *= 1.2
        self._plot_signals()

    def _zoom_out(self):
        self.gain /= 1.2
        self._plot_signals()

    def _prev_channels(self):
        self.current_chan_start = max(0, self.current_chan_start - self.n_display)
        self._plot_signals()

    def _next_channels(self):
        max_start = max(0, self.n_channels - self.n_display)
        self.current_chan_start = min(max_start, self.current_chan_start + self.n_display)
        self._plot_signals()

    def on_mouse_press(self, ev):
        pos = ev.scenePos()
        mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
        self._mouse_down_scene_pos = pos

        if self.add_train_mode:
            self.drag_t0 = mouse_point.x()
            self.drag_y0 = mouse_point.y()
            self.dragging = True
            self.selection_item = pg.RectROI(
                [self.drag_t0, self.drag_y0],
                [0.001, 0.001],
                pen=pg.mkPen((0, 0, 255), width=2),
            )
            self.plot_widget.addItem(self.selection_item)
            return

        if self.add_mode:
            self._add_marker_from_click(mouse_point.x(), mouse_point.y())
            return

        if self.rm_mode:
            self.drag_t0 = mouse_point.x()
            self.drag_y0 = mouse_point.y()
            self.dragging = True
            self.selection_item = pg.RectROI(
                [self.drag_t0, self.drag_y0],
                [0.001, 0.001],
                pen=pg.mkPen((220, 0, 0), width=2),
            )
            self.plot_widget.addItem(self.selection_item)

    def on_mouse_move(self, ev):
        if not self.dragging:
            return

        pos = ev.scenePos()
        mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)

        if self.rm_mode or self.add_train_mode:
            x0, y0 = self.drag_t0, self.drag_y0
            x1, y1 = mouse_point.x(), mouse_point.y()
            self.selection_item.setPos(min(x0, x1), min(y0, y1))
            self.selection_item.setSize((abs(x1 - x0), abs(y1 - y0)))

    def on_mouse_release(self, ev):
        if self.add_train_mode and self.dragging:
            t0, t1, y0, y1 = self._roi_bounds(self.selection_item)
            self._add_train_markers(t0, t1, y0, y1)
            self._clear_selection()
            return

        if self.rm_mode and self.dragging:
            pos = ev.scenePos()
            mouse_point = self.plot_widget.getViewBox().mapSceneToView(pos)
            moved = self._mouse_moved_enough(pos)

            if moved:
                t0, t1, y0, y1 = self._roi_bounds(self.selection_item)
                self._remove_markers_in_rect(t0, t1, y0, y1)
            else:
                self._remove_nearest_marker(mouse_point.x(), mouse_point.y())

            self._clear_selection()

    def _toggle_rm_mode(self):
        self.rm_mode = self.btn_rm.isChecked()
        if self.rm_mode:
            self._set_add_mode(False)
            self._set_add_train_mode(False)
        self.btn_rm.setStyleSheet("background-color: red; color: white;" if self.rm_mode else "")

    def _add_marker_from_click(self, t_click, y_click):
        self._ensure_markers_df()
        selected_channel, selected_idx = self._get_closest_channel(t_click, y_click)

        if selected_channel is None:
            return

        sample = self._time_to_sample(t_click)
        if sample < 0 or sample >= self.n_times:
            return

        window = 15
        s0 = max(0, sample - window)
        s1 = min(self.n_times, sample + window + 1)
        segment = self.signals[selected_idx, s0:s1]

        if len(segment) == 0:
            return

        best_sample = s0 + int(np.argmax(np.abs(segment)))
        self._push_undo()

        new_row = pd.DataFrame({"channel": [selected_channel], "sample": [best_sample]})
        self.markers_df = pd.concat([self.markers_df, new_row], ignore_index=True)
        self._update_spikes_display()

    def _add_train_markers(self, t0, t1, y0, y1):
        self._ensure_markers_df()

        s0 = max(0, self._time_to_sample(min(t0, t1)))
        s1 = min(self.n_times, self._time_to_sample(max(t0, t1)))

        if s1 <= s0:
            return

        best_score = -np.inf
        best_idx = None

        for ch_idx, offset in self._visible_channel_offsets():
            sig = self.signals[ch_idx, s0:s1] * self.gain + offset
            mask = (sig >= min(y0, y1)) & (sig <= max(y0, y1))
            score = np.sum(mask)

            if score > best_score:
                best_score = score
                best_idx = ch_idx

        if best_idx is None or best_score == 0:
            return

        selected_channel = self.channel_names[best_idx]
        segment = self.signals[best_idx, s0:s1]
        noise_level = np.median(np.abs(segment)) / 0.6745

        peaks_pos, _ = find_peaks(
            segment,
            prominence=2 * noise_level,
            distance=max(1, int(0.01 * self.fs)),
        )
        peaks_neg, _ = find_peaks(
            -segment,
            prominence=2 * noise_level,
            distance=max(1, int(0.01 * self.fs)),
        )

        peaks = np.concatenate([peaks_pos, peaks_neg])
        deriv = np.diff(segment)
        slope_threshold = np.std(deriv) * self.bp_std_deriv_train.value()

        good_peaks = []
        for p in peaks:
            if p <= 1 or p >= len(segment) - 2:
                continue

            slope_before = abs(deriv[p - 1])
            slope_after = abs(deriv[p])
            if slope_before > slope_threshold and slope_after > slope_threshold:
                good_peaks.append(p)

        peaks = np.array(good_peaks)
        if len(peaks) == 0:
            return

        peaks_global = s0 + peaks
        self._push_undo()

        new_rows = pd.DataFrame(
            {"channel": [selected_channel] * len(peaks_global), "sample": peaks_global}
        )
        self.markers_df = pd.concat([self.markers_df, new_rows], ignore_index=True)
        self._update_spikes_display()

    def _remove_nearest_marker(self, t_click, y_click):
        if self.markers_df is None or len(self.markers_df) == 0:
            return

        candidates = self._visible_marker_positions()
        if len(candidates) == 0:
            return

        time_tol = max(0.05, self.window_sec * 0.015)
        y_tol = self.channel_spacing * 0.35
        dt = np.abs(candidates["time"].to_numpy() - t_click) / time_tol
        dy = np.abs(candidates["y"].to_numpy() - y_click) / y_tol
        score = dt + dy
        best_pos = int(np.argmin(score))

        if dt[best_pos] > 1 or dy[best_pos] > 1:
            return

        self._push_undo()
        marker_index = candidates.iloc[best_pos]["marker_index"]
        self.markers_df = self.markers_df.drop(index=marker_index).reset_index(drop=True)
        self._update_spikes_display()

    def _remove_markers_in_rect(self, t0, t1, y0, y1):
        if self.markers_df is None or len(self.markers_df) == 0:
            return

        candidates = self._visible_marker_positions()
        if len(candidates) == 0:
            return

        t_min, t_max = sorted([t0, t1])
        y_min, y_max = sorted([y0, y1])
        selected_channels = self._channels_in_y_range(y_min, y_max)

        if len(selected_channels) == 0:
            return

        inside = (
            (candidates["time"] >= t_min)
            & (candidates["time"] <= t_max)
            & (candidates["channel"].isin(selected_channels))
        )
        to_delete = candidates.loc[inside, "marker_index"]

        if len(to_delete) == 0:
            return

        self._push_undo()
        self.markers_df = self.markers_df.drop(index=to_delete).reset_index(drop=True)
        self._update_spikes_display()

    def _get_closest_channel(self, t_click, y_click):
        sample = self._time_to_sample(t_click)
        if sample < 0 or sample >= self.n_times:
            return None, None

        best_dist = np.inf
        best_channel = None
        best_idx = None

        for ch_idx, offset in self._visible_channel_offsets():
            y_signal = self.signals[ch_idx, sample] * self.gain + offset
            dist = abs(y_click - y_signal)

            if dist < best_dist:
                best_dist = dist
                best_channel = self.channel_names[ch_idx]
                best_idx = ch_idx

        return best_channel, best_idx

    def _toggle_add_mode(self):
        self._set_add_mode(self.btn_add.isChecked())
        if self.add_mode:
            self._set_add_train_mode(False)
            self._set_rm_mode(False)

    def _toggle_add_train_mode(self):
        self._set_add_train_mode(self.btn_add_train.isChecked())
        if self.add_train_mode:
            self._set_add_mode(False)
            self._set_rm_mode(False)

    def _set_add_mode(self, enabled):
        self.add_mode = enabled
        self.btn_add.setChecked(enabled)
        self.btn_add.setStyleSheet("background-color: green; color: white;" if enabled else "")

    def _set_add_train_mode(self, enabled):
        self.add_train_mode = enabled
        self.btn_add_train.setChecked(enabled)
        self.btn_add_train.setStyleSheet(
            "background-color: blue; color: white;" if enabled else ""
        )

    def _set_rm_mode(self, enabled):
        self.rm_mode = enabled
        self.btn_rm.setChecked(enabled)
        self.btn_rm.setStyleSheet("background-color: red; color: white;" if enabled else "")

    def _undo_last_action(self):
        if len(self._undo_stack) == 0:
            return
        self.markers_df = self._undo_stack.pop()
        self._plot_signals()

    def _save_markers(self):
        if self.markers_df is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save markers", "", "Text (*.txt)")
        if path:
            out = self.markers_df.rename(columns={"sample": "sample_index"})
            out.to_csv(path, sep="\t", index=False)

    def _apply_bandpass(self):
        low, high = self.bp_low.value(), self.bp_high.value()
        nyquist = self.fs / 2

        if low <= 0 or high >= nyquist or low >= high:
            QtWidgets.QMessageBox.warning(
                self,
                "Filtre invalide",
                f"Choisir 0 < low < high < Nyquist ({nyquist:.1f} Hz).",
            )
            return

        b, a = butter(4, [low / nyquist, high / nyquist], btype="band")
        self.signals = filtfilt(b, a, self.signals_raw, axis=1)
        self._plot_signals()

    def _apply_notch(self):
        f0 = self.notch_freq.value()
        nyquist = self.fs / 2

        if f0 <= 0 or f0 >= nyquist:
            QtWidgets.QMessageBox.warning(
                self,
                "Notch invalide",
                f"Choisir une frequence entre 0 et Nyquist ({nyquist:.1f} Hz).",
            )
            return

        b, a = iirnotch(f0, 30, self.fs)
        self.signals = filtfilt(b, a, self.signals, axis=1)
        self._plot_signals()

    def on_wheel(self, ev):
        self.window_sec *= 0.9 if ev.delta() > 0 else 1.1
        self.window_sec = float(np.clip(self.window_sec, 1, 60))
        self._plot_signals()

    def _on_slider(self, value):
        self.start_idx = int(value)
        self._plot_signals()

    def keyPressEvent(self, event):
        match event.key():
            case QtCore.Qt.Key_Right:
                self.slider.setValue(min(self.start_idx + int(self.fs), self.n_times - 1))
            case QtCore.Qt.Key_Left:
                self.slider.setValue(max(self.start_idx - int(self.fs), 0))
            case QtCore.Qt.Key_Up:
                self._prev_channels()
            case QtCore.Qt.Key_Down:
                self._next_channels()
            case QtCore.Qt.Key_PageDown:
                step = int(self.window_sec * self.fs)
                self.slider.setValue(min(self.start_idx + step, self.n_times - 1))
            case QtCore.Qt.Key_PageUp:
                step = int(self.window_sec * self.fs)
                self.slider.setValue(max(self.start_idx - step, 0))
            case QtCore.Qt.Key_Plus:
                self._zoom_in()
            case QtCore.Qt.Key_Minus:
                self._zoom_out()
            case QtCore.Qt.Key_Z if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                self._undo_last_action()

    def _make_channel_ticks(self):
        ticks = []
        for i, offset in self._visible_channel_offsets():
            ticks.append((offset, self.channel_names[i]))
        return [ticks]

    def _plot_signals(self):
        self.plot_widget.clear()
        self.curves.clear()
        self.spike_items.clear()

        self.plot_item.getAxis("left").setTicks(self._make_channel_ticks())

        win_len = int(self.window_sec * self.fs)
        end_idx = min(self.start_idx + win_len, self.n_times)
        t = self.times[self.start_idx:end_idx]

        for ch_idx, offset in self._visible_channel_offsets():
            sig = self.signals[ch_idx, self.start_idx:end_idx] * self.gain
            scatter = pg.ScatterPlotItem(
                pen=pg.mkPen(color=(255, 0, 0, 70), width=2),
                brush=None,
                symbol="o",
                size=12,
            )
            self.plot_widget.addItem(scatter)
            self.spike_items[ch_idx] = scatter
            self.plot_widget.plot(t, sig + offset, pen=pg.mkPen("k"))

        self._update_spikes_display()

    def _update_spikes_display(self):
        if self.markers_df is None:
            return

        win_len = int(self.window_sec * self.fs)
        end_idx = min(self.start_idx + win_len, self.n_times)

        for ch_idx, offset in self._visible_channel_offsets():
            rows = self.markers_df[self.markers_df["channel"] == self.channel_names[ch_idx]]
            idx = rows["sample"].to_numpy(dtype=int)
            idx = idx[(idx >= self.start_idx) & (idx < end_idx)]

            if len(idx) == 0:
                self.spike_items[ch_idx].setData([], [])
                continue

            x = self.times[idx]
            y = self.signals[ch_idx, idx] * self.gain + offset
            self.spike_items[ch_idx].setData(x, y)

    def _visible_channel_offsets(self):
        offset = 0.0
        stop = min(self.current_chan_start + self.n_display, self.n_channels)
        for ch_idx in range(self.current_chan_start, stop):
            yield ch_idx, offset
            offset += self.channel_spacing

    def _channels_in_y_range(self, y_min, y_max):
        channels = []
        half_spacing = self.channel_spacing / 2

        for ch_idx, offset in self._visible_channel_offsets():
            channel_low = offset - half_spacing
            channel_high = offset + half_spacing
            if y_min <= channel_high and y_max >= channel_low:
                channels.append(self.channel_names[ch_idx])

        return channels

    def _visible_marker_positions(self):
        win_len = int(self.window_sec * self.fs)
        end_idx = min(self.start_idx + win_len, self.n_times)
        rows = []

        for ch_idx, offset in self._visible_channel_offsets():
            ch_name = self.channel_names[ch_idx]
            channel_rows = self.markers_df[self.markers_df["channel"] == ch_name]
            for marker_index, row in channel_rows.iterrows():
                sample = int(row["sample"])
                if sample < self.start_idx or sample >= end_idx:
                    continue
                if sample < 0 or sample >= self.n_times:
                    continue
                rows.append(
                    {
                        "marker_index": marker_index,
                        "channel": ch_name,
                        "sample": sample,
                        "time": self.times[sample],
                        "y": self.signals[ch_idx, sample] * self.gain + offset,
                    }
                )

        return pd.DataFrame(rows)

    def _time_to_sample(self, t):
        return int(np.searchsorted(self.times, t, side="left"))

    def _normalize_markers(self, markers_df):
        if markers_df is None:
            return pd.DataFrame(columns=["channel", "sample"])

        markers = markers_df.copy()
        if "sample" not in markers.columns and "sample_index" in markers.columns:
            markers = markers.rename(columns={"sample_index": "sample"})
        if "channel" not in markers.columns or "sample" not in markers.columns:
            raise ValueError("markers_df doit contenir les colonnes 'channel' et 'sample'.")

        markers = markers[["channel", "sample"]].copy()
        markers["sample"] = markers["sample"].astype(int)
        return markers.reset_index(drop=True)

    def _ensure_markers_df(self):
        if self.markers_df is None:
            self.markers_df = pd.DataFrame(columns=["channel", "sample"])

    def _push_undo(self):
        self._undo_stack.append(self.markers_df.copy())

    def _roi_bounds(self, roi):
        pos = roi.pos()
        size = roi.size()
        t0, t1 = pos.x(), pos.x() + size.x()
        y0, y1 = pos.y(), pos.y() + size.y()
        return t0, t1, y0, y1

    def _mouse_moved_enough(self, release_scene_pos):
        if self._mouse_down_scene_pos is None:
            return False
        delta = release_scene_pos - self._mouse_down_scene_pos
        return (delta.x() ** 2 + delta.y() ** 2) ** 0.5 >= 4

    def _clear_selection(self):
        if self.selection_item is not None:
            self.plot_widget.removeItem(self.selection_item)
        self.selection_item = None
        self.dragging = False
        self.drag_t0 = None
        self.drag_y0 = None
        self._mouse_down_scene_pos = None

    def closeEvent(self, event):
        self.plot_widget.clear()
        self.plot_widget.setParent(None)
        self.plot_widget.deleteLater()
        self.view_box.editor = None
        event.accept()

    def _exit_app(self):
        self.close()
        QtWidgets.QApplication.quit()


def launch_editor(
    signals,
    times,
    channel_names,
    markers_df=None,
    window_sec=20,
    n_display=60,
    resize=(1500, 800),
    move=(50, 200),
):
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("gui", "qt")
    except Exception:
        pass

    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        created_app = True

    editor = EEGEditor(
        signals=signals,
        times=times,
        channel_names=channel_names,
        markers_df=markers_df,
        window_sec=window_sec,
        n_display=n_display,
    )

    editor.show()
    editor.resize(*resize)
    editor.move(*move)

    def _on_close():
        editor.deleteLater()
        if created_app:
            app.quit()

    editor.destroyed.connect(_on_close)

    if created_app:
        app.exec()

    return editor
