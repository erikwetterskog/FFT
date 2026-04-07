#!/usr/bin/env python3
"""
Real-time Screen FFT Analyzer
==============================
Captures a 2^n × 2^n screen ROI in real time, applies a circular cosine-taper
aperture mask to suppress edge/aperture artefacts, and displays the 2D FFT
log-magnitude and phase live.

Inspired by the live-FFT feature in Gatan Digital Micrograph and Gwyddion,
but works on any on-screen content (video playback, simulations, …).

Usage
-----
    pip install -r requirements.txt
    python fft_analyzer.py

macOS note: Screen Recording permission must be granted in
    System Settings → Privacy & Security → Screen Recording
"""

from __future__ import annotations

import sys
import time
import threading
from typing import Optional

import numpy as np
import mss
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QComboBox, QPushButton, QGroupBox,
    QSpinBox, QSlider, QFrame, QSizePolicy, QCheckBox, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont

# ── Optional fast FFT backend ────────────────────────────────────────────────
try:
    from scipy.fft import fft2 as _fft2, fftshift as _fftshift
    _FFT_BACKEND = "scipy"
    def _compute_fft(arr: np.ndarray) -> np.ndarray:
        return _fftshift(_fft2(arr, workers=-1))
except ImportError:
    from numpy.fft import fft2 as _np_fft2, fftshift as _np_fftshift
    _FFT_BACKEND = "numpy"
    def _compute_fft(arr: np.ndarray) -> np.ndarray:          # type: ignore[misc]
        return _np_fftshift(_np_fft2(arr))

# ── Optional matplotlib colormaps ────────────────────────────────────────────
try:
    import matplotlib.cm as _mcm
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False

_MAG_CMAPS   = ["hot", "inferno", "viridis", "plasma", "turbo", "gray", "bone"]
_PHASE_CMAPS = ["hsv", "twilight", "twilight_shifted", "RdBu", "gray"]

_LUT_CACHE: dict[str, np.ndarray] = {}


def _build_lut(name: str) -> np.ndarray:
    """Return a 256×3 uint8 RGB lookup table."""
    if _HAVE_MPL:
        try:
            cmap = _mcm.get_cmap(name)
            return (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)
        except Exception:
            pass
    v = np.arange(256, dtype=np.uint8)
    return np.stack([v, v, v], axis=1)


def _lut(name: str) -> np.ndarray:
    if name not in _LUT_CACHE:
        _LUT_CACHE[name] = _build_lut(name)
    return _LUT_CACHE[name]


# ── Circular cosine-taper mask ───────────────────────────────────────────────

def make_circular_mask(size: int, radius_frac: float, rolloff_frac: float) -> np.ndarray:
    """
    2D circular aperture with smooth cosine-taper rolloff.

    mask = 1           inside the flat region
         = cosine taper across the rolloff band
         = 0           outside

    Parameters
    ----------
    size        : side length of the square image (pixels)
    radius_frac : outer radius as fraction of *size*  (0 < r ≤ 0.5)
    rolloff_frac: rolloff width as fraction of *size*
    """
    half = (size - 1) / 2.0
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - half) ** 2 + (y - half) ** 2)
    r_outer = radius_frac * size
    r_inner = max(r_outer - rolloff_frac * size, 0.0)
    mask = np.where(
        r <= r_inner, 1.0,
        np.where(
            r >= r_outer, 0.0,
            0.5 * (1.0 + np.cos(np.pi * (r - r_inner) / max(r_outer - r_inner, 1e-9))),
        ),
    ).astype(np.float32)
    return mask


# ── Array → QImage helpers ───────────────────────────────────────────────────

def _gray_to_qimage(arr: np.ndarray) -> QImage:
    h, w = arr.shape
    return QImage(bytes(arr.tobytes()), w, h, w, QImage.Format.Format_Grayscale8)


def _rgb_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    return QImage(bytes(arr.tobytes()), w, h, w * 3, QImage.Format.Format_RGB888)


# ── ROI overlay window ───────────────────────────────────────────────────────

class ROIOverlay(QWidget):
    """
    Frameless transparent window that the user drags over the screen to
    define the capture region.  Emits ``roi_moved(x, y)`` when dragged.
    """

    roi_moved = pyqtSignal(int, int)  # top-left in screen logical pixels

    _BORDER = 2          # border thickness (px)
    _BORDER_COLOR = QColor(0, 220, 220, 255)   # cyan
    _FILL_COLOR   = QColor(0, 220, 220, 20)    # very faint

    def __init__(self, size: int = 1024) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._drag_start: Optional[QPoint] = None
        self.set_capture_size(size)

    def set_capture_size(self, size: int) -> None:
        self.resize(size, size)

    # ── Painting ──────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:           # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Faint fill
        p.fillRect(self.rect(), self._FILL_COLOR)

        # Solid border
        pen = QPen(self._BORDER_COLOR, self._BORDER)
        p.setPen(pen)
        b = self._BORDER // 2
        p.drawRect(b, b, self.width() - self._BORDER, self.height() - self._BORDER)

        # Size label in top-left corner
        p.setPen(QColor(0, 240, 240, 200))
        font = QFont()
        font.setPixelSize(11)
        font.setBold(True)
        p.setFont(font)
        p.drawText(6, 14, f"{self.width()} × {self.height()}")

    # ── Dragging ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:       # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:        # type: ignore[override]
        if self._drag_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            new_pos = event.globalPosition().toPoint() - self._drag_start
            self.move(new_pos)
            self.roi_moved.emit(new_pos.x(), new_pos.y())

    def mouseReleaseEvent(self, event) -> None:     # type: ignore[override]
        self._drag_start = None


# ── FFT worker thread ─────────────────────────────────────────────────────────

class FFTWorker(QThread):
    """
    Background thread: screen-grab → grayscale → circular mask → FFT.

    Emits three uint8 numpy arrays each frame:
        result_ready(gray_u8, log_mag_u8, phase_u8)
    """

    result_ready      = pyqtSignal(object, object, object)
    fps_update        = pyqtSignal(float)
    permission_warning = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._running = False
        # -- configurable params (lock-protected) --
        self._size          = 1024
        self._roi_x         = 0
        self._roi_y         = 0
        self._radius_frac   = 0.45
        self._rolloff_frac  = 0.05
        self._subtract_mean = True
        self._max_fps       = 0          # 0 = unlimited
        # -- worker-private state --
        self._mask: Optional[np.ndarray] = None
        self._mask_key      = (-1, -1.0, -1.0)
        self._warned_black  = False

    def configure(self, *, size: int, roi_x: int, roi_y: int,
                  radius_frac: float, rolloff_frac: float,
                  subtract_mean: bool, max_fps: int) -> None:
        with self._lock:
            self._size          = size
            self._roi_x         = roi_x
            self._roi_y         = roi_y
            self._radius_frac   = radius_frac
            self._rolloff_frac  = rolloff_frac
            self._subtract_mean = subtract_mean
            self._max_fps       = max_fps

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        self._running = True
        sct = mss.mss()
        fps_count = 0
        fps_t0    = time.perf_counter()

        while self._running:
            t_frame = time.perf_counter()

            with self._lock:
                size          = self._size
                roi_x         = self._roi_x
                roi_y         = self._roi_y
                radius_frac   = self._radius_frac
                rolloff_frac  = self._rolloff_frac
                subtract_mean = self._subtract_mean
                max_fps       = self._max_fps

            # Rebuild mask only when its parameters change
            key = (size, radius_frac, rolloff_frac)
            if key != self._mask_key:
                self._mask     = make_circular_mask(size, radius_frac, rolloff_frac)
                self._mask_key = key

            monitor = {"top": roi_y, "left": roi_x, "width": size, "height": size}

            try:
                shot = sct.grab(monitor)

                # --- HiDPI / Retina handling -----------------------------------
                # mss on macOS may capture at 2× physical resolution.
                # We downscale to the requested logical size with Pillow.
                raw_h, raw_w = shot.height, shot.width
                if raw_h != size or raw_w != size:
                    pil_img = Image.frombytes("RGBA", (raw_w, raw_h), bytes(shot.raw))
                    pil_img = pil_img.resize((size, size), Image.Resampling.BILINEAR)
                    rgba    = np.array(pil_img, dtype=np.uint8)   # RGBA
                    # RGBA → BT.709 luminance (note: PIL frombytes gives RGBA)
                    gray_f = (
                        0.2126 * rgba[:, :, 0].astype(np.float32) +
                        0.7152 * rgba[:, :, 1].astype(np.float32) +
                        0.0722 * rgba[:, :, 2].astype(np.float32)
                    )
                else:
                    # mss raw is BGRA
                    raw = np.frombuffer(shot.raw, dtype=np.uint8).reshape(size, size, 4)
                    gray_f = (
                        0.0722 * raw[:, :, 0].astype(np.float32) +   # B
                        0.7152 * raw[:, :, 1].astype(np.float32) +   # G
                        0.2126 * raw[:, :, 2].astype(np.float32)     # R
                    )

                # macOS screen-recording permission check (black image = no permission)
                if not self._warned_black and gray_f.mean() < 1.0:
                    self._warned_black = True
                    self.permission_warning.emit()

                mask = self._mask

                # --- FFT input preparation ------------------------------------
                if subtract_mean:
                    w_sum  = float(mask.sum()) + 1e-9
                    mean_v = float((gray_f * mask).sum()) / w_sum
                    fft_in = (gray_f - mean_v) * mask
                else:
                    fft_in = gray_f * mask

                # --- 2D FFT ---------------------------------------------------
                F = _compute_fft(fft_in)

                # Log-magnitude, percentile normalised
                magnitude = np.log1p(np.abs(F)).astype(np.float32)
                p_lo = float(np.percentile(magnitude, 0.5))
                p_hi = float(np.percentile(magnitude, 99.5))
                if p_hi > p_lo:
                    log_mag_u8 = np.clip(
                        (magnitude - p_lo) / (p_hi - p_lo) * 255.0, 0, 255
                    ).astype(np.uint8)
                else:
                    log_mag_u8 = np.zeros((size, size), dtype=np.uint8)

                # Phase: map [−π, +π] → [0, 255]
                phase = np.angle(F).astype(np.float32)
                phase_u8 = ((phase + np.pi) / (2.0 * np.pi) * 255.0).clip(0, 255).astype(np.uint8)

                # Display image: original gray scaled through mask for context
                gray_u8 = np.clip(gray_f * mask, 0, 255).astype(np.uint8)

                self.result_ready.emit(gray_u8, log_mag_u8, phase_u8)

            except Exception as exc:
                print(f"[FFTWorker] {exc}", file=sys.stderr)
                time.sleep(0.05)
                continue

            # FPS accounting
            fps_count += 1
            now = time.perf_counter()
            if now - fps_t0 >= 1.0:
                self.fps_update.emit(fps_count / (now - fps_t0))
                fps_count = 0
                fps_t0    = now

            # Frame-rate cap
            if max_fps > 0:
                sleep_t = (1.0 / max_fps) - (time.perf_counter() - t_frame)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        sct.close()

    def stop(self) -> None:
        self._running = False


# ── Image display panel ───────────────────────────────────────────────────────

class ImagePanel(QFrame):
    """Titled dark frame containing a QLabel that scales images to fit."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.setStyleSheet("QFrame { background: #0c0c0c; border-radius: 5px; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 3, 4, 4)
        lay.setSpacing(2)

        lbl_title = QLabel(title)
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet(
            "color: #777; font-size: 10px; background: transparent; border: none;"
        )
        lay.addWidget(lbl_title)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setMinimumSize(200, 200)
        self._img_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._img_label.setStyleSheet("background: transparent; border: none;")
        lay.addWidget(self._img_label)

        self._pixmap: Optional[QPixmap] = None

    def show_gray(self, arr: np.ndarray) -> None:
        self._pixmap = QPixmap.fromImage(_gray_to_qimage(arr))
        self._redraw()

    def show_rgb(self, arr: np.ndarray) -> None:
        self._pixmap = QPixmap.fromImage(_rgb_to_qimage(arr))
        self._redraw()

    def _redraw(self) -> None:
        if self._pixmap is not None:
            self._img_label.setPixmap(
                self._pixmap.scaled(
                    self._img_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )

    def resizeEvent(self, event) -> None:           # type: ignore[override]
        super().resizeEvent(event)
        self._redraw()


# ── Dark stylesheet ───────────────────────────────────────────────────────────

_QSS = """
QWidget          { background: #1c1c1c; color: #d8d8d8; font-size: 12px; }
QMainWindow      { background: #1c1c1c; }
QGroupBox        {
    border: 1px solid #3a3a3a; border-radius: 4px;
    margin-top: 12px; padding: 6px 6px 6px 6px; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #999; }
QComboBox, QSpinBox {
    background: #272727; border: 1px solid #505050; border-radius: 3px;
    padding: 2px 6px; min-width: 60px;
    selection-background-color: #2e6fb5;
}
QComboBox::drop-down { border: none; }
QPushButton {
    background: #1c5393; border: none; border-radius: 4px;
    padding: 5px 14px; font-weight: bold; color: #fff;
}
QPushButton:hover   { background: #2568c0; }
QPushButton:pressed { background: #143e70; }
QSlider::groove:horizontal {
    height: 4px; background: #3a3a3a; border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px; height: 14px; background: #3d8ef5;
    border-radius: 7px; margin: -5px 0;
}
QCheckBox::indicator {
    width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid #555; background: #272727;
}
QCheckBox::indicator:checked { background: #1c5393; }
QLabel { color: #d8d8d8; }
"""


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Real-time Screen FFT Analyzer")
        self.setStyleSheet(_QSS)

        self._overlay = ROIOverlay(1024)
        self._overlay.roi_moved.connect(self._on_overlay_moved)

        self._setup_ui()
        self._setup_worker()
        self._centre_roi()   # sensible default position

        scr = QApplication.primaryScreen().geometry()
        W, H = 1380, 860
        self.resize(W, H)
        self.move((scr.width() - W) // 2, (scr.height() - H) // 2)

    # ── UI construction ───────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        # Three image panels
        panels = QHBoxLayout()
        panels.setSpacing(6)
        self._panel_real  = ImagePanel("Captured image  ·  circular mask applied")
        self._panel_mag   = ImagePanel("2D FFT  ·  log |F(u,v)|  ·  DC centred")
        self._panel_phase = ImagePanel("FFT phase  ·  ∠F(u,v)")
        panels.addWidget(self._panel_real)
        panels.addWidget(self._panel_mag)
        panels.addWidget(self._panel_phase)
        vbox.addLayout(panels, stretch=1)

        # Control bar
        ctrl = QWidget()
        ctrl_row = QHBoxLayout(ctrl)
        ctrl_row.setContentsMargins(0, 0, 0, 0)
        ctrl_row.setSpacing(8)
        ctrl_row.addWidget(self._grp_capture())
        ctrl_row.addWidget(self._grp_roi())
        ctrl_row.addWidget(self._grp_mask())
        ctrl_row.addWidget(self._grp_display())
        ctrl_row.addWidget(self._grp_status())
        ctrl_row.addStretch()
        vbox.addWidget(ctrl)

    # ── Control groups ────────────────────────────────────────────────────

    def _grp_capture(self) -> QGroupBox:
        g = QGroupBox("Capture")
        v = QVBoxLayout(g)

        v.addWidget(QLabel("Size:"))
        self._res_combo = QComboBox()
        self._res_combo.addItems(["256 × 256", "512 × 512", "1024 × 1024", "2048 × 2048"])
        self._res_combo.setCurrentIndex(2)
        self._res_combo.currentIndexChanged.connect(self._on_res_changed)
        v.addWidget(self._res_combo)

        v.addWidget(QLabel("Max FPS:"))
        self._fps_combo = QComboBox()
        self._fps_combo.addItems(["Unlimited", "60", "30", "15", "10"])
        self._fps_combo.currentIndexChanged.connect(self._push_params)
        v.addWidget(self._fps_combo)
        return g

    def _grp_roi(self) -> QGroupBox:
        g = QGroupBox("ROI Origin (px)")
        v = QVBoxLayout(g)

        for lbl_text, attr in [("X:", "_roi_x"), ("Y:", "_roi_y")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(lbl_text))
            spin = QSpinBox()
            spin.setRange(0, 9999)
            spin.valueChanged.connect(self._on_roi_spin_changed)
            setattr(self, attr, spin)
            row.addWidget(spin)
            v.addLayout(row)

        b_centre = QPushButton("Centre on Screen")
        b_centre.clicked.connect(self._centre_roi)
        v.addWidget(b_centre)

        self._overlay_btn = QPushButton("Show Overlay")
        self._overlay_btn.setCheckable(True)
        self._overlay_btn.toggled.connect(self._toggle_overlay)
        v.addWidget(self._overlay_btn)
        return g

    def _grp_mask(self) -> QGroupBox:
        g = QGroupBox("Circular Mask")
        v = QVBoxLayout(g)

        def _slider_row(lbl: str, lo: int, hi: int, val: int, unit: str, attr: str):
            row = QHBoxLayout()
            row.addWidget(QLabel(lbl))
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(val)
            vl = QLabel(f"{val}{unit}")
            vl.setFixedWidth(36)
            sl.valueChanged.connect(lambda v, lbl=vl, u=unit: lbl.setText(f"{v}{u}"))
            sl.valueChanged.connect(self._push_params)
            setattr(self, attr, sl)
            row.addWidget(sl)
            row.addWidget(vl)
            v.addLayout(row)

        _slider_row("Radius: ",  10, 50, 45, "%", "_r_sl")
        _slider_row("Rolloff:",   1, 20,  5, "%", "_ro_sl")

        self._mean_cb = QCheckBox("Subtract mean  (reduce DC spike)")
        self._mean_cb.setChecked(True)
        self._mean_cb.stateChanged.connect(self._push_params)
        v.addWidget(self._mean_cb)
        return g

    def _grp_display(self) -> QGroupBox:
        g = QGroupBox("FFT Display")
        v = QVBoxLayout(g)

        v.addWidget(QLabel("Magnitude colormap:"))
        self._mag_cmap = QComboBox()
        self._mag_cmap.addItems(_MAG_CMAPS)
        self._mag_cmap.currentIndexChanged.connect(self._push_params)
        v.addWidget(self._mag_cmap)

        v.addWidget(QLabel("Phase colormap:"))
        self._phase_cmap = QComboBox()
        self._phase_cmap.addItems(_PHASE_CMAPS)
        self._phase_cmap.currentIndexChanged.connect(self._push_params)
        v.addWidget(self._phase_cmap)
        return g

    def _grp_status(self) -> QGroupBox:
        g = QGroupBox("Status")
        v = QVBoxLayout(g)

        self._fps_lbl = QLabel("FPS: —")
        self._fps_lbl.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #4ac; background: transparent;"
        )
        v.addWidget(self._fps_lbl)

        backend_lbl = QLabel(f"FFT: {_FFT_BACKEND}")
        backend_lbl.setStyleSheet("font-size: 10px; color: #666; background: transparent;")
        v.addWidget(backend_lbl)

        self._toggle_btn = QPushButton("Pause")
        self._toggle_btn.clicked.connect(self._toggle_capture)
        v.addWidget(self._toggle_btn)
        return g

    # ── Worker wiring ─────────────────────────────────────────────────────

    def _setup_worker(self) -> None:
        self._worker = FFTWorker()
        self._worker.result_ready.connect(self._on_result)
        self._worker.fps_update.connect(lambda fps: self._fps_lbl.setText(f"FPS: {fps:.1f}"))
        self._worker.permission_warning.connect(self._warn_permission)
        self._push_params()
        self._worker.start()

    def _push_params(self) -> None:
        sizes   = [256, 512, 1024, 2048]
        fp_caps = [0, 60, 30, 15, 10]
        self._worker.configure(
            size          = sizes[self._res_combo.currentIndex()],
            roi_x         = self._roi_x.value(),
            roi_y         = self._roi_y.value(),
            radius_frac   = self._r_sl.value()  / 100.0,
            rolloff_frac  = self._ro_sl.value() / 100.0,
            subtract_mean = self._mean_cb.isChecked(),
            max_fps       = fp_caps[self._fps_combo.currentIndex()],
        )

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_result(self, gray_u8: np.ndarray,
                   mag_u8: np.ndarray, phase_u8: np.ndarray) -> None:
        self._panel_real.show_gray(gray_u8)

        mag_rgb   = _lut(self._mag_cmap.currentText())[mag_u8]
        phase_rgb = _lut(self._phase_cmap.currentText())[phase_u8]
        self._panel_mag.show_rgb(mag_rgb)
        self._panel_phase.show_rgb(phase_rgb)

    def _on_res_changed(self) -> None:
        sizes = [256, 512, 1024, 2048]
        self._overlay.set_capture_size(sizes[self._res_combo.currentIndex()])
        self._push_params()

    def _on_roi_spin_changed(self) -> None:
        # Move overlay to match spinboxes (if visible)
        if self._overlay.isVisible():
            self._overlay.blockSignals(True)
            self._overlay.move(self._roi_x.value(), self._roi_y.value())
            self._overlay.blockSignals(False)
        self._push_params()

    def _on_overlay_moved(self, x: int, y: int) -> None:
        self._roi_x.blockSignals(True)
        self._roi_y.blockSignals(True)
        self._roi_x.setValue(x)
        self._roi_y.setValue(y)
        self._roi_x.blockSignals(False)
        self._roi_y.blockSignals(False)
        self._push_params()

    def _centre_roi(self) -> None:
        scr  = QApplication.primaryScreen().geometry()
        sizes = [256, 512, 1024, 2048]
        size = sizes[self._res_combo.currentIndex()]
        x = max(0, (scr.width()  - size) // 2)
        y = max(0, (scr.height() - size) // 2)
        self._roi_x.setValue(x)
        self._roi_y.setValue(y)
        if self._overlay.isVisible():
            self._overlay.move(x, y)

    def _toggle_overlay(self, checked: bool) -> None:
        if checked:
            self._overlay.move(self._roi_x.value(), self._roi_y.value())
            self._overlay.show()
            self._overlay_btn.setText("Hide Overlay")
        else:
            self._overlay.hide()
            self._overlay_btn.setText("Show Overlay")

    def _toggle_capture(self) -> None:
        if self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
            self._toggle_btn.setText("Resume")
        else:
            self._worker.start()
            self._toggle_btn.setText("Pause")

    def _warn_permission(self) -> None:
        QMessageBox.warning(
            self, "Screen Recording Permission Required",
            "The captured area appears to be black.\n\n"
            "On macOS, grant Screen Recording permission:\n"
            "  System Settings → Privacy & Security → Screen Recording\n\n"
            "Then restart the application.",
        )

    # ── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:            # type: ignore[override]
        self._overlay.close()
        self._worker.stop()
        self._worker.wait()
        super().closeEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
