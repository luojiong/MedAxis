"""Platform-aware glass surfaces for the MedAxis workstation.

The referenced qt-liquid-glass project is macOS-only because its native
backend injects AppKit's NSGlassEffectView. MedAxis runs on Windows too, so
this adapter uses the Windows 11 DWM backdrop when available and keeps the
surface treatment in Qt for every platform.
"""

from __future__ import annotations

import sys
from ctypes import Structure, byref, c_int, sizeof

try:
    from ctypes import windll
except ImportError:  # pragma: no cover - only used on Windows
    windll = None

from PySide6.QtCore import QEvent, QPoint, QRectF, QTimer, Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsBlurEffect, QGraphicsDropShadowEffect, QGraphicsPixmapItem,
    QGraphicsScene, QWidget,
)


class _Margins(Structure):
    _fields_ = [("left", c_int), ("right", c_int),
                ("top", c_int), ("bottom", c_int)]


def apply_glass_window(window: QWidget) -> None:
    """Enable a native desktop backdrop where the host OS supports it."""
    if sys.platform != "win32" or windll is None:
        return
    try:
        hwnd = int(window.winId())
        dwm = windll.dwmapi
        mica = c_int(2)
        dwm.DwmSetWindowAttribute(hwnd, 38, byref(mica), sizeof(mica))
        margins = _Margins(-1, -1, -1, -1)
        dwm.DwmExtendFrameIntoClientArea(hwnd, byref(margins))
    except (AttributeError, OSError, TypeError, ValueError):
        return


class GlassPanel(QWidget):
    """A reusable local glass surface for a Qt Widgets layout block.

    The panel snapshots only the portion of its parent behind itself, blurs
    that cached pixmap, and paints the result beneath a translucent tint.
    Child controls remain normal Qt widgets and keep their interaction and
    accessibility behavior.
    """

    def __init__(self, parent: QWidget | None = None, *, blur_radius: float = 16.0,
                 tint: QColor | None = None) -> None:
        super().__init__(parent)
        self._blur_radius = blur_radius
        self._tint = tint or QColor(255, 255, 255, 150)
        self._background = QPixmap()
        self._background_dirty = True
        self._capture_timer = QTimer(self)
        self._capture_timer.setSingleShot(True)
        self._capture_timer.setInterval(70)
        self._capture_timer.timeout.connect(self._refresh_background)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        if parent is not None:
            parent.installEventFilter(self)

    def invalidate_glass(self) -> None:
        self._background_dirty = True
        self._schedule_capture()
        self.update()

    def _schedule_capture(self) -> None:
        if self.isVisible() and not self._capture_timer.isActive():
            self._capture_timer.start()

    def _refresh_background(self) -> None:
        if not self.isVisible() or self.width() <= 0 or self.height() <= 0:
            return
        self._background = self._capture_background()
        self._background_dirty = False
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._background_dirty = True
        self._schedule_capture()
        super().resizeEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        self._background_dirty = True
        self._capture_timer.start(0)
        super().showEvent(event)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self.parentWidget() and event.type() in (
                QEvent.Resize, QEvent.Move, QEvent.LayoutRequest):
            self._background_dirty = True
            self._schedule_capture()
            self.update()
        return super().eventFilter(watched, event)

    def _capture_background(self) -> QPixmap:
        parent = self.parentWidget()
        if parent is None or self.width() <= 0 or self.height() <= 0:
            return QPixmap()

        raw = QPixmap(self.size())
        raw.fill(Qt.transparent)
        # Hide this child while rendering the parent to prevent recursive
        # GlassPanel paint events from appearing in the captured surface.
        was_visible = self.isVisible()
        self.setVisible(False)
        painter = QPainter(raw)
        parent.render(painter, QPoint(-self.x(), -self.y()),
                      self.geometry(), QWidget.DrawWindowBackground | QWidget.DrawChildren)
        painter.end()
        self.setVisible(was_visible)

        scene = QGraphicsScene()
        item = QGraphicsPixmapItem(raw)
        effect = QGraphicsBlurEffect()
        effect.setBlurRadius(self._blur_radius)
        item.setGraphicsEffect(effect)
        scene.addItem(item)
        blurred = QImage(raw.size(), QImage.Format_ARGB32_Premultiplied)
        blurred.fill(Qt.transparent)
        blur_painter = QPainter(blurred)
        scene.render(blur_painter, QRectF(blurred.rect()), QRectF(raw.rect()))
        blur_painter.end()
        return QPixmap.fromImage(blurred)

    def paintEvent(self, event) -> None:  # noqa: N802
        # During continuous resize, keep the last blurred frame. Capturing
        # and blurring synchronously here would block the Qt GUI thread.
        if self._background.isNull() and not self._capture_timer.isActive():
            self._refresh_background()

        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.Antialiasing, True)
            bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            path = QPainterPath()
            path.addRoundedRect(bounds, 10.0, 10.0)
            painter.save()
            painter.setClipPath(path)
            if not self._background.isNull():
                painter.drawPixmap(0, 0, self._background)
            painter.fillPath(path, self._tint)
            highlight = QLinearGradient(0, 0, 0, max(1, self.height()))
            highlight.setColorAt(0.0, QColor(255, 255, 255, 62))
            highlight.setColorAt(0.42, QColor(255, 255, 255, 12))
            highlight.setColorAt(1.0, QColor(223, 238, 247, 26))
            painter.fillPath(path, highlight)
            painter.restore()
            painter.setPen(QPen(QColor(255, 255, 255, 188), 1.0))
            painter.drawPath(path)
        super().paintEvent(event)
def apply_glass_surface(widget: QWidget, *, radius: int = 10) -> None:
    """Add the reusable depth treatment used by each workstation block."""
    widget.setAttribute(Qt.WA_TranslucentBackground, True)
    widget.setProperty("glassSurface", True)
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(22)
    shadow.setOffset(0, 5)
    shadow.setColor(QColor(40, 76, 101, 34))
    widget.setGraphicsEffect(shadow)
    widget.setStyleSheet(
        f"{widget.styleSheet()} QWidget {{ border-radius: {radius}px; }}")
