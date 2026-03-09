"""
Eärendil 
--------------------------------------------------------------------------
Real-time gravitational lensing visualization around a spinning black hole
using real infrared sky survey data from 2MASS.

Named after the star of high hope in Tolkien's legendarium.

Created by K.Kostaros
"""

import sys
import numpy as np
from threading import Thread
from queue import Queue, Empty

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QGroupBox, QCheckBox,
    QComboBox, QProgressBar, QStatusBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QFont

from kerr_tracer import trace_rays_batch, get_celestial_coords, R_CELESTIAL
from sky_data import load_or_build_sky_texture, sample_sky_equirect_batch, SURVEY_NAME

@jax.jit
def _u_t_static_kerr(r, th, M, a):
    Sigma = r*r + a*a * jnp.cos(th)**2
    g_tt = -(1.0 - (2.0*M*r)/Sigma)
    return 1.0 / jnp.sqrt(jnp.maximum(-g_tt, 1e-15))


@jax.jit
def apply_gr_shift_to_sky(colors, thetas_hit, r_obs, th_obs, M, a, r_emit):
    uobs_t = _u_t_static_kerr(r_obs, th_obs, M, a)
    uem_t = _u_t_static_kerr(r_emit, thetas_hit, M, a)
    
    g = jnp.clip(uobs_t / jnp.maximum(uem_t, 1e-15), 0.0, 50.0)
    scale = (g**3)[..., None]
    return jnp.clip(colors * scale, 0.0, 1.0)

@jax.jit
def compute_magnification(thetas, phis, b_spacing, al_spacing):
    eps_b = b_spacing * 0.5
    eps_al = al_spacing * 0.5
    
    dtheta_db = jnp.gradient(thetas, axis=1) / eps_b
    dtheta_dal = jnp.gradient(thetas, axis=0) / eps_al
    dphi_db = jnp.gradient(phis, axis=1) / eps_b
    dphi_dal = jnp.gradient(phis, axis=0) / eps_al
    
    sin_theta = jnp.sin(jnp.clip(thetas, 0.01, jnp.pi - 0.01))
    jacobian = jnp.abs(dtheta_db * dphi_dal - dtheta_dal * dphi_db) * sin_theta
    
    magnification = 1.0 / jnp.clip(jacobian, 1e-6, 1e6)
    log_mag = jnp.log(jnp.clip(magnification, 1e-10, 1e10))
    log_mag = jnp.nan_to_num(log_mag, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = jnp.exp(log_mag - jnp.median(log_mag))
    
    return jnp.clip(normalized, 0.3, 3.0)


@jax.jit
def apply_magnification(colors, magnification):
    factor = jnp.power(magnification, 0.3)[..., None]
    return jnp.clip(colors * factor, 0.0, 1.0)


@jax.jit
def apply_black_hole_shadow(colors, hit_horizon):
    mask = hit_horizon[..., None]
    return jnp.where(mask, 0.0, colors)


class RenderWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    sky_ready = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.running = True
        self.sky_texture = None
        self.sky_meta = None
    
    def process_request(self, params):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break
        self.queue.put({"type": "render", "params": params})
    
    def request_build_sky(self, width=4096, height=2048, force=False):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break
        self.queue.put({
            "type": "build_sky",
            "width": width,
            "height": height,
            "force": force,
        })
    
    def run(self):
        while self.running:
            try:
                job = self.queue.get(timeout=0.1)
                job_type = job.get("type", "render")
                if job_type == "build_sky":
                    self._build_sky(job)
                else:
                    self._render_frame(job["params"])
            except Empty:
                continue
            except Exception as exc:
                import traceback
                self.error.emit(f"{exc}\n{traceback.format_exc()}")
    
    def _build_sky(self, job):
        width = int(job.get("width", 4096))
        height = int(job.get("height", 2048))
        force = bool(job.get("force", False))
        
        self.progress_update.emit(5)
        self.status_update.emit(f"Loading {SURVEY_NAME}...")
        
        sky_rgb, meta = load_or_build_sky_texture(
            width=width,
            height=height,
            cache_dir="sky_cache",
            force_download=force,
        )
        
        self.sky_texture = jnp.asarray(sky_rgb, dtype=jnp.float32)
        self.sky_meta = meta
        
        self.progress_update.emit(100)
        self.status_update.emit("Sky ready")
        self.sky_ready.emit(meta)
    
    def _render_frame(self, params):
        r_obs = params['r_obs']
        theta_obs = params['theta_obs']
        M = params['M']
        a = params['a']
        resolution = params['resolution']
        fov_angle = params['fov']
        
        self.progress_update.emit(5)
        self.status_update.emit("Creating ray grid...")
        
        fov_rad = fov_angle * jnp.pi / 180
        b_max = r_obs * jnp.tan(fov_rad / 2)
        aspect = 16 / 9
        al_max = b_max / aspect
        
        resolution_y = resolution
        resolution_x = int(resolution * aspect)
        
        b_values = jnp.linspace(-b_max, b_max, resolution_x)
        al_values = jnp.linspace(-al_max, al_max, resolution_y)
        b_grid, al_grid = jnp.meshgrid(b_values, al_values, indexing='ij')
        b_flat = b_grid.ravel()
        al_flat = al_grid.ravel()
        
        self.progress_update.emit(10)
        self.status_update.emit("Tracing rays through spacetime...")
        
        final_states = trace_rays_batch(
            b_flat, al_flat,
            r_obs, theta_obs,
            M, a,
            0.0, 15000.0, 0.025
        )
        
        self.progress_update.emit(60)
        self.status_update.emit("Processing ray intersections...")
        
        thetas, phis, hit_horizon = get_celestial_coords(final_states, M, a)
        
        thetas_2d = thetas.reshape(resolution_x, resolution_y).T
        phis_2d = phis.reshape(resolution_x, resolution_y).T
        hit_horizon_2d = hit_horizon.reshape(resolution_x, resolution_y).T
        
        self.progress_update.emit(70)
        self.status_update.emit("Rendering sky...")
        
        thetas_safe = jnp.where(jnp.isnan(thetas_2d), jnp.pi/2, thetas_2d)
        phis_safe = jnp.where(jnp.isnan(phis_2d), 0.0, phis_2d)
        thetas_safe = jnp.clip(thetas_safe, 0.01, jnp.pi - 0.01)
        
        if self.sky_texture is not None:
            colors = sample_sky_equirect_batch(thetas_safe, phis_safe, self.sky_texture)
        else:
            colors = jnp.full((*thetas_safe.shape, 3), 0.02)
        
        colors = apply_gr_shift_to_sky(colors, thetas_safe, r_obs, theta_obs, M, a, R_CELESTIAL)
        
        self.progress_update.emit(80)
        self.status_update.emit("Computing lensing magnification...")
        
        b_spacing = 2 * b_max / resolution_x
        al_spacing = 2 * al_max / resolution_y
        magnification = compute_magnification(thetas_safe, phis_safe, b_spacing, al_spacing)
        colors = apply_magnification(colors, magnification)
        
        self.progress_update.emit(90)
        self.status_update.emit("Applying shadow...")
        
        colors = apply_black_hole_shadow(colors, hit_horizon_2d)
        
        colors_np = np.array(colors)
        colors_uint8 = (np.clip(colors_np, 0, 1) * 255).astype(np.uint8)
        
        self.progress_update.emit(100)
        self.status_update.emit("Complete")
        self.frame_ready.emit(colors_uint8)
    
    def stop(self):
        self.running = False


class ImageDisplay(QLabel):
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #0a0a0f; border: 1px solid #333;")
        self._full_res_image = None
        self._show_placeholder()
    
    def _show_placeholder(self):
        self.setText("Loading sky data...\n\nFirst render includes JIT compilation\nand may take 30-60 seconds.")
        self.setStyleSheet("""
            background-color: #0a0a0f; 
            border: 1px solid #333;
            color: #666;
            font-size: 16px;
        """)
    
    def display_frame(self, image_data):
        self._full_res_image = image_data.copy()
        self._update_display()
    
    def _update_display(self):
        if self._full_res_image is None:
            return
        
        image_data = self._full_res_image
        h, w, c = image_data.shape
        bytes_per_line = 3 * w
        
        qimage = QImage(image_data.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.setPixmap(scaled)
        self.setStyleSheet("background-color: #0a0a0f; border: 1px solid #333;")
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()
    
    def get_full_res_image(self):
        return self._full_res_image


class ControlPanel(QWidget):

    
    params_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(280)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        title = QLabel("✧ Eärendil")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #fff; padding: 8px 0;")
        layout.addWidget(title)
        
        observer_group = QGroupBox("Observer Position")
        observer_layout = QVBoxLayout(observer_group)
        
        self.distance_label = QLabel("Distance: 500 M")
        observer_layout.addWidget(self.distance_label)
        
        self.distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(50, 1000)
        self.distance_slider.setValue(500)
        self.distance_slider.valueChanged.connect(self._on_distance_changed)
        observer_layout.addWidget(self.distance_slider)
        
        self.inclination_label = QLabel("Inclination: 70°")
        observer_layout.addWidget(self.inclination_label)
        
        self.inclination_slider = QSlider(Qt.Orientation.Horizontal)
        self.inclination_slider.setRange(5, 175)
        self.inclination_slider.setValue(70)
        self.inclination_slider.valueChanged.connect(self._on_inclination_changed)
        observer_layout.addWidget(self.inclination_slider)
        
        layout.addWidget(observer_group)
        
        bh_group = QGroupBox("Black Hole")
        bh_layout = QVBoxLayout(bh_group)
        
        self.spin_label = QLabel("Spin (a/M): 0.998")
        bh_layout.addWidget(self.spin_label)
        
        self.spin_slider = QSlider(Qt.Orientation.Horizontal)
        self.spin_slider.setRange(0, 999)
        self.spin_slider.setValue(998)
        self.spin_slider.valueChanged.connect(self._on_spin_changed)
        bh_layout.addWidget(self.spin_slider)
        
        self.horizon_label = QLabel("Horizon: r ≈ 1.063 M")
        self.horizon_label.setStyleSheet("color: #888; font-size: 11px;")
        bh_layout.addWidget(self.horizon_label)
        
        layout.addWidget(bh_group)
        
        render_group = QGroupBox("Rendering")
        render_layout = QVBoxLayout(render_group)
        
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Quality:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["Fast (100)", "Medium (150)", "High (200)", "Ultra (300)", "SuperUltra (500)"])
        self.resolution_combo.setCurrentIndex(1)
        self.resolution_combo.currentIndexChanged.connect(self._emit_params)
        res_layout.addWidget(self.resolution_combo)
        render_layout.addLayout(res_layout)
        
        self.fov_label = QLabel("FOV: 30°")
        render_layout.addWidget(self.fov_label)
        
        self.fov_slider = QSlider(Qt.Orientation.Horizontal)
        self.fov_slider.setRange(5, 120)
        self.fov_slider.setValue(30)
        self.fov_slider.valueChanged.connect(self._on_fov_changed)
        render_layout.addWidget(self.fov_slider)
        
        layout.addWidget(render_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        layout.addWidget(self.progress_bar)
        
        self.render_button = QPushButton("🔄 Render Frame")
        self.render_button.clicked.connect(self._emit_params)
        self.render_button.setStyleSheet("""
            QPushButton {
                background-color: #4a4a6a;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #5a5a7a; }
        """)
        layout.addWidget(self.render_button)
        
        self.save_button = QPushButton("💾 Save Image")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #3a5a4a;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #4a6a5a; }
        """)
        layout.addWidget(self.save_button)
        
        self.auto_render = QCheckBox("Auto-render on change")
        self.auto_render.setChecked(True)
        layout.addWidget(self.auto_render)
        
        layout.addStretch()
        
        self.sky_info_label = QLabel("Sky: 2MASS (CDS/Aladin)")
        self.sky_info_label.setStyleSheet("color: #555; font-size: 10px;")
        layout.addWidget(self.sky_info_label)
        
        creator_label = QLabel("Created by K. Kostaros")
        creator_label.setStyleSheet("color: #444; font-size: 10px; font-style: italic;")
        layout.addWidget(creator_label)
        
    def _update_horizon_label(self):
        a = self.spin_slider.value() / 1000
        M = 1.0
        r_h = M + np.sqrt(max(0, M**2 - a**2))
        self.horizon_label.setText(f"Horizon: r ≈ {r_h:.3f} M")
    
    def _on_distance_changed(self, value):
        self.distance_label.setText(f"Distance: {value} M")
        if self.auto_render.isChecked():
            self._emit_params()
    
    def _on_inclination_changed(self, value):
        self.inclination_label.setText(f"Inclination: {value}°")
        if self.auto_render.isChecked():
            self._emit_params()
    
    def _on_spin_changed(self, value):
        self.spin_label.setText(f"Spin (a/M): {value/1000:.3f}")
        self._update_horizon_label()
        if self.auto_render.isChecked():
            self._emit_params()
    
    def _on_fov_changed(self, value):
        self.fov_label.setText(f"FOV: {value}°")
        if self.auto_render.isChecked():
            self._emit_params()
    
    def _emit_params(self):
        resolution_map = {0: 100, 1: 150, 2: 200, 3: 300, 4: 500}
        
        params = {
            'r_obs': float(self.distance_slider.value()),
            'theta_obs': float(self.inclination_slider.value()) * np.pi / 180,
            'M': 1.0,
            'a': float(self.spin_slider.value()) / 1000,
            'resolution': resolution_map[self.resolution_combo.currentIndex()],
            'fov': float(self.fov_slider.value()),
        }
        
        self.params_changed.emit(params)
    
    def update_sky_info(self, text):
        self.sky_info_label.setText(text)
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if value < 100:
            self.progress_bar.setFormat(f"Rendering... {value}%")
        else:
            self.progress_bar.setFormat("Complete")
    
    def update_status(self, status):
        if self.progress_bar.value() < 100:
            self.progress_bar.setFormat(status)

class EarendilViewer(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eärendil")
        self.setMinimumSize(1300, 800)
        self.resize(1700, 1000)
        
        self._setup_ui()
        self._setup_worker()
        self._apply_dark_theme()
        
        QTimer.singleShot(100, self._load_sky_and_render)
    
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        self.display = ImageDisplay()
        main_layout.addWidget(self.display, stretch=1)
        
        self.controls = ControlPanel()
        self.controls.params_changed.connect(self._on_params_changed)
        self.controls.save_button.clicked.connect(self._save_image)
        main_layout.addWidget(self.controls)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing...")
    
    def _setup_worker(self):
        self.worker = RenderWorker()
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.progress_update.connect(self.controls.update_progress)
        self.worker.status_update.connect(self.controls.update_status)
        self.worker.sky_ready.connect(self._on_sky_ready)
        self.worker.error.connect(self._on_worker_error)
        
        self.worker_thread = Thread(target=self.worker.run, daemon=True)
        self.worker_thread.start()
        
        self.render_timer = QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._do_render)
        self.pending_params = None
    
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QWidget { background-color: #1a1a2e; color: #e0e0e0; }
            QStatusBar { background-color: #0f0f1a; color: #888; }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                background-color: #222;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4a6a9a;
                border-radius: 2px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #aaa;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 6px;
                background: #333;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6a6a8a;
                border: 1px solid #555;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QCheckBox { color: #ccc; }
            QComboBox {
                background-color: #333;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 3px;
                color: #ccc;
            }
        """)
    
    def _load_sky_and_render(self):
        self.status_bar.showMessage("Loading 2MASS sky data...")
        self.worker.request_build_sky(width=4096, height=2048, force=False)
    
    def _on_sky_ready(self, meta):
        width = meta.get("width", "?")
        height = meta.get("height", "?")
        self.controls.update_sky_info(f"Sky: 2MASS J/H/K ({width}×{height})")
        self.status_bar.showMessage(f"Sky loaded — {SURVEY_NAME}")
        
        QTimer.singleShot(100, self.controls._emit_params)
    
    def _on_params_changed(self, params):
        self.pending_params = params
        self.render_timer.start(100)
    
    def _do_render(self):
        if self.pending_params:
            self.controls.update_progress(0)
            self.status_bar.showMessage("Rendering...")
            self.worker.process_request(self.pending_params)
            self.pending_params = None
    
    def _on_frame_ready(self, frame):
        self.display.display_frame(frame)
        self.status_bar.showMessage(f"Eärendil — {frame.shape[1]}×{frame.shape[0]}")
    
    def _on_worker_error(self, message):
        self.status_bar.showMessage(f"Error: {message[:80]}")
    
    def _save_image(self):
        from PyQt6.QtWidgets import QFileDialog
        from PIL import Image
        import datetime
        
        image_data = self.display.get_full_res_image()
        if image_data is None:
            self.status_bar.showMessage("No image to save - render first!")
            return
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"earendil_{timestamp}.png"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image", default_name, "PNG Files (*.png);;All Files (*)"
        )
        
        if filename:
            try:
                img = Image.fromarray(image_data)
                img.save(filename)
                self.status_bar.showMessage(f"Saved: {filename}")
            except Exception as e:
                self.status_bar.showMessage(f"Error saving: {e}")
    
    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

def main():
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    print(f"Celestial sphere radius: {R_CELESTIAL} M")
    
    app = QApplication(sys.argv)
    app.setApplicationName("Eärendil")
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = EarendilViewer()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
