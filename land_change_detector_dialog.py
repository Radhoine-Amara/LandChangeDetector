"""
land_change_detector_dialog.py
LandChangeDetector — Phase 6

UI controller: connects the Qt dialog to the algorithm backend.
Handles file browsing, input validation, QThread dispatch,
progress updates, log output, and QGIS layer loading.

Author: Darius — 3rd Year AI Engineering
"""

import os
import traceback
import datetime
from pathlib import Path

import numpy as np

from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.uic import loadUiType

from qgis.core import (
    QgsRasterLayer,
    QgsProject,
    QgsColorRampShader,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsMessageLog,
    Qgis,
)

# ── Load the compiled UI ────────────────────────────────────────────────────
PLUGIN_DIR = os.path.dirname(__file__)
FORM_CLASS, _ = loadUiType(
    os.path.join(PLUGIN_DIR, "land_change_detector_dialog_base.ui")
)

# ── Algorithm imports ───────────────────────────────────────────────────────
from .algorithms.band_differencing import run_band_differencing
from .algorithms.ndvi_differencing  import run_ndvi_differencing
from .algorithms.cva                import run_cva
from .algorithms.rf_improvements    import (run_binary_rf,
                                            apply_majority_filter_and_compare)

# ── Utility imports ─────────────────────────────────────────────────────────
from .utils.raster_utils import load_band_crop, load_cloud_mask, save_raster
from .utils.stats_utils  import compute_change_statistics, export_statistics_csv

LOG_TAG = "LandChangeDetector"


def _get_result_array(results: dict, *keys):
    """Return the first key found in *results*, raising a clear KeyError if none exist."""
    for key in keys:
        if key in results and results[key] is not None:
            return results[key]
    raise KeyError(
        f"Missing required result array. Tried: {keys}. "
        f"Available: {list(results.keys())}"
    )


def _ts():
    """Return current timestamp string for log messages."""
    return datetime.datetime.now().strftime("%H:%M:%S")


def _derive_band_path(b04_path: str, target_band: str) -> str:
    """
    Derive sibling band path from a B04 path by replacing '_B04_' with
    '_B02_', '_B03_', or '_B08_'.
    """
    return b04_path.replace("_B04_", f"_{target_band}_")


# ═══════════════════════════════════════════════════════════════════════════
# WORKER — runs the algorithm in a background thread
# ═══════════════════════════════════════════════════════════════════════════

class AnalysisWorker(QObject):
    """
    Background worker that runs the selected algorithm.
    Emits progress(int), log(str), finished(dict), error(str).
    """
    progress = pyqtSignal(int)
    log      = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, params: dict):
        """
        Parameters
        ----------
        params : dict with keys:
            before_path, after_path,
            scl_before_path, scl_after_path,
            method, cloud_mask_enabled, crop_size,
            output_dir
        """
        super().__init__()
        self.params = params

    def run(self):
        try:
            p = self.params
            crop_size = p["crop_size"]

            CROP = dict(x_off=4000, y_off=4000,
                        x_size=crop_size, y_size=crop_size)
            SCL_CROP = dict(x_off=CROP["x_off"] // 2, y_off=CROP["y_off"] // 2,
                            x_size=crop_size // 2,    y_size=crop_size // 2)

            # ── Step 1: Load bands (0% → 20%) ─────────────────────────────
            self.log.emit(f"[{_ts()}] Loading before image bands...")
            self.progress.emit(0)

            b04_before = p["before_path"]
            b04_after  = p["after_path"]

            def load(path):
                d, gt, proj = load_band_crop(path, **CROP)
                return d.astype(np.float32), gt, proj

            red_b, gt, proj = load(b04_before)
            nir_b, _,  _    = load(_derive_band_path(b04_before, "B08"))
            grn_b, _,  _    = load(_derive_band_path(b04_before, "B03"))
            blu_b, _,  _    = load(_derive_band_path(b04_before, "B02"))

            self.log.emit(f"[{_ts()}] Loading after image bands...")

            red_a, _,  _    = load(b04_after)
            nir_a, _,  _    = load(_derive_band_path(b04_after, "B08"))
            grn_a, _,  _    = load(_derive_band_path(b04_after, "B03"))
            blu_a, _,  _    = load(_derive_band_path(b04_after, "B02"))

            self.progress.emit(20)
            self.log.emit(f"[{_ts()}] Bands loaded. Crop: {crop_size}x{crop_size} px")

            # ── Step 2: Cloud masking (20% → 40%) ─────────────────────────
            target_size = (crop_size, crop_size)

            if p["cloud_mask_enabled"]:
                self.log.emit(f"[{_ts()}] Applying cloud masks (SCL)...")

                _, valid_b = load_cloud_mask(
                    p["scl_before_path"], target_size=target_size, **SCL_CROP
                )
                _, valid_a = load_cloud_mask(
                    p["scl_after_path"],  target_size=target_size, **SCL_CROP
                )
                valid_both = valid_b & valid_a

                n_valid = int(valid_both.sum())
                pct     = n_valid / valid_both.size * 100
                self.log.emit(
                    f"[{_ts()}] Valid pixels: {n_valid:,} / {valid_both.size:,} "
                    f"({pct:.1f}%)"
                )
            else:
                valid_both = np.ones(target_size, dtype=bool)
                self.log.emit(f"[{_ts()}] Cloud masking disabled — using all pixels.")

            self.progress.emit(40)

            # Apply mask (NaN = cloud / no-data)
            def mask(arr):
                return np.where(valid_both, arr, np.nan).astype(np.float32)

            red_b = mask(red_b); nir_b = mask(nir_b)
            grn_b = mask(grn_b); blu_b = mask(blu_b)
            red_a = mask(red_a); nir_a = mask(nir_a)
            grn_a = mask(grn_a); blu_a = mask(blu_a)

            # ── Step 3: Run selected algorithm (40% → 60%) ────────────────
            method = p["method"]
            self.log.emit(f"[{_ts()}] Running {method}...")
            self.progress.emit(40)

            if method == "NDVI Differencing":
                results = run_ndvi_differencing(red_b, nir_b, red_a, nir_a)

            elif method == "Band Differencing":
                results = run_band_differencing(red_b, red_a)

            elif method == "CVA":
                results = run_cva(
                    bands_before=[blu_b, grn_b, red_b, nir_b],
                    bands_after =[blu_a, grn_a, red_a, nir_a],
                    band_names  =["B02", "B03", "B04", "B08"],
                )

            elif method == "Binary RF + Smoothing":
                # ── Resolve WorldCover paths ──────────────────────────────
                try:
                    from .utils.image_pair_config import DATA_ROOT
                except ImportError:
                    from utils.image_pair_config import DATA_ROOT

                wc_dir = DATA_ROOT / "worldcover"
                wc_candidates = [
                    "ESA_WorldCover_10m_2021_v200_N33E006_Map.tif",
                    "ESA_WorldCover_10m_2021_v200_N33E003_Map.tif",
                ]
                wc_paths = [
                    str(wc_dir / name)
                    for name in wc_candidates
                    if (wc_dir / name).exists()
                ]
                if not wc_paths:
                    # Also try a glob in case the tile names differ
                    if wc_dir.exists():
                        wc_paths = [
                            str(p) for p in wc_dir.glob("ESA_WorldCover*.tif")
                        ]

                if not wc_paths:
                    raise FileNotFoundError(
                        f"WorldCover reference data not found in {wc_dir}.\n"
                        "Supervised Binary RF cannot run without WorldCover labels.\n"
                        "Please ensure ESA_WorldCover_10m_2021_v200_N33E*.tif "
                        "files are present in the data/worldcover/ directory, "
                        "or set the LAND_CHANGE_DATA environment variable to "
                        "your data root."
                    )

                self.log.emit(
                    f"[{_ts()}] WorldCover: {len(wc_paths)} tile(s) found"
                )
                for p in wc_paths:
                    self.log.emit(f"[{_ts()}]   {p}")

                # ── Run Binary RF (supervised, WorldCover labels) ─────────
                bands_before = [blu_b, grn_b, red_b, nir_b]
                bands_after  = [blu_a, grn_a, red_a, nir_a]
                rf_results   = run_binary_rf(
                    bands_before=bands_before,
                    bands_after =bands_after,
                    worldcover_path=wc_paths,
                    wc_crop=CROP,
                    reference_band_path=b04_before,
                )

                # ── Diagnostic: run_binary_rf output ──────────────────────
                self.log.emit(f"[{_ts()}] Binary RF training_mode: {rf_results.get('training_mode')}")
                self.log.emit(f"[{_ts()}] Binary RF raw change_pct: {rf_results.get('change_pct')}")
                self.log.emit(f"[{_ts()}] Binary RF veg_loss_pct: {rf_results.get('veg_loss_pct')}")
                self.log.emit(f"[{_ts()}] Binary RF veg_gain_pct: {rf_results.get('veg_gain_pct')}")
                self.log.emit(f"[{_ts()}] Binary RF accuracy: {rf_results.get('accuracy')}")
                self.log.emit(f"[{_ts()}] Binary RF spatial_cv_mean: {rf_results.get('spatial_cv_mean')}")

                # ── Smooth and compare ────────────────────────────────────
                rf_valid     = rf_results.get("valid_mask", valid_both)
                class_before = _get_result_array(rf_results,
                                                 "class_before",
                                                 "binary_map_before")
                class_after  = _get_result_array(rf_results,
                                                 "class_after",
                                                 "binary_map_after")

                smoothed = apply_majority_filter_and_compare(
                    class_before, class_after, rf_valid, window=5,
                )

                self.log.emit(
                    f"[{_ts()}] Smoothed: "
                    f"raw={smoothed.get('change_pct_original'):.2f}% "
                    f"-> smoothed={smoothed.get('change_pct'):.2f}% "
                    f"(reduction {smoothed.get('reduction_pp'):.2f} pp)"
                )

                results = {
                    **rf_results,
                    **smoothed,
                    "change_mask": _get_result_array(smoothed,
                                                     "change_mask",
                                                     "change_map_smooth"),
                    "change_pct":  _get_result_array(smoothed,
                                                     "change_pct",
                                                     "change_pct_smooth"),
                }
            else:
                raise ValueError(f"Unknown method: {method!r}")

            # Normalise key
            if "change_mask" not in results and "change_map" in results:
                results["change_mask"] = results["change_map"].astype(bool)

            change_pct = results.get("change_pct", 0.0)
            self.log.emit(
                f"[{_ts()}] {method} complete — change: {change_pct:.2f}%"
            )
            self.progress.emit(60)

            # ── Step 4: Save outputs (60% → 80%) ──────────────────────────
            self.log.emit(f"[{_ts()}] Saving outputs...")

            out_dir = Path(p["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)

            safe_method  = method.replace(" ", "_").replace("+", "").replace("__", "_")
            geotiff_path = str(out_dir / f"change_map_{safe_method}.tif")
            csv_path     = str(out_dir / f"statistics_{safe_method}.csv")

            save_raster(geotiff_path,
                        results["change_mask"].astype(np.float32),
                        gt, proj)
            self.log.emit(f"[{_ts()}] GeoTIFF saved: {geotiff_path}")

            self.progress.emit(80)

            stats_df = compute_change_statistics(results, method)
            export_statistics_csv(stats_df, csv_path)
            self.log.emit(f"[{_ts()}] CSV saved:     {csv_path}")

            self.progress.emit(100)
            self.log.emit(f"[{_ts()}] Done.")

            self.finished.emit({
                "results":       results,
                "method":        method,
                "geo_transform": gt,
                "projection":    proj,
                "geotiff_path":  geotiff_path,
                "csv_path":      csv_path,
            })

        except Exception as exc:
            tb = traceback.format_exc()
            QgsMessageLog.logMessage(tb, LOG_TAG, Qgis.Critical)
            self.error.emit(str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DIALOG
# ═══════════════════════════════════════════════════════════════════════════

class LandChangeDetectorDialog(QDialog, FORM_CLASS):
    """Main plugin dialog."""

    # Emitted after a successful run so the plugin can add the layer
    analysis_complete = pyqtSignal(str)

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.iface = iface

        self._worker       = None
        self._thread       = None
        self._last_output  = None

        self._connect_signals()
        self._set_initial_state()

    # ─────────────────────────────────────────────────────────────
    # SIGNAL CONNECTIONS
    # ─────────────────────────────────────────────────────────────

    def _connect_signals(self):
        self.btnBrowseBefore.clicked.connect(self._browse_before)
        self.btnBrowseAfter.clicked.connect(self._browse_after)
        self.btnBrowseSCLBefore.clicked.connect(self._browse_scl_before)
        self.btnBrowseSCLAfter.clicked.connect(self._browse_scl_after)

        self.checkCloudMask.toggled.connect(self._on_cloud_mask_toggled)

        self.btnRun.clicked.connect(self._run_analysis)
        self.btnExport.clicked.connect(self._export_results)
        self.btnClose.clicked.connect(self.reject)

    # ─────────────────────────────────────────────────────────────
    # INITIAL STATE
    # ─────────────────────────────────────────────────────────────

    def _set_initial_state(self):
        self.progressBar.setValue(0)
        self.btnExport.setEnabled(False)
        self._on_cloud_mask_toggled(self.checkCloudMask.isChecked())

    # ─────────────────────────────────────────────────────────────
    # BROWSE HANDLERS
    # ─────────────────────────────────────────────────────────────

    def _browse(self, line_edit, caption):
        path, _ = QFileDialog.getOpenFileName(
            self, caption,
            line_edit.text() or os.path.expanduser("~"),
            "JP2 Files (*.jp2);;All Files (*)"
        )
        if path:
            line_edit.setText(path)

    def _browse_before(self):
        self._browse(self.lineEditBefore, "Select Before Image (B04 .jp2)")

    def _browse_after(self):
        self._browse(self.lineEditAfter, "Select After Image (B04 .jp2)")

    def _browse_scl_before(self):
        self._browse(self.lineEditSCLBefore, "Select SCL Before (cloud mask .jp2)")

    def _browse_scl_after(self):
        self._browse(self.lineEditSCLAfter, "Select SCL After (cloud mask .jp2)")

    # ─────────────────────────────────────────────────────────────
    # UI STATE HANDLERS
    # ─────────────────────────────────────────────────────────────

    def _on_cloud_mask_toggled(self, checked):
        self.lineEditSCLBefore.setEnabled(checked)
        self.btnBrowseSCLBefore.setEnabled(checked)
        self.lineEditSCLAfter.setEnabled(checked)
        self.btnBrowseSCLAfter.setEnabled(checked)

    # ─────────────────────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────────────────────

    def _validate(self):
        before = self.lineEditBefore.text().strip()
        after  = self.lineEditAfter.text().strip()

        if not before:
            self._show_error("Please select a Before image (B04 .jp2).")
            return False
        if not os.path.isfile(before):
            self._show_error(f"Before image not found:\n{before}")
            return False

        if not after:
            self._show_error("Please select an After image (B04 .jp2).")
            return False
        if not os.path.isfile(after):
            self._show_error(f"After image not found:\n{after}")
            return False

        if self.checkCloudMask.isChecked():
            scl_b = self.lineEditSCLBefore.text().strip()
            scl_a = self.lineEditSCLAfter.text().strip()
            if not scl_b:
                self._show_error("Cloud masking enabled — select SCL Before.")
                return False
            if not os.path.isfile(scl_b):
                self._show_error(f"SCL Before not found:\n{scl_b}")
                return False
            if not scl_a:
                self._show_error("Cloud masking enabled — select SCL After.")
                return False
            if not os.path.isfile(scl_a):
                self._show_error(f"SCL After not found:\n{scl_a}")
                return False

        if not self.comboMethod.currentText():
            self._show_error("Please select a detection method.")
            return False

        return True

    # ─────────────────────────────────────────────────────────────
    # RUN ANALYSIS
    # ─────────────────────────────────────────────────────────────

    def _run_analysis(self):
        if not self._validate():
            return

        before  = self.lineEditBefore.text().strip()
        out_dir = str(Path(before).parent / "output")

        params = {
            "before_path":       before,
            "after_path":        self.lineEditAfter.text().strip(),
            "scl_before_path":   self.lineEditSCLBefore.text().strip() or None,
            "scl_after_path":    self.lineEditSCLAfter.text().strip()  or None,
            "method":            self.comboMethod.currentText(),
            "cloud_mask_enabled": self.checkCloudMask.isChecked(),
            "crop_size":         self.spinCropSize.value(),
            "output_dir":        out_dir,
        }

        self.textLog.clear()
        self._log(f"Starting analysis: {params['method']}")
        self._log(f"Crop size: {params['crop_size']} px")
        self.progressBar.setValue(0)
        self.btnExport.setEnabled(False)
        self.btnRun.setEnabled(False)

        self._thread = QThread()
        self._worker = AnalysisWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progressBar.setValue)
        self._worker.log.connect(self._log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    # ─────────────────────────────────────────────────────────────
    # RESULT HANDLERS
    # ─────────────────────────────────────────────────────────────

    def _cleanup_thread(self):
        """Null out thread/worker references after Qt schedules deleteLater."""
        self._thread = None
        self._worker = None

    def _on_finished(self, output):
        self._last_output = output
        self.btnRun.setEnabled(True)
        self.btnExport.setEnabled(True)

        results    = output["results"]
        method     = output["method"]
        geotiff    = output["geotiff_path"]
        change_pct = results.get("change_pct", 0.0)

        self._log(f"Analysis complete — {change_pct:.2f}% change detected.")

        self.analysis_complete.emit(geotiff or "")

    def _on_error(self, error_msg):
        self.btnRun.setEnabled(True)
        self._log(f"ERROR: {error_msg}")
        self._show_error(
            f"Processing failed:\n\n{error_msg}\n\n"
            "Check the QGIS log (View > Panels > Log Messages) for the traceback."
        )

    # ─────────────────────────────────────────────────────────────
    # EXPORT RESULTS
    # ─────────────────────────────────────────────────────────────

    def _export_results(self):
        if self._last_output is None:
            return
        geotiff = self._last_output.get("geotiff_path", "")
        csv     = self._last_output.get("csv_path", "")
        msg = "Exported:\n"
        if geotiff:
            msg += f"  GeoTIFF: {geotiff}\n"
        if csv:
            msg += f"  CSV:     {csv}\n"
        QMessageBox.information(self, "Export Complete", msg)

    # ─────────────────────────────────────────────────────────────
    # QGIS LAYER LOADING
    # ─────────────────────────────────────────────────────────────

    def _load_binary_layer(self, geotiff_path: str, layer_name: str):
        """Load a binary change mask with grey=0 / red=1 colour ramp."""
        layer = QgsRasterLayer(geotiff_path, layer_name)
        if not layer.isValid():
            QgsMessageLog.logMessage(
                f"Invalid layer: {geotiff_path}", LOG_TAG, Qgis.Warning
            )
            return

        from PyQt5.QtGui import QColor
        color_ramp = QgsColorRampShader()
        color_ramp.setColorRampType(QgsColorRampShader.Exact)
        color_ramp.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(0, QColor("#aaaaaa"), "No Change"),
            QgsColorRampShader.ColorRampItem(1, QColor("#e63946"), "Change"),
        ])

        raster_shader = QgsRasterShader()
        raster_shader.setRasterShaderFunction(color_ramp)

        renderer = QgsSingleBandPseudoColorRenderer(
            layer.dataProvider(), 1, raster_shader
        )
        layer.setRenderer(renderer)
        layer.triggerRepaint()

        QgsProject.instance().addMapLayer(layer)
        QgsMessageLog.logMessage(
            f"Layer '{layer_name}' added to canvas.", LOG_TAG, Qgis.Info
        )

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _log(self, message: str):
        self.textLog.append(message)

    def _show_error(self, message: str):
        QMessageBox.critical(self, "LandChangeDetector — Error", message)

    # ─────────────────────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        try:
            if self._thread is not None and self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(3000)
        except RuntimeError:
            pass
        self._thread = None
        self._worker = None
        super().closeEvent(event)
