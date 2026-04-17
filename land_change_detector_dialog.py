"""
land_change_detector_dialog.py
LandChangeDetector — Phase 4 + Phase 6

UI controller: connects the Qt dialog to the algorithm backend.
This file handles all button signals, validation, algorithm dispatch,
progress updates, and result rendering.

Author: Darius — 3rd Year AI Engineering
"""

import os
import traceback
import importlib
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.uic import loadUiType

from qgis.core import (
    QgsRasterLayer,
    QgsProject,
    QgsColorRampShader,
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
from algorithms.band_differencing import run_band_differencing
from algorithms.ndvi_differencing  import run_ndvi_differencing
from algorithms.cva                import run_cva


def _load_random_forest_runner():
    """Resolve the Random Forest runner from either supported module name."""
    for module_name in ("algorithms.random_forst", "algorithms.random_forest"):
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "run_random_forest", None)
        except ImportError:
            continue
    return None


run_random_forest = _load_random_forest_runner()

# ── Utility imports ─────────────────────────────────────────────────────────
from utils.raster_utils  import load_band_crop, load_cloud_mask, save_raster
from utils.stats_utils   import compute_change_statistics, export_statistics_csv
from utils.visualization import (
    render_change_map_in_qgis,
    render_cva_map_in_qgis,
)

# ── Crop to use for all band loading (memory-safe for development) ──────────
CROP = dict(x_off=4000, y_off=4000, x_size=2000, y_size=2000)
SCL_CROP = dict(x_off=2000, y_off=2000, x_size=1000, y_size=1000)

LOG_TAG = "LandChangeDetector"


# ═══════════════════════════════════════════════════════════════════════════
# WORKER — runs the algorithm in a background thread so QGIS UI doesn't freeze
# ═══════════════════════════════════════════════════════════════════════════

class AnalysisWorker(QObject):
    """
    Background worker that runs the selected algorithm.
    Emits progress(int), status(str), finished(dict), error(str).
    """
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, params):
        """
        Parameters
        ----------
        params : dict with keys:
            before_path, after_path, scl_path (or None),
            method, threshold (float or None),
            smooth, cloud_mask_enabled,
            output_dir, export_geotiff, export_csv
        """
        super().__init__()
        self.params = params

    def run(self):
        try:
            p = self.params

            # ── Step 1: Load bands ─────────────────────────────────────────
            self.status.emit("Loading before image bands...")
            self.progress.emit(5)

            before_dir = os.path.dirname(p["before_path"])
            after_dir  = os.path.dirname(p["after_path"])

            def load_sibling(base_dir, band_suffix):
                """Load a sibling band from the same directory."""
                files = [f for f in os.listdir(base_dir)
                         if band_suffix in f and f.endswith(".jp2")]
                if not files:
                    raise FileNotFoundError(
                        f"Could not find band '{band_suffix}' in {base_dir}"
                    )
                return load_band_crop(
                    os.path.join(base_dir, files[0]), **CROP
                )

            # Before bands
            red_b,   gt, proj = load_band_crop(p["before_path"], **CROP)
            green_b, _,  _    = load_sibling(before_dir, "B03_10m")
            blue_b,  _,  _    = load_sibling(before_dir, "B02_10m")
            nir_b,   _,  _    = load_sibling(before_dir, "B08_10m")

            self.status.emit("Loading after image bands...")
            self.progress.emit(15)

            # After bands
            red_a,   _,  _    = load_band_crop(p["after_path"], **CROP)
            green_a, _,  _    = load_sibling(after_dir, "B03_10m")
            blue_a,  _,  _    = load_sibling(after_dir, "B02_10m")
            nir_a,   _,  _    = load_sibling(after_dir, "B08_10m")

            # ── Step 2: Cloud masking ──────────────────────────────────────
            if p["cloud_mask_enabled"] and p["scl_path"]:
                self.status.emit("Applying cloud mask (SCL)...")
                self.progress.emit(25)

                _, valid_b = load_cloud_mask(
                    p["scl_path"], **SCL_CROP, target_size=red_b.shape
                )
                # After image: assume cloud-free if no second SCL provided
                valid_both = valid_b
            else:
                valid_both = np.ones(red_b.shape, dtype=bool)

            self.progress.emit(30)

            # Apply mask (NaN = cloudy)
            def mask(arr):
                return np.where(valid_both, arr, np.nan).astype(np.float32)

            red_b_m   = mask(red_b)
            green_b_m = mask(green_b)
            blue_b_m  = mask(blue_b)
            nir_b_m   = mask(nir_b)
            red_a_m   = mask(red_a)
            green_a_m = mask(green_a)
            blue_a_m  = mask(blue_a)
            nir_a_m   = mask(nir_a)

            threshold = p["threshold"]   # None = auto (Otsu)
            smooth    = p["smooth"]

            # ── Step 3: Run selected algorithm ─────────────────────────────
            method = p["method"]
            self.status.emit(f"Running {method}...")
            self.progress.emit(35)

            if method == "Band Differencing":
                results = run_band_differencing(
                    red_b_m, red_a_m,
                    threshold=threshold,
                    smooth=smooth
                )

            elif method == "NDVI Differencing":
                results = run_ndvi_differencing(
                    red_b_m, nir_b_m, red_a_m, nir_a_m,
                    threshold=threshold,
                    smooth=smooth
                )

            elif method == "Change Vector Analysis (CVA)":
                results = run_cva(
                    bands_before=[blue_b_m, green_b_m, red_b_m, nir_b_m],
                    bands_after =[blue_a_m, green_a_m, red_a_m, nir_a_m],
                    band_names  =["B02", "B03", "B04", "B08"],
                    threshold=threshold,
                    smooth=smooth
                )

            elif method == "Random Forest (Post-Classification)":
                if run_random_forest is None:
                    raise ImportError(
                        "Random Forest method is unavailable: could not import "
                        "algorithms.random_forest or algorithms.random_forst."
                    )
                results = run_random_forest(
                    bands_before=[blue_b_m, green_b_m, red_b_m, nir_b_m],
                    bands_after =[blue_a_m, green_a_m, red_a_m, nir_a_m],
                    n_classes=4,
                    n_train_samples=5000
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            self.progress.emit(75)

            # ── Step 4: Save outputs ───────────────────────────────────────
            out_dir = p["output_dir"]
            os.makedirs(out_dir, exist_ok=True)

            safe_method = method.replace(" ", "_").replace("(", "").replace(")", "")
            geotiff_path = os.path.join(out_dir, f"change_map_{safe_method}.tif")
            csv_path     = os.path.join(out_dir, f"statistics_{safe_method}.csv")

            if p["export_geotiff"]:
                self.status.emit("Saving GeoTIFF...")
                self.progress.emit(82)
                save_raster(
                    geotiff_path,
                    results["change_mask"].astype(np.float32),
                    gt, proj
                )

            if p["export_csv"]:
                self.status.emit("Exporting statistics CSV...")
                self.progress.emit(90)
                stats_df = compute_change_statistics(results, method)
                export_statistics_csv(stats_df, csv_path)

            self.progress.emit(100)
            self.status.emit("Done.")

            # Return everything the dialog needs to render the result
            self.finished.emit({
                "results":      results,
                "method":       method,
                "geo_transform": gt,
                "projection":   proj,
                "geotiff_path": geotiff_path if p["export_geotiff"] else None,
                "csv_path":     csv_path     if p["export_csv"]     else None,
            })

        except Exception as exc:
            tb = traceback.format_exc()
            QgsMessageLog.logMessage(tb, LOG_TAG, Qgis.Critical)
            self.error.emit(str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DIALOG CLASS
# ═══════════════════════════════════════════════════════════════════════════

class LandChangeDetectorDialog(QDialog, FORM_CLASS):
    """
    Main plugin dialog.

    Connects every widget signal to a handler, validates inputs,
    dispatches the worker thread, and renders results in QGIS.
    """

    def __init__(self, iface, parent=None):
        """
        Parameters
        ----------
        iface  : QgsInterface — QGIS interface object (passed from plugin entry point)
        parent : QWidget or None
        """
        super().__init__(parent)
        self.setupUi(self)
        self.iface = iface

        self._worker  = None
        self._thread  = None
        self._last_results = None   # store last run results for later use

        self._connect_signals()
        self._set_initial_state()

    # ─────────────────────────────────────────────────────────────
    # SIGNAL CONNECTIONS
    # ─────────────────────────────────────────────────────────────

    def _connect_signals(self):
        # Browse buttons
        self.btnBrowseBefore.clicked.connect(self._browse_before)
        self.btnBrowseAfter.clicked.connect(self._browse_after)
        self.btnBrowseSCL.clicked.connect(self._browse_scl)
        self.btnBrowseOutput.clicked.connect(self._browse_output)

        # Threshold radio buttons
        self.radioThresholdAuto.toggled.connect(self._on_threshold_mode_changed)
        self.radioThresholdManual.toggled.connect(self._on_threshold_mode_changed)

        # Cloud mask checkbox toggles SCL field
        self.checkCloudMask.toggled.connect(self._on_cloud_mask_toggled)

        # Method change — update UI hints
        self.comboMethod.currentIndexChanged.connect(self._on_method_changed)

        # Run / Close
        self.btnRun.clicked.connect(self._run_analysis)
        self.btnClose.clicked.connect(self.reject)

    # ─────────────────────────────────────────────────────────────
    # INITIAL STATE
    # ─────────────────────────────────────────────────────────────

    def _set_initial_state(self):
        self.progressBar.setValue(0)
        self.spinThreshold.setEnabled(False)
        self._set_status("Ready.")
        self._on_method_changed(0)

    # ─────────────────────────────────────────────────────────────
    # BROWSE HANDLERS
    # ─────────────────────────────────────────────────────────────

    def _browse_before(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Before Image Band",
            self.lineEditBefore.text() or os.path.expanduser("~"),
            "Raster files (*.jp2 *.tif *.tiff);;All files (*)"
        )
        if path:
            self.lineEditBefore.setText(path)

    def _browse_after(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select After Image Band",
            self.lineEditAfter.text() or os.path.expanduser("~"),
            "Raster files (*.jp2 *.tif *.tiff);;All files (*)"
        )
        if path:
            self.lineEditAfter.setText(path)

    def _browse_scl(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SCL Cloud Mask",
            self.lineEditSCL.text() or os.path.expanduser("~"),
            "Raster files (*.jp2 *.tif *.tiff);;All files (*)"
        )
        if path:
            self.lineEditSCL.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder",
            self.lineEditOutput.text() or os.path.expanduser("~")
        )
        if path:
            self.lineEditOutput.setText(path)

    # ─────────────────────────────────────────────────────────────
    # UI STATE HANDLERS
    # ─────────────────────────────────────────────────────────────

    def _on_threshold_mode_changed(self):
        """Enable the spin box only when Manual mode is selected."""
        self.spinThreshold.setEnabled(self.radioThresholdManual.isChecked())

    def _on_cloud_mask_toggled(self, checked):
        """Grey out the SCL field when cloud masking is disabled."""
        self.lineEditSCL.setEnabled(checked)
        self.btnBrowseSCL.setEnabled(checked)

    def _on_method_changed(self, index):
        """Update status bar hint when the user switches methods."""
        hints = {
            0: "Band Differencing — compares raw reflectance values.",
            1: "NDVI Differencing — most reliable for vegetation change.",
            2: "CVA — detects change type (vegetation/urban/water).",
            3: "Random Forest — post-classification comparison.",
        }
        self._set_status(hints.get(index, ""))

    # ─────────────────────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────────────────────

    def _validate(self):
        """
        Check all required fields are filled and files exist.
        Returns True if valid, shows an error QMessageBox if not.
        """
        before = self.lineEditBefore.text().strip()
        after  = self.lineEditAfter.text().strip()
        out    = self.lineEditOutput.text().strip()
        scl    = self.lineEditSCL.text().strip()

        if not before:
            self._show_error("Please select a Before image band.")
            return False
        if not os.path.isfile(before):
            self._show_error(f"Before image not found:\n{before}")
            return False

        if not after:
            self._show_error("Please select an After image band.")
            return False
        if not os.path.isfile(after):
            self._show_error(f"After image not found:\n{after}")
            return False

        if not out:
            self._show_error("Please select an output folder.")
            return False

        if self.checkCloudMask.isChecked() and scl:
            if not os.path.isfile(scl):
                self._show_error(f"SCL cloud mask not found:\n{scl}")
                return False

        if not self.checkExportGeoTIFF.isChecked() and \
           not self.checkExportCSV.isChecked():
            self._show_error(
                "Please enable at least one export option\n"
                "(GeoTIFF or CSV)."
            )
            return False

        return True

    # ─────────────────────────────────────────────────────────────
    # RUN ANALYSIS
    # ─────────────────────────────────────────────────────────────

    def _run_analysis(self):
        if not self._validate():
            return

        # Build params dict
        threshold = (
            None if self.radioThresholdAuto.isChecked()
            else self.spinThreshold.value()
        )
        scl_path = self.lineEditSCL.text().strip() or None

        params = {
            "before_path":         self.lineEditBefore.text().strip(),
            "after_path":          self.lineEditAfter.text().strip(),
            "scl_path":            scl_path,
            "method":              self.comboMethod.currentText(),
            "threshold":           threshold,
            "smooth":              self.checkSmooth.isChecked(),
            "cloud_mask_enabled":  self.checkCloudMask.isChecked(),
            "output_dir":          self.lineEditOutput.text().strip(),
            "export_geotiff":      self.checkExportGeoTIFF.isChecked(),
            "export_csv":          self.checkExportCSV.isChecked(),
        }

        # ── Disable UI during run ──────────────────────────────────────────
        self._set_running(True)
        self.progressBar.setValue(0)

        # ── Spin up background thread ──────────────────────────────────────
        self._thread = QThread()
        self._worker = AnalysisWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progressBar.setValue)
        self._worker.status.connect(self._set_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        # Clean up thread when done
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    # ─────────────────────────────────────────────────────────────
    # RESULT HANDLERS
    # ─────────────────────────────────────────────────────────────

    def _on_finished(self, output):
        """
        Called in the main thread after the worker finishes.
        Adds result layer to QGIS canvas and shows a summary.
        """
        self._set_running(False)
        self._last_results = output

        results   = output["results"]
        method    = output["method"]
        gt        = output["geo_transform"]
        proj      = output["projection"]
        geotiff   = output["geotiff_path"]

        # ── Add to QGIS canvas ─────────────────────────────────────────────
        if geotiff and os.path.isfile(geotiff):
            try:
                layer_name = f"Change Map — {method}"

                if method == "Change Vector Analysis (CVA)":
                    self._load_cva_layer(geotiff, results, gt, proj, layer_name)
                else:
                    self._load_binary_layer(geotiff, layer_name)

                self.iface.mapCanvas().refresh()

            except Exception as exc:
                QgsMessageLog.logMessage(
                    f"Layer rendering error: {exc}", LOG_TAG, Qgis.Warning
                )

        # ── Summary message box ────────────────────────────────────────────
        change_pct = results.get("change_pct", 0.0)
        threshold  = results.get("threshold", "N/A")

        summary = (
            f"<b>Method:</b> {method}<br>"
            f"<b>Change detected:</b> {change_pct:.2f}%<br>"
            f"<b>Threshold used:</b> {threshold}<br>"
        )

        # Add NDVI-specific info
        if "gain_pct" in results:
            summary += (
                f"<b>Vegetation gained:</b> {results['gain_pct']:.2f}%<br>"
                f"<b>Vegetation lost:</b>   {results['loss_pct']:.2f}%<br>"
            )

        # Add CVA type breakdown
        if "type_stats" in results:
            summary += "<br><b>Change types:</b><br>"
            for label, stats in results["type_stats"].items():
                summary += f"&nbsp;&nbsp;{label}: {stats['pct']:.2f}%<br>"

        if output["csv_path"]:
            summary += f"<br><b>CSV saved:</b> {output['csv_path']}"
        if output["geotiff_path"]:
            summary += f"<br><b>GeoTIFF saved:</b> {output['geotiff_path']}"

        QMessageBox.information(self, "Analysis Complete", summary)
        self._set_status("Analysis complete.")

    def _on_error(self, error_msg):
        """Called when the worker emits an error signal."""
        self._set_running(False)
        self._set_status("Error — see message below.")
        self._show_error(
            f"An error occurred during processing:\n\n{error_msg}\n\n"
            "Check the QGIS message log (View → Panels → Log Messages) "
            "for the full traceback."
        )

    # ─────────────────────────────────────────────────────────────
    # QGIS LAYER LOADING
    # ─────────────────────────────────────────────────────────────

    def _load_binary_layer(self, geotiff_path, layer_name):
        """
        Load a binary (0/1) change mask GeoTIFF into QGIS with a
        red (changed) / grey (unchanged) colour ramp.
        """
        layer = QgsRasterLayer(geotiff_path, layer_name)
        if not layer.isValid():
            QgsMessageLog.logMessage(
                f"Invalid layer: {geotiff_path}", LOG_TAG, Qgis.Warning
            )
            return

        # Colour ramp: 0 = light grey, 1 = red
        shader = QgsColorRampShader()
        shader.setColorRampType(QgsColorRampShader.Exact)
        from PyQt5.QtGui import QColor
        shader.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(0, QColor("#cccccc"), "No Change"),
            QgsColorRampShader.ColorRampItem(1, QColor("#e63946"), "Changed"),
        ])

        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1)
        renderer.setClassificationMin(0)
        renderer.setClassificationMax(1)
        renderer.setShader(shader)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

        QgsProject.instance().addMapLayer(layer)
        QgsMessageLog.logMessage(
            f"Layer '{layer_name}' added to canvas.", LOG_TAG, Qgis.Info
        )

    def _load_cva_layer(self, geotiff_path, results, gt, proj, layer_name):
        """
        Save the CVA change_type map (0–4) as a separate GeoTIFF and load
        it into QGIS with the 5-class colour scheme.
        """
        # Save the change_type uint8 map
        cva_type_path = geotiff_path.replace(".tif", "_types.tif")
        save_raster(
            cva_type_path,
            results["change_type"].astype(np.float32),
            gt, proj
        )

        layer = QgsRasterLayer(cva_type_path, layer_name)
        if not layer.isValid():
            return

        from PyQt5.QtGui import QColor
        CVA_STYLE = {
            0: ("#888888", "No Change"),
            1: ("#e63946", "Vegetation Loss"),
            2: ("#2dc653", "Vegetation Gain"),
            3: ("#f4a261", "Urban/Soil Gain"),
            4: ("#457b9d", "Water/Shadow"),
        }
        shader = QgsColorRampShader()
        shader.setColorRampType(QgsColorRampShader.Exact)
        shader.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(v, QColor(hex_c), label)
            for v, (hex_c, label) in CVA_STYLE.items()
        ])

        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1)
        renderer.setClassificationMin(0)
        renderer.setClassificationMax(4)
        renderer.setShader(shader)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

        QgsProject.instance().addMapLayer(layer)

    # ─────────────────────────────────────────────────────────────
    # UI HELPER METHODS
    # ─────────────────────────────────────────────────────────────

    def _set_running(self, is_running):
        """Lock/unlock controls during processing."""
        self.btnRun.setEnabled(not is_running)
        self.groupInput.setEnabled(not is_running)
        self.groupAlgorithm.setEnabled(not is_running)
        self.groupOutput.setEnabled(not is_running)
        if is_running:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _set_status(self, message):
        self.labelStatus.setText(message)
        QApplication.processEvents()   # force UI refresh

    def _show_error(self, message):
        QMessageBox.critical(self, "LandChangeDetector — Error", message)

    # ─────────────────────────────────────────────────────────────
    # CLEANUP ON CLOSE
    # ─────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Stop any running worker thread before closing."""
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)   # wait max 3 seconds
        QApplication.restoreOverrideCursor()
        super().closeEvent(event)
