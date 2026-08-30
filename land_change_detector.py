"""
land_change_detector.py
LandChangeDetector — Phase 6 (Plugin Integration)

Main QGIS plugin class.
Registers the toolbar button, menu entry, and opens the dialog.

This file is the spine of the plugin:
    __init__.py  →  classFactory()  →  LandChangeDetector (this file)
                                          ↓
                                    LandChangeDetectorDialog  (dialog.py)
                                          ↓
                                    AnalysisWorker (background thread)
                                          ↓
                                    algorithms/*  +  utils/*

Author: Darius — 3rd Year AI Engineering
"""

import os

from PyQt5.QtGui     import QColor, QIcon
from PyQt5.QtWidgets import QAction

from qgis.core import (
    QgsColorRampShader,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    Qgis,
)


PLUGIN_DIR = os.path.dirname(__file__)
LOG_TAG    = "LandChangeDetector"


class LandChangeDetector:
    """
    QGIS Plugin entry-point class.

    QGIS calls:
        __init__(iface)        once, when QGIS loads the plugin
        initGui()              once, to add toolbar/menu items
        unload()               once, when the plugin is disabled/removed
    """

    def __init__(self, iface):
        """
        Parameters
        ----------
        iface : QgsInterface
            The QGIS interface object — gives access to canvas, toolbar,
            menu bar, message bar, and everything else QGIS exposes.
        """
        self.iface   = iface
        self.canvas  = iface.mapCanvas()
        self.actions = []          # keep references so we can remove them in unload()
        self.menu    = "&LandChangeDetector"
        self.dialog  = None        # created lazily on first open

        QgsMessageLog.logMessage(
            "LandChangeDetector plugin initialised.", LOG_TAG, Qgis.Info
        )

    # ─────────────────────────────────────────────────────────────
    # INIT GUI — called by QGIS after __init__
    # ─────────────────────────────────────────────────────────────

    def initGui(self):
        """
        Create and register all toolbar buttons and menu items.
        Called once by QGIS after the plugin is loaded.
        """

        # ── Main action: open the change detection dialog ─────────────────
        icon_path = os.path.join(PLUGIN_DIR, "icon.png")
        icon      = QIcon(icon_path) if os.path.isfile(icon_path) else QIcon()

        self.action_open = QAction(
            icon,
            "Land Change Detector",
            self.iface.mainWindow()
        )
        self.action_open.setObjectName("openLandChangeDetector")
        self.action_open.setStatusTip(
            "Open the LandChangeDetector dialog — "
            "bi-temporal land use/land cover change detection"
        )
        self.action_open.setWhatsThis(
            "Runs one of four change detection algorithms "
            "(Band Differencing, NDVI Differencing, CVA, or Random Forest) "
            "on a pair of Sentinel-2 L2A images."
        )
        self.action_open.triggered.connect(self.open_dialog)

        # ── About action ──────────────────────────────────────────────────
        self.action_about = QAction(
            QIcon(),
            "About LandChangeDetector",
            self.iface.mainWindow()
        )
        self.action_about.setObjectName("aboutLandChangeDetector")
        self.action_about.triggered.connect(self.show_about)

        # ── Register actions in Plugins menu and main toolbar ─────────────
        self.iface.addPluginToMenu(self.menu, self.action_open)
        self.iface.addPluginToMenu(self.menu, self.action_about)
        self.iface.addToolBarIcon(self.action_open)

        # Keep references for cleanup
        self.actions.extend([self.action_open, self.action_about])

        QgsMessageLog.logMessage(
            "LandChangeDetector GUI registered.", LOG_TAG, Qgis.Info
        )

    # ─────────────────────────────────────────────────────────────
    # UNLOAD — called by QGIS when plugin is disabled/removed
    # ─────────────────────────────────────────────────────────────

    def unload(self):
        """
        Remove all toolbar buttons and menu items.
        Called by QGIS when the plugin is disabled or uninstalled.
        """
        for action in self.actions:
            self.iface.removePluginMenu(self.menu, action)
            self.iface.removeToolBarIcon(action)
        self.actions.clear()

        # Close dialog if open
        if self.dialog is not None:
            self.dialog.close()
            self.dialog = None

        QgsMessageLog.logMessage(
            "LandChangeDetector unloaded.", LOG_TAG, Qgis.Info
        )

    # ─────────────────────────────────────────────────────────────
    # OPEN DIALOG
    # ─────────────────────────────────────────────────────────────

    def open_dialog(self):
        """
        Open (or bring to front) the main LandChangeDetector dialog.
        The dialog is created once and reused — state is preserved
        between opens in the same QGIS session.
        """
        # Import here (not at top) so QGIS loads the plugin faster
        from .land_change_detector_dialog import LandChangeDetectorDialog

        if self.dialog is None:
            self.dialog = LandChangeDetectorDialog(
                iface=self.iface,
                parent=self.iface.mainWindow()
            )
            self.dialog.analysis_complete.connect(self._on_analysis_complete)
            QgsMessageLog.logMessage(
                "LandChangeDetectorDialog created.", LOG_TAG, Qgis.Info
            )

        # Show non-modal so the user can still interact with QGIS canvas
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    # ─────────────────────────────────────────────────────────────
    # ABOUT
    # ─────────────────────────────────────────────────────────────

    def _on_analysis_complete(self, result_path: str):
        """Load the result GeoTIFF into QGIS canvas with a grey/red colour ramp."""
        if not result_path or not os.path.isfile(result_path):
            return

        layer = QgsRasterLayer(result_path, "Change Detection Result")
        if not layer.isValid():
            QgsMessageLog.logMessage(
                f"Could not load result layer: {result_path}", LOG_TAG, Qgis.Warning
            )
            return

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

        QgsProject.instance().addMapLayer(layer)
        self.iface.mapCanvas().refresh()

        QgsMessageLog.logMessage(
            "Change Detection Result layer added to canvas.", LOG_TAG, Qgis.Info
        )

    def show_about(self):
        """Show a simple About message box."""
        from PyQt5.QtWidgets import QMessageBox

        QMessageBox.about(
            self.iface.mainWindow(),
            "About LandChangeDetector",
            "<b>LandChangeDetector</b><br>"
            "Version 1.0 — April 2026<br><br>"
            "Bi-temporal land use / land cover change detection<br>"
            "using Sentinel-2 L2A satellite imagery.<br><br>"
            "<b>Study area:</b> Batna Province, Algeria (tile T31SGV)<br>"
            "<b>Time period:</b> June 2019 → August 2025<br><br>"
            "<b>Algorithms:</b><br>"
            "&nbsp;&nbsp;• Band Differencing<br>"
            "&nbsp;&nbsp;• NDVI Differencing<br>"
            "&nbsp;&nbsp;• Change Vector Analysis (CVA)<br>"
            "&nbsp;&nbsp;• Random Forest (Post-Classification)<br><br>"
            "<b>Author:</b> Darius — 3rd Year AI Engineering<br>"
            "<b>GitHub:</b> https://github.com/Radhoine-Amara/LandChangeDetector"
        )
