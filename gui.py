"""Amharic Sentiment Intelligence Studio - Modern Desktop GUI.

Powered by PySide6 (Qt6) and AfriBERTa Transformer Engine with Decoupled Dual-Axis Calibration.
Supports Dark/Light Themes, Real-Time Amharic Neural Inference, and Golden Benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

from PySide6.QtCore import (
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThread,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QPainter,
    QPalette,
    QShortcut,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QStyle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.engine import SentimentInferenceEngine
from src.preprocessor import AmharicPreprocessor


# ==============================================================================
# DESIGN SYSTEM & THEMES
# ==============================================================================

DARK_THEME_QSS = """
QMainWindow, QWidget#MainContainer {
    background-color: #12121c;
    color: #f1f5f9;
    font-family: 'Inter', 'Segoe UI', 'Noto Sans Ethiopic', 'Ubuntu', sans-serif;
}

QTabWidget::pane {
    border: 1px solid #232336;
    background-color: #181826;
    border-radius: 12px;
    top: -1px;
}

QTabBar::tab {
    background: #181826;
    color: #94a3b8;
    padding: 10px 24px;
    margin-right: 4px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    font-weight: 600;
    font-size: 13px;
    border: 1px solid transparent;
}

QTabBar::tab:selected {
    background: #232338;
    color: #818cf8;
    border-bottom: 2px solid #6366f1;
}

QTabBar::tab:hover:!selected {
    background: #1e1e30;
    color: #cbd5e1;
}

QFrame#HeaderCard {
    background-color: #1a1a2a;
    border: 1px solid #282840;
    border-radius: 14px;
}

QFrame#ContentCard, QFrame#ResultCard, QFrame#MetricCard {
    background-color: #1a1a2b;
    border: 1px solid #27273f;
    border-radius: 14px;
}

QTextEdit#InputArea {
    background-color: #131320;
    color: #f8fafc;
    border: 1.5px solid #2d2d48;
    border-radius: 10px;
    padding: 12px;
    font-size: 15px;
    selection-background-color: #4f46e5;
}

QTextEdit#InputArea:focus {
    border: 1.5px solid #6366f1;
    background-color: #151524;
}

QPushButton#PrimaryBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #4f46e5);
    color: #ffffff;
    font-weight: 700;
    font-size: 14px;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
}

QPushButton#PrimaryBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7174f4, stop:1 #5a52ec);
}

QPushButton#PrimaryBtn:pressed {
    background-color: #4338ca;
}

QPushButton#SecondaryBtn {
    background-color: #232338;
    color: #cbd5e1;
    font-weight: 600;
    font-size: 12px;
    border: 1px solid #323250;
    border-radius: 8px;
    padding: 7px 14px;
}

QPushButton#SecondaryBtn:hover {
    background-color: #2b2b44;
    border-color: #6366f1;
    color: #ffffff;
}

QPushButton#PillBtn {
    background-color: #202034;
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid #2f2f4c;
    border-radius: 14px;
    padding: 6px 14px;
}

QPushButton#PillBtn:hover {
    background-color: #2a2a46;
    border-color: #818cf8;
    color: #ffffff;
}

QProgressBar {
    background-color: #141422;
    border: 1px solid #25253c;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
    font-size: 11px;
    font-weight: bold;
    height: 16px;
}

QProgressBar::chunk {
    border-radius: 5px;
}

QTableWidget {
    background-color: #141422;
    color: #e2e8f0;
    gridline-color: #222238;
    border: 1px solid #262640;
    border-radius: 10px;
    font-size: 13px;
    selection-background-color: #312e81;
}

QHeaderView::section {
    background-color: #1c1c2e;
    color: #94a3b8;
    padding: 8px;
    border: none;
    border-right: 1px solid #27273e;
    border-bottom: 1px solid #27273e;
    font-weight: bold;
}

QScrollBar:vertical {
    border: none;
    background: #141422;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #2d2d46;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #4f46e5;
}
"""

LIGHT_THEME_QSS = """
QMainWindow, QWidget#MainContainer {
    background-color: #f8fafc;
    color: #0f172a;
    font-family: 'Inter', 'Segoe UI', 'Noto Sans Ethiopic', 'Ubuntu', sans-serif;
}

QTabWidget::pane {
    border: 1px solid #e2e8f0;
    background-color: #ffffff;
    border-radius: 12px;
    top: -1px;
}

QTabBar::tab {
    background: #f1f5f9;
    color: #64748b;
    padding: 10px 24px;
    margin-right: 4px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    font-weight: 600;
    font-size: 13px;
    border: 1px solid transparent;
}

QTabBar::tab:selected {
    background: #ffffff;
    color: #4f46e5;
    border-bottom: 2px solid #4f46e5;
}

QTabBar::tab:hover:!selected {
    background: #e2e8f0;
    color: #1e293b;
}

QFrame#HeaderCard {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
}

QFrame#ContentCard, QFrame#ResultCard, QFrame#MetricCard {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
}

QTextEdit#InputArea {
    background-color: #ffffff;
    color: #0f172a;
    border: 1.5px solid #cbd5e1;
    border-radius: 10px;
    padding: 12px;
    font-size: 15px;
    selection-background-color: #c7d2fe;
}

QTextEdit#InputArea:focus {
    border: 1.5px solid #4f46e5;
    background-color: #ffffff;
}

QPushButton#PrimaryBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #4338ca);
    color: #ffffff;
    font-weight: 700;
    font-size: 14px;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
}

QPushButton#PrimaryBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5b52ec, stop:1 #4c41d6);
}

QPushButton#SecondaryBtn {
    background-color: #f1f5f9;
    color: #334155;
    font-weight: 600;
    font-size: 12px;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 7px 14px;
}

QPushButton#SecondaryBtn:hover {
    background-color: #e2e8f0;
    border-color: #4f46e5;
    color: #0f172a;
}

QPushButton#PillBtn {
    background-color: #eef2ff;
    color: #4338ca;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid #c7d2fe;
    border-radius: 14px;
    padding: 6px 14px;
}

QPushButton#PillBtn:hover {
    background-color: #e0e7ff;
    border-color: #4f46e5;
}

QProgressBar {
    background-color: #e2e8f0;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    text-align: center;
    color: #0f172a;
    font-size: 11px;
    font-weight: bold;
    height: 16px;
}

QProgressBar::chunk {
    border-radius: 5px;
}

QTableWidget {
    background-color: #ffffff;
    color: #0f172a;
    gridline-color: #e2e8f0;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    font-size: 13px;
    selection-background-color: #e0e7ff;
}

QHeaderView::section {
    background-color: #f8fafc;
    color: #475569;
    padding: 8px;
    border: none;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    font-weight: bold;
}
"""

COLOR_PALETTE = {
    "Positive": {"bg": "#22c55e", "text": "#ffffff", "icon": "😊", "name": "Positive (አዎንታዊ)"},
    "Negative": {"bg": "#ef4444", "text": "#ffffff", "icon": "😡", "name": "Negative (አሉታዊ)"},
    "Neutral":  {"bg": "#3b82f6", "text": "#ffffff", "icon": "😐", "name": "Neutral (ገለልተኛ)"},
    "Mixed":    {"bg": "#eab308", "text": "#000000", "icon": "🔀", "name": "Mixed (ድብልቅ)"},
}


# ==============================================================================
# BACKGROUND THREAD WORKERS
# ==============================================================================

class InferenceWorker(QThread):
    """Background worker executing neural inference without blocking GUI event loop."""
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, engine: SentimentInferenceEngine, text: str) -> None:
        super().__init__()
        self.engine = engine
        self.text = text

    def run(self) -> None:
        try:
            res = self.engine.predict(self.text)
            self.finished.emit(res)
        except Exception as exc:
            self.error.emit(str(exc))


class BenchmarkWorker(QThread):
    """Background worker executing benchmark cases sequentially."""
    case_started = Signal(int, str)
    case_finished = Signal(int, dict, bool)
    all_finished = Signal(int, int, float)
    error = Signal(str)

    def __init__(self, engine: SentimentInferenceEngine, cases_path: Path) -> None:
        super().__init__()
        self.engine = engine
        self.cases_path = cases_path

    def run(self) -> None:
        try:
            if not self.cases_path.exists():
                self.error.emit(f"Benchmark file not found: {self.cases_path}")
                return

            with open(self.cases_path, "r", encoding="utf-8") as f:
                cases = json.load(f)

            passed_count = 0
            total_latency = 0.0

            for idx, item in enumerate(cases):
                self.case_started.emit(idx, item.get("text", ""))
                res = self.engine.predict(item["text"])
                is_pass = res["class"].lower() == item["expected_class"].lower()
                if is_pass:
                    passed_count += 1
                total_latency += res["latency_ms"]
                self.case_finished.emit(idx, res, is_pass)

            avg_lat = total_latency / len(cases) if cases else 0.0
            self.all_finished.emit(len(cases), passed_count, avg_lat)
        except Exception as exc:
            self.error.emit(str(exc))


# ==============================================================================
# MAIN STUDIO WINDOW
# ==============================================================================

class SentimentStudioWindow(QMainWindow):
    """Main window for Amharic Sentiment Intelligence Studio."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Amharic Sentiment Intelligence Studio — Powered by AfriBERTa")
        self.resize(1100, 820)
        self.setMinimumSize(950, 700)

        self.is_dark_mode = True
        self.engine = SentimentInferenceEngine()

        self._init_ui()
        self.apply_theme(dark=True)

        # Asynchronously load weights on startup
        self.status_badge.setText("⏳ Loading AfriBERTa Weights...")
        self.status_badge.setStyleSheet("color: #f59e0b; font-weight: bold;")
        QThreadPool.globalInstance().start(self._async_load_weights)

    def _async_load_weights(self) -> None:
        try:
            self.engine.load()
            self.status_badge.setText("● AfriBERTa Engine Ready (4 CPU Threads)")
            self.status_badge.setStyleSheet("color: #10b981; font-weight: bold;")
        except Exception as e:
            self.status_badge.setText(f"⚠ Load Error: {e}")
            self.status_badge.setStyleSheet("color: #ef4444; font-weight: bold;")

    def _init_ui(self) -> None:
        central_widget = QWidget(self)
        central_widget.setObjectName("MainContainer")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # 1. Header Section
        header_card = QFrame()
        header_card.setObjectName("HeaderCard")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(20, 14, 20, 14)

        title_vbox = QVBoxLayout()
        title_label = QLabel("Amharic Sentiment Intelligence Studio")
        title_label.setFont(QFont("Inter", 18, QFont.Bold))
        subtitle_label = QLabel("Real-Time AfriBERTa Neural Classification with Decoupled Dual-Axis Calibration")
        subtitle_label.setStyleSheet("color: #94a3b8; font-size: 12px;")
        title_vbox.addWidget(title_label)
        title_vbox.addWidget(subtitle_label)
        header_layout.addLayout(title_vbox)

        header_layout.addStretch()

        # Engine Status Badge
        self.status_badge = QLabel("● Model Initializing...")
        self.status_badge.setStyleSheet("color: #f59e0b; font-size: 12px; padding: 4px 10px; background: #1e1e30; border-radius: 8px;")
        header_layout.addWidget(self.status_badge)

        # Theme Toggle Button
        self.theme_toggle_btn = QPushButton("🌙 Dark Mode")
        self.theme_toggle_btn.setObjectName("SecondaryBtn")
        self.theme_toggle_btn.setCursor(Qt.PointingHandCursor)
        self.theme_toggle_btn.clicked.connect(self.toggle_theme)
        header_layout.addWidget(self.theme_toggle_btn)

        main_layout.addWidget(header_card)

        # 2. Main Tabs
        self.tabs = QTabWidget()
        self.tab_single = QWidget()
        self.tab_benchmark = QWidget()

        self._setup_single_analysis_tab()
        self._setup_benchmark_tab()

        self.tabs.addTab(self.tab_single, "🔍 Real-Time Analysis")
        self.tabs.addTab(self.tab_benchmark, "📊 Interactive Golden Benchmark")
        main_layout.addWidget(self.tabs)

    # --------------------------------------------------------------------------
    # TAB 1: SINGLE ANALYSIS
    # --------------------------------------------------------------------------
    def _setup_single_analysis_tab(self) -> None:
        layout = QVBoxLayout(self.tab_single)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Quick Preset Buttons Row
        samples_box = QHBoxLayout()
        samples_label = QLabel("Quick Pre-fills:")
        samples_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #94a3b8;")
        samples_box.addWidget(samples_label)

        presets = [
            ("😊 ፈጣን አገልግሎት", "የደንበኞች አገልግሎታችሁ እጅግ በጣም ፈጣን እና የሚያረካ ነው፡ በጣም አመሰግናለሁ!"),
            ("😡 ሲስተም አይሰራም", "ሲስተማችሁ ሁልጊዜ አይሰራም፡ ገንዘቤ ተቆርጦ አገልግሎት አላገኘሁም፡ በጣም አሳፋሪ ነው!"),
            ("😐 የባንክ ሰዓት", "የባንኩ ዋና መስሪያ ቤት ከሰኞ እስከ አርብ ከጠዋቱ 2:00 እስከ 11:00 ክፍት ነው"),
            ("🔀 ውበት አለው ግን...", "አዲሱ አፕሊኬሽን ውበት አለው ግን ሎግኢን ለማድረግ በጣም ያስቸግራል"),
            ("🔥 ቪዲዮው ይመቻል", "ቪዲዮው በእውነት ይመቻል፡ አሪፍ ስራ ነው ባክህ! 🔥"),
        ]

        for label, text in presets:
            btn = QPushButton(label)
            btn.setObjectName("PillBtn")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _, t=text: self.set_input_text(t))
            samples_box.addWidget(btn)

        samples_box.addStretch()
        layout.addLayout(samples_box)

        # Splitter between Input & Output
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left Panel: Input Area
        input_card = QFrame()
        input_card.setObjectName("ContentCard")
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(16, 16, 16, 16)
        input_layout.setSpacing(12)

        input_header_layout = QHBoxLayout()
        input_header_title = QLabel("Amharic Text Input (የአማርኛ ጽሑፍ)")
        input_header_title.setFont(QFont("Inter", 13, QFont.Bold))
        self.char_word_badge = QLabel("0 characters | 0 words")
        self.char_word_badge.setStyleSheet("color: #94a3b8; font-size: 11px;")
        input_header_layout.addWidget(input_header_title)
        input_header_layout.addStretch()
        input_header_layout.addWidget(self.char_word_badge)
        input_layout.addLayout(input_header_layout)

        self.input_text_edit = QTextEdit()
        self.input_text_edit.setObjectName("InputArea")
        self.input_text_edit.setPlaceholderText("የአማርኛ ጽሑፍ እዚህ ያስገቡ...\n(Type or paste Amharic text here, then click Analyze)")
        self.input_text_edit.textChanged.connect(self._on_input_text_changed)
        input_layout.addWidget(self.input_text_edit)

        # Action Buttons
        actions_layout = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("SecondaryBtn")
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.clicked.connect(lambda: self.input_text_edit.clear())
        actions_layout.addWidget(clear_btn)

        actions_layout.addStretch()

        self.analyze_btn = QPushButton("Analyze Sentiment (ተንትን)")
        self.analyze_btn.setObjectName("PrimaryBtn")
        self.analyze_btn.setCursor(Qt.PointingHandCursor)
        self.analyze_btn.clicked.connect(self.run_single_inference)
        actions_layout.addWidget(self.analyze_btn)

        input_layout.addLayout(actions_layout)
        splitter.addWidget(input_card)

        # Right Panel: Results Dashboard
        results_card = QFrame()
        results_card.setObjectName("ResultCard")
        results_layout = QVBoxLayout(results_card)
        results_layout.setContentsMargins(18, 18, 18, 18)
        results_layout.setSpacing(14)

        results_header = QLabel("Neural Analysis Result")
        results_header.setFont(QFont("Inter", 13, QFont.Bold))
        results_layout.addWidget(results_header)

        # Primary Sentiment Badge (Big Hero Card)
        self.sentiment_hero_frame = QFrame()
        self.sentiment_hero_frame.setStyleSheet("background: #232338; border-radius: 12px; padding: 14px;")
        hero_layout = QHBoxLayout(self.sentiment_hero_frame)

        self.hero_emoji_label = QLabel("✨")
        self.hero_emoji_label.setFont(QFont("Inter", 32))
        hero_layout.addWidget(self.hero_emoji_label)

        hero_text_vbox = QVBoxLayout()
        self.hero_class_label = QLabel("Awaiting Input")
        self.hero_class_label.setFont(QFont("Inter", 18, QFont.Bold))
        self.hero_conf_label = QLabel("Confidence: --%")
        self.hero_conf_label.setStyleSheet("color: #cbd5e1; font-size: 13px;")
        hero_text_vbox.addWidget(self.hero_class_label)
        hero_text_vbox.addWidget(self.hero_conf_label)
        hero_layout.addLayout(hero_text_vbox)
        hero_layout.addStretch()
        results_layout.addWidget(self.sentiment_hero_frame)

        # Probability Gauges / Progress Bars
        bars_box = QVBoxLayout()
        bars_box.setSpacing(8)

        # Positive Bar
        pos_header = QHBoxLayout()
        pos_header.addWidget(QLabel("🟢 Positive Pole Activation:"))
        self.pos_val_label = QLabel("0.0%")
        self.pos_val_label.setStyleSheet("font-weight: bold; color: #22c55e;")
        pos_header.addStretch()
        pos_header.addWidget(self.pos_val_label)
        bars_box.addLayout(pos_header)
        self.pos_bar = QProgressBar()
        self.pos_bar.setStyleSheet("QProgressBar::chunk { background-color: #22c55e; }")
        self.pos_bar.setValue(0)
        bars_box.addWidget(self.pos_bar)

        # Negative Bar
        neg_header = QHBoxLayout()
        neg_header.addWidget(QLabel("🔴 Negative Pole Activation:"))
        self.neg_val_label = QLabel("0.0%")
        self.neg_val_label.setStyleSheet("font-weight: bold; color: #ef4444;")
        neg_header.addStretch()
        neg_header.addWidget(self.neg_val_label)
        bars_box.addLayout(neg_header)
        self.neg_bar = QProgressBar()
        self.neg_bar.setStyleSheet("QProgressBar::chunk { background-color: #ef4444; }")
        self.neg_bar.setValue(0)
        bars_box.addWidget(self.neg_bar)

        # Neutral Bar
        neu_header = QHBoxLayout()
        neu_header.addWidget(QLabel("🔵 Neutral Baseline:"))
        self.neu_val_label = QLabel("0.0%")
        self.neu_val_label.setStyleSheet("font-weight: bold; color: #3b82f6;")
        neu_header.addStretch()
        neu_header.addWidget(self.neu_val_label)
        bars_box.addLayout(neu_header)
        self.neu_bar = QProgressBar()
        self.neu_bar.setStyleSheet("QProgressBar::chunk { background-color: #3b82f6; }")
        self.neu_bar.setValue(0)
        bars_box.addWidget(self.neu_bar)

        results_layout.addLayout(bars_box)

        # Metrics Metadata Grid
        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(12)
        metrics_grid.setVerticalSpacing(8)

        self.latency_metric = QLabel("-- ms")
        self.latency_metric.setStyleSheet("font-weight: bold; color: #a5b4fc;")
        self.clauses_metric = QLabel("--")
        self.clauses_metric.setStyleSheet("font-weight: bold; color: #a5b4fc;")
        self.cleaned_text_metric = QLabel("--")
        self.cleaned_text_metric.setWordWrap(True)
        self.cleaned_text_metric.setStyleSheet("font-size: 11px; color: #94a3b8;")

        metrics_grid.addWidget(QLabel("⏱ Latency:"), 0, 0)
        metrics_grid.addWidget(self.latency_metric, 0, 1)
        metrics_grid.addWidget(QLabel("🧩 Clauses:"), 0, 2)
        metrics_grid.addWidget(self.clauses_metric, 0, 3)
        metrics_grid.addWidget(QLabel("✨ Cleaned:"), 1, 0)
        metrics_grid.addWidget(self.cleaned_text_metric, 1, 1, 1, 3)

        results_layout.addLayout(metrics_grid)
        results_layout.addStretch()

        splitter.addWidget(results_card)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter)

    def _on_input_text_changed(self) -> None:
        text = self.input_text_edit.toPlainText()
        chars = len(text)
        words = len(text.split())
        self.char_word_badge.setText(f"{chars} chars | {words} words")

    def set_input_text(self, text: str) -> None:
        self.input_text_edit.setPlainText(text)
        self.run_single_inference()

    def run_single_inference(self) -> None:
        text = self.input_text_edit.toPlainText().strip()
        if not text:
            return

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Analyzing...")

        self.worker = InferenceWorker(self.engine, text)
        self.worker.finished.connect(self._on_inference_finished)
        self.worker.error.connect(self._on_inference_error)
        self.worker.start()

    def _on_inference_finished(self, result: dict[str, Any]) -> None:
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze Sentiment (ተንትን)")

        s_class = result["class"]
        conf = result["confidence"]
        p_pos = result["p_pos"]
        p_neg = result["p_neg"]
        latency = result["latency_ms"]
        cleaned = result["cleaned_text"]

        p_neu = max(0.0, 1.0 - (p_pos + p_neg)) if (p_pos + p_neg) <= 1.0 else 0.0

        meta = COLOR_PALETTE.get(s_class, COLOR_PALETTE["Neutral"])
        self.hero_emoji_label.setText(meta["icon"])
        self.hero_class_label.setText(meta["name"])
        self.hero_class_label.setStyleSheet(f"color: {meta['bg']}; font-weight: bold; font-size: 18px;")
        self.hero_conf_label.setText(f"Confidence: {conf:.2f}% (Dual-Axis Calibrated)")

        self.pos_val_label.setText(f"{p_pos * 100:.1f}%")
        self.pos_bar.setValue(int(p_pos * 100))

        self.neg_val_label.setText(f"{p_neg * 100:.1f}%")
        self.neg_bar.setValue(int(p_neg * 100))

        self.neu_val_label.setText(f"{p_neu * 100:.1f}%")
        self.neu_bar.setValue(int(p_neu * 100))

        self.latency_metric.setText(f"{latency:.2f} ms")
        clauses = self.engine._extract_clauses(raw_text=self.input_text_edit.toPlainText())
        self.clauses_metric.setText(f"{len(clauses)} detected")
        self.cleaned_text_metric.setText(cleaned)

    def _on_inference_error(self, err_msg: str) -> None:
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Analyze Sentiment (ተንትን)")
        self.hero_class_label.setText("Inference Error")
        self.hero_class_label.setStyleSheet("color: #ef4444; font-weight: bold;")
        self.hero_conf_label.setText(err_msg)

    # --------------------------------------------------------------------------
    # TAB 2: INTERACTIVE GOLDEN BENCHMARK
    # --------------------------------------------------------------------------
    def _setup_benchmark_tab(self) -> None:
        layout = QVBoxLayout(self.tab_benchmark)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Benchmark Top Control Bar
        top_bar = QHBoxLayout()
        bench_desc = QLabel("Automated 10-Case Golden Benchmark Suite covering Praise, Outrage, Negation, Mixed & Slang.")
        bench_desc.setStyleSheet("color: #94a3b8; font-size: 12px;")
        top_bar.addWidget(bench_desc)
        top_bar.addStretch()

        self.run_bench_btn = QPushButton("🚀 Run Golden Benchmark")
        self.run_bench_btn.setObjectName("PrimaryBtn")
        self.run_bench_btn.setCursor(Qt.PointingHandCursor)
        self.run_bench_btn.clicked.connect(self.run_benchmark)
        top_bar.addWidget(self.run_bench_btn)
        layout.addLayout(top_bar)

        # Benchmark Summary Metrics Bar
        self.bench_summary_card = QFrame()
        self.bench_summary_card.setObjectName("ContentCard")
        summary_layout = QHBoxLayout(self.bench_summary_card)
        summary_layout.setContentsMargins(16, 10, 16, 10)

        self.bench_total_label = QLabel("Total Cases: 10")
        self.bench_passed_label = QLabel("Passed: -- / 10")
        self.bench_acc_label = QLabel("Accuracy: --%")
        self.bench_lat_label = QLabel("Avg Latency: -- ms")

        for lbl in [self.bench_total_label, self.bench_passed_label, self.bench_acc_label, self.bench_lat_label]:
            lbl.setFont(QFont("Inter", 12, QFont.Bold))
            summary_layout.addWidget(lbl)
            summary_layout.addStretch()

        layout.addWidget(self.bench_summary_card)

        # Benchmark Table
        self.bench_table = QTableWidget()
        self.bench_table.setColumnCount(8)
        self.bench_table.setHorizontalHeaderLabels([
            "#", "Category", "Amharic Text Preview", "Expected", "Predicted", "Confidence", "P(pos) / P(neg)", "Status"
        ])
        header = self.bench_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)

        layout.addWidget(self.bench_table)

    def run_benchmark(self) -> None:
        cases_file = Path("tests/benchmark_cases.json")
        if not cases_file.exists():
            return

        with open(cases_file, "r", encoding="utf-8") as f:
            cases = json.load(f)

        self.bench_table.setRowCount(len(cases))
        for r, item in enumerate(cases):
            self.bench_table.setItem(r, 0, QTableWidgetItem(str(item.get("id", r + 1))))
            self.bench_table.setItem(r, 1, QTableWidgetItem(item.get("category", "")))
            self.bench_table.setItem(r, 2, QTableWidgetItem(item.get("text", "")))
            self.bench_table.setItem(r, 3, QTableWidgetItem(item.get("expected_class", "")))
            self.bench_table.setItem(r, 4, QTableWidgetItem("Waiting..."))
            self.bench_table.setItem(r, 5, QTableWidgetItem("--"))
            self.bench_table.setItem(r, 6, QTableWidgetItem("--"))
            self.bench_table.setItem(r, 7, QTableWidgetItem("⏳"))

        self.run_bench_btn.setEnabled(False)
        self.run_bench_btn.setText("Running Benchmark...")

        self.bench_worker = BenchmarkWorker(self.engine, cases_file)
        self.bench_worker.case_finished.connect(self._on_bench_case_finished)
        self.bench_worker.all_finished.connect(self._on_bench_all_finished)
        self.bench_worker.start()

    def _on_bench_case_finished(self, row: int, result: dict[str, Any], is_pass: bool) -> None:
        p_class = result["class"]
        conf = result["confidence"]
        p_pos = result["p_pos"]
        p_neg = result["p_neg"]

        color = COLOR_PALETTE.get(p_class, {}).get("bg", "#94a3b8")
        pred_item = QTableWidgetItem(p_class)
        pred_item.setForeground(QColor(color))
        pred_item.setFont(QFont("Inter", 11, QFont.Bold))
        self.bench_table.setItem(row, 4, pred_item)

        self.bench_table.setItem(row, 5, QTableWidgetItem(f"{conf:.1f}%"))
        self.bench_table.setItem(row, 6, QTableWidgetItem(f"{p_pos:.2f} / {p_neg:.2f}"))

        status_item = QTableWidgetItem("✔ PASS" if is_pass else "✖ FAIL")
        status_item.setForeground(QColor("#22c55e" if is_pass else "#ef4444"))
        status_item.setFont(QFont("Inter", 11, QFont.Bold))
        self.bench_table.setItem(row, 7, status_item)

    def _on_bench_all_finished(self, total: int, passed: int, avg_lat: float) -> None:
        self.run_bench_btn.setEnabled(True)
        self.run_bench_btn.setText("🚀 Run Golden Benchmark")

        acc = (passed / total) * 100.0 if total else 0.0
        self.bench_passed_label.setText(f"Passed: {passed} / {total}")
        self.bench_passed_label.setStyleSheet("color: #22c55e;" if passed == total else "color: #f59e0b;")
        self.bench_acc_label.setText(f"Accuracy: {acc:.1f}%")
        self.bench_acc_label.setStyleSheet("color: #22c55e;" if acc >= 90 else "color: #ef4444;")
        self.bench_lat_label.setText(f"Avg Latency: {avg_lat:.2f} ms")

    # --------------------------------------------------------------------------
    # THEME TOGGLE
    # --------------------------------------------------------------------------
    def toggle_theme(self) -> None:
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme(dark=self.is_dark_mode)

    def apply_theme(self, dark: bool) -> None:
        if dark:
            self.setStyleSheet(DARK_THEME_QSS)
            self.theme_toggle_btn.setText("🌙 Dark Mode")
        else:
            self.setStyleSheet(LIGHT_THEME_QSS)
            self.theme_toggle_btn.setText("☀ Light Mode")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main() -> None:
    """Launch the Amharic Sentiment Intelligence Studio application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Amharic Sentiment Intelligence Studio")
    app.setApplicationDisplayName("Amharic Sentiment Studio")

    window = SentimentStudioWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
