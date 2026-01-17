import sys
import pandas as pd
import numpy as np
import shap
import plotly.express as px
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox, QScrollArea
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
import joblib

# -------------------------------
# Load saved model and scaler
# -------------------------------
try:
    model = joblib.load("placement_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    raise Exception("Model or scaler file not found. Save your model first using joblib.")

# Features your model expects
features = [
    'work_experience_months', 'backlogs', 'soft_skill_score',
    'technical_skill_score', 'certifications', 'entrance_exam_score',
    'internship_count', 'live_projects'
]

explainer = shap.TreeExplainer(model)

# -------------------------------
# GUI Definition
# -------------------------------
class PlacementGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Student Placement Predictor")
        self.setGeometry(100, 100, 800, 700)
        self.inputs = {}
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.init_ui()

    def init_ui(self):
        # Scroll area for inputs
        scroll = QScrollArea()
        input_widget = QWidget()
        input_layout = QVBoxLayout()
        input_widget.setLayout(input_layout)
        scroll.setWidgetResizable(True)
        scroll.setWidget(input_widget)
        self.layout.addWidget(scroll)

        # Create input fields
        for feature in features:
            hlayout = QHBoxLayout()
            label = QLabel(f"{feature.replace('_',' ').title()}:")
            label.setFixedWidth(200)
            line_edit = QLineEdit()
            line_edit.setPlaceholderText("Enter value")
            hlayout.addWidget(label)
            hlayout.addWidget(line_edit)
            input_layout.addLayout(hlayout)
            self.inputs[feature] = line_edit

        # Predict button
        self.predict_btn = QPushButton("Predict Placement")
        self.predict_btn.clicked.connect(self.predict)
        self.layout.addWidget(self.predict_btn)

        # Output label
        self.output_label = QLabel("")
        self.output_label.setStyleSheet("font-size:16px; font-weight:bold; color:blue;")
        self.layout.addWidget(self.output_label)

        # Plotly Webview
        self.webview = QWebEngineView()
        self.webview.setFixedHeight(400)
        self.layout.addWidget(self.webview)

    # -------------------------------
    # Prediction function
    # -------------------------------
    def predict(self):
        try:
            # Gather inputs from user
            student_data = []
            for feature in features:
                value = float(self.inputs[feature].text())
                student_data.append(value)

            # Scale inputs
            student_scaled = scaler.transform([student_data])

            # Predict probability
            prob = model.predict_proba(student_scaled)[:, 1][0]
            placement = int(prob >= 0.59)  # threshold
            self.output_label.setText(f"Predicted Placement: {placement} (Probability: {prob * 100:.1f}%)")

            # SHAP values for explanation
            shap_values = explainer.shap_values(student_scaled)
            shap_contrib = shap_values[0][:, 1]  # Class 1 contributions

            # Pie chart of absolute contributions
            feature_contrib = pd.DataFrame({
                'Feature': features,
                'Contribution': np.abs(shap_contrib)
            })
            fig = px.pie(
                feature_contrib,
                values='Contribution',
                names='Feature',
                title='Feature Contributions to Placement Prediction',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            html = fig.to_html(include_plotlyjs='cdn')
            self.webview.setHtml(html)

        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter numeric values for all features.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# -------------------------------
# Run the application
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlacementGUI()
    window.show()
    sys.exit(app.exec())
