import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report
import shap
import plotly.express as px
import joblib

data = pd.read_csv('student_academic_placement_performance_dataset.csv')

new_student = pd.DataFrame([{
    'work_experience_months': 9,
    'backlogs': 16,
    'soft_skill_score': 73,
    'technical_skill_score': 94,
    'certifications': 4,
    'entrance_exam_score': 73,
    'internship_count': 3,
    'live_projects': 3
}])

x = data[['work_experience_months', 'backlogs', 'soft_skill_score', 'technical_skill_score', 'certifications', 'entrance_exam_score', 'internship_count', 'live_projects']]
y = data[['placement_status']]
y = y.values.ravel()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
new_student_scaled = scaler.transform(new_student)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42
)
model.fit(x_train_scaled, y_train)
probs = model.predict_proba(x_test_scaled)[:, 1]

#custom student check
prob = model.predict_proba(new_student_scaled)[:, 1][0]
placement_prediction = int(prob >= 0.59)

#predictions = model.predict(x_test_scaled)
predictions = (probs >= 0.59).astype(int)
print("Predictions:", predictions)


print(data[['placement_status']].value_counts())
print(classification_report(y_test, predictions))
print(f"Predicted custom Placement Status: {placement_prediction} (Probability: {prob:.2f})")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(new_student_scaled)

shap_contributions = shap_values[0][:, 1]  # shape = number of features

#shap_contributions = shap_contributions.flatten()

feature_contrib = pd.DataFrame({
    'Feature': x.columns,
    'Contribution': shap_contributions
})

feature_contrib['Contribution'] = feature_contrib['Contribution'].abs()

fig = px.pie(feature_contrib, 
             names='Feature', 
             values='Contribution', 
             title='Feature Contributions for this Student (Placement Prediction)',
             hole=0.3)  # donut chart

fig.show()
