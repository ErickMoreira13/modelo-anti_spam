import os
import io
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay)

# Download dataset (UCI)
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
NAMES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"

print("Baixando dataset...")
raw = urllib.request.urlopen(DATA_URL).read().decode('utf-8')
# carregar em DataFrame
# O dataset não tem headers; o arquivo .names contém descrições. Vamos criar nomes genéricos
# Existem 57 features e 1 target = 58 colunas
n_features = 57
col_names = [f'feat_{i+1}' for i in range(n_features)] + ['is_spam']
df = pd.read_csv(io.StringIO(raw), header=None, names=col_names)

print("Dimensão:", df.shape)
print(df['is_spam'].value_counts(normalize=False))

# Quick EDA
print(df.describe().T)

# Split features / target
X = df.drop(columns=['is_spam'])
y = df['is_spam']

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Pipelines - modelos
pipe_lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])
pipe_rf = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))])

# Treinando os modelos
print("Treinando Logistic Regression...")
pipe_lr.fit(X_train, y_train)
print("Treinando Random Forest...")
pipe_rf.fit(X_train, y_train)

# Voting ensemble
voting = VotingClassifier(estimators=[('lr', pipe_lr.named_steps['lr']), ('rf', pipe_rf.named_steps['rf'])], voting='soft')

pipeline_voting = Pipeline([('scaler', StandardScaler()), ('voting', voting)])
pipeline_voting.fit(X_train, y_train)

# Predições e metricas
def evaluate(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:,1]
    except:
        pass
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (spam):", precision_score(y_test, y_pred))
    print("Recall (spam):", recall_score(y_test, y_pred))
    print("F1 (spam):", f1_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)
    return y_proba, cm

probs_lr, cm_lr = evaluate(pipe_lr, X_test, y_test, "LogisticRegression")
probs_rf, cm_rf = evaluate(pipe_rf, X_test, y_test, "RandomForest")
probs_v, cm_v = evaluate(pipeline_voting, X_test, y_test, "Voting (LR+RF)")

# Curva ROC  e AUC 
plt.figure(figsize=(8,6))
models = [('Logistic', probs_lr), ('RandomForest', probs_rf), ('Voting', probs_v)]
for name, probs in models:
    if probs is None:
        continue
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# Confusion matrix plot 
plt.figure(figsize=(5,4))
sns.heatmap(cm_v, annot=True, fmt='d', cmap='Blues', xticklabels=['notspam','spam'], yticklabels=['notspam','spam'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Voting')
plt.tight_layout()
plt.savefig("confusion_voting.png")
plt.show()

# Features importantes (Random Forest)
importances = pipe_rf.named_steps['rf'].feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(20)
print("Top 20 features (RF):")
print(feat_imp)
feat_imp.plot(kind='barh', figsize=(8,6))
plt.title("Top 20 feature importances (RandomForest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()

