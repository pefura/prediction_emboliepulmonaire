# ==============================================================================
# 1. IMPORTATION DES LIBRAIRIES
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import joblib
import shap


# Preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Modèles
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb

# Métriques
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve,
    precision_recall_curve, average_precision_score
)

# Ignorer les avertissements non critiques
warnings.filterwarnings("ignore", category=UserWarning, message="Found unknown categories in columns .* during transform.")
print("Librairies importées avec succès.")


# ==============================================================================
# SECTIONS 2 à 12 (INCHANGÉES)
# ==============================================================================
# 2. CONFIGURATION ET CHARGEMENT DES DONNÉES
print("\n--- Chargement et préparation des données ---")
DATA_FILE = "df_EP_decode_clean.csv"
TARGET_VARIABLE = 'EP_CONFIRMEE'
numerical_features = ['AGE', 'SCORE_WELLS', 'D_DIMERE', 'PAPs', 'TAPSE']
categorical_features = ['SIV_PARADOXAL', 'RAPPORT_VD_VG', 'DYSPNEE', 'ALLITEMENT', 'TOUX', 'ONDE_T_NEGATIVE', 'PALPITATION', 'BBD', 'SEXE', 'ATCDS_MTEV', 'DOULEUR_THORACIQUE', 'OBESITE', 'ASPECT_S1Q3']
df = pd.read_csv(DATA_FILE)
target_mapping = {'non': 0, 'oui': 1}
df[TARGET_VARIABLE] = df[TARGET_VARIABLE].map(target_mapping)
all_features = numerical_features + categorical_features
X = df.drop(columns=[TARGET_VARIABLE])[all_features]
y = df[TARGET_VARIABLE]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 3. CRÉATION DU PIPELINE DE PRÉ-TRAITEMENT
numeric_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_pipeline, numerical_features), ('cat', categorical_pipeline, categorical_features)], remainder='passthrough')

# 4. DÉFINITION DES MODÈLES ET GESTION DU DÉSÉQUILIBRE
scale_pos_weight_value = y_train.value_counts()[0] / y_train.value_counts()[1]
models = {
    'Régression Logistique': LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000), 'SVM': SVC(random_state=42, probability=True, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'), 'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight_value),
    'KNN': KNeighborsClassifier(), 'Naive Bayes': GaussianNB(),
}
param_grids = {
    'Régression Logistique': {'model__C': np.logspace(-3, 3, 20), 'model__penalty': ['l1', 'l2'], 'model__solver': ['liblinear', 'saga']},
    'KNN': {'model__n_neighbors': list(range(3, 31)), 'model__weights': ['uniform', 'distance'], 'model__metric': ['euclidean', 'manhattan']},
    'SVM': {'model__C': np.logspace(-2, 3, 15), 'model__gamma': np.logspace(-4, 1, 15), 'model__kernel': ['rbf']}, 'Naive Bayes': {'model__var_smoothing': np.logspace(0, -9, num=100)},
    'Random Forest': {'model__n_estimators': [100, 300, 500], 'model__max_depth': [5, 10, 20, None], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4], 'model__max_features': ['sqrt', 'log2']},
    'XGBoost': {'model__n_estimators': [100, 300, 500], 'model__max_depth': [3, 5, 8], 'model__learning_rate': [0.01, 0.05, 0.1], 'model__subsample': [0.7, 1.0], 'model__colsample_bytree': [0.7, 1.0]}
}

# 5. ENTRAÎNEMENT
print("--- Lancement de l'entraînement ---")
best_estimators = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    n_iterations = 40 if name in ['Random Forest', 'XGBoost', 'SVM'] else 20
    search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], n_iter=n_iterations, scoring='roc_auc', cv=5, n_jobs=-1, verbose=0, random_state=42)
    search.fit(X_train, y_train)
    best_estimators[name] = search.best_estimator_

# 6, 7, 8...
test_results = []
for name, model in best_estimators.items():
    y_pred, y_pred_proba = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    test_results.append({'Modèle': name, 'Accuracy': accuracy_score(y_test, y_pred), 'Precision': precision_score(y_test, y_pred, zero_division=0), 'Recall': recall_score(y_test, y_pred), 'F1-Score': f1_score(y_test, y_pred), 'ROC AUC': roc_auc_score(y_test, y_pred_proba)})
test_df = pd.DataFrame(test_results)
best_recall_model_name, best_precision_model_name = test_df.loc[test_df['Recall'].idxmax()]['Modèle'], test_df.loc[test_df['Precision'].idxmax()]['Modèle']
if best_recall_model_name != best_precision_model_name:
    model_recall, model_precision = best_estimators[best_recall_model_name], best_estimators[best_precision_model_name]
    voting_clf = VotingClassifier(estimators=[('recall', model_recall), ('precision', model_precision)], voting='soft').fit(X_train, y_train)
    y_pred_vote, y_pred_proba_vote = voting_clf.predict(X_test), voting_clf.predict_proba(X_test)[:, 1]
    combined_results = {'Modèle': 'Modèle Combiné (P+R)', 'Accuracy': accuracy_score(y_test, y_pred_vote), 'Precision': precision_score(y_test, y_pred_vote), 'Recall': recall_score(y_test, y_pred_vote), 'F1-Score': f1_score(y_test, y_pred_vote), 'ROC AUC': roc_auc_score(y_test, y_pred_proba_vote)}
    test_df = pd.concat([test_df, pd.DataFrame([combined_results])], ignore_index=True)
    best_estimators['Modèle Combiné (P+R)'] = voting_clf
test_df = test_df.sort_values(by='ROC AUC', ascending=False).reset_index(drop=True)
print("\n--- Tableau d'évaluation final ---\n", test_df)


# ==============================================================================
# 13. EXPLICABILITÉ DE XGBOOST AVEC SHAP (COMPATIBLE ANCIENNE VERSION)
# ==============================================================================
print("\n" + "="*80 + "\n       ANALYSE D'EXPLICABILITÉ DE XGBOOST AVEC SHAP       \n" + "="*80)

model_to_explain_shap = 'XGBoost'
if model_to_explain_shap in best_estimators:
    
    pipeline = best_estimators[model_to_explain_shap]
    model = pipeline.named_steps['model']
    preprocessor_for_shap = pipeline.named_steps['preprocessor']

    X_train_processed = preprocessor_for_shap.transform(X_train)
    feature_names_processed = preprocessor_for_shap.get_feature_names_out()
    
    if hasattr(X_train_processed, "toarray"):
        X_train_processed_dense = X_train_processed.toarray()
    else:
        X_train_processed_dense = X_train_processed
        
    X_train_processed_df = pd.DataFrame(X_train_processed_dense, columns=feature_names_processed)

    print("Calcul des valeurs SHAP pour XGBoost...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_processed_df)
    
    # Correction pour gérer les différents formats de sortie de .shap_values()
    if isinstance(shap_values, list):
        shap_values_for_plot = shap_values[1]
    else:
        shap_values_for_plot = shap_values
    
    # --- 1. Génération et sauvegarde du Bar Plot ---
    print("Génération du Bar Plot SHAP...")
    plt.figure() # Crée une nouvelle figure
    shap.summary_plot(shap_values_for_plot, X_train_processed_df, plot_type="bar", show=False, max_display=15)
    
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.title(f"Importance Globale des Variables - {model_to_explain_shap}", fontsize=16)
    plt.tight_layout()
    bar_plot_filename = f'shap_barplot_{model_to_explain_shap}.png'
    plt.savefig(bar_plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Bar Plot SHAP sauvegardé dans : '{bar_plot_filename}'")
    plt.show()

    # --- 2. Génération et sauvegarde du Beeswarm Plot ---
    print("\nGénération du Beeswarm Plot SHAP...")
    plt.figure() # Crée une autre nouvelle figure
    shap.summary_plot(shap_values_for_plot, X_train_processed_df, plot_type="dot", show=False, max_display=15)
    
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.title(f"Impact des Variables sur la Prédiction - {model_to_explain_shap}", fontsize=16)
    plt.tight_layout()
    beeswarm_filename = f'shap_beeswarm_{model_to_explain_shap}.png'
    plt.savefig(beeswarm_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Beeswarm Plot SHAP sauvegardé dans : '{beeswarm_filename}'")
    plt.show()

else:
    print(f"Le modèle '{model_to_explain_shap}' n'a pas pu être analysé.")

print("\nAnalyse complète terminée.")