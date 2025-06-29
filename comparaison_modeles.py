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
from sklearn.inspection import permutation_importance

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
    precision_recall_curve, auc as auc_score_func
)

# Ignorer les avertissements non critiques pour une sortie plus propre
warnings.filterwarnings("ignore", category=UserWarning, message="Found unknown categories in columns .* during transform.")
print("Librairies importées avec succès.")


# ==============================================================================
# 2. CONFIGURATION ET CHARGEMENT DES DONNÉES
# ==============================================================================
print("\n--- Chargement et préparation des données ---")
DATA_FILE = "df_EP_decode_clean.csv"
TARGET_VARIABLE = 'EP_CONFIRMEE'

# Définition des listes de variables pour le prétraitement
numerical_features = ['AGE', 'SCORE_WELLS', 'D_DIMERE', 'PAPs', 'TAPSE']
categorical_features = ['SIV_PARADOXAL', 'RAPPORT_VD_VG', 'DYSPNEE', 'ALLITEMENT', 'TOUX', 'ONDE_T_NEGATIVE', 'PALPITATION', 'BBD', 'SEXE', 'ATCDS_MTEV', 'DOULEUR_THORACIQUE', 'OBESITE', 'ASPECT_S1Q3']

# Chargement du jeu de données
df = pd.read_csv(DATA_FILE)

# Encodage de la variable cible de format textuel ('non'/'oui') en format numérique (0/1)
target_mapping = {'non': 0, 'oui': 1}
df[TARGET_VARIABLE] = df[TARGET_VARIABLE].map(target_mapping)

# Définition des prédicteurs (X) et de la cible (y)
all_features = numerical_features + categorical_features
X = df.drop(columns=[TARGET_VARIABLE])[all_features]
y = df[TARGET_VARIABLE]

# Partitionnement stratifié des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Données chargées et préparées. Entraînement: {len(X_train)}, Test: {len(X_test)}")


# ==============================================================================
# 3. CRÉATION DU PIPELINE DE PRÉ-TRAITEMENT
# ==============================================================================
numeric_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_pipeline, numerical_features), ('cat', categorical_pipeline, categorical_features)], remainder='passthrough')
print("Pré-processeur (ColumnTransformer) avec imputation créé.")


# ==============================================================================
# 4. DÉFINITION DES MODÈLES ET GESTION DU DÉSÉQUILIBRE DE CLASSE
# ==============================================================================
scale_pos_weight_value = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Le poids pour la classe positive (scale_pos_weight) est de: {scale_pos_weight_value:.2f}")

models = {
    'Régression Logistique': LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000),
    'SVM': SVC(random_state=42, probability=True, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight_value),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
}
param_grids = {
    'Régression Logistique': {'model__C': np.logspace(-3, 3, 20), 'model__penalty': ['l1', 'l2'], 'model__solver': ['liblinear', 'saga']},
    'KNN': {'model__n_neighbors': list(range(3, 31)), 'model__weights': ['uniform', 'distance'], 'model__metric': ['euclidean', 'manhattan']},
    'SVM': {'model__C': np.logspace(-2, 3, 15), 'model__gamma': np.logspace(-4, 1, 15), 'model__kernel': ['rbf']},
    'Naive Bayes': {'model__var_smoothing': np.logspace(0, -9, num=100)},
    'Random Forest': {'model__n_estimators': [100, 300, 500], 'model__max_depth': [5, 10, 20, None], 'model__min_samples_split': [2, 5, 10], 'model__min_samples_leaf': [1, 2, 4], 'model__max_features': ['sqrt', 'log2']},
    'XGBoost': {'model__n_estimators': [100, 300, 500], 'model__max_depth': [3, 5, 8], 'model__learning_rate': [0.01, 0.05, 0.1], 'model__subsample': [0.7, 1.0], 'model__colsample_bytree': [0.7, 1.0]}
}


# ==============================================================================
# 5. ENTRAÎNEMENT ET RECHERCHE D'HYPERPARAMÈTRES
# ==============================================================================
print("\n--- Lancement de l'entraînement et de la recherche d'hyperparamètres ---")
best_estimators = {}
for name, model in models.items():
    print(f"Entraînement de : {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    n_iterations = 40 if name in ['Random Forest', 'XGBoost', 'SVM'] else 20
    search = RandomizedSearchCV(pipeline, param_distributions=param_grids[name], n_iter=n_iterations, scoring='roc_auc', cv=5, n_jobs=-1, verbose=0, random_state=42)
    search.fit(X_train, y_train)
    best_estimators[name] = search.best_estimator_
print("✅ Tous les modèles ont été entraînés.")


# ==============================================================================
# 6. RÉSULTATS DE LA VALIDATION CROISÉE
# ==============================================================================
cv_results = []
for name, model_pipeline in best_estimators.items():
    cv_score = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    cv_results.append({'Modèle': name, 'Mean ROC AUC (CV)': cv_score.mean(), 'Std ROC AUC (CV)': cv_score.std()})

print("\n" + "="*80 + "\n       RÉSULTATS DE LA VALIDATION CROISÉE       \n" + "="*80)
cv_df = pd.DataFrame(cv_results).sort_values(by='Mean ROC AUC (CV)', ascending=False).reset_index(drop=True)
cv_df['Formatted Result'] = cv_df.apply(lambda row: f"{row['Mean ROC AUC (CV)']:.4f} ± {row['Std ROC AUC (CV)']:.4f}", axis=1)
print(cv_df[['Modèle', 'Formatted Result']].to_string(index=False))
cv_df.to_csv('cross_validation_results.csv', index=False)
print(f"\n✅ Tableau de validation croisée sauvegardé dans : 'cross_validation_results.csv'")


# ==============================================================================
# 7. ÉVALUATION FINALE (MODÈLES INDIVIDUELS)
# ==============================================================================
test_results = []
for name, model in best_estimators.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    test_results.append({
        'Modèle': name, 'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
        'AUC-PR': auc_score_func(recall, precision) # Calcul de l'aire sous la courbe P-R
    })
test_df = pd.DataFrame(test_results)


# ==============================================================================
# 8. CRÉATION ET ÉVALUATION DU MODÈLE COMBINÉ
# ==============================================================================
best_recall_model_name = test_df.loc[test_df['Recall'].idxmax()]['Modèle']
best_precision_model_name = test_df.loc[test_df['Precision'].idxmax()]['Modèle']
if best_recall_model_name != best_precision_model_name:
    print(f"\n--- Création du modèle combiné (Spécialiste Rappel: '{best_recall_model_name}', Spécialiste Précision: '{best_precision_model_name}') ---")
    model_recall, model_precision = best_estimators[best_recall_model_name], best_estimators[best_precision_model_name]
    voting_clf = VotingClassifier(estimators=[('recall', model_recall), ('precision', model_precision)], voting='soft').fit(X_train, y_train)
    y_pred_vote, y_pred_proba_vote = voting_clf.predict(X_test), voting_clf.predict_proba(X_test)[:, 1]
    precision_vote, recall_vote, _ = precision_recall_curve(y_test, y_pred_proba_vote)
    
    combined_results = {
        'Modèle': 'Modèle Combiné (P+R)', 'Accuracy': accuracy_score(y_test, y_pred_vote),
        'Precision': precision_score(y_test, y_pred_vote), 'Recall': recall_score(y_test, y_pred_vote),
        'F1-Score': f1_score(y_test, y_pred_vote), 'AUC-ROC': roc_auc_score(y_test, y_pred_proba_vote),
        'AUC-PR': auc_score_func(recall_vote, precision_vote)
    }
    test_df = pd.concat([test_df, pd.DataFrame([combined_results])], ignore_index=True)
    best_estimators['Modèle Combiné (P+R)'] = voting_clf
else:
    print("\nLe même modèle est le meilleur pour la précision ET le rappel. Pas de combinaison nécessaire.")

test_df = test_df.sort_values(by='AUC-ROC', ascending=False).reset_index(drop=True)
print("\n" + "="*80 + "\n            ÉVALUATION FINALE (SUR L'ENSEMBLE DE TEST)            \n" + "="*80)
pd.set_option('display.float_format', '{:.4f}'.format)
print(test_df)
test_df.to_csv('final_evaluation_results.csv', index=False)
print(f"\n✅ Tableau d'évaluation finale sauvegardé dans : 'final_evaluation_results.csv'")


# ==============================================================================
# 9. VISUALISATION DES COURBES ROC
# ==============================================================================
print("\n--- Génération du graphique des courbes ROC ---")
plt.style.use('seaborn-v0_8-whitegrid')
fig_roc, ax_roc = plt.subplots(figsize=(14, 11))
palette = sns.color_palette("husl", len(best_estimators))
individual_test_df = test_df[~test_df['Modèle'].str.contains("Combiné")]
best_individual_model_name = individual_test_df.loc[individual_test_df['AUC-ROC'].idxmax()]['Modèle']

for i, (name, model) in enumerate(best_estimators.items()):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_val = roc_auc_score(y_test, y_pred_proba)
    label, linestyle, linewidth = f'{name} (AUC = {auc_val:.3f})', '-', 2.0
    if name == 'Modèle Combiné (P+R)': linewidth = 4.0
    elif name == best_individual_model_name: linewidth = 2.5
    elif name in ['KNN', 'Naive Bayes', 'Régression Logistique']: linestyle, linewidth = '--', 1.5
    ax_roc.plot(fpr, tpr, label=label, color=palette[i], linewidth=linewidth, linestyle=linestyle)

ax_roc.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.500)')
ax_roc.set_xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=14)
ax_roc.set_ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=14)
ax_roc.set_title('Comparaison des Courbes ROC sur l\'Ensemble de Test', fontsize=18, pad=20)
ax_roc.legend(fontsize=12, loc='lower right', title='Modèles', title_fontsize='13')
ax_roc.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Courbes ROC sauvegardées.")
plt.show()


# ==============================================================================
# 10. VISUALISATION DES COURBES PRÉCISION-RAPPEL
# ==============================================================================
print("\n--- Génération du graphique des courbes Précision-Rappel ---")
fig_pr, ax_pr = plt.subplots(figsize=(14, 11))
palette_pr = sns.color_palette("viridis", len(best_estimators))
no_skill = len(y_test[y_test==1]) / len(y_test)
ax_pr.plot([0, 1], [no_skill, no_skill], 'k--', label=f'Aléatoire (AUC-PR = {no_skill:.3f})')

for i, (name, model) in enumerate(best_estimators.items()):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc_score_func(recall, precision)
    label, linestyle, linewidth = f'{name} (AUC-PR = {auc_pr:.3f})', '-', 2.0
    if name == 'Modèle Combiné (P+R)': linewidth = 4.0
    elif name == best_individual_model_name: linewidth = 2.5
    elif name in ['KNN', 'Naive Bayes', 'Régression Logistique']: linestyle, linewidth = '--', 1.5
    ax_pr.plot(recall, precision, label=label, color=palette_pr[i], linewidth=linewidth, linestyle=linestyle)

ax_pr.set_xlabel('Rappel (Sensibilité)', fontsize=14)
ax_pr.set_ylabel('Précision', fontsize=14)
ax_pr.set_title('Comparaison des Courbes Précision-Rappel sur l\'Ensemble de Test', fontsize=18, pad=20)
ax_pr.legend(fontsize=12, loc='upper right', title='Modèles', title_fontsize='13')
ax_pr.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
print(f"✅ Courbes Précision-Rappel sauvegardées.")
plt.show()


# ==============================================================================
# 11. VISUALISATION DES COURBES D'APPRENTISSAGE
# ==============================================================================
def plot_learning_curve_on_ax(estimator, title, X, y, ax, cv, n_jobs=-1):
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Taille de l'ensemble d'entraînement", fontsize=10)
    ax.set_ylabel("Score (AUC-ROC)", fontsize=10)
    ax.grid(True)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, scoring='roc_auc', train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean, train_scores_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_scores_mean, test_scores_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation croisée")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(bottom=0.5, top=1.05)

print("\n--- Génération du graphique des courbes d'apprentissage ---")
n_models, n_cols = len(best_estimators), 3
n_rows = (n_models + n_cols - 1) // n_cols
fig_lc, axes_lc = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 6, n_rows * 5))
axes_lc = axes_lc.flatten()
for i, (name, model) in enumerate(best_estimators.items()):
    plot_learning_curve_on_ax(model, name, X_train, y_train, axes_lc[i], cv=5)
for i in range(n_models, len(axes_lc)): axes_lc[i].axis('off')
fig_lc.suptitle("Comparaison des Courbes d'Apprentissage des Modèles", fontsize=24, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('learning_curves_grid.png', dpi=300, bbox_inches='tight')
print(f"✅ Courbes d'apprentissage sauvegardées.")
plt.show()


# ==============================================================================
# 12. EXPLICABILITÉ DE XGBOOST AVEC SHAP (COMPATIBLE ANCIENNE VERSION)
# ==============================================================================
print("\n" + "="*80 + "\n       ANALYSE D'EXPLICABILITÉ DE XGBOOST AVEC SHAP       \n" + "="*80)
model_to_explain_shap = 'XGBoost'
if model_to_explain_shap in best_estimators:
    pipeline = best_estimators[model_to_explain_shap]
    model = pipeline.named_steps['model']
    preprocessor_for_shap = pipeline.named_steps['preprocessor']
    X_train_processed = preprocessor_for_shap.transform(X_train)
    feature_names_processed = preprocessor_for_shap.get_feature_names_out()
    if hasattr(X_train_processed, "toarray"): X_train_processed_dense = X_train_processed.toarray()
    else: X_train_processed_dense = X_train_processed
    X_train_processed_df = pd.DataFrame(X_train_processed_dense, columns=feature_names_processed)

    print("Calcul des valeurs SHAP pour XGBoost...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_processed_df)
    if isinstance(shap_values, list): shap_values = shap_values[1]
    
    # 1. Bar Plot
    print("Génération du Bar Plot SHAP...")
    plt.figure()
    shap.summary_plot(shap_values, X_train_processed_df, plot_type="bar", show=False, max_display=15)
    plt.gcf().set_size_inches(12, 8)
    plt.title(f"Importance Globale des Variables - {model_to_explain_shap}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'shap_barplot_{model_to_explain_shap}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Beeswarm Plot
    print("\nGénération du Beeswarm Plot SHAP...")
    plt.figure()
    shap.summary_plot(shap_values, X_train_processed_df, plot_type="dot", show=False, max_display=15)
    plt.gcf().set_size_inches(12, 8)
    plt.title(f"Impact des Variables sur la Prédiction - {model_to_explain_shap}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'shap_beeswarm_{model_to_explain_shap}.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# 13. IMPORTANCE DES VARIABLES POUR NAIVE BAYES (PERMUTATION)
# ==============================================================================
print("\n" + "="*80 + "\n       IMPORTANCE DES VARIABLES POUR NAIVE BAYES (PERMUTATION)       \n" + "="*80)
model_to_explain_perm = 'Naive Bayes'
if model_to_explain_perm in best_estimators:
    pipeline_nb = best_estimators[model_to_explain_perm]
    print(f"Calcul de l'importance par permutation pour '{model_to_explain_perm}'...")
    result = permutation_importance(pipeline_nb, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='roc_auc')
    perm_importance_df = pd.DataFrame({'Variable': X_test.columns, 'Importance (Baisse Moyenne AUC)': result.importances_mean}).sort_values('Importance (Baisse Moyenne AUC)', ascending=False)

    print("\nImportance des variables pour Naive Bayes :")
    print(perm_importance_df.head(10))
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance (Baisse Moyenne AUC)', y='Variable', data=perm_importance_df.head(15), palette='mako')
    plt.title(f"Importance des Variables pour le Modèle {model_to_explain_perm}", fontsize=18, pad=20)
    plt.xlabel("Baisse de Performance (Score AUC-ROC)", fontsize=14)
    plt.ylabel("Variable", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'permutation_importance_{model_to_explain_perm.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# 14. SAUVEGARDE DES ARTEFACTS POUR STREAMLIT
# ==============================================================================
print("\n" + "="*80 + "\n       SAUVEGARDE DES ARTEFACTS POUR STREAMLIT       \n" + "="*80)
MODEL_FILE = "final_model.pkl"
FEATURES_FILE = "feature_names.pkl"
if 'Modèle Combiné (P+R)' in best_estimators:
    final_model_to_save = best_estimators['Modèle Combiné (P+R)']
    model_name_saved = 'Modèle Combiné (P+R)'
else:
    best_model_name = test_df.loc[0]['Modèle']
    final_model_to_save = best_estimators[best_model_name]
    model_name_saved = best_model_name
joblib.dump(final_model_to_save, MODEL_FILE)
print(f"✅ Modèle final ('{model_name_saved}') sauvegardé dans : '{MODEL_FILE}'")
joblib.dump(all_features, FEATURES_FILE)
print(f"✅ Liste des features sauvegardée dans : '{FEATURES_FILE}'")
print("\n🎉 Artefacts sauvegardés. Vous pouvez maintenant construire l'application Streamlit.")

print("\nAnalyse complète terminée.")
