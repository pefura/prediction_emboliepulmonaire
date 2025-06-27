import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Prédiction d'Embolie Pulmonaire",
    page_icon="🩺",
    layout="wide"
)

# --- FONCTION DE CHARGEMENT (MISE EN CACHE) ---
# Utiliser st.cache_resource pour ne charger les modèles qu'une seule fois
@st.cache_resource
def load_artifacts():
    """Charge le modèle, le transformeur et la liste des features."""
    try:
        model = joblib.load("best_xgb_model.pkl")
        transformer = joblib.load("column_transformer.pkl")
        features_list = joblib.load("feature_names.pkl")
        return model, transformer, features_list
    except FileNotFoundError:
        st.error("Erreur : Un ou plusieurs fichiers (.pkl) sont manquants. Assurez-vous que 'best_xgb_model.pkl', 'column_transformer.pkl', et 'feature_names.pkl' sont dans le même dossier que le script.")
        return None, None, None

# --- CHARGEMENT DES ARTIFACTS ---
model, transformer, features_list = load_artifacts()

# --- INTERFACE UTILISATEUR ---
st.title("🩺 Prédiction du Risque d'Embolie Pulmonaire")
st.markdown("Cette application utilise un modèle de Machine Learning pour estimer la probabilité d'une embolie pulmonaire en se basant sur des données cliniques.")

st.error(
    "**Avertissement :** Cet outil est une démonstration à but éducatif et ne doit en aucun cas "
    "remplacer un avis médical professionnel. Consultez toujours un médecin pour tout diagnostic."
)

# On continue seulement si les fichiers ont été chargés correctement
if model and transformer and features_list:

    # --- BARRE LATÉRALE POUR LES INPUTS ---
    st.sidebar.header("Paramètres du patient")

    # Dictionnaire pour stocker les inputs de l'utilisateur
    user_inputs = {}

    # Création dynamique des champs de saisie
    # NOTE : Le code ci-dessous fait des suppositions sur le type de variables.
    # Vous devrez l'adapter à vos variables réelles.
    for feature in features_list:
        # Supposons que les variables catégorielles contiennent des mots comme 'sexe', 'atcd', 'oui_non', etc.
        # ou sont des chaînes de caractères après nettoyage.
        if 'sexe' in feature.lower():
            user_inputs[feature] = st.sidebar.selectbox(f"Sexe", ["Homme", "Femme"], key=feature)
        elif 'atcd' in feature.lower() or 'chirurgie' in feature.lower() or 'cancer' in feature.lower() or 'immobilisation' in feature.lower():
            user_inputs[feature] = st.sidebar.selectbox(f"{feature.replace('_', ' ').capitalize()}", ["Non", "Oui"], key=feature)
        # Supposons que les autres sont numériques
        else:
            # Adapter les valeurs min/max/default selon la logique de la variable
            if 'age' in feature.lower():
                user_inputs[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').capitalize()}", min_value=1, max_value=120, value=50, step=1, key=feature)
            elif 'fc' in feature.lower() or 'freq_card' in feature.lower():
                 user_inputs[feature] = st.sidebar.number_input(f"Fréquence Cardiaque (bpm)", min_value=30, max_value=250, value=90, step=1, key=feature)
            elif 'fr' in feature.lower() or 'freq_resp' in feature.lower():
                 user_inputs[feature] = st.sidebar.number_input(f"Fréquence Respiratoire (/min)", min_value=5, max_value=60, value=18, step=1, key=feature)
            else: # Pour les autres variables numériques (ex: D-dimères)
                user_inputs[feature] = st.sidebar.number_input(f"{feature.replace('_', ' ').capitalize()}", value=0.0, format="%.2f", key=feature)


    # --- PRÉDICTION ET AFFICHAGE DES RÉSULTATS ---
    if st.sidebar.button("Calculer la probabilité", type="primary"):
        # 1. Créer un DataFrame à partir des inputs de l'utilisateur
        # L'ordre des colonnes doit correspondre à `features_list`
        input_df = pd.DataFrame([user_inputs])
        
        # S'assurer que les colonnes sont dans le bon ordre
        input_df = input_df[features_list]
        
        st.subheader("Données saisies par l'utilisateur")
        st.dataframe(input_df)

        try:
            # 2. Appliquer le ColumnTransformer (scaling, encodage...)
            transformed_data = transformer.transform(input_df)

            # 3. Faire la prédiction de probabilité
            # predict_proba renvoie les probabilités pour chaque classe [prob_classe_0, prob_classe_1]
            prediction_proba = model.predict_proba(transformed_data)

            # On récupère la probabilité de la classe positive (embolie = 1)
            prob_embolie = prediction_proba[0][1]

            # 4. Afficher le résultat
            st.subheader("Résultat de la prédiction")
            prob_percentage = prob_embolie * 100

            # Affichage avec une jauge et un code couleur
            if prob_percentage < 30:
                st.success(f"**Risque Faible**")
            elif prob_percentage < 60:
                st.warning(f"**Risque Modéré**")
            else:
                st.error(f"**Risque Élevé**")
            
            st.metric(
                label="Probabilité d'Embolie Pulmonaire",
                value=f"{prob_percentage:.2f} %"
            )

            # Barre de progression pour un visuel simple
            st.progress(prob_embolie)
            
            st.info("Rappel : Ce résultat est basé sur un modèle statistique et ne constitue pas un diagnostic.")

        except Exception as e:
            st.error(f"Une erreur est survenue lors de la prédiction : {e}")
            st.error("Veuillez vérifier que les types de données saisis correspondent à ceux attendus par le modèle.")

else:
    st.info("En attente du chargement des fichiers du modèle...")
