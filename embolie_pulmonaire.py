import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Prédiction d'Embolie Pulmonaire",
    page_icon="🩺",
    layout="centered", # 'centered' est souvent mieux pour les formulaires
    initial_sidebar_state="expanded"
)

# --- FONCTION DE CHARGEMENT DES ARTEFACTS (MISE EN CACHE) ---
@st.cache_resource
def load_artifacts():
    """Charge le modèle-pipeline et la liste des features."""
    try:
        # NOTE: On ne charge que le pipeline complet et la liste des features.
        # Le pré-processeur est déjà inclus dans final_model.pkl
        model = joblib.load("final_model.pkl")
        features_list = joblib.load("feature_names.pkl")
        return model, features_list
    except FileNotFoundError:
        st.error("Erreur : Fichiers modèles manquants. Assurez-vous que 'final_model.pkl' et 'feature_names.pkl' sont présents.")
        return None, None

# --- CHARGEMENT ---
model, features_list = load_artifacts()

# --- INTERFACE UTILISATEUR ---
st.title("🩺 Outil d'Aide à la Décision : Risque d'Embolie Pulmonaire")
st.markdown("""
Cette application utilise un modèle de Machine Learning pour estimer la probabilité d'une embolie pulmonaire. 
Elle se base sur le **modèle combiné** développé à partir de l'analyse comparative.
""")

st.error(
    "**Avertissement Important :** Cet outil est une démonstration à but éducatif. "
    "Il ne doit **en aucun cas** remplacer un diagnostic ou un avis médical professionnel. "
    "Consultez toujours un médecin pour toute question de santé."
)

# On continue seulement si les fichiers ont été chargés correctement
if model and features_list:

    # --- BARRE LATÉRALE POUR LA SAISIE DES DONNÉES ---
    st.sidebar.header("Paramètres du Patient")

    # Dictionnaire pour stocker les inputs
    user_inputs = {}

    # Utilisons les informations de votre 'df.info()' pour créer les widgets
    with st.sidebar.form(key='patient_form'):
        st.write("Veuillez remplir les informations du patient :")

        # --- Variables Numériques ---
        st.subheader("Mesures Numériques")
        user_inputs['AGE'] = st.number_input("Âge (années)", min_value=15, max_value=100, value=55, step=1)
        user_inputs['SCORE_WELLS'] = st.number_input("Score de Wells", min_value=0.0, max_value=12.0, value=3.0, step=0.5, format="%.1f")
        user_inputs['D_DIMERE'] = st.number_input("D-Dimères (ng/mL)", min_value=0, max_value=15000, value=2800, step=100)
        user_inputs['PAPs'] = st.number_input("Pression Artérielle Pulmonaire Systolique (PAPs, mmHg)", min_value=0, max_value=120, value=38, step=1)
        user_inputs['TAPSE'] = st.number_input("Excursion Systolique de l'Anneau Tricuspide (TAPSE, mm)", min_value=5, max_value=35, value=17, step=1)
        
        # --- Variables Catégorielles ---
        st.subheader("Signes Cliniques et Antécédents")
        # Les options sont en minuscules pour correspondre aux données d'entraînement
        user_inputs['SEXE'] = st.selectbox("Sexe", options=['féminin', 'masculin'], index=0)
        user_inputs['DYSPNEE'] = st.selectbox("Dyspnée", options=['oui', 'non'], index=0)
        user_inputs['DOULEUR_THORACIQUE'] = st.selectbox("Douleur Thoracique", options=['oui', 'non'], index=0)
        user_inputs['TOUX'] = st.selectbox("Toux", options=['non', 'oui'], index=0)
        user_inputs['PALPITATION'] = st.selectbox("Palpitations", options=['non', 'oui'], index=0)
        user_inputs['OBESITE'] = st.selectbox("Obésité", options=['non', 'oui'], index=0)
        user_inputs['ALLITEMENT'] = st.selectbox("Alitement récent", options=['non', 'oui'], index=0)
        user_inputs['ATCDS_MTEV'] = st.selectbox("Antécédents de MTEV", options=['non', 'oui'], index=0)
        
        st.subheader("Signes Échographiques / ECG")
        user_inputs['ONDE_T_NEGATIVE'] = st.selectbox("Onde T négative (V1-V3)", options=['oui', 'non'], index=0)
        user_inputs['BBD'] = st.selectbox("Bloc de Branche Droit (BBD)", options=['non', 'oui'], index=0)
        user_inputs['ASPECT_S1Q3'] = st.selectbox("Aspect S1Q3", options=['non', 'oui'], index=0)
        user_inputs['SIV_PARADOXAL'] = st.selectbox("Mouvement Septal Paradoxal (SIV)", options=['oui', 'non'], index=0)
        user_inputs['RAPPORT_VD_VG'] = st.selectbox("Rapport VD/VG > 1", options=['oui', 'non'], index=0)
        
        # Bouton de soumission du formulaire
        submit_button = st.form_submit_button(label="Calculer la Probabilité", type="primary")

    # --- PRÉDICTION ET AFFICHAGE DES RÉSULTATS ---
    if submit_button:
        # 1. Créer un DataFrame à partir des inputs
        # L'ordre des colonnes doit correspondre exactement à `features_list`
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df[features_list]
        
        st.subheader("📋 Récapitulatif des Données Saisies")
        st.dataframe(input_df.T.rename(columns={0: 'Valeur'}))

        try:
            # 2. Faire la prédiction de probabilité
            # Le modèle est un pipeline complet, il gère le pré-traitement !
            prediction_proba = model.predict_proba(input_df)
            prob_embolie = prediction_proba[0][1] # Probabilité de la classe 1

            # 3. Afficher le résultat de manière claire
            st.subheader("📈 Résultat de la Prédiction")
            prob_percentage = prob_embolie * 100
            
            # Utiliser des colonnes pour un affichage plus propre
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Probabilité d'Embolie Pulmonaire",
                    value=f"{prob_percentage:.1f} %"
                )
                # Jauge de progression pour un visuel simple
                st.progress(prob_embolie)

            with col2:
                # Affichage avec un code couleur
                if prob_percentage < 20:
                    st.success("**Risque Faible**")
                elif prob_percentage < 60:
                    st.warning("**Risque Modéré**")
                else:
                    st.error("**Risque Élevé**")
                st.write("Le modèle suggère un niveau de risque basé sur les données fournies.")

            st.info("Rappel : Ce résultat est une estimation statistique. Il ne remplace en aucun cas le jugement clinique d'un professionnel de santé.")

        except Exception as e:
            st.error(f"Une erreur est survenue lors de la prédiction : {e}")

else:
    st.info("Chargement du modèle en cours... Si une erreur persiste, vérifiez la présence des fichiers .pkl dans le dossier de l'application.")
