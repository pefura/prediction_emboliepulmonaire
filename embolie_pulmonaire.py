# Fichier : embolie_pulmonaire.py

import streamlit as st
import pandas as pd
import joblib
import warnings

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Prédiction d'Embolie Pulmonaire",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Ignorer les avertissements non critiques de scikit-learn pour une interface plus propre
warnings.filterwarnings("ignore", category=UserWarning)

# --- FONCTION DE CHARGEMENT DES ARTEFACTS (MISE EN CACHE) ---
@st.cache_resource
def load_artifacts():
    """Charge le modèle-pipeline et la liste des features."""
    try:
        # On charge le pipeline complet qui inclut le pré-processeur et le modèle.
        model = joblib.load("final_model.pkl")
        features_list = joblib.load("feature_names.pkl")
        return model, features_list
    except FileNotFoundError:
        st.error("Erreur : Fichiers modèles manquants. Assurez-vous que 'final_model.pkl' et 'feature_names.pkl' sont présents dans le dépôt GitHub.")
        return None, None

# --- CHARGEMENT ---
model, features_list = load_artifacts()

# --- INTERFACE UTILISATEUR ---
st.title("🩺 Outil d'Aide au Diagnostic : Risque d'Embolie Pulmonaire")
st.markdown("""
Cette application utilise un modèle de Machine Learning pour estimer la probabilité d'une embolie pulmonaire. 
Le modèle final est un **ensemble combinant les forces** de plusieurs algorithmes pour un meilleur équilibre entre précision et sensibilité.
""")

st.error(
    "**Avertissement Important :** Cet outil est une démonstration à but éducatif. "
    "Il ne doit **en aucun cas** remplacer un diagnostic ou un avis médical professionnel. "
    "Consultez toujours un médecin pour toute question de santé."
)

# On continue seulement si les fichiers ont été chargés correctement
if model and features_list:

    st.sidebar.header("Paramètres du Patient")

    with st.sidebar.form(key='patient_form'):
        st.write("Veuillez remplir les informations du patient :")

        # --- Variables Numériques ---
        st.subheader("Mesures Numériques")
        user_inputs = {
            'AGE': st.number_input("Âge (années)", min_value=15, max_value=100, value=55, step=1),
            'SCORE_WELLS': st.number_input("Score de Wells", min_value=0.0, max_value=12.0, value=3.0, step=0.5, format="%.1f"),
            'D_DIMERE': st.number_input("D-Dimères (ng/mL)", min_value=0, max_value=15000, value=2800, step=100),
            'PAPs': st.number_input("Pression Artérielle Pulmonaire Systolique (mmHg)", min_value=0, max_value=120, value=38, step=1),
            'TAPSE': st.number_input("Excursion Systolique de l'Anneau Tricuspide (mm)", min_value=5, max_value=35, value=17, step=1)
        }
        
        # --- Variables Catégorielles ---
        st.subheader("Signes Cliniques & Antécédents")
        # Les options sont en minuscules pour correspondre aux données d'entraînement ('oui'/'non', 'féminin'/'masculin')
        categorical_inputs = {
            'SEXE': st.selectbox("Sexe", options=['féminin', 'masculin']),
            'DYSPNEE': st.selectbox("Dyspnée", options=['oui', 'non']),
            'DOULEUR_THORACIQUE': st.selectbox("Douleur Thoracique", options=['oui', 'non']),
            'TOUX': st.selectbox("Toux", options=['non', 'oui']),
            'PALPITATION': st.selectbox("Palpitations", options=['non', 'oui']),
            'OBESITE': st.selectbox("Obésité", options=['non', 'oui']),
            'ALLITEMENT': st.selectbox("Alitement récent", options=['non', 'oui']),
            'ATCDS_MTEV': st.selectbox("Antécédents de MTEV", options=['non', 'oui']),
            'ONDE_T_NEGATIVE': st.selectbox("Onde T négative (V1-V3)", options=['oui', 'non']),
            'BBD': st.selectbox("Bloc de Branche Droit (BBD)", options=['non', 'oui']),
            'ASPECT_S1Q3': st.selectbox("Aspect S1Q3", options=['non', 'oui']),
            'SIV_PARADOXAL': st.selectbox("Mouvement Septal Paradoxal (SIV)", options=['oui', 'non']),
            'RAPPORT_VD_VG': st.selectbox("Rapport VD/VG > 1", options=['oui', 'non'])
        }
        user_inputs.update(categorical_inputs)
        
        submit_button = st.form_submit_button(label="Évaluer le Risque", type="primary")

    if submit_button:
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df[features_list] # Garantir le bon ordre des colonnes
        
        st.subheader("📋 Récapitulatif des Données Saisies")
        st.dataframe(input_df.T.rename(columns={0: 'Valeur Saisie'}))

        try:
            prediction_proba = model.predict_proba(input_df)
            prob_embolie = prediction_proba[0][1]
            prob_percentage = prob_embolie * 100
            
            st.subheader("📈 Résultat de la Prédiction")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Probabilité d'Embolie Pulmonaire", value=f"{prob_percentage:.1f} %")
                st.progress(prob_embolie)
            with col2:
                if prob_percentage < 20: st.success("**Risque Faible**")
                elif prob_percentage < 60: st.warning("**Risque Modéré**")
                else: st.error("**Risque Élevé**")
                st.write("Le modèle suggère un niveau de risque basé sur les données fournies.")

            st.info("Rappel : Ce résultat est une estimation statistique basée sur des données historiques. Il ne remplace en aucun cas le jugement clinique.")

        except Exception as e:
            st.error(f"Une erreur est survenue lors de la prédiction : {e}")

else:
    st.info("Chargement du modèle en cours... Si une erreur persiste, vérifiez la présence des fichiers .pkl.")
