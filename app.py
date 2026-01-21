import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from PIL import Image
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Dépenses Bancaires",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé - Thème Rouge et Blanc
THEMES = {
    "Rouge Corporate": {
        "primary": "#DC143C",
        "secondary": "#FF6B6B",
        "light": "#F8B4B4",
        "pale": "#EF5350",
    },
    "Bleu Bancaire": {
        "primary": "#003366",
        "secondary": "#4F81BD",
        "light": "#DCE6F2",
        "pale": "#6FA8DC",
    },
    "Vert Finance": {
        "primary": "#1B5E20",
        "secondary": "#4CAF50",
        "light": "#C8E6C9",
        "pale": "#81C784",
    },
}
if "theme" not in st.session_state:
    st.session_state.theme = "Rouge Corporate"
colors = THEMES[st.session_state.theme]

st.markdown(f"""
<style>

/* =========================
   VARIABLES DE THÈME
========================= */
:root {{
    --primary-color: {colors["primary"]};
    --secondary-color: {colors["secondary"]};
    --light-color: {colors["light"]};
    --pale-color: {colors["pale"]};
    --white: #FFFFFF;
    --gray: #F5F5F5;
    --dark-gray: #3A3A3A;
}}

/* =========================
   STYLE GLOBAL
========================= */
.main {{
    background-color: var(--white);
}}

/* =========================
   TITRES
========================= */
h1, h2, h3, h4 {{
    color: var(--primary-color);
    font-family: 'Helvetica Neue', 'Inter', sans-serif;
    font-weight: 600;
}}

/* =========================
   SIDEBAR
========================= */
[data-testid="stSidebar"] {{
    background: linear-gradient(
        180deg,
        var(--primary-color) 0%,
        var(--secondary-color) 100%
    );
}}

[data-testid="stSidebar"] * {{
    color: var(--white) !important;
}}

/* =========================
   BOUTONS
========================= */
.stButton > button {{
    background-color: var(--primary-color);
    color: var(--white);
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease;
    width: 100%;
}}

.stButton > button:hover {{
    background-color: var(--secondary-color);
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.15);
}}

/* =========================
   CARTES / METRICS
========================= */
.metric-card {{
    background: linear-gradient(
        135deg,
        var(--light-color) 0%,
        var(--pale-color) 100%
    );
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid var(--primary-color);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin: 12px 0;
}}

/* =========================
   INPUTS & SELECT
========================= */
.stNumberInput input,
.stSelectbox select,
.stTextInput input {{
    border: 2px solid var(--light-color);
    border-radius: 6px;
    padding: 10px;
}}

.stNumberInput input:focus,
.stSelectbox select:focus,
.stTextInput input:focus {{
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(0,0,0,0.08);
    outline: none;
}}

/* =========================
   FILE UPLOADER
========================= */
[data-testid="stFileUploader"] {{
    background-color: var(--light-color);
    border: 2px dashed var(--primary-color);
    border-radius: 12px;
    padding: 20px;
}}

/* =========================
   EXPANDER
========================= */
.streamlit-expanderHeader {{
    background-color: var(--light-color);
    color: var(--primary-color);
    border-radius: 6px;
    font-weight: 600;
}}

/* =========================
   ALERTES (SUCCESS / INFO)
========================= */
.element-container .stAlert {{
    background-color: var(--light-color);
    border-left: 5px solid var(--primary-color);
}}

/* =========================
   HEADER PRINCIPAL
========================= */
.header-container {{
    background: linear-gradient(
        90deg,
        var(--primary-color) 0%,
        var(--secondary-color) 100%
    );
    padding: 32px;
    border-radius: 14px;
    color: var(--white);
    text-align: center;
    margin-bottom: 32px;
}}

/* =========================
   ICÔNES + TEXTE
========================= */
.icon-text {{
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 18px;
    margin: 10px 0;
}}

/* =========================
   ANIMATIONS
========================= */
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes pulse {{
    0% {{ transform: scale(1); }}
    50% {{ transform: scale(1.05); }}
    100% {{ transform: scale(1); }}
}}

.fade-in {{
    animation: fadeIn 0.6s ease-in;
}}

.pulse {{
    animation: pulse 1.5s infinite;
}}

</style>
""", unsafe_allow_html=True)



# Import des utilitaires
from utils.tensorflow_model import TensorFlowPredictor
# from utils.pytorch_model import PyTorchPredictor

# Initialisation des modèles
@st.cache_resource
def load_models():
    tf_model = TensorFlowPredictor('models/best_model.h5')
    # pt_model = PyTorchPredictor('models/best_model.pth')
    return tf_model
    # return tf_model, pt_model
# Fonction pour créer des métriques stylisées
def create_metric_card(title, value, icon):
    return f"""
    <div class="metric-card">
        <div class="icon-text">
            <span style="font-size: 24px;">{icon}</span>
            <div>
                <h4 style="margin: 0; color: var(--dark-gray);">{title}</h4>
                <h2 style="margin: 5px 0; color: var(--primary-color);">{value}</h2>
            </div>
        </div>
    </div>
    """

# Fonction pour le header
def show_header(title, subtitle):
    st.markdown(f"""
    <div class="header-container fade-in">
        <h1 style="color: white; margin: 0;">🏦 {title}</h1>
        <p style="color: white; margin: 10px 0 0 0; font-size: 18px;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# fonction pour les valeurs catégorielles

def load_frequency_encoders():
    return {
        'agence': {'Agence_Centre': 0.323, 'Agence_Sud': 0.342, 'Agence_Nord':0.335},
        'banque': {'Société Générale': 0.238, 'UBA': 0.260, 'Ecobank': 0.251, 'BGFI': 0.251},
        'lieu': {'Bafoussam': 0.266, 'Douala': 0.251, 'Yaoundé': 0.243, 'Garoua': 0.240}
    }
        
# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 48px; margin: 0;">🏦</h1>
        <h2 style="margin: 10px 0;">Deep Learning</h2>
        <h3 style="margin: 0;">Bancaire</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    menu = st.radio(
        "📋 Navigation",
        ["🏠 Accueil", "🔮 Prédictions TensorFlow", "⚡ Prédictions PyTorch", 
         "📊 Analyses & Visualisations", "⚙️ Paramètres"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <p style="font-size: 12px;">Version 1.0.0</p>
        <p style="font-size: 12px;">©2026 Banking AI</p>
    </div>
    """, unsafe_allow_html=True)

# PAGE 1: ACCUEIL
if menu == "🏠 Accueil":
    show_header(
        "Système de Prédiction des Dépenses Bancaires",
        "Intelligence Artificielle pour l'Analyse Financière"
    )
    
    # Section d'introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="fade-in">
            <h2>📌 Bienvenue dans notre Solution AI</h2>
            <p style="font-size: 18px; line-height: 1.8; color:#4A4A4A;">
                Notre application utilise des modèles de <b>Deep Learning</b> avancés pour prédire 
                avec précision les dépenses bancaires de vos clients. Grâce aux frameworks 
                <b>TensorFlow</b> et <b>PyTorch</b>, nous offrons une analyse fiable et rapide.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Objectifs de l'Application")
        
        objectives = [
            ("📈", "Prédire les dépenses futures", "Anticiper le comportement financier"),
            ("🔍", "Analyser les tendances", "Identifier les patterns de consommation"),
            ("⚠️", "Détecter les anomalies", "Prévenir les dépassements budgétaires"),
            ("📊", "Visualiser les données", "Tableaux de bord interactifs")
        ]
        
        for icon, title, desc in objectives:
            st.markdown(f"""
            <div class="icon-text" style="margin: 15px 0;">
                <span style="font-size: 32px;">{icon}</span>
                <div>
                    <strong style="color: var(--primary-color);">{title}</strong>
                    <p style="margin: 0; color: var(--dark-gray);">{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: var(--light-color); padding: 20px; border-radius: 10px; text-align: center;">
            <h1 style="font-size: 64px; margin: 20px 0;">💰</h1>
            <h3>Prédiction Intelligente</h3>
            <p>Basée sur l'IA</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Statistiques clés
        st.markdown(create_metric_card("Précision", "97.5%", "🎯"), unsafe_allow_html=True)
        st.markdown(create_metric_card("Clients", "1000+", "👥"), unsafe_allow_html=True)
        st.markdown(create_metric_card("Prédictions", "50K+", "🔮"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Description des données
    st.markdown("### 📋 Description des Données")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>💵 Données Financières</h4>
            <ul style="line-height: 2;">
                <li><b>Bilan Financier</b>: État global</li>
                <li><b>Actifs</b>: Ressources possédées</li>
                <li><b>Revenu</b>: Entrées d'argent</li>
                <li><b>Dépenses</b>: Sorties d'argent</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Indicateurs Clés</h4>
            <ul style="line-height: 2;">
                <li><b>Taux d'intérêt</b>: Coût du crédit</li>
                <li><b>Flux de trésorerie</b>: Liquidités</li>
                <li><b>Capital</b>: Fonds disponibles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🏢 Informations Contextuelles</h4>
            <ul style="line-height: 2;">
                <li><b>Agence</b>: Point de service</li>
                <li><b>Banque</b>: Institution</li>
                <li><b>Lieu</b>: Localisation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Workflow
    st.markdown("### 🔄 Comment ça marche ?")
    
    workflow_cols = st.columns(4)
    
    steps = [
        ("1️⃣", "Collecte", "Importez vos données"),
        ("2️⃣", "Saisie", "Remplissez le formulaire"),
        ("3️⃣", "Prédiction", "IA calcule les dépenses"),
        ("4️⃣", "Résultats", "Visualisez & téléchargez")
    ]
    
    for col, (num, title, desc) in zip(workflow_cols, steps):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: var(--light-color); border-radius: 10px;">
                <h1 style="font-size: 48px; margin: 0;">{num}</h1>
                <h4 style="color: var(--primary-color); margin: 10px 0;">{title}</h4>
                <p style="color: var(--dark-gray);">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Call to action
    st.markdown("""
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, var(--light-color) 0%, var(--secondary-color) 100%); border-radius: 10px;">
        <h2 style="color: var(--primary-color);">🚀 Prêt à commencer ?</h2>
        <p style="font-size: 18px; color: var(--dark-gray);">
            Choisissez votre framework préféré dans le menu latéral et commencez vos prédictions !
        </p>
    </div>
    """, unsafe_allow_html=True)

# PAGE 2: PRÉDICTIONS (TensorFlow )
elif menu == "🔮 Prédictions TensorFlow":
    framework = "TensorFlow" if "TensorFlow" in menu else "PyTorch"
    icon = "🔮" if framework == "TensorFlow" else "⚡"
    
    show_header(
        f"Prédictions avec {framework}",
        f"Modèle Deep Learning basé sur {framework}"
    )
    
    # Chargement du modèle
    try:
        tf_model = load_models()
        # tf_model, pt_model = load_models()
        # model = tf_model if framework == "TensorFlow" else pt_model
        model = tf_model if framework == "TensorFlow" else ""
        st.success(f"✅ Modèle {framework} chargé avec succès !")
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle: {str(e)}")
        st.stop()
    
    # Tabs pour les différentes options
    tab1, tab2 = st.tabs(["📁 Import Fichier", "✍️ Saisie Manuelle"])
    
    # TAB 1: Import de fichier
    with tab1:
        st.markdown("### 📤 Importez vos données")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choisissez un fichier CSV ou Excel",
                type=['csv', 'xlsx', 'xls'],
                help="Format: bilan_financier, actifs, revenu, taux_interet, flux_tresorerie, capital, agence_freq, banque_freq, lieu_freq"
            )
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>📋 Format requis</h4>
                <ul style="font-size: 14px;">
                    <li>CSV ou Excel</li>
                    <li>9 colonnes</li>
                    <li>En-têtes inclus</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                # Lecture du fichier
                if uploaded_file.name.endswith('.csv'):
                    try:
                        df = pd.read_csv(uploaded_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(uploaded_file, encoding='latin1')
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Fichier chargé: {len(df)} lignes")
                
                # Aperçu des données
                with st.expander("👀 Aperçu des données", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Vérification des colonnes
                required_cols = ['bilan_financier', 'actifs', 'revenu', 'taux_interet', 
                                'flux_tresorerie', 'capital', 'agence', 'banque', 'lieu']
                
                encoded_cols = ['bilan_financier', 'actifs', 'revenu', 'taux_interet', 
                                'flux_tresorerie', 'capital', 'agence_freq', 'banque_freq', 'lieu_freq']
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Colonnes manquantes: {', '.join(missing_cols)}")
                else:
                    st.info(f"✅ Toutes les colonnes requises sont présentes")
                    
                    if st.button("🔮 Lancer les Prédictions", key="predict_file"):
                        with st.spinner(f"🔄 Prédiction en cours avec {framework}..."):
                            # Préparation des données
                            # Charger les encodeurs
                            freq_encoders = load_frequency_encoders()
                            
                            # Encoder les colonnes catégorielles
                            df_encoded = df.copy()
                            
                            if 'agence' in df.columns:
                                df_encoded['agence_freq'] = df['agence'].map(freq_encoders['agence'])
                                df_encoded['agence_freq'] = df_encoded['agence_freq'].fillna(
                                    df_encoded['agence_freq'].mean()
                                )
                            
                            if 'banque' in df.columns:
                                df_encoded['banque_freq'] = df['banque'].map(freq_encoders['banque'])
                                df_encoded['banque_freq'] = df_encoded['banque_freq'].fillna(
                                    df_encoded['banque_freq'].mean()
                                )
                            
                            if 'lieu' in df.columns:
                                df_encoded['lieu_freq'] = df['lieu'].map(freq_encoders['lieu'])
                                df_encoded['lieu_freq'] = df_encoded['lieu_freq'].fillna(
                                    df_encoded['lieu_freq'].mean()
                                )
                            # supprimer les anciennes colonnes catégorielles
                            df_encoded.drop(['agence','banque','lieu'], axis=1, inplace=True)
                            
                            # affichage du tableau des données
                            # st.write("dataset financier chargés...")
                            
                            # st.table(df_encoded.head(5))
                            
                            X = df_encoded[encoded_cols].values
                            # Prédiction
                            predictions = model.predict(X)
                            
                            # Ajout des prédictions
                            df['depenses_predites'] = predictions
                            
                            st.balloons()
                            st.success("✅ Prédictions terminées !")
                            
                            # Statistiques
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown(create_metric_card(
                                    "Moyenne", 
                                    f"{predictions.mean():.2f} FCFA", 
                                    "💰"
                                ), unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(create_metric_card(
                                    "Maximum", 
                                    f"{predictions.max():.2f} FCFA", 
                                    "📈"
                                ), unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(create_metric_card(
                                    "Minimum", 
                                    f"{predictions.min():.2f} FCFA", 
                                    "📉"
                                ), unsafe_allow_html=True)
                            
                            with col4:
                                st.markdown(create_metric_card(
                                    "Écart-type", 
                                    f"{predictions.std():.2f} FCFA", 
                                    "📊"
                                ), unsafe_allow_html=True)
                            
                            # Graphique de distribution
                            fig = px.histogram(
                                predictions, 
                                nbins=30,
                                title="Distribution des Dépenses Prédites",
                                labels={'value': 'Dépenses (FCFA)', 'count': 'Fréquence'},
                                color_discrete_sequence=['#DC143C']
                            )
                            fig.update_layout(
                                plot_bgcolor='white',
                                paper_bgcolor='var(--light-color)',
                                font=dict(color='#333333')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Affichage des résultats
                            with st.expander("📊 Résultats détaillés", expanded=True):
                                st.dataframe(df, use_container_width=True)
                            
                            # Téléchargement
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Télécharger les Résultats (CSV)",
                                data=csv,
                                file_name=f"predictions_{framework}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                            )
            
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
    
    # TAB 2: Saisie manuelle
    with tab2:
        st.markdown("### ✍️ Saisissez les données du client")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 💵 Données Financières")
                bilan_financier = st.number_input(
                    "Bilan Financier (FCFA)", 
                    min_value=0.0, 
                    value=50000.0,
                    step=1000.0,
                    help="État financier global du client"
                )
                actifs = st.number_input(
                    "Actifs (FCFA)", 
                    min_value=0.0, 
                    value=25000.0,
                    step=1000.0,
                    help="Ressources possédées par le client"
                )
                revenu = st.number_input(
                    "Revenu (FCFA)", 
                    min_value=0.0, 
                    value=2500.0,
                    step=100.0,
                    help="Revenu mensuel du client"
                )
            
            with col2:
                st.markdown("#### 📊 Indicateurs")
                taux_interet = st.number_input(
                    "Taux d'Intérêt (%)", 
                    min_value=0.0, 
                    max_value=100.0,
                    value=5.0,
                    step=0.1,
                    help="Taux d'intérêt applicable"
                )
                flux_tresorerie = st.number_input(
                    "Flux de Trésorerie (FCFA)", 
                    min_value=0.0, 
                    value=8000.0,
                    step=500.0,
                    help="Liquidités disponibles"
                )
                capital = st.number_input(
                    "Capital (FCFA)", 
                    min_value=0.0, 
                    value=40000.0,
                    step=1000.0,
                    help="Fonds propres disponibles"
                )
            
            with col3:
                st.markdown("#### 🏢 Contexte")
                
                # Charger les encodeurs
                freq_encoders = load_frequency_encoders()
                
                # Agence
                agence = st.selectbox(
                    "🏢 Agence", 
                    options=list(freq_encoders['agence'].keys()),
                    help="Sélectionnez l'agence du client"
                )
                agence_freq = freq_encoders['agence'][agence]
                # st.caption(f"📊 Fréquence encodée: {agence_freq:.3f}")
                
                # Banque
                banque = st.selectbox(
                    "🏦 Banque", 
                    options=list(freq_encoders['banque'].keys()),
                    help="Sélectionnez la banque"
                )
                banque_freq = freq_encoders['banque'][banque]
                # st.caption(f"📊 Fréquence encodée: {banque_freq:.3f}")
                
                # Lieu
                lieu = st.selectbox(
                    "📍 Lieu", 
                    options=list(freq_encoders['lieu'].keys()),
                    help="Sélectionnez le lieu"
                )
                lieu_freq = freq_encoders['lieu'][lieu]
                # st.caption(f"📊 Fréquence encodée: {lieu_freq:.3f}")
            
            submitted = st.form_submit_button(
                f"🔮 Prédire avec {framework}",
                use_container_width=True
            )
            
            if submitted:
                with st.spinner(f"🔄 Calcul en cours avec {framework}..."):
                    # Préparation des données
                    input_data = np.array([[
                        bilan_financier, actifs, revenu, taux_interet,
                        flux_tresorerie, capital, agence_freq, banque_freq, lieu_freq
                    ]])
                    
                    # st.write(input_data)
                    
                    # Prédiction
                    
                    # st.write("prediction", model.predict(input_data))
                    prediction = model.predict(input_data)[0]
                    
                    st.balloons()
                    
                    # Affichage du résultat
                    st.markdown("---")
                    st.markdown("### 🎯 Résultat de la Prédiction")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); 
                                    padding: 40px; border-radius: 15px; text-align: center; color: white;">
                            <h4 style="margin: 0; color: white;">💰 Dépenses Prédites</h4>
                            <h1 style="margin: 20px 0; font-size: 48px; color: white;">{prediction:.2f} FCFA</h1>
                            <p style="margin: 0; color: white;">Basé sur le modèle {framework}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Analyse comparative
                        ratio_depenses_revenu = (prediction / revenu * 100) if revenu > 0 else 0
                        ratio_depenses_capital = (prediction / capital * 100) if capital > 0 else 0
                        
                        st.markdown("""
                        <div class="metric-card">
                            <h4>📊 Analyse Détaillée</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.metric(
                            "Ratio Dépenses/Revenu",
                            f"{ratio_depenses_revenu:.1f}%",
                            delta=f"{ratio_depenses_revenu - 50:.1f}% vs moyenne"
                        )
                        
                        st.metric(
                            "Ratio Dépenses/Capital",
                            f"{ratio_depenses_capital:.1f}%",
                            delta=f"{ratio_depenses_capital - 3:.1f}% vs normal"
                        )
                        
                        # Recommandation
                        if ratio_depenses_revenu > 70:
                            st.warning("⚠️ Attention: Dépenses élevées par rapport au revenu")
                        elif ratio_depenses_revenu > 50:
                            st.info("ℹ️ Dépenses dans la moyenne")
                        else:
                            st.success("✅ Dépenses maîtrisées")
                    
                    # Graphique radar
                    st.markdown("### 📊 Profil Financier")
                    
                    categories = ['Bilan', 'Actifs', 'Revenu', 'Capital', 'Flux']
                    values = [
                        (bilan_financier / 100000) * 100,
                        (actifs / 50000) * 100,
                        (revenu / 5000) * 100,
                        (capital / 100000) * 100,
                        (flux_tresorerie / 20000) * 100
                    ]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        fillcolor='rgba(220, 20, 60, 0.3)',
                        line=dict(color='#DC143C', width=2),
                        name='Profil Client'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100],
                                gridcolor='#FFE5E5'
                            ),
                            bgcolor='white'
                        ),
                        showlegend=False,
                        paper_bgcolor='var(--light-color)',
                        plot_bgcolor='white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("""
                        **Interprétation du profil financier :**

                        Ce graphique radar représente le profil financier du client sur 5 axes clés :

                        - **Bilan** : Niveau global du bilan financier du client par rapport à 100 000 FCFA de référence.  
                        - **Actifs** : Valeur totale des actifs détenus par rapport à 50 000 FCFA.  
                        - **Revenu** : Revenu mensuel ou annuel (selon vos données) par rapport à 5 000 FCFA.  
                        - **Capital** : Capital disponible par rapport à 100 000 FCFA.  
                        - **Flux de trésorerie** : Capacité de génération de liquidités par rapport à 20 000 FCFA.

                        Les valeurs sont exprimées en pourcentage par rapport aux seuils de référence.  
                        Une **zone plus étendue** sur le radar indique une **performance financière plus élevée** sur cet axe.  
                        Une **zone plus petite** montre un axe où le client pourrait avoir des **points faibles ou des opportunités d'amélioration**.

                        💡 Ce profil permet de visualiser rapidement les forces et les points à surveiller dans la situation financière du client.
                        """)

elif menu == "⚡ Prédictions PyTorch":
    show_header(
        "Prédictions avec PyTorch",
        "Application dédiée PyTorch"
    )
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FFE5E5 0%, #FFFFFF 100%); 
                padding: 40px; border-radius: 15px; text-align: center; margin: 40px 0;">
        <div style="font-size: 80px; margin-bottom: 20px;">⚡</div>
        <h2 style="color: #DC143C; margin-bottom: 20px;">Application PyTorch Dédiée</h2>
        <p style="font-size: 18px; color: #555; line-height: 1.8; margin-bottom: 30px;">
            Les prédictions avec <b>PyTorch</b> sont disponibles sur une application dédiée 
            pour une meilleure performance et optimisation des ressources GPU.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 30px; border-radius: 10px; 
                    border: 2px solid #DC143C; box-shadow: 0 4px 12px rgba(220, 20, 60, 0.2);">
            <h3 style="color: #DC143C; text-align: center; margin-top: 0;">🚀 Accéder à l'application</h3>
            <p style="text-align: center; color: #666; margin-bottom: 25px;">
                Cliquez sur le bouton ci-dessous pour être redirigé vers l'application PyTorch
            </p>
        """, unsafe_allow_html=True)
        
        # Bouton de redirection principal
        pytorch_url = "https://financialexpensesapp-2rqpkcpkdterecpnj5buy5.streamlit.app/"
        
        st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <a href="{pytorch_url}" target="_blank" style="text-decoration: none;">
                <button style="background: linear-gradient(135deg, #DC143C 0%, #FF6B6B 100%);
                               color: white; border: none; padding: 15px 40px; font-size: 18px;
                               border-radius: 10px; cursor: pointer; font-weight: bold;
                               box-shadow: 0 4px 15px rgba(220, 20, 60, 0.4);
                               transition: transform 0.3s;">
                    ⚡ Ouvrir l'Application PyTorch
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Lien texte alternatif
        st.markdown(f"""
        <p style="text-align: center; margin-top: 20px; font-size: 14px; color: #666;">
            Ou copiez ce lien : <br>
            <a href="{pytorch_url}" target="_blank" style="color: #DC143C; word-break: break-all;">
                {pytorch_url}
            </a>
        </p>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Informations complémentaires
    st.markdown("### ℹ️ Pourquoi deux applications séparées ?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Performance Optimale</h4>
            <p style="font-size: 14px; color: #555;">
                Chaque framework bénéficie de ressources dédiées 
                pour des temps de réponse optimaux.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>⚙️ Configuration Spécifique</h4>
            <p style="font-size: 14px; color: #555;">
                PyTorch utilise des optimisations GPU spécifiques 
                qui nécessitent un environnement dédié.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🔄 Flexibilité</h4>
            <p style="font-size: 14px; color: #555;">
                Mise à jour et maintenance indépendantes 
                pour chaque technologie.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Section d'aide
    with st.expander("❓ Besoin d'aide pour la navigation ?"):
        st.markdown("""
        **Comment accéder à l'application PyTorch :**
        
        1. **Option 1 (Recommandée)** : Cliquez sur le bouton rouge "Ouvrir l'Application PyTorch" ci-dessus
           - Une nouvelle fenêtre s'ouvrira automatiquement
           - Vous serez redirigé vers l'application PyTorch
        
        2. **Option 2** : Copiez le lien et collez-le dans votre navigateur
           - Faites un clic droit → "Copier l'adresse du lien"
           - Collez dans une nouvelle fenêtre de navigateur
        
        3. **Option 3** : Enregistrez le lien en favori pour un accès rapide
        
        **⚠️ Note importante :**
        - Les deux applications sont indépendantes
        - Vos données ne sont pas partagées entre les applications
        - Vous pouvez utiliser les deux en parallèle dans des onglets différents
        
        **💡 Conseil :** Gardez cette page ouverte pour revenir facilement à l'application TensorFlow
        """)
# PAGE 4: ANALYSES & VISUALISATIONS
elif menu == "📊 Analyses & Visualisations":
    show_header(
        "Analyses & Visualisations",
        "Tableaux de bord interactifs et analyses approfondies"
    )
    
    # Chargement des données d'exemple
    try:
        df = pd.read_csv('data/dataset_financier.csv')
        
        # Filtres
        st.markdown("### 🔍 Filtres")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'banque' in df.columns:
                banques = ['Toutes'] + list(df['banque'].unique())
                selected_banque = st.selectbox("🏦 Banque", banques)
        
        with col2:
            if 'agence' in df.columns:
                agences = ['Toutes'] + list(df['agence'].unique())
                selected_agence = st.selectbox("🏢 Agence", agences)
        
        with col3:
            if 'lieu' in df.columns:
                lieux =lieux = ['Tous'] + list(df['lieu'].unique())
                selected_lieu = st.selectbox("📍 Lieu", lieux)
        
        # Application des filtres
        df_filtered = df.copy()
        if selected_banque != 'Toutes' and 'banque' in df.columns:
            df_filtered = df_filtered[df_filtered['banque'] == selected_banque]
        if selected_agence != 'Toutes' and 'agence' in df.columns:
            df_filtered = df_filtered[df_filtered['agence'] == selected_agence]
        if selected_lieu != 'Tous' and 'lieu' in df.columns:
            df_filtered = df_filtered[df_filtered['lieu'] == selected_lieu]
        
        st.info(f"📊 {len(df_filtered)} clients après filtrage")
        
        st.markdown("---")
        
        # KPIs
        st.markdown("### 📈 Indicateurs Clés")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_depenses = df_filtered['depenses'].mean() if 'depenses' in df_filtered.columns else 0
            st.markdown(create_metric_card(
                "Dépenses Moyennes",
                f"{avg_depenses:.2f} FCFA",
                "💰"
            ), unsafe_allow_html=True)
        
        with col2:
            avg_revenu = df_filtered['revenu'].mean() if 'revenu' in df_filtered.columns else 0
            st.markdown(create_metric_card(
                "Revenu Moyen",
                f"{avg_revenu:.2f} FCFA",
                "💵"
            ), unsafe_allow_html=True)
        
        with col3:
            avg_capital = df_filtered['capital'].mean() if 'capital' in df_filtered.columns else 0
            st.markdown(create_metric_card(
                "Capital Moyen",
                f"{avg_capital:.2f} FCFA",
                "🏦"
            ), unsafe_allow_html=True)
        
        with col4:
            avg_taux = df_filtered['taux_interet'].mean() if 'taux_interet' in df_filtered.columns else 0
            st.markdown(create_metric_card(
                "Taux Moyen",
                f"{avg_taux:.2f} %",
                "📊"
            ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Graphiques
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Distributions", "📈 Tendances", "🔍 Corrélations", "🗺️ Géographie"
        ])
        
        with tab1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FFE5E5 0%, #FFFFFF 100%); 
                        padding: 20px; border-radius: 10px; border-left: 4px solid #DC143C; margin-bottom: 20px;">
                <h4 style="color: #DC143C; margin-top: 0;">📊 Analyse des Distributions</h4>
                <p style="color: #333; line-height: 1.8; margin-bottom: 0;">
                    Les graphiques de distribution permettent de visualiser <b>comment les valeurs sont réparties</b> 
                    dans votre portefeuille client. Ils révèlent les <b>tendances centrales</b>, les <b>valeurs extrêmes</b> 
                    et la <b>dispersion</b> des données financières.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'depenses' in df_filtered.columns:
                    st.markdown("""
                    <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <p style="margin: 0; color: #555; font-size: 14px;">
                            <b>💡 Interprétation :</b> Ce graphique montre la <b>répartition des dépenses</b> des clients.
                            <br>• <b>Nous observons un Pic élevé</b> = ce qui implique que Beaucoup de clients ont des dépenses similaires
                            <br>• <b>egalement une queue longue à droite</b> = ce qui implique la Présence de clients aux dépenses très élevées
                            <br>• <b>enfin une forme bimodale</b> = Nous avons Deux groupes distincts de comportement de dépense (basse et haute)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig = px.histogram(
                        df_filtered,
                        x='depenses',
                        nbins=30,
                        title="Distribution des Dépenses",
                        color_discrete_sequence=['#DC143C']
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='var(--light-color)',
                        xaxis_title="Dépenses (FCFA)",
                        yaxis_title="Fréquence"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'revenu' in df_filtered.columns:
                    st.markdown("""
                    <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <p style="margin: 0; color: #555; font-size: 14px;">
                            <b>💡 Interprétation :</b> Le <b>box plot</b> révèle la distribution des revenus avec ses quartiles.
                            <br>• <b>Boîte</b> = 50% des clients au centre de la distribution
                            <br>• <b>Points isolés</b> = Valeurs aberrantes (revenus exceptionnels)
                            <br>• <b>Médiane (ligne)</b> = Revenu typique des clients
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig = px.box(
                        df_filtered,
                        y='revenu',
                        title="Distribution des Revenus",
                        color_discrete_sequence=['#FF6B6B']
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='var(--light-color)',
                        yaxis_title="Revenu (FCFA)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Graphiques supplémentaires
            col3, col4 = st.columns(2)
            
            with col3:
                if 'capital' in df_filtered.columns:
                    st.markdown("""
                    <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <p style="margin: 0; color: #555; font-size: 14px;">
                            <b>💡 Interprétation :</b> Le <b>violin plot</b> combine densité et box plot pour le capital.
                            <br>• <b>Largeur</b> = à 40k la largeur du violon est énorme donc nous avons une concentration de clients avec ce capital
                            <br>• <b>Forme étroite</b> = dans les intervalles [0-20k], [60, 8k] nous avons Peu de variabilité du capital
                            <br>• <b>Deux bosses</b> = les deux bosses implique une Segmentation possible de la clientèle (clients à capital haut et à capital bas)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                     
                    fig = px.violin(
                        df_filtered,
                        y='capital',
                        title="Distribution du Capital",
                        color_discrete_sequence=['#DC143C'],
                        box=True
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='var(--light-color)',
                        yaxis_title="Capital (FCFA)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                if 'flux_tresorerie' in df_filtered.columns:
                    st.markdown("""
                    <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <p style="margin: 0; color: #555; font-size: 14px;">
                            <b>💡 Interprétation :</b> Distribution du flux de trésorerie disponible.
                            <br>• <b>Asymétrie à gauche</b> = Majorité avec flux faible, quelques-uns élevés
                            <br>• <b>Distribution uniforme</b> = Diversité équilibrée des liquidités
                            <br>• <b>Pics multiples</b> = Différents profils de gestion de trésorerie
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                     
                    fig = px.histogram(
                        df_filtered,
                        x='flux_tresorerie',
                        nbins=25,
                        title="Distribution du Flux de Trésorerie",
                        color_discrete_sequence=['#FF6B6B']
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='var(--light-color)',
                        xaxis_title="Flux de Trésorerie (FCFA)",
                        yaxis_title="Fréquence"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FFE5E5 0%, #FFFFFF 100%); 
                        padding: 20px; border-radius: 10px; border-left: 4px solid #DC143C; margin-bottom: 20px;">
                <h4 style="color: #DC143C; margin-top: 0;">📈 Analyse des Tendances</h4>
                <p style="color: #333; line-height: 1.8; margin-bottom: 0;">
                    Cette section compare les <b>performances par segment</b> (banque, agence, lieu) et révèle 
                    les <b>disparités de comportement</b> financier. Identifiez les segments les plus performants 
                    ou nécessitant une attention particulière.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if 'banque' in df_filtered.columns and 'depenses' in df_filtered.columns:
                depenses_par_banque = df_filtered.groupby('banque')['depenses'].mean().reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <p style="margin: 0; color: #555; font-size: 14px;">
                            <b>💡 Interprétation :</b> Comparaison des dépenses moyennes entre banques.
                            <br>• <b>Barres hautes au niveau de UBA</b> = UBA à une clientèle à fort pouvoir de dépense
                            <br>• <b>Couleur intense</b> = Niveau de dépenses élevé
                            <br>• <b>Différences marquées avec les autres banques</b> = Opportunité de stratégies ciblées
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig = px.bar(
                        depenses_par_banque,
                        x='banque',
                        y='depenses',
                        title="Dépenses Moyennes par Banque",
                        color='depenses',
                        color_continuous_scale=['#FFE5E5', '#DC143C']
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='var(--light-color)',
                        xaxis_title="Banque",
                        yaxis_title="Dépenses Moyennes (FCFA)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'agence' in df_filtered.columns:
                        st.markdown("""
                        <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <p style="margin: 0; color: #555; font-size: 14px;">
                                <b>💡 Interprétation :</b> Performance des agences en termes de dépenses clients.
                                <br>• <b>l'Agence dominante est celle du centre</b> = Zone à fort potentiel commercial
                                <br>• <b>l'Agences faibles</b> = ca pourrait signifier qu'elles nécessite des actions marketing ciblées
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        depenses_par_agence = df_filtered.groupby('agence')['depenses'].mean().reset_index()
                        
                        fig = px.bar(
                            depenses_par_agence,
                            x='agence',
                            y='depenses',
                            title="Dépenses Moyennes par Agence",
                            color='depenses',
                            color_continuous_scale=['#FFE5E5', '#DC143C']
                        )
                        fig.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='var(--light-color)',
                            xaxis_title="Agence",
                            yaxis_title="Dépenses Moyennes (FCFA)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Évolution temporelle (si données temporelles disponibles)
            if 'revenu' in df_filtered.columns and 'depenses' in df_filtered.columns:
                st.markdown("#### 💹 Comparaison Revenus vs Dépenses")
                
                st.markdown("""
                <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <p style="margin: 0; color: #555; font-size: 14px;">
                        <b>💡 Interprétation :</b> Évolution comparative des revenus et dépenses client par client.
                        <br>• <b>Courbes non parallèles</b> =aucune corrélation forte entre revenu et dépenses
                        <br>• <b>la courbe Verte est au-dessus sur les intervalles [0,3]; [5,22]; [25,50]</b> = Capacité d'épargne, situation saine
                        <br>• <b>la courbe Rouge est au-dessus sur les intervalles [3,5]</b> = Dépenses > revenus, risque d'endettement
                        <br>• <b>il y'a Croisements au niveau de l'intervalle [23,25]</b> = Volatilité du comportement financier
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                comparison_data = df_filtered[['revenu', 'depenses']].head(50)
                comparison_data['index'] = range(len(comparison_data))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=comparison_data['index'],
                    y=comparison_data['revenu'],
                    mode='lines+markers',
                    name='Revenus',
                    line=dict(color='#28a745', width=2),
                    marker=dict(size=6)
                ))
                fig.add_trace(go.Scatter(
                    x=comparison_data['index'],
                    y=comparison_data['depenses'],
                    mode='lines+markers',
                    name='Dépenses',
                    line=dict(color='#DC143C', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title="Évolution Revenus vs Dépenses (50 premiers clients)",
                    plot_bgcolor='white',
                    paper_bgcolor='var(--light-color)',
                    xaxis_title="Clients",
                    yaxis_title="Montant (FCFA)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FFE5E5 0%, #FFFFFF 100%); 
                        padding: 20px; border-radius: 10px; border-left: 4px solid #DC143C; margin-bottom: 20px;">
                <h4 style="color: #DC143C; margin-top: 0;">🔍 Analyse des Corrélations</h4>
                <p style="color: #333; line-height: 1.8; margin-bottom: 0;">
                    Les corrélations révèlent les <b>relations entre variables financières</b>. Une corrélation forte 
                    indique que deux variables évoluent ensemble, ce qui est crucial pour comprendre les 
                    <b>leviers d'influence</b> sur les dépenses.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                st.markdown("#### 🔗 Matrice de Corrélation")
                
                st.markdown("""
                <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <p style="margin: 0; color: #555; font-size: 14px;">
                        <b>💡 Comment lire :</b> Cette heatmap montre la force des relations entre variables.
                        <br>• <b>Rouge foncé (+1.0)</b> = Corrélation positive parfaite (les deux augmentent ensemble)
                        <br>• <b>Blanc (0.0)</b> = Aucune corrélation (variables indépendantes)
                        <br>• <b>Valeurs > 0.7</b> = Forte corrélation, attention à la multicolinéarité
                        <br>• <b>Valeurs < 0.3</b> = Faible relation entre les variables
                        <br>• <b>il n'ya donc aucune corrélation forte entre les variables de ce jeu de données
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                
                corr_matrix = df_filtered[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale=['#FFFFFF', '#FFE5E5', '#DC143C'],
                    title="Corrélations entre Variables Financières",
                    aspect='auto'
                )
                fig.update_layout(
                    paper_bgcolor='var(--light-color)',
                    plot_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plots des corrélations principales
                st.markdown("#### 📈 Relations entre Variables")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    
                    if 'revenu' in df_filtered.columns and 'depenses' in df_filtered.columns:
                        
                        st.markdown("""
                        <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <p style="margin: 0; color: #555; font-size: 14px;">
                                <b>💡 Interprétation :</b> Relation entre revenu et dépenses avec tendance linéaire.
                                <br>• <b>Points alignés</b> = Dépenses proportionnelles au revenu
                                <br>• <b>Points au-dessus</b> = Clients dépensant plus que leur revenu
                                <br>• <b>Ligne de tendance montante</b> = Corrélation positive confirmée
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        fig = px.scatter(
                            df_filtered,
                            x='revenu',
                            y='depenses',
                            title="Revenu vs Dépenses",
                            color='depenses',
                            color_continuous_scale=['#FFE5E5', '#DC143C'],
                            trendline="ols"
                        )
                        fig.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='var(--light-color)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'capital' in df_filtered.columns and 'actifs' in df_filtered.columns:
                        
                        st.markdown("""
                        <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <p style="margin: 0; color: #555; font-size: 14px;">
                                <b>💡 Interprétation :</b> Relation entre capital disponible et actifs possédés.
                                <br>• <b>Nuage concentré</b> = Relation stable et prévisible
                                <br>• <b>Dispersion large</b> = Stratégies d'investissement variées
                                <br>• <b>Pente forte</b> = Conversion efficace capital → actifs
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        fig = px.scatter(
                            df_filtered,
                            x='capital',
                            y='actifs',
                            title="Capital vs Actifs",
                            color='actifs',
                            color_continuous_scale=['#FFE5E5', '#DC143C'],
                            trendline="ols"
                        )
                        fig.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='var(--light-color)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #FFE5E5 0%, #FFFFFF 100%); 
                        padding: 20px; border-radius: 10px; border-left: 4px solid #DC143C; margin-bottom: 20px;">
                <h4 style="color: #DC143C; margin-top: 0;">🗺️ Analyse Géographique</h4>
                <p style="color: #333; line-height: 1.8; margin-bottom: 0;">
                    La dimension géographique révèle les <b>disparités territoriales</b> dans les comportements de dépense. 
                    Identifiez les <b>zones à fort potentiel</b> et optimisez votre <b>stratégie d'expansion</b> ou 
                    de <b>développement commercial</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if 'lieu' in df_filtered.columns and 'depenses' in df_filtered.columns:
                depenses_par_lieu = df_filtered.groupby('lieu')['depenses'].mean().reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <p style="margin: 0; color: #555; font-size: 14px;">
                            <b>💡 Interprétation :</b> Répartition des dépenses totales par zone géographique.
                            <br>• <b>la Grande part est celle de Yaoundé et Garoua</b> = Zone à forte activité commerciale
                            <br>• <b>Petite part correspond a celle de Douala </b> = Zone sous-exploitée ou à faible potentiel
                            <br>• <b>Distribution non équilibrée</b> = Couverture géographique non homogène
=                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig = px.pie(
                        depenses_par_lieu,
                        values='depenses',
                        names='lieu',
                        title="Répartition des Dépenses par Lieu",
                        color_discrete_sequence=px.colors.sequential.Reds
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label'
                    )
                    fig.update_layout(
                        paper_bgcolor='var(--light-color)',
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Treemap
                    if 'banque' in df_filtered.columns:
                        
                        st.markdown("""
                        <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <p style="margin: 0; color: #555; font-size: 14px;">
                                <b>💡 Interprétation :</b> Vue hiérarchique Lieu → Banque avec volumes de dépenses.
                                <br>• <b>Rectangles larges</b> = Forte contribution aux dépenses totales
                                <br>• <b>Couleur intense</b> = Niveau de dépenses élevé
                                <br>• <b>Subdivisions</b> = Part de chaque banque dans chaque zone
                                <br>• <b>Ce graphique permet d'Identifier que les banques BGFI, UBA dominent la zone de Bafoussam, La société générale celle de Yaoundé, UBA pour Douala et Ecobank pour Garoua</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        fig = px.treemap(
                            df_filtered,
                            path=['lieu', 'banque'],
                            values='depenses',
                            title="Hiérarchie Lieu → Banque → Dépenses",
                            color='depenses',
                            color_continuous_scale=['#FFE5E5', '#DC143C']
                        )
                        fig.update_layout(
                            paper_bgcolor='var(--light-color)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Tableau récapitulatif par lieu
            if 'lieu' in df_filtered.columns:
                st.markdown("#### 📋 Statistiques par Lieu")
                
                st.markdown("""
                <div style="background: #FFF9F9; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <p style="margin: 0; color: #555; font-size: 14px;">
                        <b>💡 Comment utiliser :</b> Tableau de bord comparatif des performances par zone.
                        <br>• <b>Dépenses Moy.</b> = Indicateur du pouvoir d'achat local
                        <br>• <b>Total Dépenses</b> = Volume d'affaires généré par zone
                        <br>• <b>Nb Clients</b> = Pénétration du marché local
                        <br>• <b>Ce plot permet de comparer</b> les ratios pour identifier les zones sous-performantes
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                
                stats_lieu = df_filtered.groupby('lieu').agg({
                    'depenses': ['mean', 'sum', 'count'],
                    'revenu': 'mean',
                    'capital': 'mean'
                }).round(2)
                
                stats_lieu.columns = ['Dépenses Moy.', 'Total Dépenses', 'Nb Clients', 'Revenu Moy.', 'Capital Moy.']
                st.dataframe(stats_lieu, use_container_width=True)
    
    except FileNotFoundError:
        st.warning("⚠️ Fichier de données introuvable. Veuillez placer 'dataset_financier.csv' dans le dossier 'data/'")
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")

# PAGE 5: PARAMÈTRES
elif menu == "⚙️ Paramètres":
    show_header(
        "Paramètres & Configuration",
        "Personnalisez votre expérience"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.expander("🎨 Personnalisation", expanded=True):
            st.markdown("#### Apparence")
            st.warning("⚠️ les modifications de thème nécessitent que vous switcher sur un autre thème pour voir le thème cliqué précédemment etre appliques\n par exemple pour choisir le thème bleu, je clique sur le thème bleu ensuite je clique sur un autre thème, rouge par exemple pour voir les modifications du thème bleu etre appliqué\n ce problème sera résolue dans les versions supérieur de streamlit  .")
            
            theme = st.radio(
                "Thème de couleur",
                list(THEMES.keys()),
                horizontal=True,
                index=list(THEMES.keys()).index(st.session_state.theme)
            )
            st.session_state.theme = theme
            
            # font_size = st.slider("Taille de police", 12, 20, 16)
            
            show_animations = st.checkbox(
                "✨ Afficher les animations",value=False)
            if show_animations:
                st.snow()
            animation_class = "fade-in pulse" if show_animations else ""
            
            # st.markdown("#### Préférences de prédiction")
            
            # default_framework = st.selectbox(
            #     "Framework par défaut",
            #     ["TensorFlow", "PyTorch"]
            # )
            
            # confidence_threshold = st.slider(
            #     "Seuil de confiance (%)",
            #     50, 100, 95
            # )
            
            # auto_save = st.checkbox("Sauvegarde automatique des résultats", value=True)
        
        # with st.expander("📊 Seuils Financiers"):
        #     st.markdown("#### Alertes automatiques")
            
        #     col_a, col_b = st.columns(2)
            
        #     with col_a:
        #         seuil_depenses_elevees = st.number_input(
        #             "Dépenses élevées (FCFA)",
        #             min_value=0.0,
        #             value=2000.0,
        #             step=100.0,
        #             help="Montant au-dessus duquel une alerte sera générée"
        #         )
                
        #         seuil_ratio_risque = st.slider(
        #             "Ratio dépenses/revenu à risque (%)",
        #             0, 100, 70,
        #             help="Pourcentage au-dessus duquel le client est considéré à risque"
        #         )
            
        #     with col_b:
        #         seuil_capital_faible = st.number_input(
        #             "Capital faible (FCFA)",
        #             min_value=0.0,
        #             value=10000.0,
        #             step=1000.0
        #         )
                
        #         seuil_flux_negatif = st.number_input(
        #             "Flux de trésorerie critique (FCFA)",
        #             min_value=0.0,
        #             value=5000.0,
        #             step=500.0
        #         )
            
        #     activer_notifications = st.checkbox(
        #         "Activer les notifications par email",
        #         value=False
        #     )
            
        #     if activer_notifications:
        #         email_notifications = st.text_input(
        #             "Email pour les alertes",
        #             placeholder="analyste@banque.com"
        #         )
        
        with st.expander("📁 Gestion des données"):
            st.markdown("#### Options d'export")
            
            format_export = st.multiselect(
                "Formats d'export disponibles",
                ["CSV", "Excel", "JSON", "PDF"],
                default=["CSV", "Excel"]
            )
            st.write(" ")
            
            # Vérifier si JSON ou PDF est sélectionné
            if "JSON" in format_export or "PDF" in format_export:
                st.warning(
                    "⚠️ Fonctionnalité en cours d'implémentation pour JSON et PDF. "
                    "Veuillez patienter..."
                )
            
            inclure_metadata = st.checkbox(
                "Inclure les métadonnées dans les exports",
                value=True
            )
            
            inclure_horodatage = st.checkbox(
                "Ajouter horodatage aux fichiers",
                value=True
            )
            
            st.markdown("#### Cache et Performance")
            
            col_c, col_d = st.columns(2)
            
            with col_c:
                if st.button("🗑️ Vider le cache", use_container_width=True):
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("✅ Cache vidé!")
                    
                    st.rerun()
            
            with col_d:
                cache_size = st.select_slider(
                    "Taille du cache",
                    options=["Petit", "Moyen", "Grand"],
                    value="Moyen"
                )
        
        with st.expander("🔒 Sécurité et Confidentialité"):
            st.markdown("#### Paramètres de sécurité")
            
            anonymiser_data = st.checkbox(
                "Anonymiser les données sensibles",
                value=True
            )
            
            log_predictions = st.checkbox(
                "Enregistrer l'historique des prédictions",
                value=True
            )
            
            duree_conservation = st.select_slider(
                "Durée de conservation des logs",
                options=["7 jours", "30 jours", "90 jours", "1 an"],
                value="30 jours"
            )
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>ℹ️ Informations Système</h3>
            <ul style="line-height: 2;">
                <li><b>Version</b>: 1.0.0</li>
                <li><b>Framework</b>: Streamlit 1.31</li>
                <li><b>IA</b>: TensorFlow & PyTorch</li>
                <li><b>Python</b>: 3.10+</li>
                <li><b>Dernière MAJ</b>: 19 Jan2026</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>📞 Support Technique</h3>
            <p><b>Email:</b> support@banking-ai.com</p>
            <p><b>Téléphone:</b> +237 692 647 443</p>
            <p><b>Horaires:</b> Lun-Ven 9h-18h</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>📚 Documentation</h3>
            <p>
                <a href="https://www.afristat.org/wp-content/uploads/2023/05/Documentation_Methodologie_Comptes_Financiers_Cameroun.pdf" style="color: var(--primary-color);">📖 Guide utilisateur</a><br>
                <a href="https://youtu.be/iPXffCwe0tU?si=1u5EniSUSeLDpiZB" style="color: var(--primary-color);">🎓 Tutoriels vidéo</a><br>
                <a href="https://www.afristat.org/wp-content/uploads/2023/05/Documentation_Methodologie_Comptes_Financiers_Cameroun.pdf" style="color: var(--primary-color);">❓ FAQ</a><br>
                <a href="https://www.afristat.org/wp-content/uploads/2023/05/Documentation_Methodologie_Comptes_Financiers_Cameroun.pdf" style="color: var(--primary-color);">🔧 API Documentation</a>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Bouton de sauvegarde
        if st.button("💾 Sauvegarder les Paramètres", use_container_width=True):
            st.balloons()
            st.success("✅ Paramètres sauvegardés avec succès!")
            
            # Simulation de sauvegarde
            with st.spinner("Enregistrement en cours..."):
                import time
                time.sleep(1)
            
            st.info("""
            **Paramètres sauvegardés:**
            - Thème: """ + theme + """
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: var(--dark-gray);">
    <p style="font-size: 16px;"><b>🏦 Banking AI - Deep Learning Platform</b></p>
    <p>©2026 Banking AI Solutions | Tous droits réservés</p>
    <p style="font-size: 12px;">Propulsé par TensorFlow & PyTorch | Développé avec ❤️ et Streamlit</p>
    <p style="font-size: 11px; color: #999;">
        <a href="#" style="color: var(--primary-color);">Mentions légales</a> | 
        <a href="#" style="color: var(--primary-color);">Politique de confidentialité</a> | 
        <a href="#" style="color: var(--primary-color);">Conditions d'utilisation</a>
    </p>
</div>
""", unsafe_allow_html=True)