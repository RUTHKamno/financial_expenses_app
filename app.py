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
    page_title="Torch Prédiction Dépenses Bancaires",
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
    --dark-gray: #333333;
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

from utils.pytorch_model import PyTorchPredictor

# Initialisation des modèles
@st.cache_resource
def load_models():
    pt_model = PyTorchPredictor('models/best_model.pth')
    return pt_model
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
        ["⚡ Prédictions PyTorch","🏠 Accueil"],
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
    st.write('in home page')
    # bouton de retour vers l'paplication initiale
# PAGE 2: PRÉDICTIONS (PyTorch)
elif menu == "⚡ Prédictions PyTorch":
    framework = "PyTorch"
    icon = "⚡"
    
    show_header(
        f"Prédictions avec {framework}",
        f"Modèle Deep Learning basé sur {framework}"
    )
    
    # Chargement du modèle
    try:
        pt_model = load_models()
        model = pt_model
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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: var(--dark-gray);">
    <p style="font-size: 16px;"><b>🏦 Banking AI - Deep Learning Platform</b></p>
    <p>©2026 Banking AI Solutions | Tous droits réservés</p>
    <p style="font-size: 12px;">Propulsé par PyTorch | Développé avec ❤️ et Streamlit</p>
    <p style="font-size: 11px; color: #999;">
        <a href="#" style="color: var(--primary-color);">Mentions légales</a> | 
        <a href="#" style="color: var(--primary-color);">Politique de confidentialité</a> | 
        <a href="#" style="color: var(--primary-color);">Conditions d'utilisation</a>
    </p>
</div>
""", unsafe_allow_html=True)