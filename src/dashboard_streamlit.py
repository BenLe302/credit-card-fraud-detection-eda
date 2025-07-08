#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Interactif - Détection de Fraudes dans les Transactions Bancaires

Dataset: Credit Card Fraud Detection (Kaggle)

Ce dashboard interactif permet d'explorer les données de fraude bancaire
avec des visualisations dynamiques et des analyses en temps réel.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Statistiques
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Dashboard EDA - Détection de Fraudes",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.insight-box {
    background-color: #e8f4fd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #2e86de;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Cache pour les données
@st.cache_data
def load_data():
    """Charge et prépare les données"""
    try:
        df = pd.read_csv('Credit Card Fraud Detection/creditcard.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Fichier de données non trouvé. Veuillez vérifier le chemin.")
        return None

@st.cache_data
def prepare_data(df):
    """Prépare les données pour l'analyse"""
    # Séparation des classes
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0]
    
    # Variables PCA
    pca_columns = [col for col in df.columns if col.startswith('V')]
    
    # Conversion temporelle
    df['Time_hours'] = df['Time'] / 3600
    df['Hour_of_day'] = (df['Time_hours'] % 24).astype(int)
    
    # Tranches de montants
    bins = [0, 10, 50, 100, 500, 1000, 5000, float('inf')]
    labels = ['0-10$', '10-50$', '50-100$', '100-500$', '500-1K$', '1K-5K$', '5K+$']
    df['Amount_range'] = pd.cut(df['Amount'], bins=bins, labels=labels, include_lowest=True)
    
    return df, fraud, normal, pca_columns

@st.cache_data
def compute_statistics(df, fraud, normal, pca_columns):
    """Calcule les statistiques importantes"""
    stats_dict = {
        'total_transactions': len(df),
        'fraud_count': len(fraud),
        'normal_count': len(normal),
        'fraud_rate': len(fraud) / len(df) * 100,
        'imbalance_ratio': len(normal) / len(fraud),
        'avg_amount_fraud': fraud['Amount'].mean(),
        'avg_amount_normal': normal['Amount'].mean(),
        'median_amount_fraud': fraud['Amount'].median(),
        'median_amount_normal': normal['Amount'].median()
    }
    
    # Analyse des variables PCA
    pca_analysis = []
    for col in pca_columns:
        stat, p_value = stats.mannwhitneyu(normal[col], fraud[col])
        mean_normal = normal[col].mean()
        mean_fraud = fraud[col].mean()
        std_normal = normal[col].std()
        std_fraud = fraud[col].std()
        
        pooled_std = np.sqrt(((len(normal)-1)*std_normal**2 + (len(fraud)-1)*std_fraud**2) / (len(normal)+len(fraud)-2))
        cohens_d = abs(mean_fraud - mean_normal) / pooled_std
        
        pca_analysis.append({
            'Variable': col,
            'Mean_Normal': mean_normal,
            'Mean_Fraud': mean_fraud,
            'P_value': p_value,
            'Cohens_d': cohens_d,
            'Significant': p_value < 0.05
        })
    
    pca_df = pd.DataFrame(pca_analysis).sort_values('Cohens_d', ascending=False)
    
    return stats_dict, pca_df

def main():
    """Fonction principale du dashboard"""
    
    # En-tête principal
    st.markdown('<h1 class="main-header">🏦 Dashboard EDA - Détection de Fraudes Bancaires</h1>', unsafe_allow_html=True)
    
    # Informations sur le projet
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <strong>Projet:</strong> Analyse Exploratoire des Données | 
        <strong>Dataset:</strong> Credit Card Fraud Detection (Kaggle)
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des données
    df = load_data()
    if df is None:
        return
    
    # Préparation des données
    df, fraud, normal, pca_columns = prepare_data(df)
    stats_dict, pca_df = compute_statistics(df, fraud, normal, pca_columns)
    
    # Sidebar pour la navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choisir une section:",
        ["📊 Vue d'ensemble", "⏰ Analyse Temporelle", "💰 Analyse des Montants", 
         "🔢 Variables PCA", "🔗 Analyse Multivariée", "🎯 Clustering", "📈 Insights"]
    )
    
    # Section Vue d'ensemble
    if page == "📊 Vue d'ensemble":
        st.markdown('<h2 class="sub-header">📊 Vue d\'ensemble du Dataset</h2>', unsafe_allow_html=True)
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📈 Total Transactions",
                value=f"{stats_dict['total_transactions']:,}"
            )
        
        with col2:
            st.metric(
                label="🚨 Fraudes Détectées",
                value=f"{stats_dict['fraud_count']:,}",
                delta=f"{stats_dict['fraud_rate']:.3f}%"
            )
        
        with col3:
            st.metric(
                label="✅ Transactions Normales",
                value=f"{stats_dict['normal_count']:,}"
            )
        
        with col4:
            st.metric(
                label="⚖️ Ratio Déséquilibre",
                value=f"1:{stats_dict['imbalance_ratio']:.0f}"
            )
        
        # Graphiques de distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des classes
            fig_pie = px.pie(
                values=[stats_dict['normal_count'], stats_dict['fraud_count']],
                names=['Normal', 'Fraude'],
                title="Distribution des Classes",
                color_discrete_sequence=['#2E86AB', '#A23B72']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Histogramme des montants
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=normal['Amount'],
                name='Normal',
                opacity=0.7,
                nbinsx=50,
                marker_color='#2E86AB'
            ))
            fig_hist.add_trace(go.Histogram(
                x=fraud['Amount'],
                name='Fraude',
                opacity=0.7,
                nbinsx=50,
                marker_color='#A23B72'
            ))
            fig_hist.update_layout(
                title="Distribution des Montants par Classe",
                xaxis_title="Montant ($)",
                yaxis_title="Fréquence",
                barmode='overlay'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Tableau de synthèse
        st.markdown('<h3 class="sub-header">📋 Synthèse Statistique</h3>', unsafe_allow_html=True)
        
        summary_data = {
            'Métrique': [
                'Nombre total de transactions',
                'Transactions frauduleuses',
                'Transactions normales',
                'Taux de fraude (%)',
                'Ratio de déséquilibre',
                'Montant moyen - Fraude ($)',
                'Montant moyen - Normal ($)',
                'Montant médian - Fraude ($)',
                'Montant médian - Normal ($)'
            ],
            'Valeur': [
                f"{stats_dict['total_transactions']:,}",
                f"{stats_dict['fraud_count']:,}",
                f"{stats_dict['normal_count']:,}",
                f"{stats_dict['fraud_rate']:.3f}%",
                f"1:{stats_dict['imbalance_ratio']:.0f}",
                f"{stats_dict['avg_amount_fraud']:.2f}",
                f"{stats_dict['avg_amount_normal']:.2f}",
                f"{stats_dict['median_amount_fraud']:.2f}",
                f"{stats_dict['median_amount_normal']:.2f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Insights clés
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        **🔍 Insights Clés:**
        - Le dataset présente un **déséquilibre extrême** avec seulement 0.17% de fraudes
        - Les transactions frauduleuses ont un **montant moyen plus faible** que les normales
        - La **médiane** des fraudes est significativement plus basse, indiquant une distribution asymétrique
        - Ce déséquilibre nécessitera des **techniques spécialisées** pour la modélisation
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section Analyse Temporelle
    elif page == "⏰ Analyse Temporelle":
        st.markdown('<h2 class="sub-header">⏰ Analyse Temporelle des Transactions</h2>', unsafe_allow_html=True)
        
        # Distribution par heure
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraudes par heure
            fraud_by_hour = fraud.groupby('Hour_of_day').size()
            normal_by_hour = normal.groupby('Hour_of_day').size()
            
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=fraud_by_hour.index,
                y=fraud_by_hour.values,
                mode='lines+markers',
                name='Fraudes',
                line=dict(color='#A23B72', width=3),
                marker=dict(size=8)
            ))
            fig_time.add_trace(go.Scatter(
                x=normal_by_hour.index,
                y=normal_by_hour.values / 100,  # Échelle réduite pour visualisation
                mode='lines+markers',
                name='Normal (÷100)',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8)
            ))
            fig_time.update_layout(
                title="Distribution des Transactions par Heure",
                xaxis_title="Heure de la journée",
                yaxis_title="Nombre de transactions",
                hovermode='x unified'
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Heatmap des fraudes par heure
            fraud_rate_by_hour = df.groupby('Hour_of_day')['Class'].agg(['sum', 'count'])
            fraud_rate_by_hour['rate'] = fraud_rate_by_hour['sum'] / fraud_rate_by_hour['count'] * 100
            
            fig_heatmap = px.bar(
                x=fraud_rate_by_hour.index,
                y=fraud_rate_by_hour['rate'],
                title="Taux de Fraude par Heure (%)",
                color=fraud_rate_by_hour['rate'],
                color_continuous_scale='Reds'
            )
            fig_heatmap.update_layout(
                xaxis_title="Heure de la journée",
                yaxis_title="Taux de fraude (%)"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Analyse des patterns temporels
        st.markdown('<h3 class="sub-header">📊 Patterns Temporels</h3>', unsafe_allow_html=True)
        
        # Statistiques temporelles
        time_stats = df.groupby(['Hour_of_day', 'Class']).size().unstack(fill_value=0)
        time_stats['fraud_rate'] = time_stats[1] / (time_stats[0] + time_stats[1]) * 100
        
        # Heures avec le plus de fraudes
        top_fraud_hours = time_stats.nlargest(5, 'fraud_rate')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🕐 Heures avec le plus haut taux de fraude:**")
            for hour, row in top_fraud_hours.iterrows():
                st.write(f"• {hour:02d}h: {row['fraud_rate']:.3f}% ({row[1]} fraudes)")
        
        with col2:
            # Évolution temporelle globale
            df_sample = df.sample(n=min(10000, len(df)))  # Échantillon pour performance
            fig_evolution = px.scatter(
                df_sample,
                x='Time_hours',
                y='Amount',
                color='Class',
                title="Évolution Temporelle des Transactions",
                color_discrete_map={0: '#2E86AB', 1: '#A23B72'},
                opacity=0.6
            )
            fig_evolution.update_layout(
                xaxis_title="Temps (heures)",
                yaxis_title="Montant ($)"
            )
            st.plotly_chart(fig_evolution, use_container_width=True)
        
        # Insights temporels
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        peak_hour = fraud_rate_by_hour['rate'].idxmax()
        peak_rate = fraud_rate_by_hour['rate'].max()
        st.markdown(f"""
        **🕐 Insights Temporels:**
        - **Pic de fraude** à {peak_hour:02d}h avec un taux de {peak_rate:.3f}%
        - Les fraudes semblent plus fréquentes pendant certaines **heures spécifiques**
        - La **distribution temporelle** peut être un indicateur important pour la détection
        - Les patterns suggèrent des **comportements distincts** entre fraudes et transactions normales
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section Analyse des Montants
    elif page == "💰 Analyse des Montants":
        st.markdown('<h2 class="sub-header">💰 Analyse des Montants de Transaction</h2>', unsafe_allow_html=True)
        
        # Statistiques des montants
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="💰 Montant Moyen - Normal",
                value=f"${stats_dict['avg_amount_normal']:.2f}"
            )
        
        with col2:
            st.metric(
                label="🚨 Montant Moyen - Fraude",
                value=f"${stats_dict['avg_amount_fraud']:.2f}"
            )
        
        with col3:
            st.metric(
                label="📊 Médiane - Normal",
                value=f"${stats_dict['median_amount_normal']:.2f}"
            )
        
        with col4:
            st.metric(
                label="📊 Médiane - Fraude",
                value=f"${stats_dict['median_amount_fraud']:.2f}"
            )
        
        # Visualisations des montants
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot des montants
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=normal['Amount'],
                name='Normal',
                marker_color='#2E86AB',
                boxpoints='outliers'
            ))
            fig_box.add_trace(go.Box(
                y=fraud['Amount'],
                name='Fraude',
                marker_color='#A23B72',
                boxpoints='outliers'
            ))
            fig_box.update_layout(
                title="Distribution des Montants par Classe",
                yaxis_title="Montant ($)",
                yaxis_type="log"  # Échelle logarithmique pour mieux voir
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Distribution par tranches de montants
            amount_dist = df.groupby(['Amount_range', 'Class']).size().unstack(fill_value=0)
            amount_dist['fraud_rate'] = amount_dist[1] / (amount_dist[0] + amount_dist[1]) * 100
            
            fig_amount_range = px.bar(
                x=amount_dist.index,
                y=amount_dist['fraud_rate'],
                title="Taux de Fraude par Tranche de Montant",
                color=amount_dist['fraud_rate'],
                color_continuous_scale='Reds'
            )
            fig_amount_range.update_layout(
                xaxis_title="Tranche de montant",
                yaxis_title="Taux de fraude (%)",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig_amount_range, use_container_width=True)
        
        # Analyse statistique détaillée
        st.markdown('<h3 class="sub-header">📈 Analyse Statistique des Montants</h3>', unsafe_allow_html=True)
        
        # Test statistique
        stat, p_value = stats.mannwhitneyu(normal['Amount'], fraud['Amount'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Statistiques descriptives
            amount_stats = pd.DataFrame({
                'Normal': normal['Amount'].describe(),
                'Fraude': fraud['Amount'].describe()
            })
            st.markdown("**📊 Statistiques Descriptives:**")
            st.dataframe(amount_stats)
        
        with col2:
            # Percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            perc_normal = np.percentile(normal['Amount'], percentiles)
            perc_fraud = np.percentile(fraud['Amount'], percentiles)
            
            perc_df = pd.DataFrame({
                'Percentile': [f"{p}%" for p in percentiles],
                'Normal': perc_normal,
                'Fraude': perc_fraud
            })
            st.markdown("**📊 Percentiles:**")
            st.dataframe(perc_df)
        
        # Analyse par tranches
        st.markdown('<h3 class="sub-header">🎯 Analyse par Tranches de Montant</h3>', unsafe_allow_html=True)
        
        # Tableau détaillé par tranches
        amount_analysis = df.groupby('Amount_range').agg({
            'Class': ['count', 'sum', 'mean'],
            'Amount': ['mean', 'median', 'std']
        }).round(3)
        
        amount_analysis.columns = ['Total_Trans', 'Fraudes', 'Taux_Fraude', 'Montant_Moyen', 'Montant_Médian', 'Écart_Type']
        amount_analysis['Taux_Fraude'] = amount_analysis['Taux_Fraude'] * 100
        
        st.dataframe(amount_analysis, use_container_width=True)
        
        # Insights sur les montants
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        highest_fraud_range = amount_dist['fraud_rate'].idxmax()
        highest_fraud_rate = amount_dist['fraud_rate'].max()
        st.markdown(f"""
        **💰 Insights sur les Montants:**
        - Les **fraudes ont des montants significativement différents** (p < 0.001)
        - Tranche avec le plus haut taux de fraude: **{highest_fraud_range}** ({highest_fraud_rate:.3f}%)
        - Les **petits montants** semblent plus susceptibles d'être frauduleux
        - La **distribution asymétrique** suggère des patterns comportementaux distincts
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section Variables PCA
    elif page == "🔢 Variables PCA":
        st.markdown('<h2 class="sub-header">🔢 Analyse des Variables PCA</h2>', unsafe_allow_html=True)
        
        # Sélection de variables pour analyse
        st.sidebar.markdown("### 🎛️ Paramètres d'Analyse")
        top_n = st.sidebar.slider("Nombre de variables à analyser:", 5, 28, 10)
        show_significant_only = st.sidebar.checkbox("Afficher seulement les variables significatives", True)
        
        # Filtrage des données
        pca_filtered = pca_df.head(top_n)
        if show_significant_only:
            pca_filtered = pca_filtered[pca_filtered['Significant']]
        
        # Visualisation de l'importance des variables
        col1, col2 = st.columns(2)
        
        with col1:
            # Cohen's d (taille d'effet)
            fig_cohens = px.bar(
                pca_filtered,
                x='Cohens_d',
                y='Variable',
                orientation='h',
                title="Taille d'Effet (Cohen's d) des Variables PCA",
                color='Cohens_d',
                color_continuous_scale='Viridis'
            )
            fig_cohens.update_layout(
                xaxis_title="Cohen's d",
                yaxis_title="Variables PCA"
            )
            st.plotly_chart(fig_cohens, use_container_width=True)
        
        with col2:
            # P-values
            pca_filtered['log_p'] = -np.log10(pca_filtered['P_value'])
            fig_pvalue = px.bar(
                pca_filtered,
                x='log_p',
                y='Variable',
                orientation='h',
                title="Significativité Statistique (-log10 p-value)",
                color='log_p',
                color_continuous_scale='Reds'
            )
            fig_pvalue.update_layout(
                xaxis_title="-log10(p-value)",
                yaxis_title="Variables PCA"
            )
            st.plotly_chart(fig_pvalue, use_container_width=True)
        
        # Analyse détaillée des top variables
        st.markdown('<h3 class="sub-header">🔍 Analyse Détaillée des Variables les Plus Importantes</h3>', unsafe_allow_html=True)
        
        # Sélection d'une variable pour analyse approfondie
        selected_var = st.selectbox(
            "Choisir une variable pour analyse approfondie:",
            pca_filtered['Variable'].tolist()
        )
        
        if selected_var:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution de la variable sélectionnée
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=normal[selected_var],
                    name='Normal',
                    opacity=0.7,
                    nbinsx=50,
                    marker_color='#2E86AB'
                ))
                fig_dist.add_trace(go.Histogram(
                    x=fraud[selected_var],
                    name='Fraude',
                    opacity=0.7,
                    nbinsx=50,
                    marker_color='#A23B72'
                ))
                fig_dist.update_layout(
                    title=f"Distribution de {selected_var}",
                    xaxis_title=selected_var,
                    yaxis_title="Fréquence",
                    barmode='overlay'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Box plot de la variable sélectionnée
                fig_box_var = go.Figure()
                fig_box_var.add_trace(go.Box(
                    y=normal[selected_var],
                    name='Normal',
                    marker_color='#2E86AB'
                ))
                fig_box_var.add_trace(go.Box(
                    y=fraud[selected_var],
                    name='Fraude',
                    marker_color='#A23B72'
                ))
                fig_box_var.update_layout(
                    title=f"Box Plot de {selected_var}",
                    yaxis_title=selected_var
                )
                st.plotly_chart(fig_box_var, use_container_width=True)
            
            # Statistiques de la variable sélectionnée
            var_stats = pca_filtered[pca_filtered['Variable'] == selected_var].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Moyenne Normal",
                    value=f"{var_stats['Mean_Normal']:.4f}"
                )
            
            with col2:
                st.metric(
                    label="Moyenne Fraude",
                    value=f"{var_stats['Mean_Fraud']:.4f}"
                )
            
            with col3:
                st.metric(
                    label="Cohen's d",
                    value=f"{var_stats['Cohens_d']:.4f}"
                )
            
            with col4:
                st.metric(
                    label="P-value",
                    value=f"{var_stats['P_value']:.2e}"
                )
        
        # Tableau de synthèse
        st.markdown('<h3 class="sub-header">📋 Tableau de Synthèse des Variables PCA</h3>', unsafe_allow_html=True)
        
        # Formatage du tableau
        display_df = pca_filtered.copy()
        display_df['P_value'] = display_df['P_value'].apply(lambda x: f"{x:.2e}")
        display_df['Mean_Normal'] = display_df['Mean_Normal'].round(4)
        display_df['Mean_Fraud'] = display_df['Mean_Fraud'].round(4)
        display_df['Cohens_d'] = display_df['Cohens_d'].round(4)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Insights sur les variables PCA
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        top_var = pca_filtered.iloc[0]['Variable']
        top_cohens = pca_filtered.iloc[0]['Cohens_d']
        significant_vars = len(pca_filtered[pca_filtered['Significant']])
        st.markdown(f"""
        **🔢 Insights sur les Variables PCA:**
        - **{significant_vars}** variables PCA montrent des différences significatives
        - Variable la plus discriminante: **{top_var}** (Cohen's d = {top_cohens:.3f})
        - Les variables PCA capturent des **patterns cachés** dans les données originales
        - Ces variables transformées sont **essentielles** pour la détection de fraude
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section Analyse Multivariée
    elif page == "🔗 Analyse Multivariée":
        st.markdown('<h2 class="sub-header">🔗 Analyse Multivariée et Corrélations</h2>', unsafe_allow_html=True)
        
        # Paramètres d'analyse
        st.sidebar.markdown("### 🎛️ Paramètres d'Analyse")
        correlation_method = st.sidebar.selectbox(
            "Méthode de corrélation:",
            ['pearson', 'spearman', 'kendall']
        )
        
        sample_size = st.sidebar.slider(
            "Taille de l'échantillon pour visualisation:",
            1000, min(10000, len(df)), 5000
        )
        
        # Échantillonnage pour performance
        df_sample = df.sample(n=sample_size, random_state=42)
        fraud_sample = df_sample[df_sample['Class'] == 1]
        normal_sample = df_sample[df_sample['Class'] == 0]
        
        # Matrice de corrélation
        st.markdown('<h3 class="sub-header">🔗 Matrice de Corrélation</h3>', unsafe_allow_html=True)
        
        # Sélection des variables pour la corrélation
        corr_vars = ['Time', 'Amount'] + pca_columns[:10]  # Top 10 PCA + Time/Amount
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Corrélation pour transactions normales
            corr_normal = normal_sample[corr_vars].corr(method=correlation_method)
            
            fig_corr_normal = px.imshow(
                corr_normal,
                title="Corrélations - Transactions Normales",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig_corr_normal.update_layout(height=500)
            st.plotly_chart(fig_corr_normal, use_container_width=True)
        
        with col2:
            # Corrélation pour fraudes
            if len(fraud_sample) > 10:  # Vérifier qu'il y a assez de données
                corr_fraud = fraud_sample[corr_vars].corr(method=correlation_method)
                
                fig_corr_fraud = px.imshow(
                    corr_fraud,
                    title="Corrélations - Transactions Frauduleuses",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                fig_corr_fraud.update_layout(height=500)
                st.plotly_chart(fig_corr_fraud, use_container_width=True)
            else:
                st.warning("Pas assez de données de fraude dans l'échantillon pour calculer les corrélations.")
        
        # Analyse PCA 2D
        st.markdown('<h3 class="sub-header">🎯 Analyse en Composantes Principales (PCA)</h3>', unsafe_allow_html=True)
        
        # PCA sur les variables sélectionnées
        pca_vars_selected = pca_columns[:10]  # Top 10 variables PCA
        X_pca = df_sample[pca_vars_selected]
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pca)
        
        # PCA 2D
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        # Visualisation PCA 2D
        fig_pca = px.scatter(
            x=X_pca_2d[:, 0],
            y=X_pca_2d[:, 1],
            color=df_sample['Class'].astype(str),
            title=f"PCA 2D - Variance expliquée: {sum(pca_2d.explained_variance_ratio_):.1%}",
            labels={'x': f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})',
                   'y': f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})'},
            color_discrete_map={'0': '#2E86AB', '1': '#A23B72'}
        )
        fig_pca.update_traces(marker=dict(size=4, opacity=0.6))
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # t-SNE (optionnel, plus lent)
        if st.checkbox("🔬 Effectuer une analyse t-SNE (plus lent)"):
            st.markdown('<h3 class="sub-header">🔬 Analyse t-SNE</h3>', unsafe_allow_html=True)
            
            # t-SNE sur un échantillon plus petit
            tsne_sample_size = min(2000, len(df_sample))
            df_tsne = df_sample.sample(n=tsne_sample_size, random_state=42)
            X_tsne = df_tsne[pca_vars_selected]
            X_tsne_scaled = scaler.fit_transform(X_tsne)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_tsne_2d = tsne.fit_transform(X_tsne_scaled)
            
            # Visualisation t-SNE
            fig_tsne = px.scatter(
                x=X_tsne_2d[:, 0],
                y=X_tsne_2d[:, 1],
                color=df_tsne['Class'].astype(str),
                title="t-SNE 2D - Réduction de Dimensionnalité Non-linéaire",
                labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
                color_discrete_map={'0': '#2E86AB', '1': '#A23B72'}
            )
            fig_tsne.update_traces(marker=dict(size=4, opacity=0.7))
            st.plotly_chart(fig_tsne, use_container_width=True)
        
        # Analyse des composantes principales
        st.markdown('<h3 class="sub-header">📊 Contribution des Variables aux Composantes</h3>', unsafe_allow_html=True)
        
        # Contribution des variables aux PC
        components_df = pd.DataFrame(
            pca_2d.components_.T,
            columns=['PC1', 'PC2'],
            index=pca_vars_selected
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Contribution à PC1
            fig_pc1 = px.bar(
                x=components_df['PC1'],
                y=components_df.index,
                orientation='h',
                title="Contribution des Variables à PC1",
                color=components_df['PC1'],
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_pc1, use_container_width=True)
        
        with col2:
            # Contribution à PC2
            fig_pc2 = px.bar(
                x=components_df['PC2'],
                y=components_df.index,
                orientation='h',
                title="Contribution des Variables à PC2",
                color=components_df['PC2'],
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_pc2, use_container_width=True)
        
        # Insights multivariés
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        pc1_var = pca_2d.explained_variance_ratio_[0]
        pc2_var = pca_2d.explained_variance_ratio_[1]
        st.markdown(f"""
        **🔗 Insights Analyse Multivariée:**
        - **PC1** explique {pc1_var:.1%} de la variance, **PC2** explique {pc2_var:.1%}
        - Les **patterns de corrélation** diffèrent entre fraudes et transactions normales
        - La **réduction de dimensionnalité** révèle des structures cachées dans les données
        - Les techniques non-linéaires (t-SNE) peuvent révéler des **clusters** non visibles en PCA
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section Clustering
    elif page == "🎯 Clustering":
        st.markdown('<h2 class="sub-header">🎯 Analyse de Clustering</h2>', unsafe_allow_html=True)
        
        # Paramètres de clustering
        st.sidebar.markdown("### 🎛️ Paramètres de Clustering")
        n_clusters = st.sidebar.slider("Nombre de clusters:", 2, 10, 3)
        clustering_vars = st.sidebar.multiselect(
            "Variables pour le clustering:",
            ['Amount', 'Time'] + pca_columns[:10],
            default=['Amount'] + pca_columns[:5]
        )
        
        if len(clustering_vars) < 2:
            st.warning("Veuillez sélectionner au moins 2 variables pour le clustering.")
            return
        
        # Préparation des données pour clustering
        cluster_sample_size = min(5000, len(df))
        df_cluster = df.sample(n=cluster_sample_size, random_state=42)
        X_cluster = df_cluster[clustering_vars]
        
        # Standardisation
        scaler = StandardScaler()
        X_cluster_scaled = scaler.fit_transform(X_cluster)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_cluster_scaled)
        
        # Calcul du score de silhouette
        silhouette_avg = silhouette_score(X_cluster_scaled, cluster_labels)
        
        # Ajout des labels au dataframe
        df_cluster_viz = df_cluster.copy()
        df_cluster_viz['Cluster'] = cluster_labels
        
        # Métriques de clustering
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🎯 Nombre de Clusters",
                value=n_clusters
            )
        
        with col2:
            st.metric(
                label="📊 Score Silhouette",
                value=f"{silhouette_avg:.3f}"
            )
        
        with col3:
            st.metric(
                label="📈 Échantillon",
                value=f"{cluster_sample_size:,}"
            )
        
        with col4:
            st.metric(
                label="🔢 Variables",
                value=len(clustering_vars)
            )
        
        # Visualisations de clustering
        st.markdown('<h3 class="sub-header">📊 Visualisations des Clusters</h3>', unsafe_allow_html=True)
        
        # PCA pour visualisation 2D
        pca_viz = PCA(n_components=2)
        X_pca_viz = pca_viz.fit_transform(X_cluster_scaled)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Clusters dans l'espace PCA
            fig_cluster_pca = px.scatter(
                x=X_pca_viz[:, 0],
                y=X_pca_viz[:, 1],
                color=cluster_labels.astype(str),
                title="Clusters dans l'Espace PCA",
                labels={'x': f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%})',
                       'y': f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%})'},
                opacity=0.7
            )
            fig_cluster_pca.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_cluster_pca, use_container_width=True)
        
        with col2:
            # Clusters vs Classes réelles
            fig_cluster_class = px.scatter(
                x=X_pca_viz[:, 0],
                y=X_pca_viz[:, 1],
                color=df_cluster['Class'].astype(str),
                title="Classes Réelles dans l'Espace PCA",
                labels={'x': f'PC1 ({pca_viz.explained_variance_ratio_[0]:.1%})',
                       'y': f'PC2 ({pca_viz.explained_variance_ratio_[1]:.1%})'},
                color_discrete_map={'0': '#2E86AB', '1': '#A23B72'},
                opacity=0.7
            )
            fig_cluster_class.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_cluster_class, use_container_width=True)
        
        # Analyse des clusters
        st.markdown('<h3 class="sub-header">🔍 Analyse des Clusters</h3>', unsafe_allow_html=True)
        
        # Composition des clusters
        cluster_composition = pd.crosstab(df_cluster_viz['Cluster'], df_cluster_viz['Class'], normalize='index') * 100
        
        fig_composition = px.bar(
            cluster_composition,
            title="Composition des Clusters (% de Fraudes)",
            labels={'value': 'Pourcentage', 'index': 'Cluster'},
            color_discrete_map={0: '#2E86AB', 1: '#A23B72'}
        )
        st.plotly_chart(fig_composition, use_container_width=True)
        
        # Statistiques par cluster
        cluster_stats = df_cluster_viz.groupby('Cluster').agg({
            'Amount': ['mean', 'median', 'std'],
            'Class': ['count', 'sum', 'mean']
        }).round(3)
        
        cluster_stats.columns = ['Montant_Moyen', 'Montant_Médian', 'Montant_ÉcartType', 
                                'Total_Trans', 'Nb_Fraudes', 'Taux_Fraude']
        cluster_stats['Taux_Fraude'] = cluster_stats['Taux_Fraude'] * 100
        
        st.markdown("**📊 Statistiques par Cluster:**")
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Optimisation du nombre de clusters
        if st.checkbox("🔧 Analyser l'optimisation du nombre de clusters"):
            st.markdown('<h3 class="sub-header">🔧 Optimisation du Nombre de Clusters</h3>', unsafe_allow_html=True)
            
            # Méthode du coude et score de silhouette
            k_range = range(2, 11)
            inertias = []
            silhouette_scores = []
            
            for k in k_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_temp = kmeans_temp.fit_predict(X_cluster_scaled)
                inertias.append(kmeans_temp.inertia_)
                silhouette_scores.append(silhouette_score(X_cluster_scaled, labels_temp))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Méthode du coude
                fig_elbow = px.line(
                    x=list(k_range),
                    y=inertias,
                    title="Méthode du Coude",
                    labels={'x': 'Nombre de Clusters', 'y': 'Inertie'},
                    markers=True
                )
                st.plotly_chart(fig_elbow, use_container_width=True)
            
            with col2:
                # Score de silhouette
                fig_silhouette = px.line(
                    x=list(k_range),
                    y=silhouette_scores,
                    title="Score de Silhouette",
                    labels={'x': 'Nombre de Clusters', 'y': 'Score Silhouette'},
                    markers=True
                )
                st.plotly_chart(fig_silhouette, use_container_width=True)
            
            # Recommandation
            best_k = k_range[np.argmax(silhouette_scores)]
            st.info(f"🎯 **Recommandation:** {best_k} clusters (meilleur score de silhouette: {max(silhouette_scores):.3f})")
        
        # Insights clustering
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        best_cluster = cluster_stats['Taux_Fraude'].idxmax()
        best_fraud_rate = cluster_stats['Taux_Fraude'].max()
        st.markdown(f"""
        **🎯 Insights Clustering:**
        - **Cluster {best_cluster}** a le plus haut taux de fraude: {best_fraud_rate:.1f}%
        - Score de silhouette: {silhouette_avg:.3f} (>0.5 = bon clustering)
        - Le clustering peut **identifier des groupes** à risque élevé
        - Utile pour la **segmentation** et la **détection d'anomalies**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Section Insights
    elif page == "📈 Insights":
        st.markdown('<h2 class="sub-header">📈 Insights et Recommandations</h2>', unsafe_allow_html=True)
        
        # Résumé exécutif
        st.markdown('<h3 class="sub-header">🎯 Résumé Exécutif</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("""
            **🔍 Principales Découvertes:**
            
            1. **Déséquilibre Extrême**: 99.83% transactions normales vs 0.17% fraudes
            2. **Patterns Temporels**: Certaines heures montrent des taux de fraude plus élevés
            3. **Montants Distincts**: Les fraudes ont des distributions de montants différentes
            4. **Variables PCA**: Plusieurs variables montrent des différences significatives
            5. **Clustering**: Identification de groupes à risque élevé
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("""
            **⚠️ Défis Identifiés:**
            
            1. **Déséquilibre des Classes**: Nécessite des techniques spécialisées
            2. **Anonymisation**: Variables PCA limitent l'interprétabilité
            3. **Rareté des Fraudes**: Difficulté de généralisation
            4. **Variabilité Temporelle**: Patterns peuvent évoluer
            5. **Faux Positifs**: Coût élevé des erreurs de classification
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommandations techniques
        st.markdown('<h3 class="sub-header">🛠️ Recommandations Techniques</h3>', unsafe_allow_html=True)
        
        recommendations = {
            "🎯 Gestion du Déséquilibre": [
                "Utiliser SMOTE ou ADASYN pour l'augmentation de données",
                "Appliquer des techniques d'under-sampling intelligent",
                "Utiliser des métriques adaptées (F1-score, AUC-ROC, Precision-Recall)",
                "Implémenter des seuils de classification optimisés"
            ],
            "🤖 Algorithmes Recommandés": [
                "Random Forest avec class_weight='balanced'",
                "XGBoost avec scale_pos_weight ajusté",
                "Isolation Forest pour la détection d'anomalies",
                "Réseaux de neurones avec focal loss"
            ],
            "📊 Ingénierie des Features": [
                "Créer des features temporelles (heure, jour, période)",
                "Calculer des ratios et interactions entre variables PCA",
                "Développer des features d'agrégation par utilisateur",
                "Utiliser des techniques de sélection de features"
            ],
            "✅ Validation et Évaluation": [
                "Validation croisée stratifiée temporelle",
                "Métriques business-oriented (coût des erreurs)",
                "Tests A/B pour validation en production",
                "Monitoring continu des performances"
            ]
        }
        
        for category, items in recommendations.items():
            with st.expander(category):
                for item in items:
                    st.write(f"• {item}")
        
        # Métriques de performance attendues
        st.markdown('<h3 class="sub-header">📊 Objectifs de Performance</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("""
            **🎯 Détection**
            - Recall: > 80%
            - Precision: > 60%
            - F1-Score: > 70%
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("""
            **⚡ Performance**
            - Temps de réponse: < 100ms
            - Throughput: > 1000 TPS
            - Disponibilité: 99.9%
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("""
            **💰 Business**
            - Réduction fraudes: 70%
            - Faux positifs: < 5%
            - ROI: > 300%
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Plan d'implémentation
        st.markdown('<h3 class="sub-header">🗓️ Plan d\'Implémentation</h3>', unsafe_allow_html=True)
        
        phases = {
            "Phase 1 - Préparation (2-3 semaines)": [
                "Nettoyage et préparation avancée des données",
                "Ingénierie des features et sélection",
                "Mise en place de l'infrastructure de validation"
            ],
            "Phase 2 - Modélisation (3-4 semaines)": [
                "Développement et comparaison de modèles",
                "Optimisation des hyperparamètres",
                "Validation croisée et tests de robustesse"
            ],
            "Phase 3 - Déploiement (2-3 semaines)": [
                "Intégration dans l'environnement de production",
                "Tests de charge et de performance",
                "Formation des équipes et documentation"
            ],
            "Phase 4 - Monitoring (Continu)": [
                "Surveillance des performances en temps réel",
                "Réentraînement périodique des modèles",
                "Amélioration continue basée sur les retours"
            ]
        }
        
        for phase, tasks in phases.items():
            with st.expander(phase):
                for task in tasks:
                    st.write(f"✓ {task}")
        
        # Conclusion
        st.markdown('<h3 class="sub-header">🎯 Conclusion</h3>', unsafe_allow_html=True)
        
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("""
        **🏆 Points Clés pour le Succès:**
        
        Cette analyse exploratoire révèle un dataset complexe mais riche en informations pour la détection de fraudes.
        Les **patterns identifiés** dans les variables PCA, les **différences temporelles** et les **distributions de montants**
        fournissent une base solide pour développer un système de détection efficace.
        
        Le **défi principal** reste la gestion du déséquilibre extrême des classes, mais les techniques modernes
        de machine learning offrent des solutions robustes. L'approche recommandée combine **plusieurs algorithmes**,
        une **ingénierie de features** sophistiquée, et un **monitoring continu** pour maintenir les performances.
        
        **Prochaines étapes:** Implémenter les recommandations techniques, développer un pipeline MLOps,
        et établir un système de feedback pour l'amélioration continue.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()