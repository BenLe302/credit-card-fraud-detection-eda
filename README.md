# 🏦 Détection de Fraudes Bancaires - Analyse Exploratoire des Données (EDA)

## 📋 Description du Projet

Ce projet présente une **analyse exploratoire complète** d'un dataset de transactions par cartes de crédit pour la détection de fraudes. Il s'agit d'un projet de Data Science axé sur l'exploration, la visualisation et la compréhension des patterns de fraude dans les données financières.

## 🎯 Objectifs

- **Analyser** les caractéristiques des transactions frauduleuses vs légitimes
- **Identifier** les variables les plus discriminantes pour la détection de fraude
- **Visualiser** les patterns temporels et les distributions des montants
- **Proposer** des recommandations pour la modélisation prédictive
- **Créer** un dashboard interactif pour l'exploration des données

## 📊 Dataset

- **Source** : [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Taille** : 284,807 transactions
- **Période** : 2 jours de transactions (septembre 2013)
- **Variables** : 31 features (Time, Amount, V1-V28, Class)
- **Déséquilibre** : 492 fraudes (0.17%) vs 284,315 transactions légitimes

## 🗂️ Structure du Projet

```
credit-card-fraud-detection-eda/
├── 📁 Credit Card Fraud Detection/     # Dataset original
│   ├── 📄 creditcard.csv
│   └── 📄 Explication.txt
├── 📁 notebooks/                       # Analyses Jupyter
│   ├── 📓 01_exploration_initiale.ipynb
│   ├── 📓 02_analyse_univariee.ipynb
│   ├── 📓 03_analyse_multivariee.ipynb
│   └── 📓 04_insights_recommandations.ipynb
├── 📁 src/                            # Code source
│   └── 🐍 dashboard_streamlit.py
├── 📁 reports/                        # Rapports et visualisations
│   ├── 📄 RAPPORT_ANALYSE_EDA.md
│   └── 📁 figures/
├── 📄 README.md                       # Ce fichier
├── 📄 requirements.txt               # Dépendances Python
├── 📄 PLAN_PROJET_EDA.md             # Plan détaillé du projet
├── 🐍 run_dashboard.py               # Script de lancement du dashboard
└── 📄 .gitignore                     # Fichiers à ignorer
```

## 🚀 Installation et Utilisation

### Prérequis
- Python 3.8+
- pip ou conda

### Installation

1. **Cloner le repository**
```bash
git clone https://github.com/BenLe302/credit-card-fraud-detection-eda.git
cd credit-card-fraud-detection-eda
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Télécharger le dataset**
- Télécharger `creditcard.csv` depuis [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Placer le fichier dans le dossier `Credit Card Fraud Detection/`

### Lancement du Dashboard

```bash
python run_dashboard.py
```

Ou directement :
```bash
streamlit run src/dashboard_streamlit.py
```

## 📈 Analyses Réalisées

### 1. Exploration Initiale
- Aperçu général du dataset
- Analyse des valeurs manquantes
- Distribution des classes (fraude vs légitime)
- Statistiques descriptives

### 2. Analyse Univariée
- **Variable Time** : Patterns temporels des fraudes
- **Variable Amount** : Distribution des montants par classe
- **Variables PCA (V1-V28)** : Analyse de chaque composante
- Tests statistiques (Mann-Whitney U, Cohen's d)
- Détection d'outliers

### 3. Analyse Multivariée
- Matrice de corrélation
- Analyse en Composantes Principales (PCA)
- Réduction dimensionnelle (t-SNE)
- Clustering (K-Means, Hierarchique)
- Analyse des interactions entre variables

### 4. Insights et Recommandations
- Synthèse des découvertes principales
- Recommandations pour la modélisation
- Pipeline de preprocessing suggéré
- Stratégies de gestion du déséquilibre

## 🔍 Principales Découvertes

### 📊 Caractéristiques des Fraudes
- **Montants** : Les fraudes concernent principalement des petits montants (médiane : 9.25€)
- **Temporalité** : Pas de pattern temporel clair sur 2 jours
- **Variables PCA** : 26/28 variables montrent des différences significatives
- **Variable la plus discriminante** : V14 (Cohen's d = 1.54)

### 🎯 Variables Clés
1. **V14** - Très forte discrimination (d = 1.54)
2. **V4** - Forte discrimination (d = 1.16)
3. **V11** - Forte discrimination (d = 1.01)
4. **V12** - Discrimination modérée (d = 0.98)
5. **Amount** - Discrimination faible mais significative

### 📈 Insights Techniques
- **Outliers** : 1.48% des transactions (4,205 outliers)
- **Corrélations** : Faibles entre variables PCA (par design)
- **Clustering** : 3-4 clusters optimaux identifiés
- **Séparabilité** : Bonne séparation possible avec les bonnes variables

## 🛠️ Technologies Utilisées

- **Python 3.8+**
- **Pandas** - Manipulation des données
- **NumPy** - Calculs numériques
- **Matplotlib/Seaborn** - Visualisations
- **Plotly** - Graphiques interactifs
- **Scikit-learn** - Machine Learning
- **Streamlit** - Dashboard interactif
- **Jupyter** - Notebooks d'analyse

## 📋 Recommandations pour la Modélisation

### Algorithmes Recommandés
1. **Random Forest** - Robuste aux outliers, gère le déséquilibre
2. **XGBoost** - Excellent pour les données tabulaires
3. **Isolation Forest** - Spécialisé dans la détection d'anomalies
4. **Neural Networks** - Pour capturer les interactions complexes

### Stratégies de Preprocessing
1. **Gestion du déséquilibre** : SMOTE, sous-échantillonnage, pondération
2. **Feature Engineering** : Ratios, interactions, agrégations temporelles
3. **Normalisation** : StandardScaler pour Amount et Time
4. **Sélection de features** : Focus sur V14, V4, V11, V12

### Métriques d'Évaluation
- **Precision/Recall** - Plus importantes que l'accuracy
- **F1-Score** - Équilibre entre precision et recall
- **AUC-ROC** - Performance globale du modèle
- **AUC-PR** - Particulièrement adapté aux données déséquilibrées

## 📊 Dashboard Interactif

Le dashboard Streamlit offre :
- **Vue d'ensemble** du dataset
- **Analyse temporelle** interactive
- **Distribution des montants** par classe
- **Exploration des variables PCA**
- **Visualisations multivariées**
- **Résultats de clustering**
- **Insights et recommandations**

## 📄 Rapports

- **[Rapport Complet](reports/RAPPORT_ANALYSE_EDA.md)** - Analyse détaillée (25 pages)
- **[Plan de Projet](PLAN_PROJET_EDA.md)** - Méthodologie et planning

## 🤝 Contribution

Ce projet est ouvert aux contributions ! N'hésitez pas à :
- Signaler des bugs
- Proposer des améliorations
- Ajouter de nouvelles analyses
- Améliorer les visualisations

## 📞 Contact

Pour toute question ou collaboration :
- **GitHub** : [BenLe302](https://github.com/BenLe302)
- **LinkedIn** : [Votre Profil LinkedIn]

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !**

*Projet réalisé dans le cadre d'un portfolio Data Science - 2024*