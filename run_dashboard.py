#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de lancement du Dashboard Streamlit

Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
Institution: UQTR - Université du Québec à Trois-Rivières

Ce script lance le dashboard interactif pour l'analyse exploratoire
des données de fraude bancaire.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Vérifie que toutes les dépendances sont installées"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        'plotly', 'scikit-learn', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Packages manquants: {', '.join(missing_packages)}")
        print("📦 Installation des packages manquants...")
        
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("✅ Tous les packages sont maintenant installés!")
    else:
        print("✅ Toutes les dépendances sont installées!")

def check_data_file():
    """Vérifie que le fichier de données existe"""
    data_paths = [
        "Credit Card Fraud Detection/creditcard.csv",
        "../Credit Card Fraud Detection/creditcard.csv",
        "creditcard.csv"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"✅ Fichier de données trouvé: {path}")
            return True
    
    print("❌ Fichier de données non trouvé!")
    print("📁 Veuillez vous assurer que 'creditcard.csv' est dans le bon répertoire.")
    print("📍 Chemins recherchés:")
    for path in data_paths:
        print(f"   - {os.path.abspath(path)}")
    
    return False

def launch_dashboard():
    """Lance le dashboard Streamlit"""
    dashboard_path = Path("src/dashboard_streamlit.py")
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard non trouvé: {dashboard_path}")
        return False
    
    print("🚀 Lancement du dashboard Streamlit...")
    print("🌐 Le dashboard s'ouvrira automatiquement dans votre navigateur")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Pour arrêter: Ctrl+C dans ce terminal")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard arrêté par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")
        return False
    
    return True

def main():
    """Fonction principale"""
    print("🏦 Dashboard EDA - Détection de Fraudes Bancaires")
    print("👨‍💼 Auteur: Dady Akrou Cyrille (UQTR)")
    print("=" * 55)
    print()
    
    # Vérification des prérequis
    print("🔍 Vérification des prérequis...")
    check_requirements()
    
    if not check_data_file():
        print("\n❌ Impossible de lancer le dashboard sans le fichier de données.")
        return
    
    print("\n" + "=" * 55)
    
    # Lancement du dashboard
    launch_dashboard()
    
    print("\n👋 Merci d'avoir utilisé le dashboard!")

if __name__ == "__main__":
    main()