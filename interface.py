# Import des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import joblib

# Chargement des objets préentraînés : scaler, colonnes du modèle et modèle SVM
scaler = joblib.load("scaler.joblib")
colonnes_modele = joblib.load("colonnes_modele.joblib")
modele = joblib.load("modele_svm.joblib")

# Vérifie que tous les champs d'une liste sont remplis (pas de champs vides)
def verifie_saisi(List_saisi):
    n = 0
    for i in range(len(List_saisi)):
        if List_saisi[i] == "":
            n += 1
    return n == 0

# Prépare les données pour la prédiction : transformation, encodage et normalisation
def preparer_donnees(List_saisi, Colonne_modele, Scaler):
    colonnes_initiales = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
                          'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                          'GroupSize', 'Deck', 'Cabin_Num', 'Side', 'TotalSpend']

    # Création d'un DataFrame avec les données utilisateur
    data = pd.DataFrame([List_saisi], columns=colonnes_initiales)

    # Encodage one-hot des colonnes catégorielles
    categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=False)

    # Conversion des colonnes booléennes en entiers
    bool_columns = data_encoded.select_dtypes(include='bool').columns
    data_encoded[bool_columns] = data_encoded[bool_columns].astype(int)

    # Ajoute les colonnes manquantes (par rapport à l'entraînement du modèle)
    for col in Colonne_modele:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    # Réorganise les colonnes dans le bon ordre
    data_encoded = data_encoded[Colonne_modele]

    # Normalisation des variables numériques
    numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall',
                    'Spa', 'VRDeck', 'Cabin_Num', 'GroupSize', 'TotalSpend']
    data_encoded[numeric_cols] = Scaler.transform(data_encoded[numeric_cols])

    return data_encoded

# Décompose la cabine au format "pont/numéro/côté" en 3 variables séparées
def traiter_cabin(cabin):
    try:
        deck, num, side = cabin.split("/")
        return pd.Series([deck, int(num), side])
    except:
        return pd.Series([None, None, None])
    
# Configuration de la page Streamlit
st.set_page_config(page_title="SpaceShip Titanic", layout="centered")
st.title("Interface de Prédiction")
st.markdown("Prédiction individuelle ou par lot")

# Onglets : Formulaire manuel vs Chargement CSV
onglet_formulaire, onglet_csv = st.tabs(["📝 Formulaire manuel", "📁 Chargement CSV"])

with onglet_formulaire:
    st.markdown("### Remplissez le formulaire")
    
    # Info utilisateur
    with st.expander("ℹ️ Informations importantes sur le formulaire"):
        st.markdown("""
            - Tous les champs marqués d'un `*` sont **obligatoires**.
            - Le **nom** est utilisé uniquement à des fins d'affichage.
            """)

    # Champs utilisateur
    nom = st.text_input("Nom ", placeholder="ex: Abdisamad")

    # Champs numériques et sélections
    col1, col2 = st.columns(2)
    age = col1.number_input("Age `*`", min_value=0, max_value=300, value=1)
    groupesize = col2.number_input("Taille du groupe `*`", min_value=1, max_value=300, value=1)
    planet_choice = col1.selectbox("Planète d'origine `*`", ["Earth", "Europa", "Mars"])
    destination = col2.selectbox("Destination `*`", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
    vip_fr = col1.selectbox("Service VIP `*`", ["Oui", "Non"])
    cryosleep_fr = col2.selectbox("En cryosommeil `*`", ["Oui", "Non"])
    
    # Conversion en booléens
    vip = True if vip_fr == "Oui" else False
    cryosleep = True if cryosleep_fr == "Oui" else False

    # Informations sur la cabine
    col1, col2, col3 = st.columns(3)
    deck = col1.selectbox("Pont `*`", ["A", "B", "C", "D", "E", "F", "G", "T"])
    cabin_num = col2.number_input("Numéro de cabine `*`", min_value=1, max_value=2000, step=1)
    side_fr = col3.selectbox("Côté `*`", ["Bâbord P", "tribord S"])
    side = "P" if side_fr == "Bâbord P" else "S"

    # Champs financiers
    col_dep1, col_dep2, col_dep3 = st.columns(3)
    room_service = col_dep1.number_input("Service chambre($) `*`", min_value=0.0)
    food_court = col_dep2.number_input("Restauration  ($) `*`", min_value=0.0)
    shopping_mall = col_dep3.number_input("Centre commercial  ($) `*`", min_value=0.0)

    col_dep4, col_dep5 = st.columns(2)
    spa = col_dep4.number_input("SPA ($) `*`", min_value=0.0)
    vr_deck = col_dep5.number_input("Deck VR ($) `*`", min_value=0.0)

    # Calcul de la dépense totale
    total_spend = sum([room_service, food_court, shopping_mall, spa, vr_deck])

    # Liste des valeurs saisies
    List_saisi = [planet_choice, cryosleep, destination, float(age), vip,
                  room_service, food_court, shopping_mall, spa, vr_deck,
                  groupesize, deck, cabin_num, side, total_spend]

    # Bouton prédiction individuelle
    if st.button("Prédire", key="formulaire"):
        if not verifie_saisi(List_saisi):
            st.error("Veuillez remplir tous les champs obligatoires.")
        else:
            # Préparation et prédiction
            data_input = preparer_donnees(List_saisi, colonnes_modele, scaler)
            prediction = modele.predict(data_input)

            # Affichage du résultat
            nom_affiche = f"Le passager **{nom}**" if nom.strip() != "" else "Le passager"
            if prediction[0] == 1:
                st.success(f"{nom_affiche} a été **transporté**.")
            else:
                st.error(f"{nom_affiche} **n'a pas été transporté**.")

with onglet_csv:
    st.markdown("### Chargement d’un fichier CSV ")

    # Informations à l'utilisateur
    with st.expander("ℹ️ Information sur le fichier CSV"):
        st.markdown("""...""")  # Explication des colonnes attendues

    # Colonnes attendues
    colonnes_obligatoires = [
        "PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age",
        "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
    ]
    colonnes_facultatives = ["Name"]

    # Upload du fichier CSV
    fichier = st.file_uploader("Téléversez le fichier CSV :", type=["csv"])

    if fichier is not None:
        try:
            # Lecture
            data_input = pd.read_csv(fichier)

            # Vérifie les colonnes manquantes
            colonnes_manquantes = [col for col in colonnes_obligatoires if col not in data_input.columns]
            if colonnes_manquantes:
                st.error(f"Colonnes manquantes : {', '.join(colonnes_manquantes)}")
            else:
                # Nettoyage et enrichissement des données
                data = data_input[[col for col in colonnes_obligatoires + colonnes_facultatives if col in data_input.columns]]
                data_id = data[["PassengerId"]].copy()
                if "Name" in data.columns:
                    data.drop(columns=["Name"], inplace=True)

                # Conversion des types
                data["CryoSleep"] = data["CryoSleep"].astype(bool)
                data["VIP"] = data["VIP"].map({"True": True, "False": False}).astype(bool)

                # Traitement cabine
                data[["Deck", "Cabin_Num", "Side"]] = data["Cabin"].apply(traiter_cabin)

                # Dépenses totales
                depenses = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
                data["TotalSpend"] = data[depenses].sum(axis=1)

                # Calcul de la taille du groupe à partir du PassengerId
                data["GroupId"] = data["PassengerId"].apply(lambda x: x.split('_')[0])
                group_sizes = data["GroupId"].value_counts()
                data["GroupSize"] = data["GroupId"].map(group_sizes)

                st.success("Fichier prêt pour les prédictions.")

                # Bouton prédiction en lot
                if st.button("Prédire"):
                    resultats = []

                    # Traitement ligne par ligne
                    for index, row in data.iterrows():
                        try:
                            if row.isnull().any():
                                raise ValueError("Valeurs manquantes")
                            List_saisi = [
                                row["HomePlanet"], row["CryoSleep"], row["Destination"], float(row["Age"]),
                                row["VIP"], row["RoomService"], row["FoodCourt"], row["ShoppingMall"],
                                row["Spa"], row["VRDeck"], int(row["GroupSize"]), row["Deck"],
                                int(row["Cabin_Num"]), row["Side"], row["TotalSpend"]
                            ]
                            donnees_preparees = preparer_donnees(List_saisi, colonnes_modele, scaler)
                            prediction_brute = modele.predict(donnees_preparees)[0]
                            prediction = "Transporté" if prediction_brute == 1 else "Pas transporté"
                        except Exception:
                            prediction = "Valeurs manquantes"

                        # Ajoute le résultat
                        resultats.append({
                            "PassengerId": row["PassengerId"],
                            "Prediction": prediction
                        })

                    # Affiche les résultats avec surlignage rouge en cas d’erreur
                    def surligner_erreurs(val):
                        return 'background-color: red; color: white' if val == "Valeurs manquantes" else ''
                    
                    data_resultats = pd.DataFrame(resultats)
                    st.dataframe(data_resultats.style.applymap(surligner_erreurs, subset=['Prediction']))
        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")

# Liens vers les profils GitHub et LinkedIn
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em; padding-top: 10px;'>
        🔗 Retrouvez-moi sur :
        <a href='https://github.com/Abdisamad06/Spaceship-Titanic.git' target='_blank'>GitHub</a> |
        <a href='https://www.linkedin.com/in/abdisamad-abdourahman-abdillahi-a49920330/' target='_blank'>LinkedIn</a> |
    </div>
    """,
    unsafe_allow_html=True
)
