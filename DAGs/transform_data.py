import os
import re
import typing as t
from collections import Counter
import json
import datetime
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import unidecode

pd.options.display.max_columns = 100
pd.options.display.max_rows = 60
pd.options.display.max_colwidth = 100
pd.options.display.precision = 10
pd.options.display.width = 160

DATA_PATH = "data"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 16),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def rename_columns(columns: t.List[str]) -> t.List[str]:
    """
    Rename columns, convert values to lowercase, and remove accents.
    """
    new_columns = [col.lower() for col in columns]

    rgxs = [
        (r"[°/']", "_"),
        (r"²", "2"),
        (r"[()]", ""),
        (r"[éè]", "e"),
        (r"â", "a"),
        (r"^_", "dpe_"),
        (r"_+", "_"),
    ]
    for i, col in enumerate(new_columns):
        for rgx in rgxs:
            new_columns[i] = re.sub(rgx[0], rgx[1], col)

    # Remove accents from column names
    new_columns = [unidecode.unidecode(col) for col in new_columns]

    return new_columns

def load_data_from_json(data_filename):
    """
    Charge les données à partir du fichier JSON.

    Args:
        data_filename (str): Le chemin du fichier JSON contenant les données.
    Returns:
        DataFrame: Les données chargées à partir du fichier JSON, ou None en cas d'erreur.
    """
    try:
        # Charger les données à partir du fichier JSON
        with open(data_filename, "r", encoding="utf-8") as file:
            data = pd.read_json(file)

        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données à partir du fichier JSON : {str(e)}")
        return None

def preprocess_data():
    # Trouver le fichier JSON le plus récent dans le répertoire de données
    latest_data_file = max((os.path.join(DATA_PATH, f), os.path.getctime(os.path.join(DATA_PATH, f))) for f in os.listdir(DATA_PATH) if f.endswith('.json'))[0]

    if latest_data_file:
        data = load_data_from_json(latest_data_file)

        if data is not None:
            print("Data loaded successfully from JSON.")

            # Renommer les colonnes
            data.columns = rename_columns(data.columns)

            # Supprimer les lignes avec des valeurs manquantes dans la cible
            target = "etiquette_ges"
            data.dropna(subset=[target], inplace=True)

            # Colonnes catégorielles
            columns_categorical = [
                "modele_dpe",
                "methode_application_dpe",
                "type_installation_chauffage",
                "type_installation_ecs_general",
                "classe_altitude",
                "zone_climatique_",
                "type_energie_principale_chauffage",
                "type_energie_n_1",
                "type_energie_n_2",
                "type_energie_n_3",
            ]

            for col in columns_categorical:
               data[col].fillna("non renseigne", inplace=True)

            # Regroupement des types d'énergies
            type_energie = []
            for col in [
                "type_energie_principale_chauffage",
                "type_energie_n_1",
                "type_energie_n_2",
                "type_energie_n_3",
            ]:
                type_energie += list(data[col])

            type_energie_count = Counter(type_energie)

            type_energie_map = {
                "non renseigne": "non renseigne",
                "electricite": "electricite",
                "electricite d'origine renouvelable utilisee dans le batiment": "electricite",
                "gaz naturel": "gaz naturel",
                "butane": "GPL",
                "propane": "GPL",
                "gpl": "GPL",
                "fioul Domestique": "fioul domestique",
                "reseau de chauffage urbain": "reseau de chauffage urbain",
                "charbon": "combustible fossile",
                "autre combustible fossile": "combustible fossile",
                "bois – buches": "bois",
                "bois – plaquettes forestieres": "bois",
                "bois – granules (pellets) ou briquettes": "bois",
                "bois – plaquettes d’industrie": "bois",
                "bois": "bois",
            }

            for col in [
                "type_energie_principale_chauffage",
                "type_energie_n_1",
                "type_energie_n_2",
                "type_energie_n_3",
            ]:
                data[col].fillna("non renseigne", inplace=True)
                data[col] = data[col].apply(lambda d: type_energie_map[d])

            # Encodage des colonnes catégorielles
            encoder = OrdinalEncoder()

            data[columns_categorical] = encoder.fit_transform(data[columns_categorical])
            for col in columns_categorical:
                data[col] = data[col].astype(int)

            # Sauvegarde des mappings catégoriels dans un fichier JSON
            mappings = {}
            for i, col in enumerate(encoder.feature_names_in_):
                mappings[col] = {int(value): category for value, category in enumerate(encoder.categories_[i])}

            with open("./categorical_mappings.json", "w", encoding="utf-8") as f:
                json.dump(mappings, f, ensure_ascii=False, indent=4)

            # Transformation de la date de visite
            columns_dates = ["date_visite_diagnostiqueur"]

            col = "date_visite_diagnostiqueur"
            data[col] = data[col].apply(lambda d: float(".".join(d.split("-")[:2])))

            # Traitement des nombres flottants
            columns_float = [
                "version_dpe",
                "hauteur_sous_plafond",
                "nombre_appartement",
                "surface_habitable_immeuble",
                "surface_habitable_logement",
                "conso_5_usages_e_finale_energie_n_1",
            ]

            for col in columns_float:
                data[col].fillna(0.0, inplace=True)

            # Encodage de la cible
            target_encoding = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
            data[target] = data[target].apply(lambda d: target_encoding[d])

            # Ensemble final de caractéristiques
            features = [
                "n_dpe",
                "version_dpe",
                "methode_application_dpe",
                "type_installation_chauffage",
                "type_installation_ecs_general",
                "classe_altitude",
                "zone_climatique_",
                "type_energie_n_1",
                "type_energie_n_2",
                "type_energie_n_3",
                "type_energie_principale_chauffage",
                target,
            ]

            data = data[features].copy()
            data.reset_index(inplace=True, drop=True)

            # Sauvegarde des données prétraitées dans un nouveau fichier JSON
            preprocessed_data_filename = os.path.join(DATA_PATH, "data_pretraite.json")
            data.to_json(preprocessed_data_filename, orient="records", indent=4, ensure_ascii=False)
            print(f"Preprocessed data saved to {preprocessed_data_filename}")

        else:
             print("Error loading data from JSON.")
    else:
          print("No JSON data file found in the data directory.")

# DAG Airflow
with DAG('preprocess_data', default_args=default_args, schedule_interval=timedelta(days=1)) as dag:
    preprocess_data_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data
    )
    
    preprocess_data_task