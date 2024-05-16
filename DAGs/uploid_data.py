import os
import json
import time
import logging
import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

URL_FILE = "url.json"
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

def check_environment_setup():
    # Configuration du logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Afficher le répertoire de travail actuel
    logger.info("--" * 20)
    logger.info(f"[info logger] cwd: {os.getcwd()}")

    # Vérifier si l'URL_FILE est défini et s'il pointe vers un fichier existant
    logger.info(f"[info logger] URL_FILE: {URL_FILE}")
    assert os.path.isfile(URL_FILE)

    # Vérifier si le répertoire de données existe
    logger.info(f"[info logger] DATA_PATH: {DATA_PATH}")
    assert os.path.isdir(DATA_PATH)

    logger.info("--" * 20)

def load_data():
    # Exemple d'utilisation des fonctions
    url = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe-v2-logements-neufs/lines?size=10000&format=csv&after=10000%2C965634&header=true"

    # Charger les données à partir de l'URL
    data = load_data_from_url(url)

    if data is not None:
        # Traiter les données et obtenir le JSON
        json_data = process_results(data)

        # Enregistrer le JSON dans le répertoire de données local
        data_filename = save_to_data_directory(json_data)

        # Supprimer les doublons du fichier JSON dans le répertoire de données
        if data_filename is not None:
            drop_duplicates_from_data(data_filename)

def load_data_from_url(url):
    """
    Charge les données à partir de l'URL spécifiée.

    Args:
        url (str): L'URL à partir de laquelle charger les données.

    Returns:
        DataFrame: Les données chargées à partir de l'URL, ou None en cas d'erreur.
    """
    try:
        # Charger les données à partir de l'URL
        data = pd.read_csv(url)
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données à partir de l'URL : {str(e)}")

def process_results(data):
    """
    Processes the results obtained from the previous API call,
    updates the URL file,
    and returns the data as a JSON.

    Args:
        data (DataFrame): Les données chargées à partir de l'API.

    Returns:
        str: Les données au format JSON.
    """
    # Save the data to a new JSON file
    # append current timestamp (up to the second to the filename)
    timestamp = int(time.time())
    data_filename = os.path.join(DATA_PATH, f"data_{timestamp}.json")

    # Save the data to a new JSON file
    with open(data_filename, "w", encoding="utf-8") as file:
        data.to_json(file, orient='records', indent=4)

    # Return the JSON data
    with open(data_filename, "r", encoding="utf-8") as file:
        json_data = file.read()
    print(f"Data saved to {data_filename}")
    return json_data

def save_to_data_directory(json_data):
    """
    Enregistre les données JSON dans le répertoire de données local.

    Args:
        json_data (str): Les données au format JSON.

    Returns:
        str: Le chemin du fichier dans lequel les données ont été enregistrées, ou None en cas d'erreur.
    """
    try:
        # Get current date and time
        today_datetime = time.strftime("%Y-%m-%d")

        # Append current date and time to the filename
        data_filename = os.path.join(DATA_PATH, f"data_{today_datetime}.json")

        # Save the JSON data to the data directory
        with open(data_filename, "w", encoding="utf-8") as file:
            file.write(json_data)

        print(f"Data saved to {data_filename}")
        return data_filename

    except Exception as error:
        print(f"Erreur lors de l'enregistrement des données dans le répertoire de données : {error}")
        return None

def drop_duplicates_from_data(data_filename):
    """
    Supprime les doublons du fichier JSON sauvegardé dans le répertoire de données.

    Args:
        data_filename (str): Le chemin du fichier JSON dans lequel supprimer les doublons.

    Returns:
        bool: True si les doublons ont été supprimés avec succès, False sinon.
    """
    try:
        # Charger les données à partir du fichier JSON
        with open(data_filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Supprimer les doublons en fonction du numéro de DPE
        data = [dict(t) for t in {tuple(d.items()) for d in data}]

        # Sauvegarder les données sans doublons dans le même fichier JSON
        with open(data_filename, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        print("Doublons supprimés avec succès dans le fichier JSON.")
        return True

    except Exception as error:
        print(f"Erreur lors de la suppression des doublons dans le fichier JSON : {error}")
        return False

# DAG Airflow
with DAG('load_data', default_args=default_args, schedule_interval=timedelta(days=1)) as dag:
    check_environment_setup_task = PythonOperator(
        task_id='check_environment_setup',
        python_callable=check_environment_setup
    )

    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )

    process_results_task = PythonOperator(
        task_id='process_results',
        python_callable=process_results
    )

    save_to_data_directory_task = PythonOperator(
        task_id='save_to_data_directory',
        python_callable=save_to_data_directory
    )

    drop_duplicates_task = PythonOperator(
        task_id='drop_duplicates',
        python_callable=drop_duplicates_from_data  # Assuming this function is for dropping duplicates from data file
    )

    check_environment_setup_task >> load_data_task >> process_results_task >> save_to_data_directory_task >> drop_duplicates_task
