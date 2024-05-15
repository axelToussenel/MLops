# MLops

Grâce à 3 DAGs Airflow, nous sommes capable de prédire des étiquettes GES de logement.
Nous nous basons sur le dataset : https://data.ademe.fr/datasets/dpe-v2-logements-neufs/ 


Vous trouverez dans ce Github les 3 programmes pythons correspondants aux DAGs Airflow pour les fonctionnalités suivantes :
- L'extraction des données
- La transformation des données
- L'entraînement et le test du modèle


Allez sur votre interpréteur bash, saisi la commande suivante :

```ssh -i .../axel.pem azureuser@40.66.47.201```


A présent tapez :

```cd airflow```


Puis lançons nos worker :

```docker-compose up```

Nous avons mis en place Airflow et MLflow sur notre VM Azure. Lorsque nous lançons la commande docker-compose up, nous voyons nos workers qui tournent :


![docker_compose](https://github.com/axelToussenel/MLops/assets/91553182/182ff27c-b012-4d7e-9a49-75316ca84918)


Regardons ce avec un docker ps :


![image](https://github.com/axelToussenel/MLops/assets/91553182/9179ca95-b9e9-4a47-aac3-f48d39771319)


Nous pouvons également voir nos images :


![image](https://github.com/axelToussenel/MLops/assets/91553182/84ee8f31-f4c5-4a8d-be82-b7388582b33c)


Une fois sur le navigateur, voici l'interface d'Airflow :


![image](https://github.com/axelToussenel/MLops/assets/91553182/a4df8626-c3f1-4537-8741-f627a04305f9)


Nous ne sommes malheureusement pas parvenu à aller au bout de ce projet. Vous pouvez trouver tous nos programmes python pour nos DAGs dans ```/DAGs```, nos docker files et docker-compose dans ```/Dockerfiles```.

De ce fait, nous avons décidé de stocker notre dataset dans un répertoiredu Git.
