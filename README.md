# MLops

Grâce à 3 DAGs Airflow, nous sommes capable de prédire des étiquettes GES de logement.
Nous nous basons sur le dataset : https://data.ademe.fr/datasets/dpe-v2-logements-neufs/ 


Vous trouverez dans ce Github les 3 programmes pythons correspondants aux DAGs Airflow pour les fonctionnalités suivantes :
- L'extraction des données
- La transformation des données
- L'entraînement et le test du modèle

Nous avons mis en place Airflow et MLflow sur notre VM Azure. Or, lorsque nous lançons la commande docker-compose up, nous voyons nos workers qui tournent :


![docker_compose](https://github.com/axelToussenel/MLops/assets/91553182/182ff27c-b012-4d7e-9a49-75316ca84918)


La preuve avec un docker ps :


![image](https://github.com/axelToussenel/MLops/assets/91553182/9179ca95-b9e9-4a47-aac3-f48d39771319)


Nous pouvons également voir nos images :


![image](https://github.com/axelToussenel/MLops/assets/91553182/84ee8f31-f4c5-4a8d-be82-b7388582b33c)


Mais une fois sur le navigateur, il est impossible d'accéder à l'interface Airflow :


![airflow_inaccessible](https://github.com/axelToussenel/MLops/assets/91553182/99c215b0-40a3-4582-8c05-97bd9a3beaae)


Nous avons essayé une certaine quantité de choses pour résoudre ou contourner ce problème, hélas sans résultat. Nous avons tout de même décidé de transmettre nos 3 programmes python qui seraient des DAGs sous Airflow. 

De ce fait, nous avons décidé de stocker notre dataset dans un répertoiredu Git.
