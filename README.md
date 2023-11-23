# Machine Learning Deployment Project



## Getting started

Ce projet fait usage de deux composants prinicaux:

- un container docker qui tourne l'api de serving qui est lié aux dossiers artifacts et data.

- un container docker qui tourne l'interface web qui est lié au dossier data/prod images. Le dossiers prod images sera considéré comme intermédiaire de communication entre les deux container. En effet, l'image qui sera lue depuis l'interface web et sera stocké dans le dossier data/prod images. Le container de serving lira l'image correspondantes depuis le dossier data/prod images grace au nom du fichier image qui lui sera envoyé par le container webapp via une requete de type post.

Image récapitulative
## prérequis : 
docker compose est indispensable pour la suite




mettez vous sur le dossier de base du projet.

## Container de l'api serving

lancer la commande suivante depuis un terminal

```
sudo docker-compose -f serving/docker-compose.yml up --force-recreate

```

image cap2


générer un nouveau terminal

## Container de l'interface web

lancer la commande suivante depuis le nouveau terminal

```
sudo docker-compose -f webapp/docker-compose.yml up --force-recreate

```
image cap3


## résultats
 naviguez sur votre browser vers l'adresse :

 http://0.0.0.0:8081

 vous retrouverez l'interface suivante :

cap 4

 insérer une image et entrez l'étiquette réelle de l'image (0 ou 1). Localement, observez le fichier prod_data sur le dossier data. Il sera mis à jour par une nouvelle ligne qui va apparaitre indiquante l'image qui vient d'être traitée.

cap 5



