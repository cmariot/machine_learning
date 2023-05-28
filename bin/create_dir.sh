#!/bin/bash

# Demande à l'utilisateur de saisir la plage de chiffres souhaitée
read -p "Entrez la plage de chiffres souhaitée (ex: 1-10): " plage

# Boucle pour créer les dossiers
for i in $(seq $(echo $plage | cut -d'-' -f1) $(echo $plage | cut -d'-' -f2)); do
    # Ajoute un zéro devant les chiffres inférieurs à 10
    if [ $i -lt 10 ]; then
        i="0$i"
    fi
    # Créer le nom du dossier en ajoutant le chiffre à la fin
    nom_dossier="ex${i}"
    # Créer le dossier
    mkdir $nom_dossier
done

echo "Les dossiers ont été créés avec succès."

