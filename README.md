# Réseau de Neurones Interactif avec Apprentissage Continu

Une application web interactive pour enseigner les principes fondamentaux des réseaux de neurones. Les utilisateurs peuvent dessiner des chiffres (0-3) sur une grille 8x8, et un réseau de neurones apprend progressivement à les reconnaître à travers les interactions.

## Caractéristiques

- **Apprentissage en temps réel**: Le réseau apprend uniquement des exemples fournis par l'utilisateur
- **Interface visuelle**: Visualisation des activations des neurones pendant la reconnaissance
- **Système de feedback**: Les utilisateurs peuvent confirmer ou corriger les prédictions
- **Implémentation authentique**: Un vrai réseau de neurones implémenté en JavaScript
- **Sauvegarde et importation**: Possibilité de sauvegarder et partager les modèles entraînés

## Technologies utilisées

- React 18
- Next.js 14
- TypeScript
- math.js pour les calculs matriciels
- Tailwind CSS pour l'interface

## Pour commencer

### Prérequis

- Node.js (version 18+)
- npm ou yarn

### Installation

1. Cloner ce dépôt
   ```
   git clone https://github.com/mieleif/NNvisual.git
   cd NNvisual
   ```

2. Installer les dépendances
   ```
   npm install
   # ou
   yarn
   ```

3. Lancer le serveur de développement
   ```
   npm run dev
   # ou
   yarn dev
   ```

4. Ouvrir [http://localhost:3000](http://localhost:3000) dans votre navigateur

## Architecture du réseau

- **Couche d'entrée**: 64 neurones (grille 8×8)
- **Couche cachée**: 16 neurones
- **Couche de sortie**: 4 neurones (chiffres 0-3)
- **Fonction d'activation**: Sigmoid
- **Méthode d'apprentissage**: Backpropagation

## Utilisation

1. Dessinez un chiffre (0-3) sur la grille
2. Cliquez sur "Reconnaître" pour que le réseau analyse le dessin
3. Si la prédiction est correcte, confirmez-la pour renforcer l'apprentissage
4. Si la prédiction est incorrecte, indiquez le bon chiffre
5. Répétez pour améliorer la précision du réseau
6. Utilisez "Sauvegarder" pour conserver le modèle entraîné

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Comment contribuer

Les contributions sont les bienvenues! N'hésitez pas à ouvrir une issue ou soumettre une pull request.
