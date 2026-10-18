# Classification d'images de paysages avec un CNN (TensorFlow / Keras)

## 1. Description du projet et problème métier

Une entreprise souhaite **automatiser l'analyse d'images** pour identifier le **type d'environnement** représenté (ville, forêt, mer...).
Ce projet développe un système de **Computer Vision** basé sur un réseau de neurones convolutif (CNN) qui :

- prend une image en entrée ;
- retourne la **catégorie prédite** et sa **probabilité** ;
- permet de **mesurer** les performances et d'**analyser les erreurs**.

## 2. Dataset et classes

Dataset : **Intel Image Classification** (images de paysages, taille d'origine 150 × 150).

| Classe | Description |
|---|---|
| `buildings` | bâtiments |
| `forest` | forêt |
| `glacier` | glacier |
| `mountain` | montagne |
| `sea` | mer |
| `street` | rue |

- Images étiquetées utilisées (`seg_train`) : **[...] images**, distribution : [équilibrée / préciser les écarts].
- Le dossier `seg_pred` contient des images **sans étiquette** : il sert uniquement à tester la prédiction sur de nouvelles images.

## 3. Organisation et séparation des données

Il n'y avait pas de dossier de test étiqueté : les images de `seg_train` ont été séparées **classe par classe** (la répartition des classes reste la même dans chaque ensemble) :

| Ensemble | Part | Nombre d'images | Rôle |
|---|---|---|---|
| Train | 70 % | [...] | le modèle **apprend** dessus |
| Validation | 15 % | [...] | **surveiller** l'entraînement et **choisir** le modèle (EarlyStopping, comparaison) |
| Test | 15 % | [...] | **examen final**, utilisé une seule fois pour mesurer la vraie performance |

Une graine aléatoire fixe (`SEED = 42`) garantit la **même séparation** à chaque exécution. Les images corrompues sont écartées.

## 4. Preprocessing des images

- **Redimensionnement** : toutes les images en **150 × 150** pixels (`image_size`).
- **Normalisation** : `Rescaling(1/255)` ramène les pixels de [0, 255] à [0, 1] (apprentissage plus stable). Ce calque est **dans le modèle** : la prédiction d'une nouvelle image applique donc automatiquement le même traitement.
- **Chargement par batchs** de 32 images avec `image_dataset_from_directory` (basé sur `tf.data`), et `prefetch` pour préparer le batch suivant pendant le calcul.

## 5. Stratégie de Data Augmentation

Appliquée **uniquement à l'entraînement** (les calques d'augmentation de Keras sont inactifs en validation, test et prédiction).

| Transformation | Valeur |
|---|---|
| Retournement horizontal | `RandomFlip("horizontal")` |
| Rotation | `RandomRotation(0.1)` |
| Zoom | `RandomZoom(0.1)` |
| Translation | `RandomTranslation(0.1, 0.1)` |
| Contraste | `RandomContrast(0.1)` |

Rôle : créer des variantes des images à chaque époque pour que le modèle reconnaisse la classe malgré les changements de position, d'angle ou de luminosité → **meilleure généralisation**, moins d'overfitting.

![Exemples d'augmentation](results/augmentation.png)  <!-- ajouter la figure du Jour 2 si vous l'avez sauvegardée -->

## 6. Architecture du CNN et justification

```
Entrée 150×150×3
→ Data Augmentation → Rescaling(1/255)
→ Conv2D(32, 3×3, ReLU) → MaxPooling2D
→ Conv2D(64, 3×3, ReLU) → MaxPooling2D
→ Conv2D(128, 3×3, ReLU) → MaxPooling2D
→ GlobalAveragePooling2D
→ Dense(128, ReLU) → Dropout(0.3)
→ Dense(6, softmax)
```

| Couche | Justification |
|---|---|
| **Conv2D** | des filtres 3×3 détectent des motifs locaux (bords, textures, formes) ; de plus en plus complexes en profondeur |
| **ReLU** | apporte la non-linéarité ; simple et rapide |
| **MaxPooling2D** | réduit la taille de moitié, garde les motifs importants, rend le modèle moins sensible à la position |
| **Nombre de filtres croissant (32 → 64 → 128)** | plus on avance, plus les motifs sont complexes et nombreux |
| **GlobalAveragePooling2D** | vecteur compact ; bien moins de paramètres que `Flatten` donc moins d'overfitting |
| **Dense(128)** | combine les motifs pour décider |
| **Dropout** | éteint des neurones au hasard à l'entraînement pour éviter la mémorisation |
| **Dense(6, softmax)** | 6 sorties = 6 probabilités (somme = 1) |

Nombre total de paramètres : [...]

## 7. Fonction de perte et optimiseur

- **Perte** : `sparse_categorical_crossentropy` — adaptée à la classification **multi-classes** quand les étiquettes sont des entiers (0 à 5).
- **Optimiseur** : **Adam**, learning rate initial **0.001**.
- **Métrique suivie** : accuracy.

## 8. Paramètres d'entraînement

| Paramètre | Valeur |
|---|---|
| Batch size | 32 |
| Epochs | 30 maximum (arrêt anticipé par EarlyStopping) — époques réellement effectuées : [...] |
| Learning rate | 0.001 initial, divisé par 2 par `ReduceLROnPlateau` si stagnation |

## 9. Stratégie contre l'overfitting

- **Data Augmentation** (variantes des images d'entraînement) ;
- **Dropout** (0.3 pour le modèle de base) ;
- **GlobalAveragePooling2D** au lieu de `Flatten` (moins de paramètres) ;
- **EarlyStopping** (`monitor="val_loss"`, `patience=5`, `restore_best_weights=True`) : arrête l'entraînement et restaure les meilleurs poids ;
- **ReduceLROnPlateau** (`factor=0.5`, `patience=2`) : réglage plus fin quand la validation stagne.

## 10. Métriques d'évaluation

- **Accuracy** : part des images bien classées.
- **Precision** (par classe) : quand le modèle prédit une classe, a-t-il raison ?
- **Recall** (par classe) : parmi les vraies images d'une classe, combien sont retrouvées ?
- **F1-score** : moyenne équilibrée de precision et recall.
- **Matrice de confusion** : quelles classes sont confondues.

Résultats du modèle final sur le **jeu de test** :

| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|
| [...] | [...] | [...] | [...] |

Résultats par classe :

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| buildings | [...] | [...] | [...] |
| forest | [...] | [...] | [...] |
| glacier | [...] | [...] | [...] |
| mountain | [...] | [...] | [...] |
| sea | [...] | [...] | [...] |
| street | [...] | [...] | [...] |

## 11. Courbes d'apprentissage et interprétation

![Courbes d'apprentissage](results/courbes.png)

Interprétation : [Décrire : les courbes train/validation évoluent-elles ensemble ? Quel est l'écart d'accuracy train − validation ? À quelle époque la meilleure `val_loss` ? Y a-t-il overfitting (validation qui stagne ou se dégrade alors que le train s'améliore) ou underfitting ? Le modèle a-t-il convergé ? EarlyStopping a-t-il arrêté l'entraînement ?]

## 12. Matrice de confusion et analyse des erreurs

![Matrice de confusion](results/matrice_confusion.png)

Analyse : [Classes les plus difficiles : ... ; confusions les plus fréquentes : ... (par exemple glacier ↔ mountain, buildings ↔ street) ; explication à partir des images mal classées : image ambiguë, plusieurs éléments dans la scène, photo floue...]

## 13. Comparaison des configurations testées

| Modèle | Changement | Epoques | Acc. train | Acc. val | Écart | Acc. test | F1 test |
|---|---|---|---|---|---|---|---|
| `cnn_v1` | référence : 3 blocs, dropout 0.3, avec augmentation | [...] | [...] | [...] | [...] | [...] | [...] |
| `cnn_v2_sans_aug` | **sans** Data Augmentation | [...] | [...] | [...] | [...] | [...] | [...] |
| `cnn_v3_plus_profond` | 4 blocs (256 filtres), dropout 0.5 | [...] | [...] | [...] | [...] | [...] | [...] |

Impact de la Data Augmentation : [comparer `cnn_v1` et `cnn_v2_sans_aug`, notamment l'écart train/validation]

## 14. Modèle final retenu et justification

Modèle retenu : **[...]**

Justification : [meilleure accuracy de **validation** (le choix ne se fait pas sur le test) ; écart train/validation raisonnable ; complexité adaptée ; ...]

## 15. Installation et exécution

```bash
git clone https://github.com/VOTRE_NOM/VOTRE_REPO.git
cd VOTRE_REPO
pip install -r requirements.txt
```

Les notebooks (dossier `notebooks/`) ont été exécutés sur **Kaggle** (GPU recommandé). Ordre d'exécution :

1. `kaggle_jour1_exploration.ipynb` : exploration et séparation train / validation / test
2. `kaggle_jour2_pretraitement.ipynb` : prétraitement et Data Augmentation
3. `kaggle_jour3_cnn.ipynb` : construction et entraînement du CNN
4. `kaggle_jour4_evaluation.ipynb` : évaluation, erreurs, comparaison des configurations
5. `kaggle_jour5_finalisation.ipynb` : modèle final et prédiction

Pour reproduire : ajouter le dataset Intel Image Classification en **Input** du notebook Kaggle, activer le GPU, puis exécuter les notebooks dans l'ordre.

## 16. Prédire une nouvelle image

Le modèle final est dans `models/final_model.keras`.

```bash
python predict.py chemin/vers/image.jpg
```

Sortie attendue :

```
Classe prédite : forest
Probabilité    : 97.3%
```

Ou en Python :

```python
from tensorflow import keras
from predict import predict_image

model = keras.models.load_model("models/final_model.keras")
classe, proba, probas = predict_image("mon_image.jpg", model)
```

Le prétraitement (redimensionnement 150 × 150 et normalisation) est identique à celui de l'entraînement. La normalisation est incluse dans le modèle.

## 17. Exemples de prédictions

![Exemples de prédictions](results/exemples_predictions.png)

[Commenter 2 ou 3 exemples : prédiction correcte avec forte probabilité, cas incertain, cas d'erreur.]

## 18. Limites et pistes d'amélioration

**Limites**
- Le modèle ne connaît que **6 classes** : toute autre scène (désert, intérieur...) est forcément rangée dans l'une des 6.
- Images **150 × 150** : les détails fins sont perdus.
- Sensible aux scènes **ambiguës** (plusieurs éléments : glacier + montagne, rue + bâtiments).
- Une probabilité élevée ne garantit pas une bonne prédiction ; en dessous de ~60 %, considérer la prédiction comme incertaine.
- Conditions d'utilisation : photos extérieures proches du dataset (paysages, qualité correcte).

**Pistes d'amélioration**
- **Transfer Learning** (MobileNetV2, ResNet pré-entraînés sur ImageNet) et fine-tuning ;
- augmentation plus riche, images de plus grande taille ;
- plus de données pour les classes les plus confondues ;
- ajouter une classe « autre » ou un seuil de confiance pour rejeter les images hors sujet ;
- tester d'autres architectures et réglages (learning rate, batch size).
