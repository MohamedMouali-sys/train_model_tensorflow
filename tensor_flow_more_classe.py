import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.applications import EfficientNetB4
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# Répertoires d'images
train_dir = "/kaggle/input/dataset-fetus-anomaly-augmented/data_train"
test_dir = "/kaggle/input/dataset-fetus-anomaly-augmented/data_extern"
val_dir = "/kaggle/input/dataset-fetus-anomaly-augmented/data_valid"

img_size = 224

# Chargement des images et des étiquettes multiclasse
def load_images_labels(directory):
    images = []
    labels = []
    class_names = sorted(os.listdir(directory))
    print(class_names)
    label_map = {name: idx for idx, name in enumerate(class_names)}
    print(f"Chargement des images depuis : {directory}")
    for label in class_names:
        path = os.path.join(directory, label)
        print(f"  → Classe {label} : {len(os.listdir(path))} images")
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                img_rgb = cv2.merge([img]*3)
                images.append(img_rgb)
                labels.append(label_map[label])
    return np.array(images) / 255.0, np.array(labels), class_names

# Étape 1 : Chargement des jeux de données
X_train, y_train, class_names = load_images_labels(train_dir)
X_test, y_test, _ = load_images_labels(test_dir)
X_val, y_val, _ = load_images_labels(val_dir)

num_classes = len(class_names)

# Fusion des ensembles d'entraînement et de validation
X_total = np.concatenate((X_train, X_val), axis=0)
y_total = np.concatenate((y_train, y_val), axis=0)
X_train, X_val, y_train, y_val = train_test_split(X_total, y_total, test_size=0.2, random_state=42)

# Étape 2 : Construction du modèle EfficientNetB4
base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

print("Résumé du modèle :")
model.summary()

# Étape 3 : Compilation du modèle
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Étape 4 : Préparation des données (augmentation)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Étape 5 : Calcul des poids de classe
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print(f"Poids de classe : {class_weights}")

# Étape 6 : Définition des callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
]

# Étape 7 : Lancement de l'entraînement
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=callbacks
)

# Étape 8 : Sauvegarde du modèle entraîné
model.save("efficientnet_multiclass_model.h5")
print("✅ Modèle sauvegardé sous le nom : efficientnet_multiclass_model.h5")

# Étape 9 : Évaluation sur les données de test
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Précision sur le test : {accuracy*100:.2f}%")

# Étape 10 : Affichage des courbes d'apprentissage
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Précision')
plt.xlabel('Épochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Perte')
plt.xlabel('Épochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Étape 11 : Prédictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Étape 12 : Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Prédit')
plt.ylabel('Vérité')
plt.title('Matrice de confusion')
plt.show()

# Étape 13 : Rapport de classification
print(classification_report(y_test, y_pred, target_names=class_names))

# Étape 14 : Affichage de quelques exemples de prédictions
plt.figure(figsize=(12,12))
for i in range(25):
    idx = np.random.randint(0, len(X_test))
    img = X_test[idx]
    true_label = class_names[y_test[idx]]
    pred_label = class_names[y_pred[idx]]
    plt.subplot(5,5,i+1)
    plt.imshow(img)
    plt.title(f"Vérité : {true_label}\nPrédit : {pred_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()

print("✅ Programme terminé avec succès.")
