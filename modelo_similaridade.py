import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing import image
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm import tqdm
import joblib
import seaborn as sns


class SignatureVerificationPipeline:
    def __init__(self, num_features=500, kernel="rbf", C=1.0):
        self.num_features = num_features
        self.kernel = kernel
        self.C = C
        self.feature_extractor = self._build_feature_extractor()
        self.classifier = SVC(kernel=self.kernel, C=self.C, probability=True)

    def _build_feature_extractor(self):
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(self.num_features, activation="relu")(x)
        model = Model(inputs=base_model.input, outputs=x)
        return model

    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=(128, 128), color_mode="grayscale")
        img_array = image.img_to_array(img)
        img_array = np.repeat(img_array, 3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return self.feature_extractor.predict(img_array).flatten()

    def prepare_dataset(self, pairs, labels):
        X, y = [], []
        for (img1_path, img2_path), label in tqdm(zip(pairs, labels), total=len(pairs), desc="Extraindo Features"):
            feat1 = self.extract_features(img1_path)
            feat2 = self.extract_features(img2_path)
            feature_diff = np.abs(feat1 - feat2)
            X.append(feature_diff)
            y.append(label)
        return np.array(X), np.array(y)

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
        print("Modelo treinado com sucesso!")

    def metrics(self, X_test, y_test, threshold=0.5):
        y_pred = np.array([self.classifier.predict([X_test[i]])[0] for i in range(len(X_test))])
        y_probs = np.array([self.classifier.predict_proba([X_test[i]])[0][1] for i in range(len(X_test))])
        y_pred = (y_probs > threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Genuína", "Forjada"]

        print(f"Acurácia: {acc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão")
        plt.show()

    def predict(self, img1_path, img2_path, threshold=0.5):
        feat1 = self.extract_features(img1_path)
        feat2 = self.extract_features(img2_path)
        feature_diff = np.abs(feat1 - feat2)
        prob = self.classifier.predict_proba([feature_diff])[0][1]
        prediction = 1 if prob > threshold else 0
        print(f"🔎 Probabilidade de ser forjada: {prob:.4f}")
        print(f"🔍 Resultado: {'Genuína ✅' if prediction == 0 else 'Forjada ❌'}")
        return prediction, prob

    def cross_validate(self, X, y, cv=10):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        acc_scores = cross_val_score(self.classifier, X, y, cv=skf, scoring="accuracy")
        f1_scores = cross_val_score(self.classifier, X, y, cv=skf, scoring="f1")
        recall_scores = cross_val_score(self.classifier, X, y, cv=skf, scoring="recall")
        precision = cross_val_score(self.classifier, X, y, cv=skf, scoring="precision")

        print(f"📊 Acurácia Média: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
        print(f"📊 F1-score Médio: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
        print(f"📊 Recall Médio: {recall_scores.mean():.4f} ± {recall_scores.std():.4f}")
        print(f"📊 Precision Médio: {precision.mean():.4f} ± {precision.std():.4f}")

        return acc_scores, f1_scores, recall_scores

    def save_pipeline(self, filename):
        joblib.dump({'feature_extractor': self.feature_extractor,
                     'classifier': self.classifier}, filename)
        print(f'pipeline salvo como {filename}')

    def optimize_hyperparameters(self, X, y, param_grid=None, cv=3):
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly']
            }
        grid = GridSearchCV(SVC(probability=True), param_grid, cv=cv)
        grid.fit(X, y)
        self.classifier = grid.best_estimator_
        print("🏁 Melhor modelo encontrado com GridSearchCV.")
        return grid.best_params_

    def load_pipeline(self, filename):
        pipeline = joblib.load(filename)
        self.feature_extractor = pipeline['feature_extractor']
        self.classifier = pipeline['classifier']
        print('Pipeline carregado com sucesso!')
