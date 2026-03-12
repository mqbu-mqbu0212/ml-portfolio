import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# 特徴量抽出関数
# -----------------------
def extract_features(img_path, feature_type):
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    features = []
    if feature_type in ["rgb","rgb_glcm"]:
        R = img_array[:, :, 0]; G = img_array[:, :, 1]; B = img_array[:, :, 2]
        features.extend([np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)])
    if feature_type in ["glcm","rgb_glcm"]:
        gray = img_as_ubyte(rgb2gray(img_array))
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)
        features.extend([
            np.mean(graycoprops(glcm,'contrast')),
            np.mean(graycoprops(glcm,'homogeneity')),
            np.mean(graycoprops(glcm,'energy')),
            np.mean(graycoprops(glcm,'correlation'))
        ])
    return features

# -----------------------
# データ読み込み
# -----------------------
def load_dataset(base_path, classes, feature_type):
    X, y = [], []
    for label_idx, class_name in enumerate(classes):
        class_path = os.path.join(base_path, class_name)
        for filename in os.listdir(class_path):
            if filename.endswith((".tif",".jpg")):
                X.append(extract_features(os.path.join(class_path, filename), feature_type))
                y.append(label_idx)
    return np.array(X), np.array(y)

# -----------------------
# クロスバリデーション関数
# -----------------------
def cross_val_score_with_seed(X, y, model, model_type, random_state):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    acc_scores, f1_scores, importances, y_test_list, pred_list = [], [], [], [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc_scores.append(accuracy_score(y_test, pred))
        f1_scores.append(f1_score(y_test, pred, average='macro'))
        y_test_list.append(y_test)
        pred_list.append(pred)
        if model_type=="RF":
            importances.append(model.feature_importances_)
    return acc_scores, f1_scores, importances, y_test_list, pred_list

# -----------------------
# 棒グラフ作成関数
# -----------------------
def plot_metrics_summary(results_list, save_path=None):
    df = pd.DataFrame(results_list)
    labels = [f"{row['model']}-{row['feature']}" for idx,row in df.iterrows()]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(x - width/2, df["Accuracy_Mean"], width, yerr=df["Accuracy_STD"], capsize=5, label="Accuracy", color="skyblue")
    ax.bar(x + width/2, df["F1_Mean"], width, yerr=df["F1_STD"], capsize=5, label="F1 Score", color="lightgreen")
    ax.set_ylabel("Score"); ax.set_ylim(0,1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Model Performance Comparison (Accuracy & F1)")
    ax.legend(); plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    #plt.show()

# -----------------------
# 複数シード平均計算
# -----------------------
def run_experiment_multiple_seeds(feature_type, model_type, use_scaler=False, SKF_count=20, results_list=None, save_folder=None):
    base_path = r"./UCMerced_LandUse/Images"
    classes = ["airplane", "denseresidential", "beach"]

    print(f"\n--- Feature: {feature_type}, Model: {model_type}, Scaler: {use_scaler} ---")
    X, y = load_dataset(base_path, classes, feature_type)

    if model_type == "RF":
        model = RandomForestClassifier(n_estimators=100, random_state=0)
    elif model_type == "SVM":
        if use_scaler:
            model = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=1.0, gamma="scale")
            )
        else:
            model = SVC(kernel="rbf", C=1.0, gamma="scale")
    else:
        raise ValueError("model_type must be 'RF' or 'SVM'")

    all_acc = []
    all_f1 = []
    all_y_test = []
    all_pred = []
    all_importances = []

    for seed in range(SKF_count):
        acc_list, f1_list, importances_list, y_test_list, pred_list = cross_val_score_with_seed(X, y, model, model_type, seed)
        all_acc.extend(acc_list)
        all_f1.extend(f1_list)
        all_importances.extend(importances_list)
        for yt, yp in zip(y_test_list, pred_list):
            all_y_test.extend(yt)
            all_pred.extend(yp)

    # Accuracy/F1
    overall_mean_acc = np.mean(all_acc)
    overall_std_acc = np.std(all_acc)
    overall_mean_f1 = np.mean(all_f1)
    overall_std_f1 = np.std(all_f1)

    print(f"Accuracy: {overall_mean_acc:.4f} ± {overall_std_acc:.4f}")
    print(f"F1 Score: {overall_mean_f1:.4f} ± {overall_std_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_y_test, all_pred)
    cm_df = pd.DataFrame(cm,
                         index=[f"True_{c}" for c in classes],
                         columns=[f"Pred_{c}" for c in classes])
    print("\nConfusion Matrix:")
    print(cm_df)

    # Feature importance (RFのみ)
    if model_type == "RF":
        all_importances = np.array(all_importances)
        mean_importance = np.mean(all_importances, axis=0)
        std_importance = np.std(all_importances, axis=0)

        feature_names = []
        rgb_features = ["mean_R","mean_G","mean_B","std_R","std_G","std_B"]
        glcm_features = ["contrast","homogeneity","energy","correlation"]

        if feature_type in ["rgb", "rgb_glcm"]:
            feature_names.extend(rgb_features)
        if feature_type in ["glcm", "rgb_glcm"]:
            feature_names.extend(glcm_features)

        print("\nFeature Importances:")
        for name, mean_val, std_val in zip(feature_names, mean_importance, std_importance):
            print(f"{name:15s}: {mean_val:.4f} ± {std_val:.4f}")

    print("-" * 60)

    # ヒートマップ表示
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({model_type}-{feature_type})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_folder:
        plt.savefig(os.path.join(save_folder, f"ConfusionMatrix_{model_type}_{feature_type}.png"))
    #plt.show()
    
    if results_list is not None:
        results_list.append({
            "model":model_type,
            "feature":feature_type,
            "Scaler":use_scaler,
            "Accuracy_Mean":overall_mean_acc,
            "Accuracy_STD":overall_std_acc,
            "F1_Mean":overall_mean_f1,
            "F1_STD":overall_std_f1
        })

    # RFの重要度も表示
    if model_type=="RF":
        all_importances = np.array(all_importances)
        mean_importance = np.mean(all_importances, axis=0)
        std_importance = np.std(all_importances, axis=0)
        feature_names = []
        if feature_type in ["rgb","rgb_glcm"]: feature_names.extend(["mean_R","mean_G","mean_B","std_R","std_G","std_B"])
        if feature_type in ["glcm","rgb_glcm"]: feature_names.extend(["contrast","homogeneity","energy","correlation"])
        indices = np.argsort(mean_importance)[::-1]
        plt.figure()
        plt.bar(range(len(mean_importance)), mean_importance[indices], yerr=std_importance[indices])
        plt.xticks(range(len(mean_importance)), np.array(feature_names)[indices], rotation=45)
        plt.ylabel("Feature Importance")
        plt.title(f"RF Feature Importance ({feature_type})")
        plt.tight_layout()
        if save_folder:
            plt.savefig(os.path.join(save_folder,f"RF_importance_{feature_type}.png"))
        #plt.show()

# -----------------------
# メイン処理
# -----------------------
if __name__=="__main__":
    save_folder = r"./Results"
    os.makedirs(save_folder, exist_ok=True)
    results_summary = []

    run_experiment_multiple_seeds("rgb","RF",SKF_count=20, results_list=results_summary, save_folder=save_folder)
    run_experiment_multiple_seeds("glcm","RF",SKF_count=20, results_list=results_summary, save_folder=save_folder)
    run_experiment_multiple_seeds("rgb_glcm","RF",SKF_count=20, results_list=results_summary, save_folder=save_folder)

    run_experiment_multiple_seeds("rgb","SVM",True,SKF_count=20, results_list=results_summary, save_folder=save_folder)
    run_experiment_multiple_seeds("glcm","SVM",True,SKF_count=20, results_list=results_summary, save_folder=save_folder)
    run_experiment_multiple_seeds("rgb_glcm","SVM",True,SKF_count=20, results_list=results_summary, save_folder=save_folder)

    # Accuracy/F1まとめ棒グラフ表示＆保存

    plot_metrics_summary(results_summary, save_path=os.path.join(save_folder,"Accuracy_F1_comparison.png"))
