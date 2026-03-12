# 機械学習ポートフォリオ：航空画像の分類実験

## 背景・動機

人間の動作を数値化・解析するうえで機械学習は避けて通れない分野だと考え、
基礎から実装できる状態にするために本実験を行った。
データの前処理からモデルの選定・評価まで一連のパイプラインを自分で実装することを目標とした。

## クラス選定の理由

以下の3クラスを意図的に選定した。

- **beach**：砂浜の均質な青・白系の色 → RGB統計量で特徴が出やすい
- **denseresidential**：建物の規則的な配置パターン → GLCMで特徴が出やすい
- **airplane**：空港のコンクリート面はbeachと、規則的構造はdenseresidentialと混同しやすい → 両特徴量で識別が難しいクラスとして選定

色情報（RGB）とテクスチャ情報（GLCM）それぞれが有利・不利になるクラスを
意図的に組み合わせることで、特徴量の違いが分類性能に与える影響を明確に比較できる構成にした。

## 使用技術

- Python（Pandas, NumPy, scikit-learn, scikit-image, Pillow）
- 分類モデル：SVM（RBFカーネル）、Random Forest
- 評価：Stratified 5-fold Cross Validation × 20回反復

## 特徴量

- RGB統計量（各チャネルの平均・標準偏差）
- GLCMテクスチャ特徴（contrast, energy, homogeneity, correlation）
- 比較構成：RGBのみ / GLCMのみ / RGB+GLCM

## 結果

### Accuracy / F1-score 比較

![Accuracy/F1比較](Accuracy_F1_comparison.png)

RGB+GLCMの組み合わせが両モデルで最も高いAccuracy・F1-scoreを達成。
RFではGLCM単体がRGB単体を上回り、SVMでは両者の差が小さかった。

### 混同行列

| RF-rgb | RF-rgb_glcm |
|--------|-------------|
| ![](ConfusionMatrix_RF_rgb.png) | ![](ConfusionMatrix_RF_rgb_glcm.png) |

| RF-glcm | SVM-rgb |
|---------|---------|
| ![](ConfusionMatrix_RF_glcm.png) | ![](ConfusionMatrix_SVM_rgb.png) |

| SVM-glcm | SVM-rgb_glcm |
|----------|--------------|
| ![](ConfusionMatrix_SVM_glcm.png) | ![](ConfusionMatrix_SVM_rgb_glcm.png) |

### 特徴量重要度（Random Forest）

| rgb | glcm | rgb_glcm |
|-----|------|----------|
| ![](RF_importance_rgb.png) | ![](RF_importance_glcm.png) | ![](RF_importance_rgb_glcm.png) |

## 考察

RGB+GLCMの併用により、airplaneとdenseresidentialの誤分類が大幅に改善された。
住宅地の規則的な建物配置や空港周辺の明暗差といったテクスチャ情報が識別に寄与したと考えられる。

一方、beach→airplaneの誤分類はGLCM追加後にわずかに増加した。
これはクラス選定時の想定とは逆の結果で、砂浜の均質なテクスチャが
駐機場のコンクリート面と類似した特徴として扱われたと考えられる。
RGBでは明確に区別できていたbeachが、GLCMの追加によって
むしろairplaneに近づいてしまった点は興味深い。

特徴量重要度ではcontrastとenergyの寄与が特に高く、
テクスチャの粗さと均一性が分類の主要な手がかりになっていることが確認された。

改善案として、HOGなどの局所形状特徴の追加、GLCMの多方向化、
ハイパーパラメータ探索によるモデル最適化が有効と考えられる。

## 詳細資料

- [実験レポート（PDF）](馬淵健_Pythonを用いた画像処理・機械学習経験.pdf)
- [ソースコード](MachineLearning_multiple5K.py)
