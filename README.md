# ARIM-ACADEMY-2026：Kerasを使った深層学習

Pythonの基礎文法を理解している方を対象に、Keras（Keras 3系）を使った深層学習の基本を、多重パーセプトロン（MLP）・畳み込みニューラルネットワーク（CNN）・オートエンコーダ（AE）・変分オートエンコーダ（VAE）という6本のノートブックを通じて学ぶ教材です。Fashion-MNIST・MNISTという2つの定番データセットを共通の題材とし、同じモデルをSequential API・Functional APIという2通りの書き方で実装し比較することで、Kerasの書き方の違いそのものを学べる構成にしています。

---

## 目次

| No. | ノートブック | Colabで開く | データセット | 内容 |
| --- | --- | --- | --- | --- |
| 1 | [`1_Keras_MLP.ipynb`](https://github.com/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras/blob/main/1_Keras_MLP.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras/blob/main/1_Keras_MLP.ipynb) | Fashion-MNIST | 多重パーセプトロン（Sequential API）：`Dense`層の積み方、`compile`・`fit`・`evaluate`の基本、学習曲線・混同行列の見方 |
| 2 | [`2_Keras_CNN_Sequential-API.ipynb`](https://github.com/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras/blob/main/2_Keras_CNN_Sequential-API.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras/blob/main/2_Keras_CNN_Sequential-API.ipynb) | Fashion-MNIST | 畳み込みニューラルネットワーク（Sequential API）：`Conv2D`・`MaxPooling2D`、MLPとの精度比較 |
| 3 | [`2_Keras_CNN_Functional-API.ipynb`](https://github.com/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras/blob/main/2_Keras_CNN_Functional-API.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras/blob/main/2_Keras_CNN_Functional-API.ipynb) | Fashion-MNIST | 同じCNNをFunctional APIで実装：Sequential編とのパラメータ数一致検証、書き方の比較 |
| 4 | [`3_Keras_AE_Sequential-API.ipynb`](https://github.com/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras/blob/main/3_Keras_AE_Sequential-API.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras/blob/main/3_Keras_AE_Sequential-API.ipynb) | MNIST | オートエンコーダ（Sequential API）：教師なし学習、潜在空間、PCAとの再構成精度比較 |
| 5 | [`3_Keras_AE_Functional-API.ipynb`](https://github.com/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras/blob/main/3_Keras_AE_Functional-API.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras/blob/main/3_Keras_AE_Functional-API.ipynb) | MNIST | 同じAEをFunctional APIで実装：Sequential編・PCAとの3者比較 |
| 6 | [`2_Keras_VAE_Functional-API.ipynb`](https://github.com/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras/blob/main/2_Keras_VAE_Functional-API.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras/blob/main/2_Keras_VAE_Functional-API.ipynb) | MNIST → Fashion-MNIST | 変分オートエンコーダ（Functional API）：再パラメータ化トリック・KLダイバージェンス損失・`train_step`のサブクラス化、生成画像の可視化 |

ノートブックは1→6の順に読み進めることを想定しています。1〜3（MLP・CNN）は分類（教師あり学習）、4〜6（AE・VAE）は教師なし学習・生成モデルという2部構成です。2・3（CNN）と4・5（AE）はそれぞれ同じモデルをSequential API・Functional APIの2通りで実装しており、「同じ結果が書き方を変えても得られること」を実際に確認できるようにしています。6（VAE）は4・5のAEをさらに発展させた内容で、5を読んだ後に進むことを前提としています。

---

## 対象読者・前提知識

- Python基礎文法（変数、関数、for文、リストなど）を理解している方
- Keras・TensorFlow・ニューラルネットワークの構築に初めて触れる方
- 統計学・機械学習の予備知識は前提としません。ニューラルネットワークの基本概念（損失関数、勾配降下法など）は各ノートブック内でその都度説明します

## 動作環境

- Python 3.10以降
- TensorFlow 2.16以降・Keras 3系（`tensorflow.keras`は現在Keras 3を指します。旧世代のtf.keras（Keras 2系）とは`Sequential`モデルの`.input`/`.output`アクセスなど一部の挙動が異なるため、本シリーズはKeras 3を前提に書かれています）
- scikit-learn（PCA比較、4・5番のAE編で使用）
- `matplotlib_fontja`（日本語フォント表示用）
- GPUがなくてもCPUで動作しますが、6番（VAE）は本シリーズで最も学習に時間がかかります

いずれもGoogle Colabの標準環境（2026年時点）であれば満たされます。

## 使い方（Google Colab）

各ノートブックの冒頭にある「教材への接続」セルを実行すると、このリポジトリを自動的にクローンし、共通ヘルパー（`module/`）を読み込む準備が整います。

```
!git clone https://github.com/ARIM-ACADEMY-2026/Advanced_Tutorial_3_Keras.git
%cd Advanced_Tutorial_3_Keras
```

ローカル環境でこのフォルダを直接開いている場合は、このセルを実行する必要はありません（すでにカレントディレクトリが正しい場所になっています）。

> **注意（ノートブックをまたぐ受け渡しについて）：** 3番（CNN Functional編）は2番（CNN Sequential編）が保存した学習済みモデルを、5番（AE Functional編）は4番（AE Sequential編）が保存した結果を、`output/`フォルダ経由で読み込んで比較します。Google Colabは開くたびに新しいセッション（まっさらな実行環境）になるため、この比較を行うには**同じColabセッション内で、先に2番→3番、4番→5番の順に実行してください**。未実行のまま3番・5番だけを実行した場合もエラーにはならず、「比較をスキップしました」というメッセージが表示されるだけです。

---

## データセットと出典

### Fashion-MNIST（1・2・3番、6番の5節）

10種類の衣料品（Tシャツ、ズボン、コート、スニーカーなど）の28×28グレースケール画像（訓練6万枚・テスト1万枚）。手書き数字のMNISTより難易度の高い代替データセットとして、Zalando Researchが作成・公開しました（MITライセンス）。`keras.datasets.fashion_mnist`から取得します。
> Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. <https://github.com/zalandoresearch/fashion-mnist>

### MNIST（4・5番、6番の2〜4節）

手書き数字（0〜9）の28×28グレースケール画像（訓練6万枚・テスト1万枚）。画像分類・教師なし学習の題材として最も広く使われるデータセットの1つです。`keras.datasets.mnist`から取得します。
> LeCun, Y., Cortes, C., & Burges, C. J. (1998). The MNIST database of handwritten digits. <http://yann.lecun.com/exdb/mnist/>

いずれのデータセットも、各ノートブックの初回実行時にKeras経由でインターネットから自動ダウンロードされ、`~/.keras/`にキャッシュされます（2回目以降はキャッシュから読み込むため高速です）。

---

## 共通モジュール（`module/`）

- `seed_utils.py`: 乱数シード固定（`set_seed(42)`）
- `data_utils.py`: MNIST/Fashion-MNISTの読み込み・前処理、学習履歴の保存/読込
- `viz_utils.py`: 学習曲線・混同行列・潜在空間散布図・再構成画像グリッドなどの描画ヘルパー

いずれも「定型処理」のみを切り出したもので、モデル構築・学習コードは各ノートブックのセルに残しています（このシリーズが教えたい内容そのものであるため）。

## 横断比較（`comparison_output/`）

- `classification_comparison.csv`: MLP・CNN(Sequential/Functional)のFashion-MNISTでのテスト精度比較。CNNノートブック実行時に自動生成・追記
- `reconstruction_comparison.csv`: AE(Sequential/Functional)とPCAの再構成MSE比較。AEノートブック実行時に自動生成・追記

これらは1冊のノートブックに閉じない、複数ノートブックをまたいだ成果物のため、各ノートブックの`output/`ではなく、このリポジトリの一段上（クローン先フォルダの外側）にある共有フォルダに保存しています。3番（CNN Functional編）は2番（CNN Sequential編）が保存した学習済みモデル（`output/cnn_seq_trained.keras`）を、5番（AE Functional編）は4番（AE Sequential編）が保存した結果（`output/ae_seq_results.npz`）を、同じ`output/`フォルダから読み込むため、実行する場合は各ペアのSequential編を先に実行してください（未実行でもエラーにはならず、比較をスキップする旨のメッセージが出ます）。

---

## ライセンス

各ノートブックのコード部分はMITライセンスで提供します。Fashion-MNIST・MNISTのライセンス・利用条件は上記の出典元に従ってください。

## 更新履歴

- 2026-08-03: 初学者向けの解説充実・ヘルパー関数のモジュール化・セルの細分化を行い全面再構成（1冊=1ノートブックの構成、`module/`によるヘルパー切り出し）
- 2026-08-03: ユーザー環境での実行時に見つかった不具合を修正——(1) `Sequential`モデルに`input_shape=`kwargを渡す旧式の書き方が、Keras 3では`.input`/`get_layer().output`アクセスで`AttributeError`になる問題（MLP・CNN Sequential・AE Sequential編で修正）、(2) CNN Functional編・AE Functional編が、存在しないノートブック別サブフォルダを参照していたパスの誤り（共有の`output/`フォルダを見るよう修正）
- 2026-08-03: Google Colabで開けるよう各ノートブックにセットアップセル（`git clone`＋`%cd`）とColabバッジを追加し、本READMEを新規作成

各ノートブックの詳細な変更点は、ノートブック内の記述および`reference/known-corrections.md`（プロジェクト側の記録）を参照してください。
