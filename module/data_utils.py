"""MNIST・Fashion-MNISTの読み込み・前処理・保存/読込のための共通ヘルパー関数集。

MLP・CNN（Sequential/Functional）・VAE・AEの6冊すべてで、
「``keras.datasets``から画像を読み込む」「0-1に正規化する」
「チャネル次元を追加する、あるいは1次元に潰す」「ラベルをone-hot化する」
という同じ前処理が繰り返し登場します。これは各ノートブックが教えたい
Kerasの概念（層の組み方、損失関数の設計など）そのものではなく、
どの本でも同じ形になる定型的な前処理なので、このファイルに集約しました。

一方で、「モデルをどう組むか」「学習をどう回すか」は各ノートブックの
セルにそのまま残しています。関数の中身を読まなくても呼び出し方さえ
分かれば使えるように、docstringに使用例を書いています。

学習済みモデル・学習履歴（``history``）は、1つのトピックの中でも
「02_model」「03_train」「04_evaluate」のようにノートブックをまたいで
受け渡しします。ノートブックを実行するたびにKernelは初期化される（＝前の
ノートブックで作った変数は残っていない）ので、``output/``フォルダに
保存・読込するための関数もここにまとめています。
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Sequence, Tuple

import numpy as np

# Fashion-MNISTの10クラスのラベル名（クラス番号0〜9の順）
FASHION_MNIST_CLASS_NAMES: Sequence[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# MNIST（手書き数字）の10クラスのラベル名。数字そのものがラベルなので文字列化するだけ
MNIST_CLASS_NAMES: Sequence[str] = [str(i) for i in range(10)]


def _to_one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """整数ラベル配列をone-hot行列に変換する（内部ヘルパー）。

    ``tensorflow.keras.utils.to_categorical``と等価な処理をnumpyだけで行う。
    Kerasが未導入の環境でもデータ準備の動作確認ができるように、あえて
    TensorFlowに依存しない実装にしている。
    """
    labels = np.asarray(labels).astype(int).reshape(-1)
    one_hot = np.zeros((labels.shape[0], num_classes), dtype="float32")
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def load_fashion_mnist(
    add_channel: bool = False, one_hot: bool = True
) -> SimpleNamespace:
    """Fashion-MNISTを読み込み、正規化・（必要なら）チャネル追加・one-hot化まで行う。

    MLP編では``add_channel=False``（画像は``(28, 28)``のまま、モデル側の
    ``Flatten``層で1次元に潰す）、CNN編では``add_channel=True``
    （画像を``(28, 28, 1)``にして``Conv2D``に渡せる形にする）を指定する。

    Args:
        add_channel: Trueならグレースケールのチャネル次元（末尾に1）を追加する。
        one_hot: Trueならラベルをone-hot行列にする（``to_categorical``相当）。
            Falseなら整数ラベル（0〜9）のまま返す。

    Returns:
        SimpleNamespace: 以下の属性を持つ名前空間。
            - ``train_images``, ``test_images``: 0〜1に正規化された``float32``配列
            - ``train_labels``, ``test_labels``: one-hotまたは整数ラベル
            - ``train_labels_int``, ``test_labels_int``: 常に整数ラベル
              （混同行列や散布図の色分けなど、one-hotでは扱いにくい場面用）
            - ``class_names``: クラス名のリスト（``FASHION_MNIST_CLASS_NAMES``）

    使用例:
        >>> data = load_fashion_mnist(add_channel=True)
        >>> model.fit(data.train_images, data.train_labels, ...)

    コラム：なぜこの関数は「正規化」までしかしないのか
        画像を0〜1に正規化する（``/ 255.0``）のは、どのモデルでも必ず必要な
        定型処理です。一方で「どんな層を積むか」はノートブックごとに違う、
        まさに教えたい内容そのものなので、あえてこの関数には含めていません。
        「前処理はどこまで共通化し、どこから先を自分のコードとして書くか」
        という線引きは、実務のコードを書くときにも役立つ判断です。
    """
    from tensorflow import keras

    (train_images, train_labels_int), (test_images, test_labels_int) = (
        keras.datasets.fashion_mnist.load_data()
    )

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    if add_channel:
        train_images = np.expand_dims(train_images, -1)
        test_images = np.expand_dims(test_images, -1)

    if one_hot:
        train_labels = _to_one_hot(train_labels_int)
        test_labels = _to_one_hot(test_labels_int)
    else:
        train_labels = train_labels_int
        test_labels = test_labels_int

    return SimpleNamespace(
        train_images=train_images,
        test_images=test_images,
        train_labels=train_labels,
        test_labels=test_labels,
        train_labels_int=train_labels_int,
        test_labels_int=test_labels_int,
        class_names=list(FASHION_MNIST_CLASS_NAMES),
    )


def load_mnist_flat() -> SimpleNamespace:
    """MNIST（手書き数字）を読み込み、正規化して1次元ベクトル（784次元）に潰す。

    オートエンコーダ（AE）編で使う。AEは教師なし学習（入力自身を再構成する）
    なので、ラベルは可視化の色分けにしか使わない。したがって元のノートブックに
    あった「ラベルをone-hot化してから``argmax``で整数に戻す」という
    往復処理（意味のない計算の無駄）は行わず、最初から整数ラベルのまま返す。

    Returns:
        SimpleNamespace: 以下の属性を持つ名前空間。
            - ``train_images``, ``test_images``: 形状``(N, 784)``、0〜1に正規化済み
            - ``train_labels``, ``test_labels``: 整数ラベル（0〜9）。可視化の色分け専用
            - ``class_names``: ``MNIST_CLASS_NAMES``

    使用例:
        >>> data = load_mnist_flat()
        >>> autoencoder.fit(data.train_images, data.train_images, ...)  # 入力=正解
    """
    from tensorflow import keras

    (train_images, train_labels), (test_images, test_labels) = (
        keras.datasets.mnist.load_data()
    )

    train_images = train_images.reshape(train_images.shape[0], -1).astype("float32") / 255.0
    test_images = test_images.reshape(test_images.shape[0], -1).astype("float32") / 255.0

    return SimpleNamespace(
        train_images=train_images,
        test_images=test_images,
        train_labels=train_labels,
        test_labels=test_labels,
        class_names=list(MNIST_CLASS_NAMES),
    )


def load_digits_for_vae() -> SimpleNamespace:
    """VAE編用に、MNIST（手書き数字）を画像形式のまま読み込む。

    KerasのVAE公式サンプル（https://keras.io/examples/generative/vae/）に
    ならい、訓練データとテストデータを連結した``all_digits``（生成モデルの
    学習には「未知データでの汎化性能」ではなく「データ全体の分布をどれだけ
    よく捉えられるか」が重要なため、分割せず全件を使う）と、潜在空間の
    可視化専用に訓練データ単体（ラベル付き）の両方を返す。

    Returns:
        SimpleNamespace:
            - ``all_digits``: 訓練+テストを連結した形状``(70000, 28, 28, 1)``の画像
            - ``train_images``: 訓練データのみ、形状``(60000, 28, 28, 1)``
            - ``train_labels``: 訓練データのラベル（整数、可視化の色分け専用）
            - ``class_names``: ``MNIST_CLASS_NAMES``
    """
    from tensorflow import keras

    (train_images, train_labels), (test_images, _test_labels) = (
        keras.datasets.mnist.load_data()
    )

    all_digits = np.concatenate([train_images, test_images], axis=0)
    all_digits = np.expand_dims(all_digits, -1).astype("float32") / 255.0
    train_images = np.expand_dims(train_images, -1).astype("float32") / 255.0

    return SimpleNamespace(
        all_digits=all_digits,
        train_images=train_images,
        train_labels=train_labels,
        class_names=list(MNIST_CLASS_NAMES),
    )


def load_fashion_mnist_for_vae() -> SimpleNamespace:
    """VAE編の後半（Fashion-MNISTへの転用）用のデータ読み込み関数。

    ``load_digits_for_vae``と対になる関数で、Fashion-MNIST版。
    """
    from tensorflow import keras

    (train_images, train_labels), (test_images, _test_labels) = (
        keras.datasets.fashion_mnist.load_data()
    )

    all_images = np.concatenate([train_images, test_images], axis=0)
    all_images = np.expand_dims(all_images, -1).astype("float32") / 255.0
    train_images = np.expand_dims(train_images, -1).astype("float32") / 255.0

    return SimpleNamespace(
        all_images=all_images,
        train_images=train_images,
        train_labels=train_labels,
        class_names=list(FASHION_MNIST_CLASS_NAMES),
    )


def save_history(output_dir, filename: str, history) -> Path:
    """``model.fit()``の戻り値``history``を、後続のノートブックのために保存する。

    Kerasの``History``オブジェクトはPythonプロセスをまたいで保存できないため、
    中身の辞書（``history.history``）だけを``.npz``形式で保存する。

    Args:
        output_dir: 保存先フォルダ（例: ``OUTPUT_DIR``）。
        filename: 拡張子込みのファイル名（例: ``"history_mlp.npz"``）。
        history: ``model.fit()``の戻り値、または``history.history``相当の辞書。

    Returns:
        Path: 保存したファイルのパス。

    使用例:
        >>> history = model.fit(...)
        >>> save_history(OUTPUT_DIR, "history.npz", history)
    """
    h = history.history if hasattr(history, "history") else history
    path = Path(output_dir) / filename
    np.savez(path, **h)
    return path


def load_history(path) -> SimpleNamespace:
    """``save_history``で保存した``.npz``を読み込み、``history.history``と同じ形で返す。

    ``module.viz_utils``の各プロット関数は``history.history[指標名]``という
    形でアクセスするため、``SimpleNamespace(history=...)``で同じインタフェースに
    揃えている（呼び出し側はKerasの``History``オブジェクトと区別せずに使える）。

    Args:
        path: ``save_history``で保存したファイルのパス（``.npz``拡張子は省略可）。

    Returns:
        SimpleNamespace: ``.history``属性に指標名→値のリストの辞書を持つ。

    使用例:
        >>> history = load_history(OUTPUT_DIR / "history.npz")
        >>> viz_utils.plot_metric_curve(history, metric="loss", fig_num=3)
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".npz")
    data = np.load(path)
    return SimpleNamespace(history={key: data[key].tolist() for key in data.files})
