"""図の体裁を整えるための描画ヘルパー関数集。

MLP・CNN（Sequential/Functional）・VAE・AEの6冊すべてで、
「学習曲線を描く」「混同行列を描く」「潜在空間の散布図を描く」
「元画像と再構成画像を並べて表示する」という同じ種類の作図が繰り返し登場します。
これらは各ノートブックが教えたいKerasの概念（層の積み方、損失関数の設計など）
そのものではなく、matplotlib/seabornの体裁を整えるだけの定型処理なので、
このファイルに集約しました。

一方で、「何を描くか」（潜在空間とは何か、再構成画像とは何か、など）は
各ノートブックのマークダウンとコードで説明しています。関数の中身を読まなくても
呼び出し方さえ分かれば使えるように、docstringに使用例を書いています。
"""

from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


def _save_if_needed(fig, output_dir, filename) -> None:
    """output_dirが指定されていれば、図をPNGとして保存する（内部ヘルパー）。"""
    if output_dir is not None and filename is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / filename, dpi=150, bbox_inches="tight")


def plot_metric_curve(
    history,
    metric: str = "loss",
    fig_num: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
) -> None:
    """``model.fit()``が返す``history``から、指定した指標の学習曲線を描く。

    訓練データ（黒線）と検証・テストデータ（赤線）の両方をプロットし、
    過学習（訓練だけ良くなり続け、検証が頭打ち・悪化する現象）の有無を
    目で確認できるようにします。

    Args:
        history: ``model.fit()``の戻り値（``keras.callbacks.History``）。
        metric: プロットする指標名。``'loss'``や``'accuracy'``など。
            検証データ側は``'val_' + metric``というキー名で自動的に探す。
        fig_num: 図番号（例: 3 なら「図3」とタイトルに付す）。省略可。
        output_dir: 指定するとこのフォルダにPNGを保存する（例: ``OUTPUT_DIR``）。
        filename: 保存するファイル名。省略時は指標名から自動生成する。

    使用例:
        >>> plot_metric_curve(history, metric="loss", fig_num=3, output_dir=OUTPUT_DIR)
        >>> plot_metric_curve(history, metric="accuracy", fig_num=4, output_dir=OUTPUT_DIR)
    """
    if metric not in history.history:
        raise KeyError(
            f"'{metric}' is not in history.history. "
            f"Available keys: {list(history.history.keys())}"
        )

    train_values = history.history[metric]
    val_key = f"val_{metric}"

    fig = plt.figure()
    plt.plot(train_values, "black", label="training")
    if val_key in history.history:
        plt.plot(history.history[val_key], "red", label="test")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    if fig_num is not None:
        plt.title(f"図{fig_num}: {metric} の学習曲線")
    plt.legend()

    _save_if_needed(fig, output_dir, filename or f"fig{fig_num or ''}_{metric}_curve.png")
    plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: Optional[Sequence[str]] = None,
    fig_num: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
):
    """正解ラベルと予測ラベルから混同行列を計算し、ヒートマップとして描く。

    ``confusion_matrix``の引数は「正解が先・予測が後」という順序が
    scikit-learnの規約です（このシリーズのコーディング規約でも明記）。
    この関数の引数もそれに合わせて``y_true, y_pred``の順にしています。

    Args:
        y_true: 正解クラスの配列（1次元、整数ラベル）。
        y_pred: 予測クラスの配列（1次元、整数ラベル）。
        class_names: 軸に表示するクラス名。省略時は数字のまま表示する。
        fig_num: 図番号。省略可。
        output_dir: 指定するとこのフォルダにPNGを保存する。
        filename: 保存するファイル名。省略時は自動生成する。

    Returns:
        numpy.ndarray: 計算した混同行列（後段で数値として使いたい場合用）。
    """
    from sklearn.metrics import confusion_matrix

    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        square=True,
        annot=True,
        cmap="jet",
        fmt=".0f",
        xticklabels=class_names if class_names is not None else "auto",
        yticklabels=class_names if class_names is not None else "auto",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if fig_num is not None:
        plt.title(f"図{fig_num}: 混同行列")

    _save_if_needed(fig, output_dir, filename or f"fig{fig_num or ''}_confusion_matrix.png")
    plt.show()
    return cm


def plot_latent_scatter(
    z: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    fig_num: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
) -> None:
    """2次元の潜在表現``z``を、ラベルで色分けした散布図として描く。

    VAE・AEどちらも「784次元の画像を2次元に圧縮した空間で、
    同じクラスの画像同士が近くに集まるか」を確認するのに使う定番の図です。

    Args:
        z: 形状``(n_samples, 2)``の潜在座標。
        labels: 各点のクラスラベル（整数、色分けに使う）。
        class_names: カラーバーに表示するクラス名（0, 1, 2, ... の順）。
        fig_num: 図番号。省略可。
        output_dir: 指定するとこのフォルダにPNGを保存する。
        filename: 保存するファイル名。省略時は自動生成する。
    """
    fig = plt.figure(figsize=(10, 8))
    sc = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap="viridis", s=8)
    if class_names is not None:
        cbar = plt.colorbar(sc, ticks=range(len(class_names)))
        cbar.ax.set_yticklabels(class_names)
    else:
        plt.colorbar(sc)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    if fig_num is not None:
        plt.title(f"図{fig_num}: 潜在空間の分布")

    _save_if_needed(fig, output_dir, filename or f"fig{fig_num or ''}_latent_scatter.png")
    plt.show()


def plot_latent_manifold(
    decode_fn: Callable[[np.ndarray], np.ndarray],
    n: int = 30,
    digit_size: int = 28,
    scale: float = 1.0,
    fig_num: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
) -> None:
    """2次元潜在空間を格子状にサンプリングし、デコード結果を敷き詰めて表示する。

    VAEの潜在空間が「なめらかに意味が変化する空間」になっているかを
    確認するための可視化です（潜在空間の連続性はVAEの重要な性質の一つ）。

    Args:
        decode_fn: 形状``(1, 2)``の潜在座標を受け取り、
            ``digit_size × digit_size``の画像を返す関数
            （例: ``lambda z: vae.decoder.predict(z, verbose=0)[0]``）。
        n: 縦横それぞれ何点サンプリングするか（``n × n``枚を並べる）。
        digit_size: 1枚あたりの画像の一辺のピクセル数。
        scale: 潜在座標をサンプリングする範囲（``-scale``から``scale``）。
        fig_num: 図番号。省略可。
        output_dir: 指定するとこのフォルダにPNGを保存する。
        filename: 保存するファイル名。省略時は自動生成する。
    """
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            digit = decode_fn(z_sample).reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    fig = plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    plt.xticks(pixel_range, np.round(grid_x, 1))
    plt.yticks(pixel_range, np.round(grid_y, 1))
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    if fig_num is not None:
        plt.title(f"図{fig_num}: 潜在空間から復元した画像の格子")
    plt.imshow(figure, cmap="Greys_r")

    _save_if_needed(fig, output_dir, filename or f"fig{fig_num or ''}_latent_manifold.png")
    plt.show()


def plot_reconstruction_grid(
    originals: np.ndarray,
    reconstructions: Union[np.ndarray, dict],
    n: int = 10,
    image_shape: Sequence[int] = (28, 28),
    fig_num: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
) -> None:
    """元画像と、1つまたは複数の手法による再構成画像を並べて比較表示する。

    AE（オートエンコーダ）とPCAのように、複数の圧縮・復元手法を
    同じ元画像に対して見比べたい場合は、``reconstructions``に
    ``{"AE": ae_result, "PCA": pca_result}``のような辞書を渡す。

    Args:
        originals: 元画像。形状``(n_samples, H*W)``または``(n_samples, H, W)``。
        reconstructions: 再構成画像の配列、または手法名をキーとした辞書。
        n: 表示する画像の枚数。
        image_shape: 1枚の画像の``(高さ, 幅)``。
        fig_num: 図番号。省略可。
        output_dir: 指定するとこのフォルダにPNGを保存する。
        filename: 保存するファイル名。省略時は自動生成する。
    """
    if isinstance(reconstructions, dict):
        recon_dict = reconstructions
    else:
        recon_dict = {"reconstruction": reconstructions}

    row_names = ["original", *recon_dict.keys()]
    n_rows = len(row_names)

    fig = plt.figure(figsize=(20, 2.2 * n_rows))
    for row_idx, name in enumerate(row_names):
        data = originals if row_idx == 0 else recon_dict[name]
        for i in range(n):
            ax = plt.subplot(n_rows, n, row_idx * n + i + 1)
            plt.imshow(np.asarray(data[i]).reshape(image_shape))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                ax.set_ylabel(name, rotation=0, labelpad=40, va="center")
    if fig_num is not None:
        fig.suptitle(f"図{fig_num}: 元画像と再構成画像の比較")

    _save_if_needed(fig, output_dir, filename or f"fig{fig_num or ''}_reconstruction_grid.png")
    plt.show()
