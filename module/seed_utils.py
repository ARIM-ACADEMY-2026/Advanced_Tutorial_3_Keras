"""乱数シードを固定するための小さなユーティリティ。

このファイルに置いているのは「本シリーズ（MLP・CNN・VAE・AE）が教えたい内容」
そのものではなく、ノートブックを実行するたびに同じ結果を再現するための
環境設定（ボイラープレート）です。前処理やモデル構築のコードは、
学習内容そのものなので各ノートブックのセルに書いたまま残しています。
"""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """NumPy・Python標準の``random``・TensorFlowの乱数シードをまとめて固定する。

    Kerasのモデル学習では、重みの初期値・訓練データのシャッフル順序・
    Dropout層の挙動などいたるところで乱数が使われます。シードを固定しないと、
    同じコードを2回実行しただけで結果（正解率やグラフ）が微妙に変わってしまい、
    「本のとおりに実行したのに数字が違う」という混乱の原因になります。
    このシリーズでは慣例として ``seed=42`` を既定値にしています。

    Args:
        seed: 固定する乱数シードの値。

    Note:
        GPU上でのTensorFlow実行は、演算の並列実行順序に起因して
        シードを固定しても完全な再現性が保証されない場合があります
        （CPU実行であれば通常は再現します）。「必ず毎回同じ結果になる」と
        断定はできない点に注意してください。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        # TensorFlow未導入の環境でも、numpy/random部分だけは固定できるようにする
        pass
