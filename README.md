# MMVC クライアント

## 概要

**MMVC クライアント**は、[MMVC (Many to Many Voice Conversion)](https://github.com/isletennos/MMVC_Trainer) のリアルタイム音声変換用クライアントです。このツールは、Windows環境でONNXモデルを使用して、入力音声をリアルタイムで変換し、変換後の音声を出力するアプリケーションです。

本プロジェクトは、[isletennos/MMVC_Client](https://github.com/isletennos/MMVC_Client) をフォークし、Rustに置き換えることで高速化を図っています。

## 特徴

* **リアルタイム処理**: 入力音声を即座に処理し、変換後の音声をリアルタイムで出力します。
* **ONNXモデル対応**: 任意のONNX形式の音声変換モデルを使用可能。
* **柔軟な設定**: サンプルレート、バッファサイズ、オーバーラップ長など、複数のパラメータをコマンドラインから調整可能。
* **SOLAアルゴリズム**: スムーズな音声クロスフェードを実現するSOLA（Synchronous Overlap-Add）アルゴリズムの実装。

## ダウンロード

最新バージョンのバイナリは以下のリンクからダウンロードできます：

* [mmvc_client_v0.1.1_x86_64_win.zip](https://github.com/kuuchan-code/MMVC_Client/releases/download/v0.1.1/mmvc_client_v0.1.1_x86_64_win.zip)

### ダウンロード手順

1. [リリースページ](https://github.com/kuuchan-code/MMVC_Client/releases)にアクセスします。
2. 最新バージョンのZIPファイルをダウンロードします。
3. ダウンロードしたZIPファイルを解凍し、実行ファイルを適当なディレクトリに配置します。

## 使い方

### コマンドライン引数

以下のオプションを使用してアプリケーションを実行できます。

| オプション | 短縮形 | 説明 | デフォルト値 |
| --- | --- | --- | --- |
| `--model` | `-m` | 使用するONNXモデルファイルのパス。必須項目。 | - |
| `--source` | `-s` | ソーススピーカーのID。必須項目。 | - |
| `--target` | `-t` | ターゲットスピーカーのID。必須項目。 | - |
| `--model_sample_rate` | `-r` | モデルへ入力するオーディオのサンプルレート（Hz）。 | `24000` |
| `--buffer_size` | `-b` | バッファサイズ。音声処理の単位となるサンプル数。 | `6144` |
| `--overlap_length` | `-l` | SOLAアルゴリズムのオーバーラップ長。 | `1024` |
| `--cutoff` |  | カットオフフィルターを有効にする。 | 無効 |
| `--cutoff_freq` |  | カットオフ周波数をHzで指定。フィルターが有効な場合に使用。 | `150.0` |
| `--input` | `-i` | 入力オーディオデバイスの番号。指定しない場合は選択を促される。 | - |
| `--output` | `-o` | 出力オーディオデバイスの番号。指定しない場合は選択を促される。 | - |

### 実行例

以下のコマンドは、指定されたONNXモデルを使用して音声をリアルタイムで変換します。`--input`および`--output`オプションを指定しない場合、実行時に利用可能なオーディオデバイスのリストが表示され、ユーザーに選択を促します。

```bash
mmvc_client.exe \
    --model path/to/model.onnx \
    --source 0 \
    --target 107 \
    --model_sample_rate 48000 \
    --buffer_size 8192 \
    --overlap_length 2048 \
    --cutoff \
    --cutoff_freq 200.0 \
    --input 0 \
    --output 1
```

* `--model`: 使用するONNXモデルのパスを指定します。
* `--source`: ソーススピーカーのIDを指定します。
* `--target`: ターゲットスピーカーのIDを指定します。
* `--model_sample_rate`: モデルへの入力用サンプルレートを指定します（例：48000 Hz）。
* `--buffer_size`: バッファサイズを指定します（例：8192）。
* `--overlap_length`: SOLAアルゴリズムのオーバーラップ長を指定します（例：2048）。
* `--cutoff`: カットオフフィルターを有効にします。
* `--cutoff_freq`: カットオフ周波数を指定します（例：200.0 Hz）。
* `--input`: 入力オーディオデバイスの番号を指定します。
* `--output`: 出力オーディオデバイスの番号を指定します。

### デバイスの選択

`--input`および`--output`オプションを指定しない場合、プログラムは実行時に利用可能なオーディオデバイスのリストを表示し、ユーザーに選択を促します。

```bash
入力デバイス:
  0: マイクロフォン (Realtek High Definition Audio)
  1: ステレオミキサー (Realtek High Definition Audio)
入力デバイスの番号を選択してください:
```

同様に、出力デバイスも選択します。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。オリジナルの著作権は[Isle Tennos](https://github.com/isletennos/MMVC_Client)に帰属します。詳細は以下の内容を参照してください。

```sql
The MIT License (MIT)

Copyright (c) 2022 Isle Tennos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

また、本プロジェクトでは以下の著作権とライセンスも適用されています。

```sql
The MIT License (MIT)

Copyright (c) 2019, Tim Sainburg

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```sql
The MIT License (MIT)

Copyright (c) 2021 Jaehyeon Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
