# MMVC クライアント

## 概要

**MMVC クライアント**は、[MMVC (Many to Many Voice Conversion)](https://github.com/isletennos/MMVC_Trainer) のリアルタイム音声変換用クライアントです。このツールは、Windows環境でONNXモデルを使用して、入力音声をリアルタイムで変換し、変換後の音声を出力するアプリケーションです。

本プロジェクトは、[isletennos/MMVC_Client](https://github.com/isletennos/MMVC_Client) をフォークし、Rustに置き換えることで高速化を図っています。また、GUI（グラフィカルユーザーインターフェース）を追加し、設定や操作がより直感的になりました。

## 特徴

* **リアルタイム処理**: 入力音声を即座に処理し、変換後の音声をリアルタイムで出力します。
* **ONNXモデル対応**: 任意のONNX形式の音声変換モデルを使用可能。
* **柔軟な設定**: サンプルレート、バッファサイズ、オーバーラップ長など、複数のパラメータをGUIから調整可能。
* **SOLAアルゴリズム**: スムーズな音声クロスフェードを実現するSOLA（Synchronous Overlap-Add）アルゴリズムの実装。
* **GUIサポート**: 簡単なクリック操作で、モデルの選択、入力・出力デバイスの設定、変換パラメータの変更が可能。

## ダウンロード

最新バージョンのバイナリは以下のリンクからダウンロードできます：

* [mmvc_client_v0.2.0_x86_64_win.zip](https://github.com/kuuchan-code/MMVC_Client/releases/download/v0.2.0/mmvc_client_v0.2.0_x86_64_win.zip)

### ダウンロード手順

1. [リリースページ](https://github.com/kuuchan-code/MMVC_Client/releases)にアクセスします。
2. 最新バージョンのZIPファイルをダウンロードします。
3. ダウンロードしたZIPファイルを解凍し、実行ファイルを適当なディレクトリに配置します。

## 使い方

### GUIの使用

1. `mmvc_client.exe` をダブルクリックしてアプリケーションを起動します。
2. 画面上部の「モデル選択」ボタンをクリックして、使用するONNXモデルファイルを選択します。
3. 「ソースID」と「ターゲットID」のフィールドに、それぞれのスピーカーIDを入力します。
4. 「入力デバイス」と「出力デバイス」のドロップダウンから、使用するデバイスを選択します。
5. 必要に応じてサンプルレートやバッファサイズなどのパラメータを調整します。
6. 「開始」ボタンをクリックして、リアルタイム音声変換を開始します。

### デバイスの選択

アプリケーション起動後、デバイス選択のドロップダウンから、使用する入力および出力デバイスを選択します。選択したデバイスを設定後、「開始」ボタンをクリックすることで、リアルタイム音声変換が開始されます。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は LICENSE ファイルを参照してください。

本プロジェクトは以下のライセンスも遵守しています：

* [isletennos/MMVC_Client](https://github.com/isletennos/MMVC_Client) - MIT License
* [ONNX Runtime](https://github.com/microsoft/onnxruntime) - MIT License

## 作者

* **ku-chan** - [kuuchan-code](https://github.com/kuuchan-code)

## お問い合わせ

ご質問やフィードバックがありましたら、[issues](https://github.com/kuuchan-code/MMVC_Client/issues)に投稿してください。