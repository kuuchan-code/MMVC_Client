# MMVC クライアント

## 概要

**MMVC クライアント**は、[MMVC (Many to Many Voice Conversion)](https://github.com/isletennos/MMVC_Trainer) のリアルタイム音声変換用クライアントです。このツールは、Windows環境でONNXモデルを使用して、入力音声をリアルタイムで変換し、変換後の音声を出力するアプリケーションです。

本プロジェクトは、[isletennos/MMVC_Client](https://github.com/isletennos/MMVC_Client) をフォークし、Rustに置き換えることで高速化を図っています。また、GUI（グラフィカルユーザーインターフェース）を追加し、設定や操作がより直感的になりました。

## 特徴

* **リアルタイム処理**: 入力音声を即座に処理し、変換後の音声をリアルタイムで出力します。
* **ONNXモデル対応**: 任意のONNX形式の音声変換モデルを使用可能。
* **柔軟な設定**: サンプルレート、バッファサイズなど、複数のパラメータをGUIから調整可能。
* **SOLAアルゴリズム**: スムーズな音声クロスフェードを実現するSOLA（Synchronous Overlap-Add）アルゴリズムの実装。
* **GUIサポート**: 簡単なクリック操作で、モデルの選択、入力・出力デバイスの設定、変換パラメータの変更が可能。

## 必要なファイル
このクライアントを使用するためには、[MMVC_Trainer v1.3](https://github.com/isletennos/MMVC_Trainer) で訓練したONNXファイルが必要です。事前にモデルを訓練し、ONNX形式でエクスポートしてください。

## 動作確認済み環境
[MMVC_Trainer v1.3.2.11](https://github.com/isletennos/MMVC_Trainer/releases/tag/v1.3.2.11)で訓練したモデルで動作確認済み。

### ダウンロード手順

1. [リリースページ](https://github.com/kuuchan-code/MMVC_Client/releases)にアクセスします。
2. 最新バージョンのZIPファイルをダウンロードします。
3. ダウンロードしたZIPファイルを解凍します。

## 使い方

### GUIの使用

1. `mmvc_client.exe` をダブルクリックしてアプリケーションを起動します。
2. 画面上部の「モデル選択」ボタンをクリックして、使用するONNXモデルファイルを選択します。
3. 「ソースID」と「ターゲットID」のフィールドに、それぞれのスピーカーIDを入力します。
4. 「入力デバイス」と「出力デバイス」のドロップダウンから、使用するデバイスを選択します。
5. 必要に応じてサンプルレートやバッファサイズなどのパラメータを調整します。
6. 「開始」ボタンをクリックして、リアルタイム音声変換を開始します。
![スクリーンショット 2024-10-07 232041](https://github.com/user-attachments/assets/9696dd0d-0fcd-4315-80dc-2b52b5c668ee)


## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は LICENSE ファイルを参照してください。

本プロジェクトは以下のライセンスも遵守しています：

* [isletennos/MMVC_Client](https://github.com/isletennos/MMVC_Client) - MIT License
* [ONNX Runtime](https://github.com/microsoft/onnxruntime) - MIT License

### フォントライセンス

本プロジェクトには、NotoSansJP-Regular.ttf フォントが含まれています。このフォントは、SIL Open Font License (OFL) バージョン1.1に基づいて使用されています。詳細なライセンス情報については、`OFL.txt` ファイルを参照してください。

## 作者

* **ku-chan** - [kuuchan-code](https://github.com/kuuchan-code)

## お問い合わせ

ご質問やフィードバックがありましたら、[issues](https://github.com/kuuchan-code/MMVC_Client/issues)に投稿してください。
