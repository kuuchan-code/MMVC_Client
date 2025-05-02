# MMVC クライアント

## 概要

**MMVC クライアント**は、[MMVC (Many to Many Voice Conversion)](https://github.com/isletennos/MMVC_Trainer) のリアルタイム音声変換用クライアントです。このツールは、ONNXモデルを使用して、入力音声をリアルタイムで変換し、変換後の音声を出力するアプリケーションです。

本プロジェクトは、[isletennos/MMVC_Client](https://github.com/isletennos/MMVC_Client) をフォークし、Rustに置き換えることで高速化を図っています。また、eguiを使用したモダンなGUIを実装し、設定や操作がより直感的になりました。

## 主な機能

* **リアルタイム音声変換**: 入力音声を即座に処理し、変換後の音声をリアルタイムで出力
* **ONNXモデル対応**: MMVC_Trainerで訓練したONNX形式の音声変換モデルを使用可能
* **SOLAアルゴリズム**: スムーズな音声クロスフェードを実現するSOLA（Synchronous Overlap-Add）アルゴリズムの実装
* **カスタマイズ可能なパラメータ**:
  * サンプルレート（8000Hz〜48000Hz）
  * バッファサイズ（2048〜16384サンプル）
  * カットオフフィルター（1Hz〜300Hz）
  * ソース/ターゲットスピーカーID
* **遅延モニタリング**: 処理遅延をリアルタイムで表示し、最適な設定をサポート
* **マルチプラットフォーム対応**: Windows/Linux/macOSで動作

## システム要件

* **OS**: Windows 10/11, Linux, macOS
* **CPU**: 64ビットプロセッサ
* **メモリ**: 4GB以上推奨
* **GPU**: NVIDIA GPU（CUDA対応）推奨
* **オーディオ**: 入力/出力デバイス

## インストール方法

### バイナリのダウンロード

1. [リリースページ](https://github.com/kuuchan-code/MMVC_Client/releases)にアクセス
2. 最新バージョンのZIPファイルをダウンロード
3. ZIPファイルを解凍

### ビルド方法

```bash
# 依存関係のインストール
cargo install cargo-build-deps

# ビルド
cargo build --release
```

## 使い方

1. アプリケーションを起動
2. 「ONNXモデルファイル」ボタンをクリックして、使用するモデルを選択
3. 以下のパラメータを設定:
   * モデルのサンプルレート
   * ソーススピーカーID
   * ターゲットスピーカーID
   * 入力/出力デバイス
   * カットオフフィルター（必要な場合）
   * バッファサイズ
4. 「開始」ボタンをクリックして音声変換を開始
5. 「停止」ボタンで処理を終了

![MMVC Client Screenshot](https://github.com/user-attachments/assets/9696dd0d-0fcd-4315-80dc-2b52b5c668ee)

### パラメータの調整ガイド

* **バッファサイズ**: 
  * 小さい値: 遅延が少ないが、音質が低下する可能性あり
  * 大きい値: 音質は良いが、遅延が増加
  * 推奨: 6144サンプル（デフォルト）

* **カットオフフィルター**:
  * 声の基本周波数（約80Hz〜300Hz）より低い値に設定
  * ノイズや不要な低音を除去可能

## トラブルシューティング

* **遅延が大きい場合**:
  * バッファサイズを小さくする
  * GPUが使用可能な場合は、CUDAが有効になっていることを確認

* **音質が悪い場合**:
  * バッファサイズを大きくする
  * カットオフフィルターの設定を調整

* **エラーが発生する場合**:
  * 入力/出力デバイスが正しく選択されているか確認
  * ONNXモデルが正しいバージョンか確認

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は LICENSE ファイルを参照してください。

### 使用ライブラリのライセンス

* [ONNX Runtime](https://github.com/microsoft/onnxruntime) - MIT License
* [egui](https://github.com/emilk/egui) - MIT License
* [cpal](https://github.com/RustAudio/cpal) - Apache License 2.0

### フォントライセンス

本プロジェクトには、NotoSansJP-Regular.ttf フォントが含まれています。このフォントは、SIL Open Font License (OFL) バージョン1.1に基づいて使用されています。

## 作者

* **ku-chan** - [kuuchan-code](https://github.com/kuuchan-code)

## お問い合わせ

ご質問やフィードバックがありましたら、[issues](https://github.com/kuuchan-code/MMVC_Client/issues)に投稿してください。
