use ort::{Environment, SessionBuilder, Value, GraphOptimizationLevel};
use ndarray::{Array, Array2, Array3, IxDyn, CowArray}; // Array2, Array3をインポート
use std::sync::Arc;
use hound; // WAVファイル処理用

// ハイパーパラメータ構造体
struct Hyperparameters {
    sample_rate: u32,
    max_wav_value: f32,
    hop_length: usize,
    dispose_stft_specs: usize,
    dispose_conv1d_specs: usize,
    overlap: usize,
    target_id: i64,
    channels: usize, // 3次元入力のためにチャネルを追加
}

impl Hyperparameters {
    fn new() -> Self {
        Hyperparameters {
            sample_rate: 24000,
            max_wav_value: 32768.0,
            hop_length: 128,
            dispose_stft_specs: 2,
            dispose_conv1d_specs: 2,
            overlap: 64,
            target_id: 2,
            channels: 80, // ここでフィルタバンクの数（例: 80）を設定
        }
    }
}

// WAVファイルをロードする関数
fn load_wav(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV file.");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "Only mono WAV files are supported.");
    reader.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

// ONNXモデルを推論する関数
fn run_onnx_model(
    session: &ort::Session,
    spec: &Array3<f32>, // 3次元配列
    spec_lengths: &Array2<i64>, // 2次元配列
    sid_src: &Array2<i64>, // 2次元配列
    sid_target: i64,
) -> Vec<f32> {
    // 各CowArrayを事前に変数として保持する
    let spec_cow = CowArray::from(spec.clone().into_dyn());
    let spec_lengths_cow = CowArray::from(spec_lengths.clone().into_dyn());
    let sid_src_cow = CowArray::from(sid_src.clone().into_dyn());
    let sid_tgt_array_cow = CowArray::from(Array::from_elem(IxDyn(&[]), sid_target).into_dyn());

    let inputs = vec![
        Value::from_array(session.allocator(), &spec_cow).unwrap(),
        Value::from_array(session.allocator(), &spec_lengths_cow).unwrap(),
        Value::from_array(session.allocator(), &sid_src_cow).unwrap(),
        Value::from_array(session.allocator(), &sid_tgt_array_cow).unwrap(),
    ];

    let outputs: Vec<Value> = session.run(inputs).unwrap();
    let audio_output: ort::tensor::OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();

    // OrtOwnedTensor を ndarray の Array に変換し、それを Vec に変換
    audio_output.view().to_owned().into_raw_vec()
}

// 音声処理関数
fn audio_trans(
    hparams: &Hyperparameters,
    session: &ort::Session,
    signal: Vec<f32>,
    target_id: i64,
) -> Vec<f32> {
    let signal_len = signal.len();
    let signal = Array::from_shape_vec((1, hparams.channels, signal_len / hparams.channels), signal).unwrap(); // 3次元配列に変換
    let spec = signal.mapv(|x| x / hparams.max_wav_value); // 正規化

    // ダミーの長さ（サンプルの長さ）
    let spec_lengths = Array::from_elem((1, 1), spec.shape()[2] as i64); // 3次元目の長さ（時間ステップ）
    let sid_src = Array::from_elem((1, 1), hparams.target_id); // 2次元配列

    // モデルの実行
    let audio = run_onnx_model(session, &spec, &spec_lengths, &sid_src, target_id);
    
    audio
}

// メイン関数
fn main() {
    // ハイパーパラメータの初期化
    let hparams = Hyperparameters::new();

    // ONNX環境とセッションの初期化
    let environment = Arc::new(Environment::builder()
        .with_name("MMVC_Client")
        .build()
        .unwrap());
    
    let session = SessionBuilder::new(&environment)
        .unwrap() // SessionBuilder を取得
        .with_optimization_level(GraphOptimizationLevel::Level3) // Optimization Level を Level3 に
        .unwrap() // Optimization Level の設定後に unwrap
        .with_model_from_file("G_best.onnx") // モデルのロード
        .unwrap();

    // WAVファイルを読み込み
    let input_wav = load_wav("emotion001.wav");

    // 音声処理を実行
    let output_wav = audio_trans(&hparams, &session, input_wav, hparams.target_id);

    // 処理結果をファイルに保存
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: hparams.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("output.wav", spec).unwrap();
    for sample in output_wav {
        writer.write_sample((sample * hparams.max_wav_value) as i16).unwrap();
    }
}
