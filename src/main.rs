use hound;
use ndarray::{Array, Array1, Array2, Array3, CowArray, IxDyn}; // IxDyn, CowArrayを追加
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc; // WAVファイル処理用

// ハイパーパラメータ構造体
struct Hyperparameters {
    sample_rate: u32,
    max_wav_value: f32,
    filter_length: usize, // FFTのウィンドウサイズ
    hop_length: usize,
    channels: usize, // 257
    target_id: i64,  // target_id フィールドを追加
}

impl Hyperparameters {
    fn new() -> Self {
        Hyperparameters {
            sample_rate: 24000,
            max_wav_value: 32768.0,
            filter_length: 512, // Pythonコードに基づく
            hop_length: 128,
            channels: 257, // モデルのチャネル数（257）
            target_id: 2,  // 任意の話者ID
        }
    }
}

// WAVファイルをロードする関数
fn load_wav(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV file.");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "Only mono WAV files are supported.");
    reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

// 音声信号にパディングを追加
fn pad_signal(signal: &mut Vec<f32>, filter_length: usize, hop_length: usize) {
    let pad_size = (filter_length - hop_length) / 2;
    let mut padded_signal = vec![0.0; pad_size];
    padded_signal.extend(signal.iter());
    padded_signal.extend(vec![0.0; pad_size]);
    *signal = padded_signal;
}

// STFTを実行
fn stft(signal: &Vec<f32>, hparams: &Hyperparameters) -> Array3<f32> {
    let fft_size = hparams.filter_length;
    let hop_size = hparams.hop_length;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);

    let num_frames = (signal.len() - fft_size) / hop_size + 1;
    let mut spectrogram = Array3::<f32>::zeros((1, hparams.channels, num_frames));

    let window: Vec<f32> = (0..fft_size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / fft_size as f32).cos()))
        .collect();

    for (i, frame) in signal.windows(fft_size).step_by(hop_size).enumerate() {
        let mut buffer: Vec<Complex<f32>> = frame
            .iter()
            .zip(&window)
            .map(|(x, w)| Complex::new(x * w, 0.0))
            .collect();
        fft.process(&mut buffer);

        // パワースペクトログラムを計算
        for (j, bin) in buffer.iter().take(hparams.channels).enumerate() {
            let power = (bin.re.powi(2) + bin.im.powi(2)).sqrt();
            spectrogram[[0, j, i]] = power;
        }
    }

    spectrogram
}

// ONNXモデルを推論する関数
fn run_onnx_model(
    session: &ort::Session,
    spec: &Array3<f32>,  // 3次元配列
    spec_lengths: &Array1<i64>,  // 1次元配列に修正
    sid_src: &Array2<i64>,  // 2次元配列
    sid_target: i64,
) -> Vec<f32> {
    // 各CowArrayを事前に変数として保持する
    let spec_cow = CowArray::from(spec.clone().into_dyn());
    let spec_lengths_cow = CowArray::from(spec_lengths.clone().into_dyn());  // 1次元の`spec_lengths`を渡す
    let sid_src_cow = CowArray::from(sid_src.clone().into_dyn());
    let sid_tgt_array_cow = CowArray::from(Array::from_elem(IxDyn(&[]), sid_target).into_dyn());

    let inputs = vec![
        Value::from_array(session.allocator(), &spec_cow).unwrap(),
        Value::from_array(session.allocator(), &spec_lengths_cow).unwrap(),  // 修正済み
        Value::from_array(session.allocator(), &sid_src_cow).unwrap(),
        Value::from_array(session.allocator(), &sid_tgt_array_cow).unwrap(),
    ];

    let outputs: Vec<Value> = session.run(inputs).unwrap();
    let audio_output: ort::tensor::OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();

    // OrtOwnedTensor を ndarray の Array に変換し、それを Vec に変換
    audio_output.view().to_owned().into_raw_vec()
}


// 音声処理関数
// 音声処理関数
fn audio_trans(
    hparams: &Hyperparameters,
    session: &ort::Session,
    signal: Vec<f32>,
    target_id: i64,
) -> Vec<f32> {
    let mut padded_signal = signal.clone();
    pad_signal(
        &mut padded_signal,
        hparams.filter_length,
        hparams.hop_length,
    );

    // STFTを実行してスペクトログラムを得る
    let spec = stft(&padded_signal, hparams);

    println!("spec shape: {:?}", spec.shape()); // 形状を確認

    // spec_lengthsを1次元に修正
    let spec_lengths = Array1::from_elem(1, spec.shape()[2] as i64); // 1次元配列に変更
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
    let environment = Arc::new(
        Environment::builder()
            .with_name("MMVC_Client")
            .build()
            .unwrap(),
    );

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
        writer
            .write_sample((sample * hparams.max_wav_value) as i16)
            .unwrap();
    }
}
