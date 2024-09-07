use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, StreamConfig};
use hound;
use ndarray::{Array1, Array3, CowArray};
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use rubato::{FftFixedInOut, Resampler};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};

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

// マイク入力から音声を取得しリサンプリングする関数
fn record_and_resample(hparams: &Hyperparameters) -> Vec<f32> {
    let host = cpal::default_host();
    let input_device = host
        .default_input_device()
        .expect("Failed to get default input device.");

    let input_config = input_device.default_input_config().unwrap();

    // モノラルに変換し、必要なサンプルレートにリサンプリング
    let channels = input_config.channels();
    let input_sample_rate = input_config.sample_rate().0;

    // リサンプラーの設定
    let mut resampler = FftFixedInOut::<f32>::new(
        input_sample_rate as usize,
        hparams.sample_rate as usize,
        480,
        1,
    )
    .unwrap();
    let signal = Arc::new(Mutex::new(Vec::new()));

    let signal_clone = Arc::clone(&signal);
    let input_stream_config: StreamConfig = input_config.into(); // 変換

    let stream = input_device
        .build_input_stream(
            &input_stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut input_signal = signal_clone.lock().unwrap();
                let mono_signal: Vec<f32> = data
                    .chunks(channels as usize)
                    .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                    .collect();
                let resampled = resampler.process(&[mono_signal], None).unwrap();
                input_signal.extend_from_slice(&resampled[0]);
            },
            move |err| {
                eprintln!("Error occurred on input stream: {}", err);
            },
            None, // ここでOption<Duration>を指定
        )
        .unwrap();

    stream.play().unwrap();

    std::thread::sleep(std::time::Duration::from_secs(5)); // 5秒間録音

    let final_signal = signal.lock().unwrap().clone();
    final_signal
}
// 処理結果を出力デバイスに送る関数
fn play_output(output_signal: Vec<f32>, hparams: &Hyperparameters) {
    let host = cpal::default_host();
    let output_device = host
        .default_output_device()
        .expect("Failed to get default output device.");

    let output_config = output_device.default_output_config().unwrap();

    let output_stream_config: StreamConfig = output_config.into(); // 変換

    let stream = output_device.build_output_stream(
        &output_stream_config,
        move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let output_length = output.len().min(output_signal.len());
            output[..output_length].copy_from_slice(&output_signal[..output_length]);
        },
        move |err| {
            eprintln!("Error occurred on output stream: {}", err);
        },
        None,
    ).unwrap();

    stream.play().unwrap();

    std::thread::sleep(std::time::Duration::from_secs(5)); // 出力を5秒間再生
}
// ONNXモデルを推論する関数
fn run_onnx_model(
    session: &ort::Session,
    spec: &Array3<f32>,         // 3次元配列
    spec_lengths: &Array1<i64>, // 1次元配列
    sid_src: &Array1<i64>,      // 1次元配列
    sid_target: i64,
) -> Vec<f32> {
    // 各CowArrayを事前に変数として保持する
    let spec_cow = CowArray::from(spec.clone().into_dyn());
    let spec_lengths_cow = CowArray::from(spec_lengths.clone().into_dyn());
    let sid_src_cow = CowArray::from(sid_src.clone().into_dyn());
    // sid_tgtを1次元配列に修正
    let sid_tgt_array_cow = CowArray::from(Array1::from_elem(1, sid_target).into_dyn());

    let inputs = vec![
        Value::from_array(session.allocator(), &spec_cow).unwrap(),
        Value::from_array(session.allocator(), &spec_lengths_cow).unwrap(),
        Value::from_array(session.allocator(), &sid_src_cow).unwrap(),
        Value::from_array(session.allocator(), &sid_tgt_array_cow).unwrap(), // 修正済み
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
    let spec_lengths = Array1::from_elem(1, spec.shape()[2] as i64); // 1次元配列
    let sid_src = Array1::from_elem(1, 0); // `sid_src`を0に設定、1次元配列
    let sid_target = hparams.target_id; // 既存のターゲットID

    // モデルの実行
    let audio = run_onnx_model(session, &spec, &spec_lengths, &sid_src, sid_target);

    audio
}
fn save_wav(signal: Vec<f32>, file_path: &str, hparams: &Hyperparameters) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: hparams.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(file_path, spec).unwrap();

    for sample in signal {
        let scaled_sample = (sample * hparams.max_wav_value) as i16; // スケーリングしてi16に変換
        writer.write_sample(scaled_sample).unwrap();
    }

    writer.finalize().unwrap();
}
// メイン関数
fn main() {
    let hparams = Hyperparameters::new();

    let environment = Arc::new(
        Environment::builder()
            .with_name("MMVC_Client")
            .build()
            .unwrap(),
    );

    let session = SessionBuilder::new(&environment)
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_model_from_file("G_best.onnx")
        .unwrap();

    // マイクから音声を取得してリサンプリング
    let input_signal = record_and_resample(&hparams);
    // 取得した信号をファイルに保存
    save_wav(input_signal.clone(), "input_signal.wav", &hparams);
    // 音声処理
    let output_signal = audio_trans(&hparams, &session, input_signal, hparams.target_id);

    // 処理後の音声を出力デバイスに再生
    save_wav(output_signal.clone(), "output_signal.wav", &hparams);
    // 処理後の音声をスピーカーに出力
    play_output(output_signal, &hparams);
}
