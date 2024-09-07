use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use ndarray::{Array1, Array3, CowArray};
use ort::{GraphOptimizationLevel, Value};
use rubato::{FftFixedInOut, Resampler};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};
use ort::{Environment, ExecutionProvider, SessionBuilder, OrtResult};


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
fn record_and_resample(
    hparams: &Hyperparameters,
    signal: Arc<Mutex<Vec<f32>>>,
    buffer_size: usize
) -> cpal::Stream {
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
        buffer_size,
        1,
    )
    .unwrap();

    let input_stream_config: StreamConfig = input_config.into(); // 変換

    let mut buffer = Vec::new();

    let stream = input_device
        .build_input_stream(
            &input_stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut input_signal = signal.lock().unwrap();
                let mono_signal: Vec<f32> = data
                    .chunks(channels as usize)
                    .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                    .collect();

                buffer.extend_from_slice(&mono_signal);
                
                // バッファが指定サイズに達したらリサンプリングして追加
                if buffer.len() >= buffer_size {
                    let resampled = resampler.process(&[buffer.clone()], None).unwrap();
                    input_signal.extend_from_slice(&resampled[0]);
                    buffer.clear(); // バッファをクリア
                }
            },
            move |err| {
                eprintln!("Error occurred on input stream: {}", err);
            },
            None, // ここでOption<Duration>を指定
        )
        .unwrap();

    stream
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

// 音声をスピーカーに出力
fn play_output(
    hparams: Arc<Hyperparameters>,  // `Arc`で所有権を共有
    session: Arc<ort::Session>,     // `Arc`で所有権を共有
    input_signal: Arc<Mutex<Vec<f32>>>,
    buffer_size: usize
) -> cpal::Stream {
    let host = cpal::default_host();
    let output_device = host
        .default_output_device()
        .expect("Failed to get default output device.");

    let output_config = output_device.default_output_config().unwrap();
    let output_sample_rate = output_config.sample_rate().0;

    // リサンプラーの設定（モデルのサンプリングレートから出力デバイスのサンプリングレートにリサンプル）
    let mut resampler = FftFixedInOut::<f32>::new(
        hparams.sample_rate as usize,
        output_sample_rate as usize,
        2048,
        1,
    )
    .unwrap();

    let output_signal = Arc::clone(&input_signal);
    let mut sample_index = 0;
    let mut processed_signal_cache: Option<Vec<f32>> = None; // キャッシュを保持

    // `output_config`のクローンを作成してから使用
    let output_config_clone = output_config.clone();
    let output_config_for_closure = output_config.clone(); // クロージャ内で使うためのクローン

    let stream = output_device
        .build_output_stream(
            &output_config_clone.into(), // ここで`into()`を呼び出して所有権を移動
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut input_signal = output_signal.lock().unwrap();
                if !input_signal.is_empty() {
                    // キャッシュがない場合は新しい信号を処理
                    if processed_signal_cache.is_none() {
                        let processed_signal = audio_trans(
                            &hparams,    // `Arc`で共有された所有権を渡す
                            &session,    // `Arc`で共有された所有権を渡す
                            input_signal.clone(),
                            hparams.target_id
                        );
                        let resampled = resampler.process(&[processed_signal], None).unwrap();
                        processed_signal_cache = Some(resampled[0].clone());
                        input_signal.clear(); // 信号が処理された後にクリア
                    }

                    let resampled_signal = processed_signal_cache.as_ref().unwrap();

                    // クロージャ内で別クローンを使用
                    for frame in data.chunks_mut(output_config_for_closure.channels() as usize) {
                        let sample = if sample_index < resampled_signal.len() {
                            resampled_signal[sample_index]
                        } else {
                            0.0 // 信号が終わったら無音にする
                        };
                        for channel in frame.iter_mut() {
                            *channel = sample;
                        }
                        sample_index += 1;
                    }

                    // 全てのサンプルを再生し終わったらキャッシュをクリア
                    if sample_index >= resampled_signal.len() {
                        sample_index = 0;
                        processed_signal_cache = None;
                    }
                }
            },
            move |err| {
                eprintln!("Error occurred on output stream: {}", err);
            },
            None
        )
        .unwrap();

    stream
}


fn main() -> OrtResult<()> {
    let hparams = Arc::new(Hyperparameters::new()); // `Arc`で共有する

    // 環境の構築とCUDAプロバイダーの追加
    let environment = Environment::builder()
        .with_name("MMVC_Client")
        .with_execution_providers([ExecutionProvider::CUDA(Default::default())]) // CUDAプロバイダーを追加
        .build()?
        .into_arc();

    // セッションの構築
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file("G_best.onnx")?;

    let buffer_size = 4096; // 0.05秒程度のバッファサイズ
    let input_signal = Arc::new(Mutex::new(Vec::new()));

    // ストリームの作成
    let input_stream = record_and_resample(&hparams, Arc::clone(&input_signal), buffer_size);
    let output_stream = play_output(Arc::clone(&hparams), Arc::new(session), Arc::clone(&input_signal), buffer_size);

    // ストリームの開始
    input_stream.play().unwrap();
    output_stream.play().unwrap();

    // 永続ループで音声を処理し続ける
    loop {
        std::thread::sleep(std::time::Duration::from_millis(50)); // 50msごとに処理
    }
}