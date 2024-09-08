use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use ndarray::{s, Array1, Array3, CowArray};
use ort::{Environment, ExecutionProvider, OrtResult, SessionBuilder};
use ort::{GraphOptimizationLevel, Value};
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
    channels: usize,             // 257
    target_id: i64,              // target_id フィールドを追加
    dispose_stft_specs: usize,   // STFT の dispose フィールド
    dispose_conv1d_specs: usize, // Conv1D の dispose フィールド
}

impl Hyperparameters {
    fn new() -> Self {
        Hyperparameters {
            sample_rate: 24000,
            max_wav_value: 32768.0,
            filter_length: 512,
            hop_length: 128,
            channels: 257,
            target_id: 2,
            dispose_stft_specs: 2,
            dispose_conv1d_specs: 10,
        }
    }
}

// STFTを実行し、ハンウィンドウを適用したスペクトログラムを生成
fn stft_with_hann_window(
    signal: &Vec<f32>,
    n_fft: usize,
    hop_size: usize,
    win_size: usize,
) -> Array3<f32> {
    // ハンウィンドウを生成
    let hann_window: Vec<f32> = (0..win_size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / win_size as f32).cos()))
        .collect();

    // FFT プランナー
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let num_frames = (signal.len() - n_fft) / hop_size + 1;
    let mut spectrogram = Array3::<f32>::zeros((1, n_fft / 2 + 1, num_frames));

    for (i, frame) in signal.windows(n_fft).step_by(hop_size).enumerate() {
        // フレームにハンウィンドウを適用
        let mut buffer: Vec<Complex<f32>> = frame
            .iter()
            .zip(&hann_window)
            .map(|(x, w)| Complex::new(x * w, 0.0))
            .collect();

        // FFTを実行
        fft.process(&mut buffer);

        // パワースペクトログラムを計算
        for (j, bin) in buffer.iter().take(n_fft / 2 + 1).enumerate() {
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
    buffer_size: usize,
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

fn dispose_stft_padding(spec: &mut Array3<f32>, dispose_stft_specs: usize) {
    let spec_len = spec.shape()[2];
    if dispose_stft_specs > 0 && spec_len > 2 * dispose_stft_specs {
        *spec = spec
            .slice(s![
                ..,
                ..,
                dispose_stft_specs..spec_len - dispose_stft_specs
            ])
            .to_owned();
    }
}

fn dispose_conv1d_padding(audio: &mut Vec<f32>, dispose_conv1d_length: usize) {
    let audio_len = audio.len();
    if dispose_conv1d_length > 0 && audio_len > 2 * dispose_conv1d_length {
        *audio = audio[dispose_conv1d_length..audio_len - dispose_conv1d_length].to_vec();
    }
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
    dispose_stft_specs: usize,   // dispose_stft_specs 引数を追加
    dispose_conv1d_specs: usize, // dispose_conv1d_specs 引数を追加
) -> Vec<f32> {
    // ハンウィンドウを適用したSTFTを実行
    let mut spec = stft_with_hann_window(
        &signal,
        hparams.filter_length, // n_fft
        hparams.hop_length,    // hop_size
        hparams.filter_length, // win_size
    );

    // STFT パディング削除
    dispose_stft_padding(&mut spec, dispose_stft_specs); // dispose_stft_specs を適用

    println!("spec shape after dispose: {:?}", spec.shape());

    // spec_lengthsを1次元に修正
    let spec_lengths = Array1::from_elem(1, spec.shape()[2] as i64); // 1次元配列
    let sid_src = Array1::from_elem(1, 0); // `sid_src`を0に設定、1次元配列
    let sid_target = hparams.target_id; // 既存のターゲットID

    // モデルの実行
    let mut audio = run_onnx_model(session, &spec, &spec_lengths, &sid_src, sid_target);

    // Conv1D パディング削除
    dispose_conv1d_padding(&mut audio, dispose_conv1d_specs); // dispose_conv1d_specs を適用

    audio
}

// 音声をスピーカーに出力
fn play_output(
    hparams: Arc<Hyperparameters>,
    session: Arc<ort::Session>,
    input_signal: Arc<Mutex<Vec<f32>>>,
    buffer_size: usize,
) -> cpal::Stream {
    let host = cpal::default_host();
    let output_device = host
        .default_output_device()
        .expect("Failed to get default output device.");

    let output_config = output_device.default_output_config().unwrap();
    let output_sample_rate = output_config.sample_rate().0;

    let mut resampler = FftFixedInOut::<f32>::new(
        hparams.sample_rate as usize,
        output_sample_rate as usize,
        buffer_size / 2 - 2 * hparams.filter_length + hparams.hop_length
            - hparams.dispose_conv1d_specs * hparams.dispose_stft_specs,
        1,
    )
    .unwrap();

    let output_signal = Arc::clone(&input_signal);
    let mut sample_index = 0;
    let mut processed_signal_cache: Option<Vec<f32>> = None; // キャッシュを保持
    let mut prev_trans_wav: Vec<f32> = Vec::new(); // 前回の音声データを保持
    let overlap_length = hparams.filter_length / 2;

    let output_config_clone = output_config.clone(); // クローンして再利用
    let stream = output_device
        .build_output_stream(
            &output_config_clone.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut input_signal = output_signal.lock().unwrap();
                if !input_signal.is_empty() {
                    if processed_signal_cache.is_none() {
                        let processed_signal = audio_trans(
                            &hparams,
                            &session,
                            input_signal.clone(),
                            hparams.target_id,
                            hparams.dispose_stft_specs, // STFT の dispose 値を渡す
                            hparams.dispose_conv1d_specs, // Conv1D の dispose 値を渡す
                        );
                        let resampled = resampler.process(&[processed_signal], None).unwrap();

                        // オーバーラップ合成
                        let merged_signal =
                            overlap_merge(&resampled[0], &prev_trans_wav, overlap_length);
                        processed_signal_cache = Some(merged_signal.clone());
                        prev_trans_wav = resampled[0].clone(); // 次回のために保存
                        input_signal.clear();
                    }

                    let resampled_signal = processed_signal_cache.as_ref().unwrap();

                    for frame in data.chunks_mut(output_config.channels() as usize) {
                        let sample = if sample_index < resampled_signal.len() {
                            resampled_signal[sample_index]
                        } else {
                            0.0
                        };
                        for channel in frame.iter_mut() {
                            *channel = sample;
                        }
                        sample_index += 1;
                    }

                    if sample_index >= resampled_signal.len() {
                        sample_index = 0;
                        processed_signal_cache = None;
                    }
                }
            },
            move |err| {
                eprintln!("Error occurred on output stream: {}", err);
            },
            None,
        )
        .unwrap();

    stream
}

fn overlap_merge(now_wav: &Vec<f32>, prev_wav: &Vec<f32>, overlap_length: usize) -> Vec<f32> {
    if overlap_length == 0 || prev_wav.is_empty() {
        return now_wav.clone();
    }

    let overlap_len = std::cmp::min(overlap_length, std::cmp::min(now_wav.len(), prev_wav.len()));
    let now_head = &now_wav[..overlap_len];
    let prev_tail = &prev_wav[prev_wav.len() - overlap_len..];

    let gradation: Vec<f32> = (0..overlap_len)
        .map(|i| (i as f32 / overlap_len as f32) * PI * 0.5)
        .collect();

    let mut merged: Vec<f32> = Vec::with_capacity(now_wav.len());

    for i in 0..overlap_len {
        let cos_grad = gradation[i].cos().powi(2);
        let prev_cos_grad = (PI * 0.5 - gradation[i]).cos().powi(2);
        let merged_sample = prev_tail[i] * prev_cos_grad + now_head[i] * cos_grad;
        merged.push(merged_sample);
    }

    merged.extend_from_slice(&now_wav[overlap_len..]);
    merged
}

fn main() -> OrtResult<()> {
    let hparams = Arc::new(Hyperparameters::new()); // `Arc`で共有する

    // 環境の構築とCUDAプロバイダーの追加
    let environment = Environment::builder()
        .with_name("MMVC_Client")
        .with_execution_providers([ExecutionProvider::DirectML(Default::default())]) // CUDAプロバイダーを追加
        .build()?
        .into_arc();

    // セッションの構築
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file("G_best.onnx")?;

    let buffer_size = 8192;
    let input_signal = Arc::new(Mutex::new(Vec::new()));

    // ストリームの作成
    let input_stream = record_and_resample(&hparams, Arc::clone(&input_signal), buffer_size);
    let output_stream = play_output(
        Arc::clone(&hparams),
        Arc::new(session),
        Arc::clone(&input_signal),
        buffer_size,
    );

    // ストリームの開始
    input_stream.play().unwrap();
    output_stream.play().unwrap();

    // 永続ループで音声を処理し続ける
    loop {
        std::thread::sleep(std::time::Duration::from_millis(50)); // 50msごとに処理
    }
}
