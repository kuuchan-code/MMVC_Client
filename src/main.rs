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
struct AudioParams {
    sample_rate: u32,
    fft_window_size: usize,
    hop_size: usize,
    target_speaker_id: i64,
    stft_padding_frames: usize,
    conv1d_padding_frames: usize,
}

impl AudioParams {
    fn new() -> Self {
        AudioParams {
            sample_rate: 24000,
            fft_window_size: 512,
            hop_size: 128,
            target_speaker_id: 2,
            stft_padding_frames: 2,
            conv1d_padding_frames: 10,
        }
    }
}
// STFTを実行してスペクトログラムを計算
fn apply_stft_with_hann_window(
    audio_signal: &Vec<f32>,
    fft_size: usize,
    hop_size: usize,
    window_size: usize,
) -> Array3<f32> {
    // ハンウィンドウの作成
    let hann_window: Vec<f32> = (0..window_size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / window_size as f32).cos()))
        .collect();

    // オーディオ信号にパディングを追加
    let padded_signal = pad_audio_signal(audio_signal, fft_size, hop_size);

    // FFTの準備
    let mut fft_planner = FftPlanner::<f32>::new();
    let fft = fft_planner.plan_fft_forward(fft_size);

    let num_frames = (padded_signal.len() - fft_size) / hop_size + 1;
    let mut spectrogram = Array3::<f32>::zeros((1, fft_size / 2 + 1, num_frames));

    for (i, frame) in padded_signal
        .windows(fft_size)
        .step_by(hop_size)
        .enumerate()
    {
        // 各フレームにハンウィンドウを適用
        let mut complex_buffer: Vec<Complex<f32>> = frame
            .iter()
            .zip(&hann_window)
            .map(|(sample, window)| Complex::new(sample * window, 0.0))
            .collect();

        // FFTの実行
        fft.process(&mut complex_buffer);

        // パワースペクトログラムを計算
        for (j, bin) in complex_buffer.iter().take(fft_size / 2 + 1).enumerate() {
            let power = (bin.re.powi(2) + bin.im.powi(2)).sqrt();
            spectrogram[[0, j, i]] = power;
        }
    }

    spectrogram
}

// オーディオ信号にパディングを追加
fn pad_audio_signal(signal: &Vec<f32>, fft_size: usize, hop_size: usize) -> Vec<f32> {
    let pad_size = (fft_size - hop_size) / 2;
    let mut padded_signal = vec![0.0; pad_size];
    padded_signal.extend_from_slice(signal);
    padded_signal.extend(vec![0.0; pad_size]);
    padded_signal
}
fn dispose_stft_padding(spec: &mut Array3<f32>, stft_padding_frames: usize) {
    let spec_len = spec.shape()[2];
    if stft_padding_frames > 0 && spec_len > 2 * stft_padding_frames {
        *spec = spec
            .slice(s![
                ..,
                ..,
                stft_padding_frames..spec_len - stft_padding_frames
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
fn run_onnx_model_inference(
    session: &ort::Session,
    spectrogram: &Array3<f32>,
    spectrogram_lengths: &Array1<i64>,
    source_speaker_id: &Array1<i64>,
    target_speaker_id: i64,
) -> Vec<f32> {
    let spec_cow = CowArray::from(spectrogram.clone().into_dyn());
    let spec_lengths_cow = CowArray::from(spectrogram_lengths.clone().into_dyn());
    let source_id_cow = CowArray::from(source_speaker_id.clone().into_dyn());
    let target_id_cow = CowArray::from(Array1::from_elem(1, target_speaker_id).into_dyn());

    let inputs = vec![
        Value::from_array(session.allocator(), &spec_cow).unwrap(),
        Value::from_array(session.allocator(), &spec_lengths_cow).unwrap(),
        Value::from_array(session.allocator(), &source_id_cow).unwrap(),
        Value::from_array(session.allocator(), &target_id_cow).unwrap(),
    ];

    let outputs: Vec<Value> = session.run(inputs).unwrap();
    let audio_output: ort::tensor::OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();

    audio_output.view().to_owned().into_raw_vec()
}

// 音声処理関数
fn audio_trans(
    hparams: &AudioParams,
    session: &ort::Session,
    signal: Vec<f32>,
    target_speaker_id: i64,
    stft_padding_frames: usize,   // stft_padding_frames 引数を追加
    conv1d_padding_frames: usize, // conv1d_padding_frames 引数を追加
) -> Vec<f32> {
    // ハンウィンドウを適用したSTFTを実行
    let mut spec = apply_stft_with_hann_window(
        &signal,
        hparams.fft_window_size, // n_fft
        hparams.hop_size,        // hop_size
        hparams.fft_window_size, // win_size
    );

    // STFT パディング削除
    dispose_stft_padding(&mut spec, stft_padding_frames); // stft_padding_frames を適用

    println!("spec shape after dispose: {:?}", spec.shape());

    // spec_lengthsを1次元に修正
    let spec_lengths = Array1::from_elem(1, spec.shape()[2] as i64); // 1次元配列
    let sid_src = Array1::from_elem(1, 0); // `sid_src`を0に設定、1次元配列
    let sid_target = hparams.target_speaker_id; // 既存のターゲットID

    // モデルの実行
    let mut audio = run_onnx_model_inference(session, &spec, &spec_lengths, &sid_src, sid_target);

    // Conv1D パディング削除
    dispose_conv1d_padding(&mut audio, conv1d_padding_frames); // conv1d_padding_frames を適用

    audio
}

// マイク入力から音声を取得しリサンプリングする関数（デバイス選択対応）
fn record_and_resample(
    hparams: &AudioParams,
    input_device: cpal::Device,
    signal: Arc<Mutex<Vec<f32>>>,
    buffer_size: usize,
) -> cpal::Stream {
    let input_config = input_device.default_input_config().unwrap();

    let frequency_bins = input_config.channels();
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
                    .chunks(frequency_bins as usize)
                    .map(|chunk| chunk.iter().sum::<f32>() / frequency_bins as f32)
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
            None,
        )
        .unwrap();

    stream
}

// 音声をスピーカーに出力（デバイス選択対応）
fn play_output(
    hparams: Arc<AudioParams>,
    session: Arc<ort::Session>,
    output_device: cpal::Device,
    input_signal: Arc<Mutex<Vec<f32>>>,
    buffer_size: usize,
) -> cpal::Stream {
    let output_config = output_device.default_output_config().unwrap();
    let output_sample_rate = output_config.sample_rate().0;

    let mut resampler = FftFixedInOut::<f32>::new(
        hparams.sample_rate as usize,
        output_sample_rate as usize,
        buffer_size / 2 - 2 * hparams.fft_window_size + hparams.hop_size
            - hparams.conv1d_padding_frames * hparams.stft_padding_frames,
        1,
    )
    .unwrap();

    let output_signal = Arc::clone(&input_signal);
    let mut sample_index = 0;
    let mut processed_signal_cache: Option<Vec<f32>> = None; // キャッシュを保持
    let mut prev_trans_wav: Vec<f32> = Vec::new(); // 前回の音声データを保持
    let overlap_length = hparams.fft_window_size / 2;

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
                            hparams.target_speaker_id,
                            hparams.stft_padding_frames, // STFT の dispose 値を渡す
                            hparams.conv1d_padding_frames, // Conv1D の dispose 値を渡す
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
// デバイス選択用の関数
fn select_device(devices: Vec<cpal::Device>, label: &str) -> cpal::Device {
    println!("{} デバイスを選択してください:", label);
    for (i, device) in devices.iter().enumerate() {
        println!("{}: {}", i, device.name().unwrap());
    }
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let index: usize = input.trim().parse().unwrap();
    devices.into_iter().nth(index).unwrap()
}
fn main() -> OrtResult<()> {
    let hparams = Arc::new(AudioParams::new()); // `Arc`で共有する

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

    // デバイス選択
    let host = cpal::default_host();
    let input_device = select_device(host.input_devices().unwrap().collect(), "入力");
    let output_device = select_device(host.output_devices().unwrap().collect(), "出力");

    // ストリームの作成
    let input_stream = record_and_resample(
        &hparams,
        input_device,
        Arc::clone(&input_signal),
        buffer_size,
    );
    let output_stream = play_output(
        Arc::clone(&hparams),
        Arc::new(session),
        output_device,
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
