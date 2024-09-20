use anyhow::{Context, Result};
use clap::{Arg, Command};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, StreamConfig};
use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::{Array1, Array3, CowArray};
use ort::{
    tensor::OrtOwnedTensor, Environment, ExecutionProvider, GraphOptimizationLevel, Session,
    SessionBuilder, Value,
};
use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, FftPlanner};
use speexdsp_resampler::State as SpeexResampler;
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::thread;

#[cfg(target_os = "windows")]
use windows::Win32::System::Threading::{
    GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_TIME_CRITICAL,
};

// 定数の定義
const FFT_WINDOW_SIZE: usize = 512;
const HOP_SIZE: usize = 128;

// AudioParams構造体の定義
struct AudioParams {
    model_sample_rate: u32, // サンプルレートはモデルへの入力用
    buffer_size: usize,
    overlap_length: usize,
    source_speaker_id: i64,
    target_speaker_id: i64,
    hann_window: Vec<f32>,
    cutoff_enabled: bool,
    cutoff_freq: f32,
}

// AudioParamsの実装
impl AudioParams {
    fn new(
        model_sample_rate: u32,
        buffer_size: usize,
        overlap_length: usize,
        source_speaker_id: i64,
        target_speaker_id: i64,
        cutoff_enabled: bool,
        cutoff_freq: f32,
    ) -> Self {
        let hann_window: Vec<f32> = (0..FFT_WINDOW_SIZE)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / FFT_WINDOW_SIZE as f32).cos()))
            .collect();

        Self {
            model_sample_rate,
            buffer_size,
            overlap_length,
            source_speaker_id,
            target_speaker_id,
            hann_window,
            cutoff_enabled,
            cutoff_freq,
        }
    }
}

// 処理スレッド
fn processing_thread(
    hparams: Arc<AudioParams>,
    session: Arc<Session>,
    input_rx: Receiver<Vec<f32>>,
    output_tx: Sender<Vec<f32>>,
) -> Result<()> {
    // Set sola_search_frame equal to overlap_length
    let sola_search_frame = hparams.overlap_length;
    let overlap_size = hparams.overlap_length;
    let mut sola = Sola::new(overlap_size, sola_search_frame);
    let mut prev_input_tail: Vec<f32> = Vec::new();

    while let Ok(mut input_signal) = input_rx.recv() {
        // 入力信号の前後処理
        if !prev_input_tail.is_empty() {
            let mut extended_signal =
                Vec::with_capacity(prev_input_tail.len() + input_signal.len());
            extended_signal.extend_from_slice(&prev_input_tail);
            extended_signal.extend_from_slice(&input_signal);
            input_signal = extended_signal;
        }

        // 音声変換処理
        let processed_signal = audio_transform(&hparams, &session, &input_signal);

        // SOLAによるマージ
        let merged_signal = sola.merge(&processed_signal);

        // 次回のために入力の終端を保持
        if input_signal.len() >= overlap_size {
            prev_input_tail = input_signal[input_signal.len() - overlap_size..].to_vec();
        } else {
            prev_input_tail = input_signal.clone();
        }

        // マージした信号を送信
        if output_tx.send(merged_signal).is_err() {
            break;
        }
    }
    Ok(())
}

// 音声変換処理のメイン関数
fn audio_transform(hparams: &AudioParams, session: &Session, signal: &[f32]) -> Vec<f32> {
    // STFTパディングの追加
    let pad_size = (FFT_WINDOW_SIZE - HOP_SIZE) / 2;
    let mut padded_signal = vec![0.0; pad_size];
    padded_signal.extend_from_slice(signal);
    padded_signal.extend(vec![0.0; pad_size]);

    // STFTの適用
    let spec = apply_stft_with_hann_window(&padded_signal, hparams);

    let spec_lengths = Array1::from_elem(1, spec.shape()[2] as i64);
    let source_speaker_id_src = Array1::from_elem(1, hparams.source_speaker_id);

    let audio_result = run_onnx_model_inference(
        session,
        &spec,
        &spec_lengths,
        &source_speaker_id_src,
        hparams.target_speaker_id,
    );

    let audio = match audio_result {
        Some(a) => a,
        None => return vec![0.0; signal.len()],
    };

    audio
}

// ONNXモデルでの推論を実行
fn run_onnx_model_inference(
    session: &Session,
    spectrogram: &Array3<f32>,
    spectrogram_lengths: &Array1<i64>,
    source_speaker_id: &Array1<i64>,
    target_speaker_id: i64,
) -> Option<Vec<f32>> {
    // CowArrayに変換
    let spec_cow = CowArray::from(spectrogram.clone().into_dyn());
    let spec_lengths_cow = CowArray::from(spectrogram_lengths.clone().into_dyn());
    let source_id_cow = CowArray::from(source_speaker_id.clone().into_dyn());
    let target_id_cow = CowArray::from(Array1::from_elem(1, target_speaker_id).into_dyn());

    // Value::from_arrayに渡す
    let inputs = vec![
        Value::from_array(session.allocator(), &spec_cow).ok()?,
        Value::from_array(session.allocator(), &spec_lengths_cow).ok()?,
        Value::from_array(session.allocator(), &source_id_cow).ok()?,
        Value::from_array(session.allocator(), &target_id_cow).ok()?,
    ];

    let outputs = session.run(inputs).ok()?;

    let audio_output: OrtOwnedTensor<f32, _> = outputs[0].try_extract().ok()?;

    Some(audio_output.view().to_owned().into_raw_vec())
}

// ハン窓を使用したSTFTの適用
fn apply_stft_with_hann_window(audio_signal: &[f32], hparams: &AudioParams) -> Array3<f32> {
    let fft_size = FFT_WINDOW_SIZE;
    let hop_size = HOP_SIZE;
    let window = &hparams.hann_window;
    let num_frames = (audio_signal.len() - fft_size) / hop_size + 1;
    let mut spectrogram = Array3::<f32>::zeros((1, fft_size / 2 + 1, num_frames));

    // FFTのプランナーを初期化
    let mut fft_planner = FftPlanner::<f32>::new();
    let fft = fft_planner.plan_fft_forward(fft_size);

    let mut input_buffer = vec![Complex::zero(); fft_size];
    // scratchバッファの追加
    let mut scratch_buffer = vec![Complex::zero(); fft.get_inplace_scratch_len()];

    // 周波数解像度の計算
    let freq_resolution = hparams.model_sample_rate as f32 / fft_size as f32;
    let cutoff_freq = hparams.cutoff_freq;
    let cutoff_bin = (cutoff_freq / freq_resolution).ceil() as usize;

    for (i, frame) in audio_signal.windows(fft_size).step_by(hop_size).enumerate() {
        // 窓関数の適用と複素数への変換
        for (j, &sample) in frame.iter().enumerate() {
            input_buffer[j] = Complex::new(sample * window[j], 0.0);
        }

        // FFTの実行
        fft.process_with_scratch(&mut input_buffer, &mut scratch_buffer);

        // スペクトログラムの取得とフィルタリング
        for (j, bin) in input_buffer.iter().take(fft_size / 2 + 1).enumerate() {
            if hparams.cutoff_enabled && j < cutoff_bin {
                spectrogram[[0, j, i]] = 0.0; // カットオフが有効であり、カットオフ周波数以下の場合
            } else {
                spectrogram[[0, j, i]] = bin.norm();
            }
        }
    }

    spectrogram
}

// 音声の録音とリサンプリングを行う関数
fn record_and_resample(
    hparams: Arc<AudioParams>,
    input_device: Device,
    input_tx: Sender<Vec<f32>>,
) -> Result<cpal::Stream> {
    let input_config = input_device
        .default_input_config()
        .context("入力デバイスのデフォルト設定を取得できませんでした")?;
    let channels = input_config.channels() as usize;
    let input_sample_rate = input_config.sample_rate().0;
    let mut resampler = SpeexResampler::new(
        1,
        input_sample_rate as usize,
        hparams.model_sample_rate as usize,
        5,
    )
    .map_err(|e| anyhow::anyhow!("リサンプリザーの初期化に失敗しました: {:?}", e))?;

    let input_stream_config: StreamConfig = input_config.into();
    let mut buffer = Vec::new();

    let buffer_size = hparams.buffer_size;

    let stream = input_device
        .build_input_stream(
            &input_stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // モノラル化
                let mono_signal: Vec<f32> = data
                    .chunks(channels)
                    .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                    .collect();

                buffer.extend_from_slice(&mono_signal);

                while buffer.len() >= buffer_size {
                    let chunk = buffer.drain(..buffer_size).collect::<Vec<f32>>();
                    let mut resampled = vec![
                        0.0;
                        (chunk.len() as f64 * hparams.model_sample_rate as f64
                            / input_sample_rate as f64)
                            .ceil() as usize
                    ];
                    // リサンプリング
                    if let Ok((_, output_generated)) =
                        resampler.process_float(0, &chunk, &mut resampled)
                    {
                        resampled.truncate(output_generated);
                        if input_tx.send(resampled).is_err() {
                            break;
                        }
                    } else {
                        eprintln!("リサンプリングに失敗しました");
                        break;
                    }
                }
            },
            move |err| eprintln!("入力ストリームエラー: {}", err),
            None,
        )
        .context("入力ストリームの構築に失敗しました")?;

    Ok(stream)
}

// 出力を再生する関数
fn play_output(
    hparams: Arc<AudioParams>,
    output_device: Device,
    output_rx: Receiver<Vec<f32>>,
) -> Result<cpal::Stream> {
    let output_config = output_device
        .default_output_config()
        .context("出力デバイスのデフォルト設定を取得できませんでした")?;
    let output_sample_rate = output_config.sample_rate().0;
    let model_sample_rate = hparams.model_sample_rate;
    let resampling_ratio = output_sample_rate as f64 / model_sample_rate as f64;

    // リサンプリング後の最大バッファサイズを計算
    let max_buffer_size = (hparams.buffer_size as f64 * resampling_ratio).ceil() as usize;
    let mut resampler = SpeexResampler::new(
        1,
        model_sample_rate as usize,
        output_sample_rate as usize,
        5,
    )
    .map_err(|e| anyhow::anyhow!("リサンプリザーの初期化に失敗しました: {:?}", e))?;
    let mut output_buffer: VecDeque<f32> = VecDeque::with_capacity(max_buffer_size);
    let output_stream_config = output_config.config();
    let channels = output_config.channels() as usize;

    let stream = output_device
        .build_output_stream(
            &output_stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // 出力データを取得
                while let Ok(processed_signal) = output_rx.try_recv() {
                    let mut resampled = vec![
                        0.0;
                        (processed_signal.len() as f64 * resampling_ratio).ceil()
                            as usize
                    ];
                    if let Ok((_, output_generated)) =
                        resampler.process_float(0, &processed_signal, &mut resampled)
                    {
                        resampled.truncate(output_generated);
                        output_buffer.extend(resampled);
                    } else {
                        eprintln!("リサンプリングに失敗しました");
                    }
                }

                // 出力バッファからデータを供給
                for frame in data.chunks_mut(channels) {
                    let sample = output_buffer.pop_front().unwrap_or(0.0);
                    for channel in frame.iter_mut() {
                        *channel = sample;
                    }
                }
            },
            move |err| eprintln!("出力ストリームエラー: {}", err),
            None,
        )
        .context("出力ストリームの構築に失敗しました")?;

    Ok(stream)
}

// SOLAアルゴリズムの実装
struct Sola {
    overlap_size: usize,
    sola_search_frame: usize,
    prev_wav: Vec<f32>,
}

impl Sola {
    fn new(overlap_size: usize, sola_search_frame: usize) -> Self {
        Self {
            overlap_size,
            sola_search_frame,
            prev_wav: Vec::new(),
        }
    }

    fn merge(&mut self, wav: &[f32]) -> Vec<f32> {
        if self.prev_wav.is_empty() {
            let output_wav = wav[..wav.len() - self.overlap_size].to_vec();
            self.prev_wav = wav[wav.len() - self.overlap_size..].to_vec();
            return output_wav;
        }

        let search_range = self.sola_search_frame.min(self.prev_wav.len());
        let max_offset = self.prev_wav.len().saturating_sub(search_range);
        let (best_offset, _) = (0..=max_offset)
            .map(|offset| {
                let prev_segment = &self.prev_wav[offset..offset + search_range];
                let current_segment = &wav[..search_range];

                let corr = Self::calculate_correlation(prev_segment, current_segment);
                (offset, corr)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((0, 0.0));

        // クロスフェードの適用
        let prev_tail = &self.prev_wav[best_offset..];
        let current_head = &wav[..prev_tail.len()];
        let crossfaded = Self::crossfade(prev_tail, current_head);

        // マージ
        let mut output_wav = self.prev_wav[..best_offset].to_vec();
        output_wav.extend(crossfaded);
        output_wav.extend(&wav[prev_tail.len()..wav.len() - self.overlap_size]);

        // 次回のためにデータを保持
        self.prev_wav = wav[wav.len() - self.overlap_size..].to_vec();

        output_wav
    }

    fn calculate_correlation(a: &[f32], b: &[f32]) -> f32 {
        let dot_product = a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f32>();
        let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
        dot_product / (norm_a * norm_b + 1e-8)
    }

    fn crossfade(prev_wav: &[f32], cur_wav: &[f32]) -> Vec<f32> {
        let len = prev_wav.len();
        (0..len)
            .map(|i| {
                let t = i as f32 / (len - 1) as f32;
                let fade_in = 0.5 - 0.5 * (PI * t).cos();
                let fade_out = 1.0 - fade_in;
                prev_wav[i] * fade_out + cur_wav[i] * fade_in
            })
            .collect()
    }
}

// デバイス選択のための関数
fn select_device(devices: &[Device], label: &str) -> Result<Device> {
    println!("{} デバイス:", label);
    for (i, device) in devices.iter().enumerate() {
        println!("  {}: {}", i, device.name().unwrap_or_default());
    }

    let stdin = io::stdin();
    print!("{}デバイスの番号を選択してください: ", label);
    io::stdout()
        .flush()
        .context("プロンプトのフラッシュに失敗しました")?;

    let input = stdin
        .lock()
        .lines()
        .next()
        .context("入力の取得に失敗しました")?
        .context("入力の読み取りに失敗しました")?;

    let index: usize = input
        .trim()
        .parse()
        .context("有効なデバイス番号を入力してください")?;
    devices
        .get(index)
        .cloned()
        .with_context(|| format!("デバイス番号 {} は存在しません", index))
}

// メイン関数
fn main() -> Result<()> {
    // ロガーの初期化
    env_logger::init();

    // コマンドライン引数の定義
    let matches = Command::new(env!("CARGO_PKG_NAME"))
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about(env!("CARGO_PKG_DESCRIPTION"))
        .arg(
            Arg::new("onnx")
                .short('m')
                .long("model")
                .value_name("ONNX_FILE")
                .help("使用するONNXファイルのパス")
                .value_parser(clap::value_parser!(String))
                .required(true),
        )
        .arg(
            Arg::new("source_id")
                .short('s')
                .long("source")
                .value_name("SOURCE_ID")
                .help("ソーススピーカーID")
                .required(true)
                .value_parser(clap::value_parser!(i64)),
        )
        .arg(
            Arg::new("target_id")
                .short('t')
                .long("target")
                .value_name("TARGET_ID")
                .help("ターゲットスピーカーID")
                .required(true)
                .value_parser(clap::value_parser!(i64)),
        )
        .arg(
            Arg::new("input_device")
                .short('i')
                .long("input")
                .value_name("INPUT_DEVICE")
                .help("入力オーディオデバイスの番号")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("output_device")
                .short('o')
                .long("output")
                .value_name("OUTPUT_DEVICE")
                .help("出力オーディオデバイスの番号")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("cutoff")
                .long("cutoff")
                .help("カットオフフィルターを有効にする")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("cutoff_freq")
                .long("cutoff_freq")
                .value_name("CUTOFF_FREQ")
                .help("カットオフ周波数をHzで指定する（デフォルト: 150.0）")
                .value_parser(clap::value_parser!(f32))
                .default_value("150.0"),
        )
        // 新しい引数を追加
        .arg(
            Arg::new("model_sample_rate")
                .short('r')
                .long("model_sample_rate")
                .value_name("MODEL_SAMPLE_RATE")
                .help("モデルへ入力するオーディオのサンプルレートを指定する（デフォルト: 24000）")
                .value_parser(clap::value_parser!(u32))
                .default_value("24000"),
        )
        .arg(
            Arg::new("buffer_size")
                .short('b')
                .long("buffer_size")
                .value_name("BUFFER_SIZE")
                .help("バッファサイズを指定する（デフォルト: 6144）")
                .value_parser(clap::value_parser!(usize))
                .default_value("6144"),
        )
        .arg(
            Arg::new("overlap_length")
                .short('l')
                .long("overlap_length")
                .value_name("OVERLAP_LENGTH")
                .help("SOLAアルゴリズムのオーバーラップ長を指定する（デフォルト: 1024）")
                .value_parser(clap::value_parser!(usize))
                .default_value("1024"),
        )
        .get_matches();

    // 引数の取得
    let onnx_file = matches
        .get_one::<String>("onnx")
        .expect("ONNXファイルを指定してください");
    let source_speaker_id: i64 = *matches
        .get_one::<i64>("source_id")
        .expect("ソーススピーカーIDを指定してください");
    let target_speaker_id: i64 = *matches
        .get_one::<i64>("target_id")
        .expect("ターゲットスピーカーIDを指定してください");
    let input_device_arg: Option<usize> = matches.get_one::<usize>("input_device").copied();
    let output_device_arg: Option<usize> = matches.get_one::<usize>("output_device").copied();
    let cutoff_enabled = matches.get_flag("cutoff");
    let cutoff_freq: f32 = *matches
        .get_one::<f32>("cutoff_freq")
        .expect("cutoff_freqのデフォルト値が設定されていません");
    let model_sample_rate: u32 = *matches
        .get_one::<u32>("model_sample_rate")
        .expect("model_sample_rateのデフォルト値が設定されていません");
    let buffer_size: usize = *matches
        .get_one::<usize>("buffer_size")
        .expect("buffer_sizeのデフォルト値が設定されていません");
    let overlap_length: usize = *matches
        .get_one::<usize>("overlap_length")
        .expect("overlap_lengthのデフォルト値が設定されていません");

    println!("使用するONNXファイル: {}", onnx_file);
    println!(
        "カットオフフィルター: {}",
        if cutoff_enabled {
            format!("有効 (周波数: {} Hz)", cutoff_freq)
        } else {
            "無効".to_string()
        }
    );
    println!("モデルのサンプルレート: {} Hz", model_sample_rate);
    println!("バッファサイズ: {}", buffer_size);
    println!("オーバーラップ長: {}", overlap_length);

    let hparams = Arc::new(AudioParams::new(
        model_sample_rate,
        buffer_size,
        overlap_length,
        source_speaker_id,
        target_speaker_id,
        cutoff_enabled,
        cutoff_freq,
    ));

    // 環境とセッションの構築
    println!("ONNX Runtimeの環境を構築中...");
    let environment = Environment::builder()
        .with_name("MMVC_Client")
        .with_execution_providers([ExecutionProvider::TensorRT(Default::default())])
        .build()
        .context("ONNX Runtimeの環境構築に失敗しました")?
        .into_arc();

    println!("ONNX Runtimeセッションを構築中...");
    let session = Arc::new(
        SessionBuilder::new(&environment)
            .context("ONNX Runtimeセッションビルダーの作成に失敗しました")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("ONNX Runtimeの最適化レベル設定に失敗しました")?
            .with_model_from_file(&onnx_file)
            .context("ONNXモデルの読み込みに失敗しました")?,
    );

    // デバイス選択
    println!("オーディオデバイスを取得中...");
    let host = cpal::default_host();

    let input_devices: Vec<Device> = host
        .input_devices()
        .context("入力デバイスの取得に失敗しました")?
        .filter(|d| d.name().is_ok())
        .collect();
    let output_devices: Vec<Device> = host
        .output_devices()
        .context("出力デバイスの取得に失敗しました")?
        .filter(|d| d.name().is_ok())
        .collect();

    let input_device = if let Some(index) = input_device_arg {
        input_devices
            .get(index)
            .cloned()
            .with_context(|| format!("入力デバイス番号 {} は存在しません", index))?
    } else {
        select_device(&input_devices, "入力").context("入力デバイスの選択に失敗しました")?
    };

    let output_device = if let Some(index) = output_device_arg {
        output_devices
            .get(index)
            .cloned()
            .with_context(|| format!("出力デバイス番号 {} は存在しません", index))?
    } else {
        select_device(&output_devices, "出力").context("出力デバイスの選択に失敗しました")?
    };

    println!(
        "選択された入力デバイス: {}",
        input_device
            .name()
            .unwrap_or_else(|_| "Unknown".to_string())
    );
    println!(
        "選択された出力デバイス: {}",
        output_device
            .name()
            .unwrap_or_else(|_| "Unknown".to_string())
    );

    // チャネルの作成
    let (input_tx, input_rx) = bounded::<Vec<f32>>(0);
    let (output_tx, output_rx) = bounded::<Vec<f32>>(0);

    // 入力ストリームの作成
    println!("入力ストリームを作成中...");
    let input_stream = record_and_resample(Arc::clone(&hparams), input_device, input_tx)?;

    // 処理スレッドの開始
    println!("処理スレッドを開始します...");
    let hparams_clone = Arc::clone(&hparams);
    let session_clone = Arc::clone(&session);
    thread::spawn(move || {
        #[cfg(target_os = "windows")]
        {
            // 処理スレッド内で優先度を設定
            let current_thread = unsafe { GetCurrentThread() };
            let success =
                unsafe { SetThreadPriority(current_thread, THREAD_PRIORITY_TIME_CRITICAL) };
            match success {
                Ok(_) => {
                    println!(
                        "処理スレッドの優先度を THREAD_PRIORITY_TIME_CRITICAL に設定しました。"
                    )
                }
                Err(e) => eprintln!("処理スレッドの優先度設定に失敗しました。エラー: {:?}", e),
            }
        }

        if let Err(e) = processing_thread(hparams_clone, session_clone, input_rx, output_tx) {
            eprintln!("処理スレッドでエラーが発生しました: {:?}", e);
        }
    });

    // 出力ストリームの作成
    println!("出力ストリームを作成中...");
    let output_stream = play_output(Arc::clone(&hparams), output_device, output_rx)?;

    // ストリームの開始
    println!("入力ストリームを再生開始...");
    input_stream
        .play()
        .context("入力ストリームの再生に失敗しました")?;
    println!("出力ストリームを再生開始...");
    output_stream
        .play()
        .context("出力ストリームの再生に失敗しました")?;

    println!("プログラムが正常に起動しました。処理を続けています...");
    // メインスレッドをブロック
    loop {
        thread::sleep(std::time::Duration::from_millis(1000));
    }
}
