use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, StreamConfig};
use crossbeam_channel::{bounded, Receiver, Sender};
use eframe::{self};
use egui::{self, Color32, ComboBox, ProgressBar, Slider};
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
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;

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

// 遅延計算用の構造体
struct Delays {
    processing_delay_ms: f32,
    inference_delay_ms: f32,
    resampling_delay_ms: f32,
}
// Implement YIN pitch detection
fn yin_pitch_detect(signal: &[f32], sample_rate: u32) -> Result<f32> {
    let tau_max = (sample_rate as f32 / 50.0) as usize; // Minimum f0 = 50 Hz
    let tau_min = (sample_rate as f32 / 1000.0) as usize; // Maximum f0 = 1000 Hz

    let mut yin_buffer = vec![0.0; tau_max];
    let signal_length = signal.len();

    // Step 1: Difference function
    for tau in tau_min..tau_max {
        for i in 0..(signal_length - tau) {
            let delta = signal[i] - signal[i + tau];
            yin_buffer[tau] += delta * delta;
        }
    }

    // Step 2: Cumulative mean normalized difference function
    let mut cumulative_mean_normalized = vec![0.0; tau_max];
    cumulative_mean_normalized[tau_min] = 1.0;
    let mut running_sum = 0.0;
    for tau in tau_min..tau_max {
        running_sum += yin_buffer[tau];
        cumulative_mean_normalized[tau] = yin_buffer[tau] / (running_sum / (tau as f32));
    }

    // Step 3: Absolute threshold
    let threshold = 0.1;
    let mut tau_estimate = None;
    for tau in tau_min..tau_max {
        if cumulative_mean_normalized[tau] < threshold {
            // Step 4: Parabolic interpolation for better accuracy
            if tau + 1 < tau_max && tau > tau_min {
                let better_tau = parabolic_interpolation(
                    cumulative_mean_normalized[tau - 1],
                    cumulative_mean_normalized[tau],
                    cumulative_mean_normalized[tau + 1],
                );
                tau_estimate = Some(tau as f32 + better_tau);
                break;
            } else {
                tau_estimate = Some(tau as f32);
                break;
            }
        }
    }

    if let Some(tau) = tau_estimate {
        let f0 = sample_rate as f32 / tau;
        Ok(f0)
    } else {
        // Fallback if no pitch detected
        Ok(0.0)
    }
}

fn parabolic_interpolation(y0: f32, y1: f32, y2: f32) -> f32 {
    let denominator = 2.0 * (y0 - 2.0 * y1 + y2);
    if denominator.abs() < std::f32::EPSILON {
        0.0
    } else {
        (y0 - y2) / denominator
    }
}fn processing_thread(
    hparams: Arc<AudioParams>,
    session: Arc<Session>,
    input_rx: Receiver<Vec<f32>>,
    output_tx: Sender<Vec<f32>>,
    delays: Arc<Mutex<Delays>>, // 追加
) -> Result<()> {
    let overlap_size = hparams.overlap_length;
    let mut psola = PSola::new(overlap_size, hparams.model_sample_rate);
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

        // 音声変換処理の遅延計測開始
        let start_processing = std::time::Instant::now();
        let processed_signal = audio_transform(&hparams, &session, &input_signal, &delays);
        let processing_duration = start_processing.elapsed().as_secs_f32() * 1000.0; // ミリ秒

        // 遅延を更新
        {
            let mut delays_guard = delays.lock().unwrap();
            delays_guard.processing_delay_ms =
                (delays_guard.processing_delay_ms + processing_duration) / 2.0;
        }

        // PSOLAによるマージ
        let merged_signal = psola.merge(&processed_signal);

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
fn audio_transform(
    hparams: &AudioParams,
    session: &Session,
    signal: &[f32],
    delays: &Arc<Mutex<Delays>>, // 追加
) -> Vec<f32> {
    // STFTパディングの追加
    let pad_size = (FFT_WINDOW_SIZE - HOP_SIZE) / 2;
    let mut padded_signal = vec![0.0; pad_size];
    padded_signal.extend_from_slice(signal);
    padded_signal.extend(vec![0.0; pad_size]);

    // STFTの適用
    let spec = apply_stft_with_hann_window(&padded_signal, hparams);

    let spec_lengths = Array1::from_elem(1, spec.shape()[2] as i64);
    let source_speaker_id_src = Array1::from_elem(1, hparams.source_speaker_id);

    // ONNX推論の遅延計測開始
    let start_inference = std::time::Instant::now();
    let audio_result = run_onnx_model_inference(
        session,
        &spec,
        &spec_lengths,
        &source_speaker_id_src,
        hparams.target_speaker_id,
    );
    let inference_duration = start_inference.elapsed().as_secs_f32() * 1000.0; // ミリ秒

    // 推論遅延を更新
    {
        let mut delays_guard = delays.lock().unwrap();
        delays_guard.inference_delay_ms =
            (delays_guard.inference_delay_ms + inference_duration) / 2.0;
    }

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
    delays: Arc<Mutex<Delays>>, // 追加
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
        10,
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

                    // リサンプリングの遅延計測開始
                    let start_resampling = std::time::Instant::now();

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

                    let resampling_duration = start_resampling.elapsed().as_secs_f32() * 1000.0; // ミリ秒

                    // リサンプリング遅延を更新
                    {
                        let mut delays_guard = delays.lock().unwrap();
                        delays_guard.resampling_delay_ms =
                            (delays_guard.resampling_delay_ms + resampling_duration) / 2.0;
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
        10,
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

// メインアプリケーションクラス
struct MyApp {
    onnx_file: String,
    source_speaker_id: i64,
    target_speaker_id: i64,
    input_device_index: Option<usize>,
    output_device_index: Option<usize>,
    cutoff_enabled: bool,
    cutoff_freq: f32,
    model_sample_rate: u32,
    buffer_size: usize,
    overlap_length: usize,

    input_device_names: Vec<String>,
    output_device_names: Vec<String>,

    input_stream: Option<cpal::Stream>,
    output_stream: Option<cpal::Stream>,
    processing_handle: Option<JoinHandle<()>>,

    is_running: bool,
    error_message: Option<String>,

    delays: Arc<Mutex<Delays>>, // 追加
}

use egui::{FontData, FontDefinitions, FontFamily};

impl MyApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // カスタムフォントの読み込み
        let mut fonts = FontDefinitions::default();

        // フォントデータを追加（パスは実際のフォントファイルの場所に合わせて変更してください）
        fonts.font_data.insert(
            "NotoSansJP".to_owned(),
            FontData::from_static(include_bytes!("../assets/fonts/NotoSansJP-Regular.ttf")),
        );
        // スタイルのカスタマイズ
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(8.0, 4.0); // アイテム間のスペースを調整
        style.spacing.window_margin = egui::Margin::same(10.0); // ウィンドウの余白を調整
        cc.egui_ctx.set_style(style);
        // プロポーショナルフォントファミリーに追加
        fonts
            .families
            .get_mut(&FontFamily::Proportional)
            .unwrap()
            .insert(0, "NotoSansJP".to_owned());

        // 必要に応じてモノスペースフォントファミリーにも追加
        fonts
            .families
            .get_mut(&FontFamily::Monospace)
            .unwrap()
            .push("NotoSansJP".to_owned());

        // フォントを設定
        cc.egui_ctx.set_fonts(fonts);

        // デバイスリストの取得（既存のコード）
        let host = cpal::default_host();

        let input_devices: Vec<Device> = host
            .input_devices()
            .map(|devices| devices.collect())
            .unwrap_or_else(|_| Vec::new());

        let output_devices: Vec<Device> = host
            .output_devices()
            .map(|devices| devices.collect())
            .unwrap_or_else(|_| Vec::new());

        let input_device_names = input_devices
            .iter()
            .map(|d| d.name().unwrap_or_else(|_| "Unknown".to_string()))
            .collect();

        let output_device_names = output_devices
            .iter()
            .map(|d| d.name().unwrap_or_else(|_| "Unknown".to_string()))
            .collect();

        Self {
            onnx_file: "".to_string(),
            source_speaker_id: 0,
            target_speaker_id: 107,
            input_device_index: None,
            output_device_index: None,
            cutoff_enabled: false,
            cutoff_freq: 150.0,
            model_sample_rate: 24000,
            buffer_size: 6144,
            overlap_length: 1536,

            input_device_names,
            output_device_names,

            input_stream: None,
            output_stream: None,
            processing_handle: None,

            is_running: false,
            error_message: None,

            delays: Arc::new(Mutex::new(Delays {
                processing_delay_ms: 0.0,
                inference_delay_ms: 0.0,
                resampling_delay_ms: 0.0,
            })),
        }
    }

    fn calculate_latency_ms(&self) -> f32 {
        let buffer_delay = self.buffer_size as f32 / self.model_sample_rate as f32 * 1000.0;
        let overlap_delay = self.overlap_length as f32 / self.model_sample_rate as f32 * 1000.0;

        let delays_guard = self.delays.lock().unwrap();
        let resampling_delay = delays_guard.resampling_delay_ms;
        let processing_delay = delays_guard.processing_delay_ms;

        buffer_delay + overlap_delay + resampling_delay + processing_delay
    }

    fn start_processing(&mut self) -> Result<()> {
        // パラメータのチェック
        if self.onnx_file.is_empty() {
            return Err(anyhow::anyhow!("ONNXファイルを選択してください"));
        }

        if self.input_device_index.is_none() || self.output_device_index.is_none() {
            return Err(anyhow::anyhow!("入力および出力デバイスを選択してください"));
        }

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
                .with_model_from_file(&self.onnx_file)
                .context("ONNXモデルの読み込みに失敗しました")?,
        );

        // デバイスの取得
        let host = cpal::default_host();

        let input_device = host
            .input_devices()?
            .nth(self.input_device_index.unwrap())
            .context("入力デバイスの選択に失敗しました")?;

        let output_device = host
            .output_devices()?
            .nth(self.output_device_index.unwrap())
            .context("出力デバイスの選択に失敗しました")?;

        // ハイパーパラメータの設定
        let hparams = Arc::new(AudioParams::new(
            self.model_sample_rate,
            self.buffer_size,
            self.overlap_length,
            self.source_speaker_id,
            self.target_speaker_id,
            self.cutoff_enabled,
            self.cutoff_freq,
        ));

        // チャネルの作成
        let (input_tx, input_rx) = bounded::<Vec<f32>>(0);
        let (output_tx, output_rx) = bounded::<Vec<f32>>(0);

        // 入力ストリームの作成
        println!("入力ストリームを作成中...");
        let delays_clone = Arc::clone(&self.delays); // 追加
        let input_stream =
            record_and_resample(Arc::clone(&hparams), input_device, input_tx, delays_clone)?; // 修正

        // 処理スレッドの開始
        println!("処理スレッドを開始します...");
        let hparams_clone = Arc::clone(&hparams);
        let session_clone = Arc::clone(&session);
        let delays_clone = Arc::clone(&self.delays); // 追加
        let processing_handle = thread::spawn(move || {
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

            if let Err(e) = processing_thread(
                hparams_clone,
                session_clone,
                input_rx,
                output_tx,
                delays_clone,
            ) {
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

        // ハンドルの保存
        self.input_stream = Some(input_stream);
        self.output_stream = Some(output_stream);
        self.processing_handle = Some(processing_handle);

        Ok(())
    }

    fn calculate_quality(&self) -> f32 {
        let min_buffer = 2048.0;
        let max_buffer = 16384.0;
        let min_overlap = min_buffer / 4.0;
        let max_overlap = (max_buffer / 2.0) - 128.0;

        let buffer_factor =
            ((self.buffer_size as f32 - min_buffer) / (max_buffer - min_buffer)).clamp(0.0, 1.0);
        let overlap_factor = ((self.overlap_length as f32 - min_overlap)
            / (max_overlap - min_overlap))
            .clamp(0.0, 1.0);

        (buffer_factor + overlap_factor) / 2.0
    }

    fn stop_processing(&mut self) {
        // ストリームを停止
        self.input_stream.take();
        self.output_stream.take();

        // 処理スレッドを終了
        if let Some(handle) = self.processing_handle.take() {
            // 処理スレッドが終了するまで待機
            handle.join().expect("処理スレッドの終了に失敗しました");
        }

        self.is_running = false;
    }
}
impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("MMVC クライアント");
            // スクロール可能な領域にUI要素を配置
            egui::ScrollArea::vertical().show(ui, |ui| {

                // エラーメッセージの表示
                if let Some(ref msg) = self.error_message {
                    ui.colored_label(egui::Color32::RED, msg);
                }

                ui.separator();

                // ONNXモデルファイルの選択
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("ONNXモデルファイル:");
                        ui.add_enabled_ui(!self.is_running, |ui| {
                            if ui.button("選択...").clicked() {
                                // ファイルダイアログを開く
                                if let Some(path) = rfd::FileDialog::new().pick_file() {
                                    self.onnx_file = path.to_string_lossy().to_string();
                                }
                            }
                        });
                    });
                    ui.label(&self.onnx_file);
                });

                ui.separator();

                // スピーカーIDの入力
                ui.group(|ui| {
                    egui::Grid::new("model_settings")
                        .num_columns(3)
                        .show(ui, |ui| {
                            ui.add_enabled_ui(!self.is_running, |ui| {
                                ui.label("モデルのサンプルレート:");
                                ui.add(
                                    egui::DragValue::new(&mut self.model_sample_rate)
                                        .speed(1000)
                                        .range(8000..=48000),
                                );
                                ui.end_row();
                                ui.label("ソーススピーカーID:");
                                ui.add(
                                    egui::DragValue::new(&mut self.source_speaker_id)
                                        .speed(1)
                                        .range(0..=1000),
                                );
                                ui.end_row();

                                ui.label("ターゲットスピーカーID:");
                                ui.add(
                                    egui::DragValue::new(&mut self.target_speaker_id)
                                        .speed(1)
                                        .range(0..=1000),
                                );
                                ui.end_row();
                            });
                        });
                });

                ui.separator();

                // デバイスの選択
                ui.group(|ui| {
                    ui.add_enabled_ui(!self.is_running, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("入力デバイス:");
                            ComboBox::from_id_source("input_device")
                                .selected_text(
                                    self.input_device_index
                                        .and_then(|i| self.input_device_names.get(i))
                                        .unwrap_or(&"入力デバイスを選択".to_string())
                                        .clone(),
                                )
                                .width(200.0) // 幅を固定
                                .show_ui(ui, |ui| {
                                    for (i, name) in self.input_device_names.iter().enumerate() {
                                        ui.selectable_value(&mut self.input_device_index, Some(i), name);
                                    }
                                });
                        });

                        ui.horizontal(|ui| {
                            ui.label("出力デバイス:");
                            ComboBox::from_id_source("output_device")
                                .selected_text(
                                    self.output_device_index
                                        .and_then(|i| self.output_device_names.get(i))
                                        .unwrap_or(&"出力デバイスを選択".to_string())
                                        .clone(),
                                )
                                .width(200.0) // 幅を固定
                                .show_ui(ui, |ui| {
                                    for (i, name) in self.output_device_names.iter().enumerate() {
                                        ui.selectable_value(&mut self.output_device_index, Some(i), name);
                                    }
                                });
                        });
                    });
                });

                ui.separator();

                // カットオフフィルター
                ui.add_enabled_ui(!self.is_running, |ui| {
                    ui.checkbox(&mut self.cutoff_enabled, "カットオフフィルターを有効にする");
                    if self.cutoff_enabled {
                        ui.horizontal(|ui| {
                            ui.label("カットオフ周波数:");
                            ui.add(egui::Slider::new(&mut self.cutoff_freq, 1.0..=300.0).text("Hz"));
                        });
                    }
                });

                ui.separator();
                ui.group(|ui| {
                    ui.add_enabled_ui(!self.is_running, |ui| {
                        // バッファサイズ（チャンクサイズ）の設定
                        ui.horizontal(|ui| {
                            ui.label("バッファサイズ:");
                            ui.add(
                                Slider::new(&mut self.buffer_size, 2048..=16384)
                                    .step_by(512.0) // 512刻みで調整可能に
                                    .text("バイト")
                            )
                            .on_hover_text("バッファサイズが小さいと遅延が低く、大きいと音質が向上します。");
                        });

                        // オーバーラップ長の設定
                        ui.horizontal(|ui| {
                            ui.label("オーバーラップ長:");
                            ui.add(
                                Slider::new(
                                    &mut self.overlap_length,
                                    (self.buffer_size / 4)..=(self.buffer_size / 2 - 128),
                                )
                                .step_by(128.0) // 128刻みで調整可能に
                                .text("バイト")
                            )
                            .on_hover_text("オーバーラップ長が短いと遅延が小さく、大きいと音質が向上します。");
                        });
                    });

                    // 音質と遅延のバランスを示すインディケーター
                    ui.separator();

                    // 音質と遅延の計算
                    let quality = self.calculate_quality();
                    let latency_ms = self.calculate_latency_ms();

                    // バッファ遅延とオーバーラップ遅延の計算
                    let buffer_delay = self.buffer_size as f32 / self.model_sample_rate as f32 * 1000.0;
                    let overlap_delay = self.overlap_length as f32 / self.model_sample_rate as f32 * 1000.0;

                    // 縦に並べる
                    ui.vertical(|ui| {
                        ui.label("音質と遅延のバランス:");

                        // 音質のプログレスバー
                        ui.horizontal(|ui| {
                            ui.label("音質:");
                            ui.add(
                                ProgressBar::new(quality)
                                    .fill(Color32::from_rgb(0, 200, 0)) // 緑色
                                    .show_percentage()
                            )
                            .on_hover_text("音質の指標です。バッファサイズとオーバーラップ長が大きいほど高くなります。");
                        });

                        // 遅延の表示
                        ui.horizontal(|ui| {
                            ui.label("遅延:");
                            ui.label(format!("{:.2} ms", latency_ms));
                        });

                        ui.label("※ 音質や遅延は推定値です。実際の環境により異なる場合があります。");

                        // 遅延詳細の表示を折りたたみ可能にする
                        ui.collapsing("遅延の詳細を見る", |ui| {
                            let delays_guard = self.delays.lock().unwrap();
                            ui.separator();
                            ui.label(format!("バッファ遅延: {:.2} ms", buffer_delay));
                            ui.label(format!("オーバーラップ遅延: {:.2} ms", overlap_delay));
                            ui.label(format!("リサンプリング遅延: {:.2} ms", delays_guard.resampling_delay_ms));
                            ui.label(format!("推論遅延: {:.2} ms", delays_guard.inference_delay_ms));
                        })
                    });
                });

                ui.separator();

                // スタート/ストップボタンを中央揃えにする
                ui.with_layout(
                    egui::Layout::centered_and_justified(egui::Direction::TopDown),
                    |ui| {
                        if !self.is_running {
                            let is_ready_to_start = !self.onnx_file.is_empty()
                                && self.input_device_index.is_some()
                                && self.output_device_index.is_some();

                            if ui
                                .add_enabled(
                                    is_ready_to_start,
                                    egui::Button::new("開始").min_size(egui::Vec2::new(100.0, 30.0)),
                                )
                                .clicked()
                            {
                                // 処理を開始
                                match self.start_processing() {
                                    Ok(_) => {
                                        self.is_running = true;
                                        self.error_message = None;
                                    }
                                    Err(e) => {
                                        self.error_message =
                                            Some(format!("処理の開始に失敗しました: {:?}", e));
                                    }
                                }
                            }
                        } else {
                            if ui.button("停止").clicked() {
                                // 処理を停止
                                self.stop_processing();
                            }
                        }
                    },
                );
            });
        });
    }

    fn save(&mut self, _storage: &mut dyn eframe::Storage) {}

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {}

    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(30)
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        // NOTE: a bright gray makes the shadows of the windows look weird.
        // We use a bit of transparency so that if the user switches on the
        // `transparent()` option they get immediate results.
        egui::Color32::from_rgba_unmultiplied(12, 12, 12, 180).to_normalized_gamma_f32()

        // _visuals.window_fill() would also be a natural choice
    }

    fn persist_egui_memory(&self) -> bool {
        true
    }

    fn raw_input_hook(&mut self, _ctx: &egui::Context, _raw_input: &mut egui::RawInput) {}
}

fn main() -> Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "MMVC Client",
        options,
        Box::new(|cc| Ok(Box::new(MyApp::new(cc)))),
    )
    .unwrap_or_else(|e| {
        eprintln!("エラーが発生しました: {}", e);
    });
    Ok(())
}
// PSOLAアルゴリズムの実装
struct PSola {
    overlap_size: usize,
    prev_wav: Vec<f32>,
    prev_pitch_period: usize,
    sample_rate: u32,
}

impl PSola {
    fn new(overlap_size: usize, sample_rate: u32) -> Self {
        Self {
            overlap_size,
            prev_wav: Vec::new(),
            prev_pitch_period: 0,
            sample_rate,
        }
    }

    fn merge(&mut self, wav: &[f32]) -> Vec<f32> {
        // ピッチ検出
        let f0 = yin_pitch_detect(wav, self.sample_rate).unwrap_or(0.0);
        let current_pitch_period = if f0 > 0.0 {
            (self.sample_rate as f32 / f0) as usize
        } else {
            self.prev_pitch_period
        };

        if self.prev_wav.is_empty() {
            let output_wav = wav[..wav.len() - self.overlap_size].to_vec();
            self.prev_wav = wav[wav.len() - self.overlap_size..].to_vec();
            self.prev_pitch_period = current_pitch_period;
            return output_wav;
        }

        // PSOLAのためのオーバーラップ位置の決定
        let search_range = current_pitch_period;
        let max_offset = self.prev_wav.len().saturating_sub(search_range);
        let (best_offset, _) = (0..=max_offset)
            .map(|offset| {
                let prev_segment = &self.prev_wav[offset..offset + search_range];
                let current_segment = &wav[..search_range.min(wav.len())];

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
        self.prev_pitch_period = current_pitch_period;

        output_wav
    }

    fn calculate_correlation(a: &[f32], b: &[f32]) -> f32 {
        let dot_product = a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f32>();
        let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
        dot_product / (norm_a * norm_b + 1e-8)
    }

    fn crossfade(prev_wav: &[f32], cur_wav: &[f32]) -> Vec<f32> {
        let len = prev_wav.len().min(cur_wav.len());
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
