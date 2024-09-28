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
use std::cell::RefCell;
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;

#[cfg(target_os = "windows")]
use windows::Win32::System::Threading::{
    GetCurrentProcess, SetPriorityClass, REALTIME_PRIORITY_CLASS,
};

// 定数の定義
const FFT_WINDOW_SIZE: usize = 512;
const HOP_SIZE: usize = 128;
thread_local! {
    static LOCAL_FFT_PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
}
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

// AudioParamsの実装（new関数の変更）
impl AudioParams {
    fn new(
        model_sample_rate: u32,
        buffer_size: usize,
        source_speaker_id: i64,
        target_speaker_id: i64,
        cutoff_enabled: bool,
        cutoff_freq: f32,
    ) -> Self {
        // overlap_lengthをbuffer_sizeに基づいて計算
        let overlap_length = (buffer_size / 4) + (buffer_size / 16) - 256;

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

// 処理スレッド
fn processing_thread(
    hparams: Arc<AudioParams>,
    session: Arc<Session>,
    input_rx: Receiver<Vec<f32>>,
    output_tx: Sender<Vec<f32>>,
    delays: Arc<Mutex<Delays>>, // 追加
) -> Result<()> {
    // Set sola_search_frame equal to overlap_length
    let sola_search_frame = hparams.overlap_length / 2;
    let overlap_size = hparams.overlap_length;
    let mut sola = Sola::new(overlap_size, sola_search_frame);
    let mut prev_input_tail: Vec<f32> = Vec::new();

    while let Ok(mut input_signal) = input_rx.recv() {
        // 入力信号の前後処理
        if !prev_input_tail.is_empty() {
            input_signal.splice(0..0, prev_input_tail.iter().cloned());
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

    match audio_result {
        Some(a) => a,
        None => vec![0.0; signal.len()],
    }
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

    // グローバルなFFTプランナーを取得
    LOCAL_FFT_PLANNER.with(|planner| {
        let mut planner = planner.borrow_mut();
        let fft = planner.plan_fft_forward(fft_size);

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
    });
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
            prev_wav: vec![0.0; overlap_size],
        }
    }

    fn merge(&mut self, wav: &[f32]) -> Vec<f32> {
        if self.prev_wav.is_empty() {
            self.prev_wav = wav[wav.len() - self.overlap_size..].to_vec();
            return wav[..wav.len() - self.overlap_size].to_vec();
        }

        let sola_search_region = &wav[..self.sola_search_frame];
        let cor_nom = Self::convolve(&self.prev_wav, sola_search_region);
        let cor_den = Self::calculate_root_energy(&self.prev_wav, self.sola_search_frame);
        let sola_offset = Self::calculate_sola_offset(&cor_nom, &cor_den);

        let prev_sola_match_region =
            &self.prev_wav[sola_offset..sola_offset + self.sola_search_frame];
        let crossfaded = Self::crossfade(sola_search_region, prev_sola_match_region);

        let total_len = sola_offset + crossfaded.len() + wav[self.sola_search_frame..].len();
        let mut sola_merged_wav = Vec::with_capacity(total_len);
        sola_merged_wav.extend_from_slice(&self.prev_wav[..sola_offset]);
        sola_merged_wav.extend_from_slice(&crossfaded);
        sola_merged_wav.extend_from_slice(&wav[self.sola_search_frame..]);

        let output_len = sola_merged_wav.len() - (sola_offset + self.overlap_size);
        let output_wav = sola_merged_wav[..output_len].to_vec();

        let prev_wav_start = wav.len() - (self.overlap_size + sola_offset);
        let prev_wav_end = wav.len() - sola_offset;
        self.prev_wav = wav[prev_wav_start..prev_wav_end].to_vec();

        output_wav
    }

    fn calculate_sola_offset(cor_nom: &[f32], cor_den: &[f32]) -> usize {
        cor_nom
            .iter()
            .zip(cor_den)
            .enumerate()
            .max_by(|(_, (nom, den)), (_, (nom2, den2))| {
                let val1 = *nom / *den;
                let val2 = *nom2 / *den2;
                val1.partial_cmp(&val2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    fn calculate_root_energy(a: &[f32], len: usize) -> Vec<f32> {
        let n = a.len();
        let size = n - len + 1;
        let mut result = Vec::with_capacity(size);
        let mut sum = 0.0;

        for item in a.iter().take(len) {
            let val = item;
            sum += val * val;
        }
        result.push(sum);

        for i in len..n {
            let val_add = a[i];
            let val_sub = a[i - len];
            sum += val_add * val_add - val_sub * val_sub;
            result.push(sum);
        }

        result
    }

    fn convolve(a: &[f32], b: &[f32]) -> Vec<f32> {
        let n = a.len();
        let m = b.len();
        let size = n - m + 1;
        let mut result = Vec::with_capacity(size);

        for i in 0..size {
            let mut sum = 0.0;
            for j in 0..m {
                sum += a[i + j] * b[j];
            }
            result.push(sum);
        }

        result
    }

    fn crossfade(cur_wav: &[f32], prev_wav: &[f32]) -> Vec<f32> {
        let len = prev_wav.len();
        let mut output_wav = Vec::with_capacity(len);

        for i in 0..len {
            let percent = i as f32 / len as f32;
            let cos_val = (percent * PI).cos();
            let prev_strength = (1.0 + cos_val) * 0.5;
            let cur_strength = (1.0 - cos_val) * 0.5;
            output_wav.push(prev_wav[i] * prev_strength + cur_wav[i] * cur_strength);
        }

        output_wav
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

    input_device_names: Vec<String>,
    output_device_names: Vec<String>,

    input_stream: Option<cpal::Stream>,
    output_stream: Option<cpal::Stream>,
    processing_handle: Option<JoinHandle<()>>,

    is_running: bool,
    error_message: Option<String>,

    delays: Arc<Mutex<Delays>>,
    environment: Option<Arc<Environment>>,
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
            buffer_size: 8192,

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
            environment: None,
        }
    }

    fn calculate_latency_ms(&self) -> f32 {
        let buffer_delay = self.buffer_size as f32 / self.model_sample_rate as f32 * 1000.0;

        // overlap_length を計算
        let overlap_length = (self.buffer_size / 4) + (self.buffer_size / 16) - 256;
        let overlap_delay = overlap_length as f32 / self.model_sample_rate as f32 * 1000.0;

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
            .with_execution_providers([ExecutionProvider::CUDA(Default::default())])
            .build()
            .context("ONNX Runtimeの環境構築に失敗しました")?
            .into_arc();
        self.environment = Some(environment.clone());

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
        let min_buffer = 4096.0;
        let max_buffer = 16384.0;
        let min_overlap = min_buffer / 8.0;
        let max_overlap = (max_buffer / 4.0) + (max_buffer / 16.0) - 256.0;

        let buffer_factor =
            ((self.buffer_size as f32 - min_buffer) / (max_buffer - min_buffer)).clamp(0.0, 1.0);

        // overlap_length を計算
        let overlap_length = (self.buffer_size / 4) + (self.buffer_size / 16) - 256;
        let overlap_factor =
            ((overlap_length as f32 - min_overlap) / (max_overlap - min_overlap)).clamp(0.0, 1.0);

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
        self.environment.take();

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
                                Slider::new(&mut self.buffer_size, 4096..=16384)
                                    .step_by(512.0) // 512刻みで調整可能に
                                    .text("サンプル数")
                            )
                            .on_hover_text("バッファサイズが小さいと遅延が低く、大きいと音質が向上します。");
                        });
                    });

                    // 音質と遅延のバランスを示すインディケーター
                    ui.separator();

                    // 音質と遅延の計算
                    let quality = self.calculate_quality();
                    let latency_ms = self.calculate_latency_ms();

                    // バッファ遅延とオーバーラップ遅延の計算
                    let buffer_delay = self.buffer_size as f32 / self.model_sample_rate as f32 * 1000.0;

                    // overlap_length を計算
                    let overlap_length = (self.buffer_size / 4) + (self.buffer_size / 16) - 256;
                    let overlap_delay = overlap_length as f32 / self.model_sample_rate as f32 * 1000.0;

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
                        } else if ui.button("停止").clicked() {
                            // 処理を停止
                            self.stop_processing();
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
        // transparent() option they get immediate results.
        egui::Color32::from_rgba_unmultiplied(12, 12, 12, 180).to_normalized_gamma_f32()

        // _visuals.window_fill() would also be a natural choice
    }

    fn persist_egui_memory(&self) -> bool {
        true
    }

    fn raw_input_hook(&mut self, _ctx: &egui::Context, _raw_input: &mut egui::RawInput) {}
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    #[cfg(target_os = "windows")]
    {
        // 現在のプロセスを取得
        let current_process = unsafe { GetCurrentProcess() };

        // プロセスの優先度をリアルタイムに設定
        if unsafe { SetPriorityClass(current_process, REALTIME_PRIORITY_CLASS) }.is_err() {
            eprintln!("プロセスの優先度設定に失敗しました。");
        } else {
            println!("プロセスの優先度をリアルタイムに設定しました。");
        }
    }
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
