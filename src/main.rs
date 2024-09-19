use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, StreamConfig};
use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::{Array1, Array3, CowArray, IxDyn};
use ort::{
    tensor::OrtOwnedTensor, Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult,
    Session, SessionBuilder, Value,
};
use rustfft::Fft;
use rustfft::{num_complex::Complex, FftPlanner};
use speexdsp_resampler::State as SpeexResampler;
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::{env, thread};

const BUFFER_SIZE: usize = 8192; // バッファサイズ

/// オーディオ処理に必要なパラメータを保持する構造体
struct AudioParams {
    sample_rate: u32,
    fft_window_size: usize,
    hop_size: usize,
    source_speaker_id: i64,
    target_speaker_id: i64,
    dispose_stft_frames: usize,
    dispose_conv1d_samples: usize,
    hann_window: Vec<f32>,
    fft: Arc<dyn Fft<f32> + Send + Sync>,
}

impl AudioParams {
    fn new(source_speaker_id: i64, target_speaker_id: i64) -> Self {
        let fft_window_size = 512;
        let hann_window: Vec<f32> = (0..fft_window_size)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / fft_window_size as f32).cos()))
            .collect();

        let mut fft_planner = FftPlanner::<f32>::new();
        let fft_instance = fft_planner.plan_fft_forward(fft_window_size);
        let fft: Arc<dyn Fft<f32> + Send + Sync> = fft_instance;

        Self {
            sample_rate: 24000,
            fft_window_size,
            hop_size: 128,
            source_speaker_id,
            target_speaker_id,
            dispose_stft_frames: 0,
            dispose_conv1d_samples: 0,
            hann_window,
            fft,
        }
    }
}

/// オーディオ処理を行うスレッド
fn processing_thread(
    hparams: Arc<AudioParams>,
    session: Arc<Session>,
    input_rx: Receiver<Vec<f32>>,
    output_tx: Sender<Vec<f32>>,
) -> Result<()> {
    let sola_search_frame = 1024;
    let overlap_size = 1024;
    let mut sola = Sola::new(overlap_size, sola_search_frame);
    let mut prev_input_tail: Vec<f32> = Vec::new();

    while let Ok(mut input_signal) = input_rx.recv() {
        // 前回の入力の終端を現在の入力の先頭に結合
        if !prev_input_tail.is_empty() {
            let mut extended_signal = prev_input_tail.clone();
            extended_signal.extend(&input_signal); // 修正箇所
            input_signal = extended_signal;
        }

        // 音声変換処理
        let processed_signal = audio_transform(&hparams, &session, &input_signal);

        // 出力が無音の場合、スキップ
        if processed_signal.iter().all(|&x| x == 0.0) {
            continue;
        }

        // SOLAによるマージ
        let merged_signal = sola.merge(&processed_signal);

        // 次回のために入力の終端を保持
        let dispose_length = (hparams.dispose_stft_frames * hparams.hop_size)
            + hparams.dispose_conv1d_samples
            + overlap_size;
        let tail_len = input_signal.len().min(dispose_length);
        prev_input_tail = input_signal[input_signal.len() - tail_len..].to_vec();

        // マージした信号を送信
        if output_tx.send(merged_signal).is_err() {
            break;
        }
    }
    Ok(())
}

/// 音声変換処理のメイン関数
fn audio_transform(hparams: &AudioParams, session: &Session, signal: &[f32]) -> Vec<f32> {
    // STFTパディングの追加
    let pad_size = (hparams.fft_window_size - hparams.hop_size) / 2;
    let mut padded_signal = vec![0.0; pad_size];
    padded_signal.extend_from_slice(signal);
    padded_signal.extend(vec![0.0; pad_size]);

    // STFTの適用
    let mut spec = apply_stft_with_hann_window(
        &padded_signal,
        hparams, // 修正
    );

    // STFTパディングによる影響を削除
    let total_frames = spec.shape()[2];
    let dispose_frames = hparams.dispose_stft_frames;

    if total_frames > 2 * dispose_frames {
        let start = dispose_frames;
        let end = total_frames - dispose_frames;
        spec = spec.slice(ndarray::s![.., .., start..end]).to_owned();
    }

    let spec_lengths = Array1::from_elem(1, spec.shape()[2] as i64);
    let source_speaker_id_src = Array1::from_elem(1, hparams.source_speaker_id);

    let audio_result = run_onnx_model_inference(
        session,
        &spec,
        &spec_lengths,
        &source_speaker_id_src,
        hparams.target_speaker_id,
    );

    let mut audio = match audio_result {
        Some(a) => a,
        None => return vec![0.0; signal.len()],
    };

    // Conv1Dパディングによる影響を削除
    let dispose_samples = hparams.dispose_conv1d_samples;
    if audio.len() > 2 * dispose_samples {
        let start = dispose_samples;
        let end = audio.len() - dispose_samples;
        audio = audio[start..end].to_vec();
    }

    audio
}

/// ONNXモデルでの推論を実行
fn run_onnx_model_inference(
    session: &Session,
    spectrogram: &Array3<f32>,
    spectrogram_lengths: &Array1<i64>,
    source_speaker_id: &Array1<i64>,
    target_speaker_id: i64,
) -> Option<Vec<f32>> {
    // ArrayBase を動的次元に変換し、CowArray として扱う
    let spec_cow: CowArray<f32, IxDyn> = CowArray::from(spectrogram.clone().into_dyn());
    let spec_lengths_cow: CowArray<i64, IxDyn> =
        CowArray::from(spectrogram_lengths.clone().into_dyn());
    let source_id_cow: CowArray<i64, IxDyn> = CowArray::from(source_speaker_id.clone().into_dyn());
    let target_id: CowArray<i64, IxDyn> =
        CowArray::from(Array1::from_elem(1, target_speaker_id).into_dyn());

    // Value::from_array に渡す際に CowArray として扱う
    let inputs = vec![
        Value::from_array(session.allocator(), &spec_cow).ok()?,
        Value::from_array(session.allocator(), &spec_lengths_cow).ok()?,
        Value::from_array(session.allocator(), &source_id_cow).ok()?,
        Value::from_array(session.allocator(), &target_id).ok()?,
    ];

    let outputs = session.run(inputs).ok()?;

    let audio_output: OrtOwnedTensor<f32, _> = outputs[0].try_extract().ok()?;

    Some(audio_output.view().to_owned().into_raw_vec())
}

/// ハン窓を使用したSTFTの適用
fn apply_stft_with_hann_window(
    audio_signal: &[f32],
    hparams: &AudioParams, // AudioParamsを引数に追加
) -> Array3<f32> {
    let fft_size = hparams.fft_window_size;
    let hop_size = hparams.hop_size;
    let window = &hparams.hann_window;
    let fft = Arc::clone(&hparams.fft);

    let num_frames = (audio_signal.len() - fft_size) / hop_size + 1;
    let mut spectrogram = Array3::<f32>::zeros((1, fft_size / 2 + 1, num_frames));

    for (i, frame) in audio_signal.windows(fft_size).step_by(hop_size).enumerate() {
        let mut complex_buffer: Vec<Complex<f32>> = frame
            .iter()
            .zip(window)
            .map(|(&sample, &w)| Complex::new(sample * w, 0.0))
            .collect();

        fft.process(&mut complex_buffer);

        for (j, bin) in complex_buffer.iter().take(fft_size / 2 + 1).enumerate() {
            let power = bin.norm();
            spectrogram[[0, j, i]] = power;
        }
    }

    spectrogram
}

/// 音声の録音とリサンプリングを行う関数
fn record_and_resample(
    hparams: Arc<AudioParams>,
    input_device: Device,
    input_tx: Sender<Vec<f32>>,
) -> cpal::Stream {
    let input_config = input_device.default_input_config().unwrap();
    let channels = input_config.channels() as usize;
    let input_sample_rate = input_config.sample_rate().0;
    let mut resampler = SpeexResampler::new(
        1,
        input_sample_rate as usize,
        hparams.sample_rate as usize,
        10,
    )
    .unwrap();

    let input_stream_config: StreamConfig = input_config.into();
    let mut buffer = Vec::new();

    input_device
        .build_input_stream(
            &input_stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mono_signal: Vec<f32> = data
                    .chunks(channels)
                    .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                    .collect();

                buffer.extend_from_slice(&mono_signal);

                while buffer.len() >= BUFFER_SIZE {
                    let chunk = buffer.drain(..BUFFER_SIZE).collect::<Vec<f32>>(); // 修正箇所
                    let mut resampled = vec![
                        0.0;
                        (chunk.len() as f64 * hparams.sample_rate as f64 / input_sample_rate as f64)
                            .ceil() as usize
                    ];
                    // リサンプリング
                    let (_, output_generated) =
                        resampler.process_float(0, &chunk, &mut resampled).unwrap();
                    resampled.truncate(output_generated);

                    if input_tx.send(resampled).is_err() {
                        break;
                    }
                }
            },
            move |err| eprintln!("Input stream error: {}", err),
            None,
        )
        .unwrap()
}

/// 出力を再生する関数
fn play_output(
    hparams: Arc<AudioParams>,
    output_device: Device,
    output_rx: Receiver<Vec<f32>>,
) -> cpal::Stream {
    let output_config = output_device.default_output_config().unwrap();
    let output_sample_rate = output_config.sample_rate().0;
    let mut resampler = SpeexResampler::new(
        1,
        hparams.sample_rate as usize,
        output_sample_rate as usize,
        10,
    )
    .unwrap();
    let mut output_buffer: VecDeque<f32> = VecDeque::with_capacity(BUFFER_SIZE);
    let output_stream_config = output_config.config();
    let channels = output_config.channels() as usize;

    output_device
        .build_output_stream(
            &output_stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                while let Ok(processed_signal) = output_rx.try_recv() {
                    let mut resampled = vec![
                        0.0;
                        (processed_signal.len() as f64 * output_sample_rate as f64
                            / hparams.sample_rate as f64)
                            .ceil() as usize
                    ];
                    let (_, output_generated) = resampler
                        .process_float(0, &processed_signal, &mut resampled)
                        .unwrap();
                    resampled.truncate(output_generated);
                    output_buffer.extend(resampled);
                }

                for frame in data.chunks_mut(channels) {
                    let sample = output_buffer.pop_front().unwrap_or(0.0);
                    for channel in frame.iter_mut() {
                        *channel = sample;
                    }
                }
            },
            move |err| eprintln!("Output stream error: {}", err),
            None,
        )
        .unwrap()
}

/// SOLAアルゴリズムの実装
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
            // 初回はそのまま出力
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

                let dot_product = prev_segment
                    .iter()
                    .zip(current_segment)
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>();

                let prev_norm = prev_segment.iter().map(|&a| a * a).sum::<f32>().sqrt();
                let current_norm = current_segment.iter().map(|&b| b * b).sum::<f32>().sqrt();

                let corr = dot_product / (prev_norm * current_norm + 1e-8);
                (offset, corr)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((0, 0.0));

        // クロスフェードの適用
        let prev_tail = &self.prev_wav[best_offset..];
        let current_head = &wav[..prev_tail.len()];
        let crossfaded = self.crossfade(prev_tail, current_head);

        // マージ
        let mut output_wav = self.prev_wav[..best_offset].to_vec();
        output_wav.extend(crossfaded);
        output_wav.extend(&wav[prev_tail.len()..wav.len() - self.overlap_size]);

        // 次回のためにデータを保持
        self.prev_wav = wav[wav.len() - self.overlap_size..].to_vec();

        output_wav
    }

    fn crossfade(&self, prev_wav: &[f32], cur_wav: &[f32]) -> Vec<f32> {
        prev_wav
            .iter()
            .zip(cur_wav)
            .enumerate()
            .map(|(i, (&prev_sample, &cur_sample))| {
                let t = i as f32 / (prev_wav.len() - 1) as f32;
                let fade_in = 0.5 - 0.5 * (PI * t).cos();
                let fade_out = 1.0 - fade_in;
                prev_sample * fade_out + cur_sample * fade_in
            })
            .collect()
    }
}

/// デバイス選択のための関数
fn select_device(devices: Vec<Device>, label: &str) -> Device {
    println!("{} デバイスを選択してください:", label);
    for (i, device) in devices.iter().enumerate() {
        println!("{}: {}", i, device.name().unwrap_or_default());
    }
    let stdin = io::stdin();
    let index = stdin
        .lock()
        .lines()
        .next()
        .and_then(|line| line.ok())
        .and_then(|input| input.trim().parse::<usize>().ok())
        .unwrap_or(0);
    devices
        .into_iter()
        .nth(index)
        .expect("選択されたデバイスが存在しません。")
}

fn main() -> OrtResult<()> {
    println!("プログラムを開始します");

    // コマンドライン引数からONNXファイル名を取得
    let args: Vec<String> = env::args().collect();
    let onnx_file = if args.len() > 1 {
        args[1].clone()
    } else {
        "G_best.onnx".to_string() // デフォルトのONNXファイル名
    };
    println!("使用するONNXファイル: {}", onnx_file);

    // ユーザーにsource_speaker_idとtarget_speaker_idを入力させる
    let source_speaker_id = read_input("ソーススピーカーIDを入力してください: ")
        .expect("有効なソーススピーカーIDを入力してください");
    let target_speaker_id = read_input("ターゲットスピーカーIDを入力してください: ")
        .expect("有効なターゲットスピーカーIDを入力してください");

    println!("ソーススピーカーID: {}", source_speaker_id);
    println!("ターゲットスピーカーID: {}", target_speaker_id);

    let hparams = Arc::new(AudioParams::new(source_speaker_id, target_speaker_id));

    // 環境とセッションの構築
    println!("ONNX Runtimeの環境を構築中...");
    let environment = Environment::builder()
        .with_name("MMVC_Client")
        .with_execution_providers([ExecutionProvider::CUDA(Default::default())])
        .build()?
        .into_arc();

    println!("ONNX Runtimeセッションを構築中...");
    let session = Arc::new(
        SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_model_from_file(&onnx_file)?, // 指定されたONNXファイルを使用
    );

    // デバイス選択
    println!("オーディオデバイスを取得中...");
    let host = cpal::default_host();

    // 使用しているオーディオAPIの名前をデバッグフォーマットで取得
    let host_id = host.id();
    let host_name = format!("{:?}", host_id);
    println!("使用しているオーディオAPI: {}", host_name);

    // オーディオAPIに基づいてモードを表示（WASAPIの場合のみ共有モードを表示）
    if host_name.contains("Wasapi") {
        // cpalはWASAPIで共有モードをデフォルトで使用します
        println!("WASAPIの共有モードを使用しています");
    }
    let input_devices: Vec<Device> = host
        .input_devices()
        .expect("入力デバイスの取得に失敗しました")
        .filter(|d| d.name().is_ok())
        .collect();
    let output_devices: Vec<Device> = host
        .output_devices()
        .expect("出力デバイスの取得に失敗しました")
        .filter(|d| d.name().is_ok())
        .collect();

    let input_device = select_device(input_devices, "入力");
    let output_device = select_device(output_devices, "出力");

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
    let input_stream = record_and_resample(Arc::clone(&hparams), input_device, input_tx);

    // 処理スレッドの開始
    println!("処理スレッドを開始します...");
    let hparams_clone = Arc::clone(&hparams);
    let session_clone = Arc::clone(&session);
    thread::spawn(move || {
        if let Err(e) = processing_thread(hparams_clone, session_clone, input_rx, output_tx) {
            eprintln!("処理スレッドでエラーが発生しました: {:?}", e);
        }
    });

    // 出力ストリームの作成
    println!("出力ストリームを作成中...");
    let output_stream = play_output(Arc::clone(&hparams), output_device, output_rx);

    // ストリームの開始
    println!("入力ストリームを再生開始...");
    input_stream
        .play()
        .expect("入力ストリームの再生に失敗しました");
    println!("出力ストリームを再生開始...");
    output_stream
        .play()
        .expect("出力ストリームの再生に失敗しました");

    println!("プログラムが正常に起動しました。処理を続けています...");
    // メインスレッドをブロック
    loop {
        thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// ユーザーからの入力を読み取る関数
fn read_input(prompt: &str) -> Result<i64, String> {
    let mut input = String::new();
    print!("{}", prompt);
    // プロンプトを即座に表示
    io::stdout()
        .flush()
        .map_err(|_| "プロンプトのフラッシュに失敗しました".to_string())?;
    // 入力を読み取る
    io::stdin()
        .read_line(&mut input)
        .map_err(|_| "入力の読み取りに失敗しました".to_string())?;
    // 入力を整数に変換
    input
        .trim()
        .parse::<i64>()
        .map_err(|_| "入力が整数ではありません".to_string())
}
