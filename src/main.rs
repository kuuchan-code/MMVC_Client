use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::{Array1, Array3, CowArray};
use ort::{
    tensor::OrtOwnedTensor, Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult,
    Session, SessionBuilder, Value,
};
use rustfft::{num_complex::Complex, FftPlanner};
use speexdsp_resampler::State as SpeexResampler;
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::io;
use std::sync::Arc;
use std::thread;

const BUFFER_SIZE: usize = 8192; // バッファサイズを増加

// ハイパーパラメータ構造体
struct AudioParams {
    sample_rate: u32,
    fft_window_size: usize,
    hop_size: usize,
    source_speaker_id: i64,
    target_speaker_id: i64,
    dispose_stft_frames: usize,
    dispose_conv1d_samples: usize,
}

impl AudioParams {
    fn new(source_speaker_id: i64, target_speaker_id: i64) -> Self {
        Self {
            sample_rate: 24000,
            fft_window_size: 512,
            hop_size: 128,
            source_speaker_id,
            target_speaker_id,
            dispose_stft_frames: 0,    // フレーム削除数を最小限に設定
            dispose_conv1d_samples: 0, // 必要に応じて設定
        }
    }
}

// 処理スレッド
fn processing_thread(
    hparams: Arc<AudioParams>,
    session: Arc<Session>,
    input_rx: Receiver<Vec<f32>>,
    output_tx: Sender<Vec<f32>>,
) {
    let sola_search_frame = 1024;
    let overlap_size = 512;
    let mut sola = Sola::new(overlap_size, sola_search_frame);
    let mut prev_input_tail: Vec<f32> = Vec::new();

    loop {
        match input_rx.recv() {
            Ok(mut input_signal) => {
                // 前回の入力の終端を現在の入力の先頭に結合
                if !prev_input_tail.is_empty() {
                    let mut extended_signal = prev_input_tail.clone();
                    extended_signal.extend(input_signal);
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
                if input_signal.len() >= dispose_length {
                    prev_input_tail = input_signal[input_signal.len() - dispose_length..].to_vec();
                } else {
                    prev_input_tail = input_signal.clone();
                }

                // マージした信号を送信
                if output_tx.send(merged_signal).is_err() {
                    break;
                }
            }
            Err(_) => {
                break;
            }
        }
    }
}

// 音声変換処理
fn audio_transform(hparams: &Arc<AudioParams>, session: &Session, signal: &[f32]) -> Vec<f32> {
    // STFTパディングの追加
    let pad_size = (hparams.fft_window_size - hparams.hop_size) / 2;
    let mut padded_signal = vec![0.0; pad_size];
    padded_signal.extend_from_slice(signal);
    padded_signal.extend(vec![0.0; pad_size]);

    // STFTの適用
    let mut spec = apply_stft_with_hann_window(
        &padded_signal,
        hparams.fft_window_size,
        hparams.hop_size,
        hparams.fft_window_size,
    );

    // STFTパディングによる影響を削除
    let total_frames = spec.shape()[2];
    let dispose_frames = hparams.dispose_stft_frames;

    // フレーム数が十分かチェック
    if total_frames > 2 * dispose_frames {
        let start = dispose_frames;
        let end = total_frames - dispose_frames;
        spec = spec
            .slice_axis(ndarray::Axis(2), ndarray::Slice::from(start..end))
            .to_owned();
    }

    let spec_lengths = Array1::from_elem(1, spec.shape()[2] as i64);
    let sid_src = Array1::from_elem(1, hparams.source_speaker_id);

    let audio_result = run_onnx_model_inference(
        session,
        &spec,
        &spec_lengths,
        &sid_src,
        hparams.target_speaker_id,
    );

    let mut audio = match audio_result {
        Some(a) => a,
        None => {
            return vec![0.0; signal.len()];
        }
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

// ONNXモデルの推論実行
fn run_onnx_model_inference(
    session: &Session,
    spectrogram: &Array3<f32>,
    spectrogram_lengths: &Array1<i64>,
    source_speaker_id: &Array1<i64>,
    target_speaker_id: i64,
) -> Option<Vec<f32>> {
    let spec_cow = CowArray::from(spectrogram.clone().into_dyn());
    let spec_lengths_cow = CowArray::from(spectrogram_lengths.clone().into_dyn());
    let source_id_cow = CowArray::from(source_speaker_id.clone().into_dyn());
    let target_id_cow = CowArray::from(Array1::from_elem(1, target_speaker_id).into_dyn());

    let inputs = vec![
        Value::from_array(session.allocator(), &spec_cow).ok()?,
        Value::from_array(session.allocator(), &spec_lengths_cow).ok()?,
        Value::from_array(session.allocator(), &source_id_cow).ok()?,
        Value::from_array(session.allocator(), &target_id_cow).ok()?,
    ];

    let outputs = match session.run(inputs) {
        Ok(o) => o,
        Err(_) => {
            return None;
        }
    };

    let audio_output: OrtOwnedTensor<f32, _> = match outputs[0].try_extract() {
        Ok(a) => a,
        Err(_) => {
            return None;
        }
    };

    Some(audio_output.view().to_owned().into_raw_vec())
}

// STFTの適用
fn apply_stft_with_hann_window(
    audio_signal: &[f32],
    fft_size: usize,
    hop_size: usize,
    window_size: usize,
) -> Array3<f32> {
    let hann_window: Vec<f32> = (0..window_size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / window_size as f32).cos()))
        .collect();

    let mut fft_planner = FftPlanner::<f32>::new();
    let fft = fft_planner.plan_fft_forward(fft_size);

    let num_frames = (audio_signal.len() - fft_size) / hop_size + 1;
    let mut spectrogram = Array3::<f32>::zeros((1, fft_size / 2 + 1, num_frames));

    for (i, frame) in audio_signal.windows(fft_size).step_by(hop_size).enumerate() {
        let mut complex_buffer: Vec<Complex<f32>> = frame
            .iter()
            .zip(&hann_window)
            .map(|(&sample, &window)| Complex::new(sample * window, 0.0))
            .collect();

        fft.process(&mut complex_buffer);

        for (j, bin) in complex_buffer.iter().take(fft_size / 2 + 1).enumerate() {
            let power = bin.norm();
            spectrogram[[0, j, i]] = power;
        }
    }

    spectrogram
}

// 音声の録音とリサンプリング
fn record_and_resample(
    hparams: Arc<AudioParams>,
    input_device: cpal::Device,
    input_tx: Sender<Vec<f32>>,
) -> cpal::Stream {
    let input_config = input_device.default_input_config().unwrap();
    let channels = input_config.channels() as usize;
    let input_sample_rate = input_config.sample_rate().0;
    let mut resampler = SpeexResampler::new(
        1,
        input_sample_rate as usize,
        hparams.sample_rate as usize,
        5,
    )
    .unwrap();

    let input_stream_config: StreamConfig = input_config.into();
    let mut buffer = Vec::new();

    let stream = input_device
        .build_input_stream(
            &input_stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mono_signal: Vec<f32> = data
                    .chunks(channels)
                    .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                    .collect();

                buffer.extend_from_slice(&mono_signal);

                while buffer.len() >= BUFFER_SIZE {
                    let chunk: Vec<f32> = buffer.drain(..BUFFER_SIZE).collect::<Vec<f32>>();
                    let mut resampled = vec![
                        0.0;
                        (chunk.len() as f64 * hparams.sample_rate as f64 / input_sample_rate as f64)
                            .ceil() as usize
                    ];
                    // 実際のリサンプリングされたサンプル数を取得
                    let (_, output_generated) =
                        resampler.process_float(0, &chunk, &mut resampled).unwrap();
                    // 実際にリサンプリングされた部分だけを使用
                    resampled.truncate(output_generated);

                    if input_tx.send(resampled).is_err() {
                        break;
                    }
                }
            },
            move |_| {},
            None,
        )
        .unwrap();

    stream
}

fn play_output(
    hparams: Arc<AudioParams>,
    output_device: cpal::Device,
    output_rx: Receiver<Vec<f32>>,
) -> cpal::Stream {
    let output_config = output_device.default_output_config().unwrap();
    let output_sample_rate = output_config.sample_rate().0;
    let mut resampler = SpeexResampler::new(
        1,
        hparams.sample_rate as usize,
        output_sample_rate as usize,
        5,
    )
    .unwrap();
    let mut output_buffer: VecDeque<f32> = VecDeque::with_capacity(BUFFER_SIZE * 10);
    let output_stream_config = output_config.config();
    let channels = output_config.channels() as usize;

    let stream = output_device
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
                    // 実際のリサンプリングされたサンプル数を取得
                    let (_, output_generated) = resampler
                        .process_float(0, &processed_signal, &mut resampled)
                        .unwrap();
                    // 実際にリサンプリングされた部分だけを使用
                    resampled.truncate(output_generated);
                    output_buffer.extend(resampled.iter());
                }
                if output_buffer.len() < data.len() {
                    // バッファに十分なデータがない場合は無音を再生
                    for frame in data.chunks_mut(channels) {
                        for channel in frame.iter_mut() {
                            *channel = 0.0;
                        }
                    }
                    return;
                }

                for frame in data.chunks_mut(channels) {
                    let sample = output_buffer.pop_front().unwrap_or(0.0);
                    for channel in frame.iter_mut() {
                        *channel = sample;
                    }
                }
            },
            move |_| {},
            None,
        )
        .unwrap();

    stream
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
            // 初回はそのまま出力
            let output_wav = wav[..wav.len() - self.overlap_size].to_vec();
            self.prev_wav = wav[wav.len() - self.overlap_size..].to_vec();
            return output_wav;
        }

        let search_range = self.sola_search_frame.min(self.prev_wav.len());
        let max_offset = self.prev_wav.len() - search_range;
        let mut best_offset = 0;
        let mut max_corr = f32::MIN;

        // 正規化された相互相関の計算
        for offset in 0..=max_offset {
            let prev_segment = &self.prev_wav[offset..offset + search_range];
            let current_segment = &wav[..search_range];

            let dot_product = prev_segment
                .iter()
                .zip(current_segment)
                .map(|(&a, &b)| a * b)
                .sum::<f32>();

            let prev_norm = prev_segment.iter().map(|&a| a * a).sum::<f32>().sqrt();
            let current_norm = current_segment.iter().map(|&b| b * b).sum::<f32>().sqrt();

            let corr = dot_product / (prev_norm * current_norm + 1e-8); // ゼロ除算を防ぐために小さな値を加算

            if corr > max_corr {
                max_corr = corr;
                best_offset = offset;
            }
        }

        // クロスフェードの適用
        let prev_tail = &self.prev_wav[best_offset..];
        let current_head = &wav[..prev_tail.len()];
        let crossfaded = self.crossfade(prev_tail, current_head);

        // マージ
        let mut output_wav = Vec::new();
        output_wav.extend_from_slice(&self.prev_wav[..best_offset]);
        output_wav.extend_from_slice(&crossfaded);
        output_wav.extend_from_slice(&wav[prev_tail.len()..wav.len() - self.overlap_size]);

        // 次回のためにデータを保持
        self.prev_wav = wav[wav.len() - self.overlap_size..].to_vec();

        output_wav
    }

    fn crossfade(&self, prev_wav: &[f32], cur_wav: &[f32]) -> Vec<f32> {
        let crossfade_size = prev_wav.len();
        (0..crossfade_size)
            .map(|i| {
                let t = i as f32 / (crossfade_size - 1) as f32;
                // イージング関数を使用して滑らかなフェードを実現
                let fade_in = t.powf(2.0) * (3.0 - 2.0 * t); // Smoothstep関数
                let fade_out = 1.0 - fade_in;
                prev_wav[i] * fade_out + cur_wav[i] * fade_in
            })
            .collect()
    }
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
    // ユーザーにsidとtarget_speaker_idを入力させる
    println!("Enter source speaker ID (sid): ");
    let mut sid_input = String::new();
    io::stdin().read_line(&mut sid_input).unwrap();
    let sid: i64 = sid_input
        .trim()
        .parse()
        .expect("Invalid input for source speaker ID");

    println!("Enter target speaker ID: ");
    let mut target_speaker_id_input = String::new();
    io::stdin().read_line(&mut target_speaker_id_input).unwrap();
    let target_speaker_id: i64 = target_speaker_id_input
        .trim()
        .parse()
        .expect("Invalid input for target speaker ID");

    let hparams = Arc::new(AudioParams::new(sid, target_speaker_id));

    // 環境とセッションの構築
    let environment = Environment::builder()
        .with_name("MMVC_Client")
        .with_execution_providers([ExecutionProvider::CUDA(Default::default())])
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file("G_best.onnx")?;

    // デバイス選択
    let host = cpal::default_host();
    let input_device = select_device(host.input_devices().unwrap().collect(), "入力");
    let output_device = select_device(host.output_devices().unwrap().collect(), "出力");

    // チャネルの作成
    let (input_tx, input_rx) = bounded(10);
    let (output_tx, output_rx) = bounded(10);

    // 入力ストリームの作成
    let input_stream = record_and_resample(Arc::clone(&hparams), input_device, input_tx);

    // 処理スレッドの開始
    let hparams_clone = Arc::clone(&hparams);
    let session_clone = Arc::new(session);
    thread::spawn(move || {
        processing_thread(hparams_clone, session_clone, input_rx, output_tx);
    });

    // 出力ストリームの作成
    let output_stream = play_output(Arc::clone(&hparams), output_device, output_rx);

    // ストリームの開始
    input_stream.play().unwrap();
    output_stream.play().unwrap();

    // メインスレッドをブロック
    loop {
        thread::sleep(std::time::Duration::from_millis(50));
    }
}
