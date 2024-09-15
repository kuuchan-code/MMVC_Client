use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::StreamConfig;
use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::{Array1, Array3, CowArray};
use ort::{
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder, Value,
};
use rustfft::{num_complex::Complex, FftPlanner};
use speexdsp_resampler::State as SpeexResampler;
use std::collections::VecDeque;
use std::f32;
use std::f32::consts::PI;
use std::sync::Arc;
use std::thread; // Use speexdsp-resampler for resampling

const BUFFER_SIZE: usize = 8192;

// ハイパーパラメータ構造体
struct AudioParams {
    sample_rate: u32,
    fft_window_size: usize,
    hop_size: usize,
    target_speaker_id: i64,
    conv1d_padding_frames: usize,
}

impl AudioParams {
    fn new() -> Self {
        AudioParams {
            sample_rate: 24000,
            fft_window_size: 512,
            hop_size: 128,
            target_speaker_id: 2,
            conv1d_padding_frames: 0,
        }
    }
}

// オーディオ信号にパディングを追加
fn pad_audio_signal(signal: &Vec<f32>, fft_size: usize, hop_size: usize) -> Vec<f32> {
    let pad_size = (fft_size - hop_size) / 2;
    let mut padded_signal = vec![0.0; pad_size];
    padded_signal.extend_from_slice(signal);
    padded_signal.extend(vec![0.0; pad_size]);
    padded_signal
}

fn dispose_padding(signal: &mut Vec<f32>, fft_size: usize, hop_size: usize) {
    let pad_size = (fft_size - hop_size) / 2;
    if signal.len() > 2 * pad_size {
        // 信号の両端から pad_size を削除
        *signal = signal[pad_size..signal.len() - pad_size].to_vec();
    }
}

fn dispose_conv1d_padding(audio: &mut Vec<f32>, dispose_conv1d_length: usize) {
    let audio_len = audio.len();
    if dispose_conv1d_length > 0 && audio_len > 2 * dispose_conv1d_length {
        *audio = audio[dispose_conv1d_length..audio_len - dispose_conv1d_length].to_vec();
    }
}
use hound::{WavSpec, WavWriter};
use std::time::Instant;

// WAVファイルとして保存する関数
fn save_as_wav(filename: &str, sample_rate: u32, data: &[f32]) {
    let spec = WavSpec {
        channels: 1, // モノラル
        sample_rate,
        bits_per_sample: 16, // 16ビットPCM
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(filename, spec).unwrap();
    for &sample in data {
        // 16ビット整数に変換して書き込む
        let scaled_sample = (sample * i16::MAX as f32) as i16;
        writer.write_sample(scaled_sample).unwrap();
    }
    writer.finalize().unwrap();
}

fn processing_thread(
    hparams: Arc<AudioParams>,
    session: Arc<ort::Session>,
    input_rx: Receiver<Vec<f32>>,
    output_tx: Sender<Vec<f32>>,
) {
    // SOLAのパラメータ設定
    let sola_search_frame = 128;
    let overlap_size = 384;
    let mut sola = Sola::new(overlap_size, sola_search_frame);

    loop {
        match input_rx.recv() {
            Ok(input_signal) => {
                // デバッグ: 受信した信号の長さを出力
                println!("Processing signal of length: {}", input_signal.len());

                // 入力信号をWAVファイルに保存
                save_as_wav("input_signal.wav", hparams.sample_rate, &input_signal);
                println!("Input signal saved to input_signal.wav");

                // 処理時間計測の開始
                let start_time = Instant::now();

                // 音声変換処理
                let processed_signal = audio_trans(
                    &hparams,
                    &session,
                    input_signal,
                    hparams.target_speaker_id,
                    hparams.conv1d_padding_frames,
                );

                // SOLAによるマージ
                let merged_signal = sola.merge(&processed_signal);

                // マージした信号を送信
                if let Err(err) = output_tx.send(merged_signal) {
                    eprintln!("Failed to send output data: {}", err);
                }

                // 処理時間の計測終了
                let processing_duration = start_time.elapsed();
                println!("Total processing time: {:?}", processing_duration);

                if processing_duration.as_secs_f32()
                    > (BUFFER_SIZE as f32 / hparams.sample_rate as f32)
                {
                    eprintln!("Processing is taking too long and may cause underruns.");
                }
            }
            Err(err) => {
                eprintln!("Failed to receive input data: {}", err);
                break;
            }
        }
    }
}

fn audio_trans(
    hparams: &AudioParams,
    session: &ort::Session,
    signal: Vec<f32>,
    _target_speaker_id: i64,
    conv1d_padding_frames: usize,
) -> Vec<f32> {
    let mut signal_copy = signal.clone();

    let spec = apply_stft_with_hann_window(
        &signal_copy,
        hparams.fft_window_size,
        hparams.hop_size,
        hparams.fft_window_size,
    );

    dispose_padding(&mut signal_copy, hparams.fft_window_size, hparams.hop_size);

    println!("STFT spectrogram shape: {:?}", spec.shape());

    let spec_lengths = Array1::from_elem(1, spec.shape()[2] as i64);
    let sid_src = Array1::from_elem(1, 0);
    let sid_target = hparams.target_speaker_id;

    let mut audio = run_onnx_model_inference(session, &spec, &spec_lengths, &sid_src, sid_target);

    println!("Model output audio length: {}", audio.len());

    dispose_conv1d_padding(&mut audio, conv1d_padding_frames);

    // モデル出力をWAVファイルに保存
    save_as_wav("model_output_audio.wav", hparams.sample_rate, &audio);
    println!("Model output audio saved to model_output_audio.wav");

    audio
}

fn run_onnx_model_inference(
    session: &ort::Session,
    spectrogram: &Array3<f32>,
    spectrogram_lengths: &Array1<i64>,
    source_speaker_id: &Array1<i64>,
    target_speaker_id: i64,
) -> Vec<f32> {
    let start_time = Instant::now(); // Start measuring time for ONNX inference

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

    let duration = start_time.elapsed(); // Measure the elapsed time for inference
    println!("ONNX model inference time: {:?}", duration);

    audio_output.view().to_owned().into_raw_vec()
}

fn apply_stft_with_hann_window(
    audio_signal: &Vec<f32>,
    fft_size: usize,
    hop_size: usize,
    window_size: usize,
) -> Array3<f32> {
    let start_time = Instant::now(); // Start measuring time for STFT

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

    let duration = start_time.elapsed(); // Measure the elapsed time for STFT
    println!("STFT processing time: {:?}", duration);

    spectrogram
}

fn record_and_resample(
    hparams: Arc<AudioParams>, // Use Arc here
    input_device: cpal::Device,
    input_tx: Sender<Vec<f32>>,
) -> cpal::Stream {
    let input_config = input_device.default_input_config().unwrap();
    let frequency_bins = input_config.channels();
    let input_sample_rate = input_config.sample_rate().0;

    let mut resampler = SpeexResampler::new(
        1,
        input_sample_rate as usize,
        hparams.sample_rate as usize,
        1,
    )
    .unwrap();
    let input_stream_config: StreamConfig = input_config.into();
    let mut buffer = Vec::new();

    let hparams_clone = Arc::clone(&hparams); // Clone Arc for the closure
    let stream = input_device
        .build_input_stream(
            &input_stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                println!("Received {} frames from input device", data.len());

                let mono_signal: Vec<f32> = data
                    .chunks(frequency_bins as usize)
                    .map(|chunk| chunk.iter().sum::<f32>() / frequency_bins as f32)
                    .collect();

                buffer.extend_from_slice(&mono_signal);
                println!("Input buffer size: {}", buffer.len());

                if buffer.len() >= BUFFER_SIZE {
                    let mut resampled = vec![
                        0.0;
                        buffer.len() * hparams_clone.sample_rate as usize
                            / input_sample_rate as usize
                    ];
                    let (_in_len, out_len) =
                        resampler.process_float(0, &buffer, &mut resampled).unwrap(); // Ignore in_len

                    println!("Resampled data length: {}", out_len);

                    if let Err(err) = input_tx.send(resampled[..out_len].to_vec()) {
                        eprintln!("Failed to send input data: {}", err);
                    }
                    buffer.clear();
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

fn play_output(
    hparams: Arc<AudioParams>, // Use Arc here
    output_device: cpal::Device,
    output_rx: Receiver<Vec<f32>>,
) -> cpal::Stream {
    let output_config = output_device.default_output_config().unwrap();
    let output_sample_rate = output_config.sample_rate().0;

    let mut resampler = SpeexResampler::new(
        1,
        hparams.sample_rate as usize,
        output_sample_rate as usize,
        1,
    )
    .unwrap();
    let mut output_buffer: VecDeque<f32> = VecDeque::with_capacity(BUFFER_SIZE * 10);
    let output_stream_config = output_config.config().clone();
    let channels = output_config.channels() as usize;

    let hparams_clone = Arc::clone(&hparams); // Clone Arc for the closure
    let stream = output_device
        .build_output_stream(
            &output_stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                while let Ok(processed_signal) = output_rx.try_recv() {
                    println!(
                        "Received processed signal of length: {}",
                        processed_signal.len()
                    );

                    let mut resampled = vec![
                        0.0;
                        processed_signal.len() * output_sample_rate as usize
                            / hparams_clone.sample_rate as usize
                    ];
                    let (_in_len, out_len) = resampler
                        .process_float(0, &processed_signal, &mut resampled)
                        .unwrap(); // Ignore in_len

                    println!("Resampled output signal length: {}", out_len);

                    output_buffer.extend(resampled[..out_len].iter());
                }

                if output_buffer.len() < BUFFER_SIZE * 2 {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    return;
                }

                for frame in data.chunks_mut(channels) {
                    let sample = if let Some(s) = output_buffer.pop_front() {
                        s
                    } else {
                        0.0
                    };
                    for channel in frame.iter_mut() {
                        *channel = sample;
                    }
                }

                println!("Output buffer size after playback: {}", output_buffer.len());
            },
            move |err| {
                eprintln!("Error occurred on output stream: {}", err);
            },
            None,
        )
        .unwrap();

    stream
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
        // Check if this is the first chunk
        if self.prev_wav.iter().all(|&x| x == 0.0) {
            // No previous data, so skip crossfade
            let output_wav = wav[..wav.len() - self.overlap_size].to_vec();
            // Save overlap for next chunk
            self.prev_wav = wav[wav.len() - self.overlap_size..].to_vec();
            return output_wav;
        } // デバッグ：入力信号の長さを出力
        println!("SOLA merge input wav length: {}", wav.len());

        let sola_search_region = &wav[..self.sola_search_frame];
        let cor_nom = self.convolve(&self.prev_wav, sola_search_region);
        let cor_den = self.calculate_root_energy(&self.prev_wav, self.sola_search_frame);
        let sola_offset = self.calculate_sola_offset(&cor_nom, &cor_den);

        // デバッグ：SOLAオフセットを出力
        println!("SOLA offset: {}", sola_offset);

        let prev_sola_match_region =
            &self.prev_wav[sola_offset..sola_offset + self.sola_search_frame];
        let crossfade_wav = self.crossfade(sola_search_region, prev_sola_match_region);
        let sola_merge_wav = [
            &self.prev_wav[..sola_offset],
            &crossfade_wav,
            &wav[self.sola_search_frame..],
        ]
        .concat();

        // デバッグ：SOLAマージ後の信号の長さを出力
        println!("SOLA merged wav length: {}", sola_merge_wav.len());

        let output_wav = sola_merge_wav[..sola_merge_wav.len() - self.overlap_size].to_vec();

        // 次のチャンク用に保存
        let start = wav.len() - self.overlap_size - sola_offset;
        let end = wav.len() - sola_offset;
        self.prev_wav = wav[start..end].to_vec();

        output_wav
    }

    fn calculate_sola_offset(&self, cor_nom: &[f32], cor_den: &[f32]) -> usize {
        let mut idx = 0;
        let mut max = f32::MIN;

        for i in 0..cor_nom.len() {
            let scaled_value = cor_nom[i] / (cor_den[i] + 1e-8); // 0除算を防ぐため
            if scaled_value > max {
                max = scaled_value;
                idx = i;
            }
        }
        idx
    }

    fn calculate_root_energy(&self, a: &[f32], len: usize) -> Vec<f32> {
        let n = a.len();
        if len == 0 || n < len {
            panic!("Input arrays have incompatible sizes.");
        }

        let mut result = vec![0.0; n - len + 1];

        for i in 0..result.len() {
            let sum: f32 = a[i..i + len].iter().map(|&x| x.powi(2)).sum();
            result[i] = sum.sqrt();
        }

        result
    }

    fn convolve(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        let n = a.len();
        let m = b.len();
        if m == 0 || n < m {
            panic!("Input arrays have incompatible sizes.");
        }

        let mut result = vec![0.0; n - m + 1];

        for i in 0..result.len() {
            let sum: f32 = a[i..i + m].iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            result[i] = sum;
        }

        result
    }

    fn crossfade(&self, cur_wav: &[f32], prev_wav: &[f32]) -> Vec<f32> {
        if prev_wav.len() != cur_wav.len() {
            panic!("prev_wav.len() != cur_wav.len()");
        }

        let crossfade_size = prev_wav.len();
        let mut output_wav = vec![0.0; crossfade_size];

        for i in 0..crossfade_size {
            let percent = i as f32 / crossfade_size as f32;
            let prev_strength = 0.5 * (1.0 + (std::f32::consts::PI * percent).cos());
            let cur_strength = 1.0 - prev_strength;
            output_wav[i] = prev_wav[i] * prev_strength + cur_wav[i] * cur_strength;
        }

        output_wav
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
    let hparams = Arc::new(AudioParams::new());

    println!("Initializing environment and session...");

    // 環境とセッションの構築
    let environment = Environment::builder()
        .with_name("MMVC_Client")
        .with_execution_providers([ExecutionProvider::CUDA(Default::default())])
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file("G_best.onnx")?;

    println!("Session initialized successfully.");

    // デバイス選択
    let host = cpal::default_host();
    let input_device = select_device(host.input_devices().unwrap().collect(), "入力");
    let output_device = select_device(host.output_devices().unwrap().collect(), "出力");

    println!("Selected input device: {}", input_device.name().unwrap());
    println!("Selected output device: {}", output_device.name().unwrap());

    // チャネルの作成
    let (input_tx, input_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(10);
    let (output_tx, output_rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = bounded(10);

    // 入力ストリームの作成
    let input_stream = record_and_resample(Arc::clone(&hparams), input_device, input_tx.clone());

    println!("Input stream created.");

    // 処理スレッドの開始
    let hparams_clone = Arc::clone(&hparams);
    let session_clone = Arc::new(session);
    thread::spawn(move || {
        println!("Processing thread started.");
        processing_thread(hparams_clone, session_clone, input_rx, output_tx);
    });

    // 出力ストリームの作成
    let output_stream = play_output(Arc::clone(&hparams), output_device, output_rx);

    println!("Output stream created.");

    // ストリームの開始
    input_stream.play().unwrap();
    output_stream.play().unwrap();

    println!("Audio streams are now playing.");

    // メインスレッドをブロック
    loop {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}
