use tract_onnx::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use rustfft::num_traits::Zero;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use smallvec::SmallVec;
use std::f32::consts::PI;

// 音声信号を前処理して正規化
fn preprocess_audio(input: &[i16], max_value: f32) -> Vec<f32> {
    input.iter().map(|&x| (x as f32) / max_value).collect()
}

// STFT（短時間フーリエ変換）の計算
fn compute_spectrogram(input: &[f32], n_fft: usize, hop_length: usize, win_size: usize, window: &[f32]) -> Vec<Complex<f32>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut stft_result = vec![];

    for frame in input.chunks(hop_length) {
        let mut padded_frame: Vec<Complex<f32>> = frame.iter()
            .zip(window)
            .map(|(&sample, &w)| Complex::new(sample * w, 0.0))
            .collect();
        
        // `Complex::zero()` は `num_traits::Zero` トレイトの関数を使う
        padded_frame.resize(n_fft, Complex::zero());
        fft.process(&mut padded_frame);

        stft_result.extend(padded_frame);
    }

    stft_result
}

// ハン窓の生成
fn generate_hann_window(size: usize) -> Vec<f32> {
    (0..size).map(|i| (PI * i as f32 / size as f32).sin().powi(2)).collect()
}

// ONNXモデルの入力データ準備
fn prepare_model_input(input_data: Vec<f32>, n_fft: usize, hop_size: usize, win_size: usize) -> TractResult<SmallVec<[TValue; 4]>> {
    let num_frames = input_data.len() / 257;
    let input_tensor = Tensor::from_shape(&[1, 257, num_frames], &input_data)?;

    let n_fft_tensor = Tensor::from(n_fft as i64);
    let hop_size_tensor = Tensor::from(hop_size as i64);
    let win_size_tensor = Tensor::from(win_size as i64);

    Ok(tvec![input_tensor.into(), n_fft_tensor.into(), hop_size_tensor.into(), win_size_tensor.into()])
}

// ONNXモデルの実行
fn run_onnx_model(model: &TypedRunnableModel<TypedModel>, inputs: SmallVec<[TValue; 4]>) -> TractResult<SmallVec<[TValue; 4]>> {
    let result = model.run(inputs)?;
    Ok(result)
}

fn main() -> TractResult<()> {
    // ONNXモデルの読み込み
    let model_path = r"C:\Users\ku-chan\mmvc_client\logs\runa\G_best.onnx";
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .into_optimized()?
        .into_runnable()?;

    // CPALを使用してデフォルトの入力デバイスを取得
    let host = cpal::default_host();
    let input_device = host.default_input_device().expect("入力デバイスが見つかりません");
    let config = input_device.default_input_config().unwrap();
    
    let input_sample_rate = config.sample_rate().0 as usize;  // デバイスのサンプリングレート
    let target_sample_rate = 24000;  // モデルが期待するサンプリングレート
    let max_value = 32768.0;  // 正規化のための最大値

    let n_fft = 512;
    let hop_size = 128;
    let win_size = 512;
    let hann_window = generate_hann_window(win_size);

    // 音声ストリームの設定
    let err_fn = |err| eprintln!("エラーが発生しました: {}", err);
    let stream = input_device.build_input_stream(
        &config.into(),
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            // 音声データの前処理
            let preprocessed_data = preprocess_audio(data, max_value);
            
            // リサンプリング（必要であれば）
            let resampled_data = if input_sample_rate != target_sample_rate {
                // リサンプリング処理（省略可能）
                preprocessed_data // TODO: 実際のリサンプリング処理を実装
            } else {
                preprocessed_data
            };

            // スペクトログラムを計算
            let spectrogram = compute_spectrogram(&resampled_data, n_fft, hop_size, win_size, &hann_window);
            
            // モデルへの入力準備
            let input_data: Vec<f32> = spectrogram.into_iter().flat_map(|c| vec![c.re, c.im]).collect();
            let model_inputs = prepare_model_input(input_data, n_fft, hop_size, win_size).unwrap();

            // モデルの実行
            let result = run_onnx_model(&model, model_inputs).unwrap();
            
            // 結果を処理（ここでは一時的に表示）
            println!("モデルの結果: {:?}", result);
        },
        err_fn,
        None,  // Option<Duration> を追加
    ).unwrap();

    // ストリームを再生
    stream.play().unwrap();
    
    // ストリームの実行を10秒間待機
    std::thread::sleep(std::time::Duration::from_secs(10));

    Ok(())
}
