use tract_onnx::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use rustfft::num_traits::Zero;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use smallvec::SmallVec;
use std::f32::consts::PI;

// 音声信号を前処理して正規化
fn preprocess_audio(input: Vec<f32>, n_fft: usize, hop_size: usize) -> Vec<f32> {
    let pad_size = (n_fft - hop_size) / 2;
    let padded_input = pad_reflect(&input, pad_size);
    padded_input
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

// 反射パディング
fn pad_reflect(input: &Vec<f32>, pad_size: usize) -> Vec<f32> {
    let mut padded_input = Vec::with_capacity(input.len() + pad_size * 2);
    let reflect_start = &input[0..pad_size];
    let reflect_start_reversed: Vec<f32> = reflect_start.iter().rev().cloned().collect();
    padded_input.extend(reflect_start_reversed);
    padded_input.extend(input.iter().cloned());
    let reflect_end = &input[input.len() - pad_size..];
    let reflect_end_reversed: Vec<f32> = reflect_end.iter().rev().cloned().collect();
    padded_input.extend(reflect_end_reversed);
    padded_input
}

// 文字列を数値列に変換する関数
fn string_to_ids(text: &str) -> Vec<i64> {
    text.chars().map(|c| c as i64).collect()
}

// ONNXモデルの入力データ準備
fn prepare_model_input(text: Vec<i64>, spec: Vec<f32>, wav: Vec<f32>, sid: i64) -> TractResult<TVec<TValue>> {
    // スペクトログラムの長さが257で割り切れるか確認
    let num_frames = spec.len() / 257;
    if spec.len() % 257 != 0 {
        panic!("スペクトログラムのデータ長が257で割り切れません。データ長: {}", spec.len());
    }

    // テンソルに変換
    let text_tensor = Tensor::from_shape(&[1, text.len()], &text)?;
    let spec_tensor = Tensor::from_shape(&[1, 257, num_frames], &spec)?;
    let wav_tensor = Tensor::from_shape(&[1, wav.len()], &wav)?;
    let sid_tensor = Tensor::from(sid);

    // モデルへの入力をまとめる
    let model_inputs = tvec![
        text_tensor.into(),
        spec_tensor.into(),
        wav_tensor.into(),
        sid_tensor.into(),
    ];

    Ok(model_inputs)
}


// ONNXモデルの実行
fn run_onnx_model(model: &TypedRunnableModel<TypedModel>, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
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

    // `text` に "m" を渡す
    let text = string_to_ids("m");  // 文字列 "m" を数値化
    let sid = 2_i64;  // 話者IDを 2 に設定

    // 音声ストリームの設定
    let err_fn = |err| eprintln!("エラーが発生しました: {}", err);
    let stream = input_device.build_input_stream(
        &config.into(),
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            // 1. `data` を `i16` から `f32` に正規化
            let wav: Vec<f32> = data.iter().map(|&x| x as f32 / max_value).collect();

            // 2. スペクトログラムを計算
            let spec = compute_spectrogram(&wav, n_fft, hop_size, win_size, &hann_window);
            let spec_data: Vec<f32> = spec.into_iter().flat_map(|c| vec![c.re, c.im]).collect();

            // 3. モデルの入力データを準備
            let model_inputs = prepare_model_input(text.clone(), spec_data, wav.clone(), sid).unwrap();

            // 4. モデルの実行
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
