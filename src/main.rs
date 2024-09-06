use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tract_onnx::prelude::*;
use tract_ndarray::Array1;
use rubato::{FftFixedInOut, Resampler};
use realfft::RealFftPlanner;
use cpal::BufferSize;

fn main() -> TractResult<()> {
    // ONNXモデルを読み込み
    let model_path = r"C:\Users\ku-chan\mmvc_client\logs\runa\G_best.onnx";
    let model = load_onnx_model(model_path)?;

    let host = cpal::default_host();
    let input_device = host.default_input_device().expect("入力デバイスが見つかりません");
    let _output_device = host.default_output_device().expect("出力デバイスが見つかりません");

    let config = input_device.default_input_config().unwrap();
    let input_sample_rate = config.sample_rate().0 as usize;  // デバイスのサンプリングレートを取得
    let target_sample_rate = 24000;  // モデルが期待するサンプリングレート
    let config_clone = config.clone();  // クローンを作成
    
    let err_fn = |err| eprintln!("エラーが発生しました: {}", err);

    // リサンプラーの設定
    let mut resampler = if input_sample_rate != target_sample_rate {
        Some(FftFixedInOut::<f32>::new(input_sample_rate, target_sample_rate, 1024, 1).unwrap())  // 1チャンネルに設定
    } else {
        None
    };

    // ストリーム設定時にバッファサイズを指定
    let stream_config = cpal::StreamConfig {
        channels: config.channels(),
        sample_rate: config.sample_rate(),
        buffer_size: BufferSize::Fixed(960),  // 960にバッファサイズを固定
    };

    // 入力ストリームを設定
    let stream = input_device.build_input_stream(
        &stream_config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // 入力データをリサンプリング
            let data_resampled = if let Some(ref mut resampler) = resampler {
                resampler.process(&[data.to_vec()], None).unwrap()[0].clone()  // リサンプリング処理
            } else {
                data.to_vec()
            };
    
            // ONNXモデルを介して音声を処理
            let n_fft = 1024;
            let hop_size = 256;
            let win_size = 1024;
            let result = run_voice_conversion(&model, data_resampled, n_fft, hop_size, win_size, target_sample_rate).unwrap();
    
            // 変換された音声を出力デバイスに再生
            output_audio(result, &config_clone.config());
        },
        err_fn,
        None,  // Option<Duration> の引数を追加
    ).unwrap();
    
    stream.play().unwrap();

    std::thread::sleep(std::time::Duration::from_secs(10));

    Ok(())
}

fn load_onnx_model(model_path: &str) -> TractResult<TypedRunnableModel<TypedModel>> {
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .into_optimized()?
        .into_runnable()?;
    Ok(model)
}

fn calculate_spectrogram(y: &[f32], n_fft: usize, hop_size: usize, win_size: usize) -> Vec<Array1<f32>> {
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);

    let hann_window: Vec<f32> = (0..win_size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / win_size as f32).cos()))
        .collect();

    let padded: Vec<f32> = vec![0.0; (n_fft - hop_size) / 2]
        .into_iter()
        .chain(y.iter().cloned())
        .chain(vec![0.0; (n_fft - hop_size) / 2])
        .collect();

    let mut spectrum = vec![];
    for chunk in padded.chunks(hop_size) {
        let mut input: Vec<f32> = chunk.iter().zip(&hann_window).map(|(&s, &w)| s * w).collect();
        input.resize(n_fft, 0.0);
        let mut output = r2c.make_output_vec();
        r2c.process(&mut input, &mut output).unwrap();

        let power_spectrum: Array1<f32> = Array1::from(output.iter().map(|c| c.norm_sqr().sqrt()).collect::<Vec<_>>());
        spectrum.push(power_spectrum);
    }
    spectrum
}

fn run_voice_conversion(model: &TypedRunnableModel<TypedModel>, input_data: Vec<f32>, n_fft: usize, hop_size: usize, win_size: usize, sample_rate: usize) -> TractResult<Vec<f32>> {
    // スペクトログラムを計算 (必要に応じて処理する)
    let _spectrogram = calculate_spectrogram(&input_data, n_fft, hop_size, win_size);
    
    let input_tensor = Array1::from(input_data).into_tensor(); 
    let n_fft_tensor = Tensor::from(n_fft as i64);
    let hop_size_tensor = Tensor::from(hop_size as i64);
    let win_size_tensor = Tensor::from(win_size as i64);
    let sample_rate_tensor = Tensor::from(sample_rate as i64);  // サンプリングレートを追加

    let result = model.run(tvec![
        input_tensor.into(),
        n_fft_tensor.into(),
        hop_size_tensor.into(),
        win_size_tensor.into(),
        sample_rate_tensor.into(),  // モデルにサンプリングレートを渡す
    ])?;

    // 配列を Vec<f32> に変換
    let output: Vec<f32> = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .collect();

    Ok(output)
}


fn output_audio(data: Vec<f32>, config: &cpal::StreamConfig) {
    let host = cpal::default_host();
    let device = host.default_output_device().expect("出力デバイスが見つかりません");

    let err_fn = |err| eprintln!("出力音声ストリームでエラーが発生しました: {}", err);

    let stream = device.build_output_stream(
        config,
        move |out: &mut [f32], _: &cpal::OutputCallbackInfo| {
            for (i, sample) in out.iter_mut().enumerate() {
                *sample = data[i % data.len()];
            }
        },
        err_fn,
        None,
    ).unwrap();

    stream.play().unwrap();
}
