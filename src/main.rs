use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tract_onnx::prelude::*;
use rubato::{FftFixedInOut, Resampler};

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
    let _config_clone = config.clone();  // クローンを作成（_config_cloneに変更して警告を回避）
    
    let err_fn = |err| eprintln!("エラーが発生しました: {}", err);

    // リサンプラーの設定
    let mut resampler = if input_sample_rate != target_sample_rate {
        Some(FftFixedInOut::<f32>::new(input_sample_rate, target_sample_rate, 916, 1).unwrap())  // 1チャンネルに設定
    } else {
        None
    };

    // ストリーム設定時にバッファサイズを指定
    let stream_config = cpal::StreamConfig {
        channels: config.channels(),
        sample_rate: config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,  // デバイスのデフォルトのバッファサイズを使用
    };

    // 入力ストリームを設定
    let stream = input_device.build_input_stream(
        &stream_config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // 入力データをリサンプリング
            let data_resampled = if let Some(ref mut resampler) = resampler {
                let resampled_data = resampler.process(&[data.to_vec()], None).unwrap()[0].clone();  // リサンプリング処理
                println!("リサンプリング後のデータサイズ: {}", resampled_data.len());
                resampled_data
            } else {
                data.to_vec()
            };
    
            // ONNXモデルを介して音声を処理
            let n_fft = 512;
            let hop_size = 128;
            let win_size = 512;
            let _result = run_voice_conversion(&model, data_resampled, n_fft, hop_size, win_size).unwrap();
    
            // 変換された音声を出力デバイスに再生（コメントアウトされた部分を後で実装）
            // output_audio(result, &config_clone.config());
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

    // モデルの入力情報を確認
    for (ix, input) in model.model().input_outlets()?.iter().enumerate() {
        let fact = model.model().outlet_fact(*input)?;
        println!("入力{}: 形状: {:?}", ix, fact.shape);
    }

    Ok(model)
}

fn run_voice_conversion(
    model: &TypedRunnableModel<TypedModel>,
    input_data: Vec<f32>,
    n_fft: usize,
    hop_size: usize,
    win_size: usize,
) -> TractResult<Vec<f32>> {
    // モデルに渡すデータの形をモデルが期待する形状 (1, 257, length) に変換する
    let num_frames = input_data.len() / 257;
    let mut reshaped_data = Vec::new();

    // データを1x257x(num_frames)に変換
    for frame in 0..num_frames {
        reshaped_data.push(&input_data[frame * 257..(frame + 1) * 257]);
    }

    // ONNXに渡す形状に変換
    let input_tensor = Tensor::from_shape(&[1, 257, num_frames], &reshaped_data.concat())?;

    // スカラ値をテンソルに変換し、形を1に揃える
    let n_fft_tensor = Tensor::from_shape(&[1], &[n_fft as i64])?;
    let hop_size_tensor = Tensor::from_shape(&[1], &[hop_size as i64])?;
    let win_size_tensor = Tensor::from_shape(&[1], &[win_size as i64])?;
    println!("Input tensor: {:?}", input_tensor);
    println!("n_fft_tensor: {:?}", n_fft_tensor);
    println!("hop_size_tensor: {:?}", hop_size_tensor);
    println!("win_size_tensor: {:?}", win_size_tensor);
    
    // モデルを実行 (4つの入力に変更)
    let result = model.run(tvec![
        input_tensor.into(),
        n_fft_tensor.into(),
        hop_size_tensor.into(),
        win_size_tensor.into(),
    ])?;

    // 結果を Vec<f32> に変換して返す
    let output: Vec<f32> = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .collect();

    Ok(output)
}
