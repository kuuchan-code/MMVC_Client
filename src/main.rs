use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tract_onnx::prelude::*;
use tract_ndarray::Array1;

fn main() -> TractResult<()> {
    // ONNXモデルを読み込み
    let model_path = r"C:\Users\ku-chan\mmvc_client\logs\runa\G_best.onnx";
    let model = load_onnx_model(model_path)?;
    // モデルの期待する入力を確認
    print_model_input_info(&model);
    let host = cpal::default_host();
    let input_device = host.default_input_device().expect("入力デバイスが見つかりません");
    let output_device = host.default_output_device().expect("出力デバイスが見つかりません");

    let config = input_device.default_input_config().unwrap();
    let config_clone = config.clone();  // クローンを作成
    
    let err_fn = |err| eprintln!("エラーが発生しました: {}", err);

    // 入力ストリームを設定
    let stream = input_device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // ONNXモデルを介して音声を処理
            let result = run_voice_conversion(&model, data.to_vec()).unwrap();
    
            // 変換された音声を出力デバイスに再生
            output_audio(result.into_tensor().as_slice::<f32>().unwrap().to_vec(), &config_clone.config());
        },
        err_fn,
        None,  // Option<Duration> の引数を追加
    ).unwrap();
    

    stream.play().unwrap();

    std::thread::sleep(std::time::Duration::from_secs(10));

    Ok(())
}
fn print_model_input_info(model: &TypedRunnableModel<TypedModel>) {
    for input in model.input_outlets().unwrap() {
        println!("{:?}", model.node(input.node).name);
    }
}

fn load_onnx_model(model_path: &str) -> TractResult<TypedRunnableModel<TypedModel>> {
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .into_optimized()?
        .into_runnable()?;
    Ok(model)
}

fn run_voice_conversion(model: &TypedRunnableModel<TypedModel>, input_data: Vec<f32>) -> TractResult<Array1<f32>> {
    let input_tensor = Array1::from(input_data).into_tensor();
    let result = model.run(tvec![input_tensor.into()])?;
    
    // 出力を1次元配列に変換
    let output = result[0].to_array_view::<f32>()?.into_shape(result[0].len())?.to_owned();
    
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
