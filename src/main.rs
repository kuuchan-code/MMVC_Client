use hound;
use ndarray::{Array1, Array2};
use realfft::{RealFftPlanner, num_complex::Complex32};
use tract_onnx::prelude::*;
use smallvec::smallvec;

fn read_wav_file(file_path: &str) -> Result<Array1<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(file_path)?;
    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    
    // 正規化: i16 -> f32 [-1.0, 1.0] の範囲
    let wav_data: Array1<f32> = samples.iter().map(|&x| x as f32 / i16::MAX as f32).collect();
    Ok(wav_data)
}

fn generate_spectrogram(wav_data: Array1<f32>, filter_length: usize, hop_length: usize, win_length: usize) -> Array2<f32> {
    let mut fft = RealFftPlanner::<f32>::new();
    let r2c = fft.plan_fft_forward(filter_length);
    
    let hann_window: Array1<f32> = Array1::from_iter((0..win_length).map(|x| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * x as f32 / win_length as f32).cos())));

    // 音声データをSTFTで変換
    let mut spectrogram = Vec::new();
    let output_size = (filter_length / 2) + 1;  // 出力サイズを計算

    let mut idx = 0;
    while idx + win_length <= wav_data.len() {
        let frame = wav_data.slice(ndarray::s![idx..(idx + win_length)]);
        let mut input: Vec<f32> = frame.to_vec();
        for (i, win) in hann_window.iter().enumerate() {
            input[i] *= win;
        }
        let mut output = vec![Complex32::default(); output_size];
        r2c.process(&mut input, &mut output).unwrap();
        
        let magnitude: Vec<f32> = output.iter().map(|c| c.norm()).collect();
        spectrogram.push(magnitude);
        idx += hop_length;
    }
    
    let spec_len = spectrogram.len();
    let spec_flattened: Vec<f32> = spectrogram.into_iter().flatten().collect();
    Array2::from_shape_vec((spec_len, output_size), spec_flattened).unwrap()
}

fn run_onnx_inference(model_path: &str, spec: Array2<f32>, sid: i64) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // モデルをロード
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .into_optimized()?
        .into_runnable()?;

    // specをTensorに変換
    let spec_shape = spec.shape().to_vec();
    let spec_tensor = Tensor::from_shape(&spec_shape, spec.as_slice().unwrap())?;

    // sidをTensorに変換
    let sid_tensor = Tensor::from(sid);

    // 入力データを準備
    let input = smallvec![spec_tensor.into(), sid_tensor.into()];
    
    // 推論実行
    let result = model.run(input)?;

    // 推論結果を取得
    let output: Tensor = result[0].clone().into_tensor();
    let output_array: Array1<f32> = output.into_array::<f32>()?.into_dimensionality()?;

    Ok(output_array.to_vec())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // パラメータ
    let sid = 2;
    let _text = "m"; // 現在は使用しない
    let model_path = "logs//runa//G_best.onnx";
    let wav_file_path = "emotion001.wav";
    let filter_length = 512;
    let hop_length = 128;
    let win_length = 512;

    // 1. wavファイルの読み込み
    let wav_data = read_wav_file(wav_file_path)?;

    // 2. スペクトログラムの生成
    let spec = generate_spectrogram(wav_data, filter_length, hop_length, win_length);

    // 3. ONNXモデルを用いた音声変換
    let result = run_onnx_inference(model_path, spec, sid)?;

    // 4. 結果を処理（例: 結果の長さを表示）
    println!("推論結果の長さ: {}", result.len());
    
    Ok(())
}
