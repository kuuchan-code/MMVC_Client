use ort::{environment::Environment, session::SessionBuilder, tensor::OrtOwnedTensor, Value};
use ndarray::{Array, CowArray};
use std::sync::Arc;

struct AudioTransformer {
    hop_length: usize,
    max_wav_value: f32,
    sample_rate: usize,
    gpu_id: Option<i32>,
    ort_session: Arc<Environment>,
}

impl AudioTransformer {
    pub fn new(hop_length: usize, max_wav_value: f32, sample_rate: usize, gpu_id: Option<i32>, ort_session: Arc<Environment>) -> Self {
        Self {
            hop_length,
            max_wav_value,
            sample_rate,
            gpu_id,
            ort_session,
        }
    }

    pub fn audio_transform(
        &self,
        input_data: &[u8],
        target_id: i64,
        stft_padding: usize,
        conv1d_padding: usize,
    ) -> Vec<u8> {
        let _stft_length = stft_padding * self.hop_length;
        let conv1d_length = conv1d_padding * self.hop_length;

        // 入力バイトデータをf32に変換
        let signal: Vec<f32> = input_data
            .chunks(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / self.max_wav_value)
            .collect();

        // 音声変換処理
        let audio = self.perform_voice_conversion_onnx(&signal, target_id);

        // conv1d paddingを考慮した音声のトリミング
        let trimmed_audio = &audio[conv1d_length..audio.len() - conv1d_padding];

        // 結果をint16に変換してバイトに変換
        trimmed_audio
            .iter()
            .flat_map(|&sample| ((sample * self.max_wav_value) as i16).to_le_bytes().to_vec())
            .collect()
    }

    fn perform_voice_conversion_onnx(&self, signal: &[f32], target_id: i64) -> Vec<f32> {
        let session = SessionBuilder::new(&self.ort_session)
            .unwrap()
            .with_model_from_file("G_best.onnx")
            .unwrap();

        // `CowArray` を使用して、所有権のない形でデータを渡す
        let input_array = Array::from_shape_vec((1, signal.len()), signal.to_vec()).unwrap();
        let cow_array = CowArray::from(input_array.view());
        let allocator_ptr = std::ptr::null_mut(); // 既存のアロケータがない場合はnullポインタを使用
        let binding = cow_array.into_dyn(); // cow_array から into_dyn() を一時的ではなく保持
        let input_tensor = Value::from_array(allocator_ptr, &binding).expect("Failed to create input tensor");
        
        // ONNXモデルで推論を実行
        let outputs = session.run(vec![input_tensor]).unwrap();

        // 推論結果を取得してベクトルに変換
        let output_tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
        let audio: Vec<f32> = output_tensor.view().to_owned().into_raw_vec();

        audio
    }
}

fn main() {
    // 環境の作成
    let environment = Arc::new(Environment::builder().with_name("audio_transformer").build().unwrap());

    // オーディオトランスフォーマーの初期化
    let transformer = AudioTransformer::new(
        128,           // hop_length
        32768.0,       // max_wav_value
        24000,         // sample_rate
        None,          // gpu_id
        environment,   // ort_session
    );

    // サンプルデータを用いて変換を実行
    let input_data = vec![0u8; 1024];  // 仮の入力データ
    let target_id = 2;                 // 話者IDのサンプル
    let output = transformer.audio_transform(&input_data, target_id, 2, 2);

    // 結果を処理
    println!("変換されたオーディオデータ: {:?}", output);
}
