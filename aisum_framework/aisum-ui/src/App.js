import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [resultImages, setResultImages] = useState([]);
  const [model, setModel] = useState('resnet');

  // 이미지 선택 시
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPreviewURL(URL.createObjectURL(file));
  };

  // FastAPI에 전송하여 임베딩 요청
  const handleEmbed = async () => {
    if (!selectedFile) return;
  
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model_name", model);
  
    try {
      const response = await axios.post("http://127.0.0.1:8000/embed/", formData);
  
      const imagePaths = response.data.similar_images;
  
      // FastAPI는 상대경로 주므로, 절대경로로 변환
      const fullUrls = imagePaths.map(
        (path) => `http://127.0.0.1:8000/${path.replace(/\\/g, "/")}`
      );
  
      setResultImages(fullUrls); // 이제 resultImages는 이미지 URL 배열
    } catch (error) {
      alert("에러 발생: " + error.message);
    }
  };
  

  // 리셋 버튼 누르면 초기화
  const handleReset = () => {
    setSelectedFile(null);
    setPreviewURL(null);
    setResultImages([]);
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* 왼쪽: 컨트롤 영역 */}
      <div
        style={{
          flex: '0 0 220px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '20px',
          borderRight: '1px solid #ccc',
        }}
      >
        <div>
          {/* 이미지 업로드 */}
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ marginBottom: '20px' }}
          />

          {/* 모델 선택 + 실행 + 리셋 */}
          <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="resnet152">ResNet152</option>
              <option value="ViT">ViT (Vision Transformer)</option>
              <option value="convnextv2">ConvNeXt V2</option>
              <option value="efnet">EfficientNet</option>
            </select>
            <button onClick={handleEmbed} disabled={!selectedFile}>
              임베딩 실행
            </button>
            <button onClick={handleReset}>리셋</button>
          </div>
        </div>

        {/* 미리보기 이미지 (아래 고정) */}
        {previewURL && (
          <div>
            <p style={{ marginTop: '30px' }}>미리보기</p>
            <img
              src={previewURL}
              alt="미리보기"
              style={{
                maxWidth: '100%',
                maxHeight: '200px',
                border: '1px solid #ccc',
                objectFit: 'contain',
              }}
            />
          </div>
        )}
      </div>

      {/* 오른쪽: 결과 이미지들 */}
      <div style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
        <h3>처리된 결과 이미지</h3>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "15px",
              marginTop: "15px",
            }}
          >
            {resultImages.map((url, index) => (
              <img
                key={index}
                src={url}
                alt={`결과 ${index + 1}`}
                style={{
                  width: "200px",
                  height: "200px",
                  objectFit: "cover",
                  border: "2px solid green",
                }}
              />
            ))}
          </div>
      </div>
    </div>
  );
}

export default App;
