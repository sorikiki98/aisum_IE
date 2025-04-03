import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [resultImages, setResultImages] = useState([]);
  const [distances, setDistances] = useState([]);
  const [model, setModel] = useState('convnextv2_base');
  const [category1, setCategory1] = useState("");
  const [category2, setCategory2] = useState("");

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPreviewURL(URL.createObjectURL(file));
  };

  const handleEmbed = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model_name", model);
    formData.append("category1", category1);
    formData.append("category2", category2);

    try {
      const response = await axios.post("http://127.0.0.1:8000/embed/", formData);
      const imagePaths = response.data.similar_images;
      const distances = response.data.distances;

      const fullUrls = imagePaths.map(
        (path) => `http://127.0.0.1:8000/${path.replace(/\\/g, "/")}`
      );

      setResultImages(fullUrls);
      setDistances(distances);
    } catch (error) {
      alert("에러 발생: " + error.message);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewURL(null);
    setResultImages([]);
    setDistances([]);
    setCategory1("");
    setCategory2("");
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* 왼쪽: 컨트롤 영역 */}
      <div style={{
        flex: '0 0 220px',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        padding: '20px',
        borderRight: '1px solid #ccc'
      }}>
        <div>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ marginBottom: '20px' }}
          />

          <input
            type="text"
            placeholder="Category 1"
            value={category1}
            onChange={(e) => setCategory1(e.target.value)}
            style={{ marginBottom: '10px', width: '100%' }}
          />
          <input
            type="text"
            placeholder="Category 2"
            value={category2}
            onChange={(e) => setCategory2(e.target.value)}
            style={{ marginBottom: '20px', width: '100%' }}
          />

          <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="efnet">EfficientNetV2</option>
              <option value="vit">ViT</option>
              <option value="convnextv2_base">ConvNeXtV2 Base</option>
              <option value="convnextv2_large">ConvNeXtV2 Large</option>
              <option value="magiclens_base">Magiclens Base</option>
              <option value="magiclens_large">Magiclens Large</option>
              <option value="openai_clip">openai_clip</option>
              <option value="laion_clip">laion_clip</option>
              <option value="blip2">blip2</option>
              <option value="densenet121">densenet121</option>
              <option value="fashionclip">fashionclip</option>
            </select>
            <button onClick={handleEmbed} disabled={!selectedFile}>임베딩 실행</button>
            <button onClick={handleReset}>리셋</button>
          </div>
        </div>

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
                objectFit: 'contain'
              }}
            />
          </div>
        )}
      </div>

      {/* 오른쪽: 결과 이미지 */}
      <div style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
        <h3>처리된 결과 이미지</h3>
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '15px',
          marginTop: '15px'
        }}>
          {resultImages.map((url, index) => (
            <div key={index} style={{ textAlign: 'center' }}>
              <img
                src={url}
                alt={`결과 ${index + 1}`}
                style={{
                  width: '200px',
                  height: '200px',
                  objectFit: 'cover',
                  border: '2px solid green'
                }}
              />
              <p>Distance: {distances[index]?.toFixed(4)}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
