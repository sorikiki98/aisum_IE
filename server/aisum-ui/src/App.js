import React, { useState, useRef } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [selectedModels, setSelectedModels] = useState([]);
  const [category1, setCategory1] = useState("");
  const [category2, setCategory2] = useState("");
  const [resultsByModel, setResultsByModel] = useState({});
  const fileInputRef = useRef(null);
  const modelOptions = [
    "magiclens", "unicom", "imagebind", "coca_mscoco", "coca_laion2b", "openai_clip", "laion_clip",
    "fashion_clip", "vit", "swin", "efnet", "siglip", "blip2","dreamsim"];
    
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPreviewURL(URL.createObjectURL(file));
  };

  const handleModelSelection = (e) => {
    const value = e.target.value;
    setSelectedModels((prev) =>
      prev.includes(value) ? prev.filter((m) => m !== value) : [...prev, value]
    );
  };

  const handleEmbed = async () => {
    if (!selectedFile || selectedModels.length === 0) return;

    const newResults = {};

    for (const model of selectedModels) {
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
          (path) => `http://127.0.0.1:8000${path.replace(/\\/g, "/")}`
        );

        newResults[model] = {
          urls: fullUrls,
          distances: distances
        };
      } catch (error) {
        alert(`모델 [${model}] 처리 중 에러 발생: ${error.message}`);
      }
    }

    setResultsByModel(newResults);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewURL(null);
    setCategory1("");
    setCategory2("");
    setSelectedModels([]);
    setResultsByModel({});
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* 왼쪽: 컨트롤 영역 */}
      <div style={{
        flex: '0 0 250px',
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
            ref={fileInputRef}
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

          <div style={{ marginBottom: '20px', maxHeight: '200px', overflowY: 'auto' }}>
            {modelOptions.map((model) => (
              <div key={model}>
                <label>
                  <input
                    type="checkbox"
                    value={model}
                    checked={selectedModels.includes(model)}
                    onChange={handleModelSelection}
                  />
                  {model}
                </label>
              </div>
            ))}
          </div>

          <button onClick={handleEmbed} disabled={!selectedFile}>이미지 검색</button>
          <button onClick={handleReset} style={{ marginTop: '10px' }}>리셋</button>
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

      {/* 오른쪽: 결과 영역 */}
      <div style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
        {Object.keys(resultsByModel).length === 0 ? (
          <h3>처리된 결과 이미지가 없습니다.</h3>
        ) : (
          Object.entries(resultsByModel).map(([model, data]) => (
            <div key={model} style={{ marginBottom: '20px' }}>
              <h4 style={{ margin: '5px 0' }}>
              🔍 Model: {model}</h4>
              <div style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '4px',
                justifyContent: 'flex-start'
              }}>
                {data.urls.map((url, index) => (
                  <div key={index} style={{ 
                    width: 'calc(6.5% - 4px)',
                    textAlign: 'center',
                    boxSizing: 'border-box'}}>
                    <img
                      src={url}
                      alt={`결과 ${index + 1}`}
                      style={{
                        width: '100%',
                        height: 'auto',
                        objectFit: 'cover',
                        border: '1px solid #999'
                      }}
                    />
                    <p style={{ fontSize: '11px', margin: '2px 0', fontWeight: '500' }}>
                    Distance: {data.distances[index]?.toFixed(4)}</p>
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default App;