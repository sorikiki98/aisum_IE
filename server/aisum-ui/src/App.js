import React, { useState, useRef } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [detections, setDetections] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [category1, setCategory1] = useState("");
  const [category2, setCategory2] = useState("");
  const [searchResults, setSearchResults] = useState({});
  const [queryCropURL, setQueryCropURL] = useState(null);
  const fileInputRef = useRef(null);

  const modelOptions = [
    "magiclens", "unicom", "imagebind", "coca_mscoco", "coca_laion2b", "openai_clip", "laion_clip",
    "fashion_clip", "vit", "swin", "efnet", "siglip", "blip2", "dreamsim", "marqo_fashionclip"
  ];
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPreviewURL(URL.createObjectURL(file));
    setDetections([]);
    setSearchResults({});
    setQueryCropURL(null);
  };

  const handleDetect = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("http://127.0.0.1:8000/detect/", formData);
      setDetections(response.data.detections || []);
    } catch (error) {
      alert("Detection 실패: " + error.message);
    }
  };

  const handleBBoxClick = async (bbox) => {
    if (!selectedFile) {
      alert("파일이 업로드되지 않았습니다.");
      return;
    }
    if (selectedModels.length === 0) {
      alert("모델을 하나 이상 선택하세요.");
      return;
    }

    // 클릭한 부분 crop 미리보기 생성
    const imageElement = new Image();
    imageElement.src = previewURL;
    imageElement.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      const [xmin, ymin, xmax, ymax] = bbox;
      canvas.width = xmax - xmin;
      canvas.height = ymax - ymin;
      ctx.drawImage(imageElement, xmin, ymin, xmax - xmin, ymax - ymin, 0, 0, xmax - xmin, ymax - ymin);
      setQueryCropURL(canvas.toDataURL());
    };

    const newResults = {};

    for (const model of selectedModels) {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("model_name", model);
      formData.append("bbox_xmin", bbox[0]);
      formData.append("bbox_ymin", bbox[1]);
      formData.append("bbox_xmax", bbox[2]);
      formData.append("bbox_ymax", bbox[3]);
      formData.append("category1", category1);
      formData.append("category2", category2);

      try {
        const response = await axios.post("http://127.0.0.1:8000/search_bbox/", formData);
        const imagePaths = response.data.similar_images || [];
        const distances = response.data.distances || [];
        const fullUrls = imagePaths.map(path => `http://127.0.0.1:8000/${path.replace(/\\/g, "/")}`);
        newResults[model] = fullUrls.map((url, idx) => ({
          url,
          distance: distances[idx]?.toFixed(4)
        }));
      } catch (error) {
        console.error(`모델 ${model} 검색 실패: ${error.message}`);
      }
    }

    setSearchResults(newResults);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewURL(null);
    setDetections([]);
    setSelectedModels([]);
    setSearchResults({});
    setQueryCropURL(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* 왼쪽: 컨트롤 패널 */}
      <div style={{ flex: '0 0 300px', padding: '20px', borderRight: '1px solid #ccc', overflowY: 'auto' }}>
        <input type="file" accept="image/*" onChange={handleFileChange} ref={fileInputRef} style={{ marginBottom: '10px' }} />

        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
          <button onClick={handleDetect} disabled={!selectedFile}>Search</button>
          <button onClick={handleReset}>Reset</button>
        </div>

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
          style={{ marginBottom: '10px', width: '100%' }}
        />

        <div style={{ marginBottom: '10px' }}>
          {modelOptions.map((model) => (
            <div key={model}>
              <label>
                <input
                  type="checkbox"
                  value={model}
                  checked={selectedModels.includes(model)}
                  onChange={(e) => {
                    const value = e.target.value;
                    setSelectedModels(prev => prev.includes(value) ? prev.filter(m => m !== value) : [...prev, value]);
                  }}
                />
                {model}
              </label>
            </div>
          ))}
        </div>

        {/* 왼쪽 하단: 전체 이미지 + bbox */}
        {previewURL && (
          <div style={{ position: 'relative', marginTop: '20px', width: '100%', height: 'auto' }}>
            <h4>전체 이미지</h4>
            <img
              src={previewURL}
              alt="미리보기"
              style={{
                width: '100%',
                height: 'auto',
                border: '1px solid #ccc',
                objectFit: 'contain',
                maxHeight: '400px'
              }}
            />
            {detections.map((det, idx) => {
              const [xmin, ymin, xmax, ymax] = det.bbox;
              const boxStyle = {
                position: 'absolute',
                left: `${xmin}px`,
                top: `${ymin}px`,
                width: `${xmax - xmin}px`,
                height: `${ymax - ymin}px`,
                border: '2px solid red',
                cursor: 'pointer',
                boxSizing: 'border-box'
              };
              return (
                <div
                  key={idx}
                  style={boxStyle}
                  onClick={() => handleBBoxClick(det.bbox)}
                  title={det.class}
                />
              );
            })}
          </div>
        )}

        {/* 클릭한 bbox crop 미리보기 */}
        {queryCropURL && (
          <div style={{ marginTop: '20px' }}>
            <h4>쿼리 이미지</h4>
            <img
              src={queryCropURL}
              alt="쿼리 이미지"
              style={{
                width: '100%',
                maxHeight: '200px',
                objectFit: 'contain',
                border: '2px solid blue',
                marginTop: '10px'
              }}
            />
          </div>
        )}
      </div>

      {/* 오른쪽: 검색 결과 */}
      <div style={{ flex: 1, padding: '20px', overflowY: 'auto', position: 'relative' }}>
        {Object.keys(searchResults).length > 0 && (
          <div style={{ marginTop: '20px' }}>
            <h3>검색 결과:</h3>
            {Object.entries(searchResults).map(([model, results]) => (
              <div key={model} style={{ marginBottom: '20px' }}>
                <h4>모델: {model}</h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 1fr)', gap: '8px' }}>
                  {results.map((result, idx) => (
                    <div key={idx} style={{ width: '150px', textAlign: 'center' }}>
                      <img src={result.url} alt={`결과 ${idx}`} style={{ width: '100%', height: 'auto', border: '1px solid #999' }} />
                      <div>Distance: {result.distance}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;