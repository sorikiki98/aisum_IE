import React, { useState, useRef } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [selectedModels, setSelectedModels] = useState([]);
  const [useEnsemble, setUseEnsemble] = useState(false);
  const [category, setCategory] = useState("");
  const [resultsByModel, setResultsByModel] = useState({});
  const fileInputRef = useRef(null);
  const modelOptions = ["dreamsim", "imagebind", "unicom"]

  const handleFileChange = async (event) => {
    try {
        const response = await axios.get("http://127.0.0.1:8000/reset");
        console.log(response.data.message);
    } catch(error) {
        console.error('Error processing reset:', error);
    }
    const file = event.target.files[0];
    setSelectedFile(file);
    setPreviewURL(URL.createObjectURL(file));
    setResultsByModel({});
  };

  const handleModelSelection = (e) => {
    const value = e.target.value;
    setSelectedModels((prev) => {
        let updated;
        if (prev.includes(value)) {
            updated = prev.filter((m) => m !== value);
        } else {
            updated = [...prev, value];
        }
        return updated;
    });
  };

  const handleSearch = () => {
  if (!selectedFile || (selectedModels.length === 0 && !useEnsemble)) return;

  const results = {};
  const modelPromises = selectedModels.map((model) => {
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("embedding_model_name", model);

    return axios.post("http://127.0.0.1:8000/search/", formData)
      .then((response) => {
        const result = response.data;
        const imagePaths = result.similar_images || [];
        const distances = result.distances || [];

        const fullUrls = imagePaths.map((path) =>
          `http://127.0.0.1:8000${path.replace(/\\/g, "/")}`
        );

        results[model] = {
          urls: fullUrls,
          distances: distances
        };
      })
      .catch((error) => {
        console.error(`Error processing model ${model}:`, error);
        let errorMessage = error.message;

        if (error.response) {
          errorMessage = `Server Error (${error.response.status}): ${JSON.stringify(error.response.data)}`;
        } else if (error.request) {
          errorMessage = '서버로부터 응답을 받지 못했습니다.';
        }

        results[model] = {
          urls: [],
          distances: [],
          error: errorMessage
        };
      });
  });

  Promise.all(modelPromises)
    .then(() => {
      setResultsByModel(results);

      if (!useEnsemble) return;

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("embedding_model_name", "ensemble");

      return axios.post("http://127.0.0.1:8000/search/", formData)
        .then((response) => {
          const result = response.data;
          if (!result) throw new Error('앙상블 결과를 받지 못했습니다.');
          console.log(result)

          const imagePaths = result.similar_images || [];
          const distances = result.distances || [];

          if (!imagePaths.length || !distances.length) {
            throw new Error('앙상블 검색 결과가 없습니다.');
          }

          const fullUrls = imagePaths.map((path, idx) =>
             `http://127.0.0.1:8000${path.replace(/\\/g, "/")}?v=${Date.now()}_${idx}`
          );

          setResultsByModel((prev) => ({
            ...prev,
            ensemble: {
              urls: fullUrls,
              distances: distances
            }
          }));
        })
        .catch((error) => {
          console.error('Error processing ensemble:', error);
          let errorMessage = error.message;

          if (error.response) {
            errorMessage = `Server Error (${error.response.status}): ${JSON.stringify(error.response.data)}`;
          } else if (error.request) {
            errorMessage = '서버로부터 응답을 받지 못했습니다.';
          }

          setResultsByModel((prev) => ({
            ...prev,
            ensemble: {
              urls: [],
              distances: [],
              error: errorMessage
            }
          }));
        });
    });
};


  const handleReset = async () => {
    try {
        const response = await axios.get("http://127.0.0.1:8000/reset_all");
        console.log(response.data.message);
    } catch(error) {
        console.error('Error processing reset:', error);
    } finally {
        setSelectedFile(null);
        setPreviewURL(null);
        setCategory("");
        setSelectedModels([]);
        setUseEnsemble(false);
        setResultsByModel({});
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
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

          <div style={{ marginBottom: '20px', maxHeight: '200px', overflowY: 'auto' }}>
            {['ensemble', ...modelOptions].map((model) => (
              <div key={model}>
                <label>
                  <input
                    type="checkbox"
                    value={model}
                    checked={model === 'ensemble' ? useEnsemble : selectedModels.includes(model)}
                    onChange={model === 'ensemble' ?
                      (e) => setUseEnsemble(e.target.checked) :
                      handleModelSelection
                    }
                  />
                  {model}
                </label>
              </div>
            ))}
          </div>

          <button
            onClick={handleSearch}
            disabled={!selectedFile || (selectedModels.length === 0 && !useEnsemble)}
          >
            이미지 검색
          </button>
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
          <>
            {/* Ensemble 결과 */}
            {resultsByModel['ensemble'] && (
              <div style={{ marginBottom: '20px' }}>
                <h4 style={{ margin: '5px 0' }}>
                  🔍 Model: ensemble
                </h4>
                {resultsByModel['ensemble'].error ? (
                  <div style={{ color: 'red', marginBottom: '10px' }}>
                    Error: {resultsByModel['ensemble'].error}
                  </div>
                ) : (
                  <div style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '4px',
                    justifyContent: 'flex-start'
                  }}>
                    {resultsByModel['ensemble'].urls.map((url, index) => (
                      <div key={url} style={{
                        width: 'calc(6.5% - 4px)',
                        textAlign: 'center',
                        boxSizing: 'border-box'
                      }}>
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
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* 개별 모델 결과 표시 */}
            {Object.entries(resultsByModel)
              .filter(([modelName]) => modelName !== 'ensemble')
              .map(([modelName, data]) => (
                <div key={modelName} style={{ marginBottom: '20px' }}>
                  <h4 style={{ margin: '5px 0' }}>
                    🔍 Model: {modelName}
                  </h4>
                  {data.error ? (
                    <div style={{ color: 'red', marginBottom: '10px' }}>
                      Error: {data.error}
                    </div>
                  ) : (
                    <div style={{
                      display: 'flex',
                      flexWrap: 'wrap',
                      gap: '4px',
                      justifyContent: 'flex-start'
                    }}>
                      {data.urls.map((url, index) => (
                        <div key={url} style={{
                          width: 'calc(6.5% - 4px)',
                          textAlign: 'center',
                          boxSizing: 'border-box'
                        }}>
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
                            Distance: {data.distances[index]?.toFixed(4)}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
          </>
        )}
      </div>
    </div>
  );
}

export default App;