import React, { useState, useRef } from "react";
import axios from "axios";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [selectedModels, setSelectedModels] = useState([]);
  const [useEnsemble, setUseEnsemble] = useState(false);
  const [detections, setDetections] = useState([]);
  const [category, setCategory] = useState("");
  const [resultsByModel, setResultsByModel] = useState({});
  const fileInputRef = useRef(null);
  const modelOptions = ["dreamsim", "marqo_fashionsiglip", "marqo_ecommerce_l"];

  const imageRef = useRef(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  const handleFileChange = async (event) => {
    try {
      const response = await axios.get("http://127.0.0.1:8000/reset");
      console.log(response.data.message);
    } catch (error) {
      console.error("Error processing reset:", error);
    }
    const file = event.target.files[0];
    setSelectedFile(file);
    setPreviewURL(URL.createObjectURL(file));
    setResultsByModel({});
    setDetections([]);
    setImageLoaded(false);
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

  const handleSearch = async () => {
    if (!selectedFile || (selectedModels.length === 0 && !useEnsemble)) return;

    const results = {};
    const modelPromises = selectedModels.map((model) => {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("embedding_model_name", model);

      return axios
        .post("http://127.0.0.1:8000/search/", formData)
        .then((response) => {
          const result = response.data;
          const imagePaths = result.similar_images || [];
          const p_scores = result.p_scores || [];

          const fullUrls = imagePaths.map(
            (path, idx) =>{
              // MySQL에서 가져온 URL이 이미 완전한 URL인지 확인
              if (path.startsWith('http://') || path.startsWith('https://')) {
                return `${path}?v=${Date.now()}_${idx}`;
              } else {
                // 로컬 경로인 경우 (기존 방식)
                return `http://127.0.0.1:8000${path.replace(/\\/g, "/")}?v=${Date.now()}_${idx}`;
              }
          });

          results[model] = {
            urls: fullUrls,
            p_scores: p_scores,
          };
        })
        .catch((error) => {
          console.error(`Error processing model ${model}:`, error);
          let errorMessage = error.message;

          if (error.response) {
            errorMessage = `Server Error (${
              error.response.status
            }): ${JSON.stringify(error.response.data)}`;
          } else if (error.request) {
            errorMessage = "서버로부터 응답을 받지 못했습니다.";
          }

          results[model] = {
            urls: [],
            p_scores: [],
            error: errorMessage,
          };
        });
    });

    Promise.all(modelPromises).then(() => {
      setResultsByModel(results);

      if (!useEnsemble) return;

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("embedding_model_name", "ensemble");

      return axios
        .post("http://127.0.0.1:8000/search/", formData)
        .then((response) => {
          const result = response.data;
          if (!result) throw new Error("앙상블 결과를 받지 못했습니다.");
          console.log(result);

          const imagePaths = result.similar_images || [];
          const p_scores = result.p_scores || [];

          if (!imagePaths.length || !p_scores.length) {
            throw new Error("앙상블 검색 결과가 없습니다.");
          }

          const fullUrls = imagePaths.map(
            (path, idx) =>{
              if (path.startsWith('http://') || path.startsWith('https://')) {  
                return `${path}?v=${Date.now()}_${idx}`; 
              } else {
                return `http://127.0.0.1:8000${path.replace(/\\/g, "/")}?v=${Date.now()}_${idx}`;
              }
          });

          setResultsByModel((prev) => ({
            ...prev,
            ensemble: {
              urls: fullUrls,
              p_scores: p_scores,
            },
          }));
        })
        .catch((error) => {
          console.error("Error processing ensemble:", error);
          let errorMessage = error.message;

          if (error.response) {
            errorMessage = `Server Error (${
              error.response.status
            }): ${JSON.stringify(error.response.data)}`;
          } else if (error.request) {
            errorMessage = "서버로부터 응답을 받지 못했습니다.";
          }

          setResultsByModel((prev) => ({
            ...prev,
            ensemble: {
              urls: [],
              p_scores: [],
              error: errorMessage,
            },
          }));
        });
    });
  };

  const handleSearchByBox = async (bbox, category) => {
    try {
      const response = await axios.get("http://127.0.0.1:8000/reset");
      console.log(response.data.message);
    } catch (error) {
      console.error("Error processing reset:", error);
    }
    const [xmin, ymin, xmax, ymax] = bbox;
    const results = {};
    const modelPromises = selectedModels.map((model) => {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("model_name", model);
      formData.append("bbox_xmin", xmin);
      formData.append("bbox_ymin", ymin);
      formData.append("bbox_xmax", xmax);
      formData.append("bbox_ymax", ymax);
      formData.append("category", category);

      return axios
        .post("http://127.0.0.1:8000/search_bbox/", formData)
        .then((response) => {
          const result = response.data;
          const imagePaths = result.similar_images || [];
          const p_scores = result.p_scores || [];

          const fullUrls = imagePaths.map(
            (path, idx) =>{
              // MySQL에서 가져온 URL이 이미 완전한 URL인지 확인
              if (path.startsWith('http://') || path.startsWith('https://')) {
                return `${path}?v=${Date.now()}_${idx}`;
              } else {
                // 로컬 경로인 경우 (기존 방식)
                return `http://127.0.0.1:8000${path.replace(/\\/g, "/")}?v=${Date.now()}_${idx}`;
              }
          });


          results[model] = {
            urls: fullUrls,
            p_scores: p_scores,
          };
        })
        .catch((error) => {
          console.error(`Error processing model ${model}:`, error);
          let errorMessage = error.message;

          if (error.response) {
            errorMessage = `Server Error (${
              error.response.status
            }): ${JSON.stringify(error.response.data)}`;
          } else if (error.request) {
            errorMessage = "서버로부터 응답을 받지 못했습니다.";
          }

          results[model] = {
            urls: [],
            p_scores: [],
            error: errorMessage,
          };
        });
    });

    Promise.all(modelPromises).then(() => {
      setResultsByModel(results);

      if (!useEnsemble) return;
      console.log("useEnsemble", useEnsemble)

      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("model_name", "ensemble");
      formData.append("bbox_xmin", xmin);
      formData.append("bbox_ymin", ymin);
      formData.append("bbox_xmax", xmax);
      formData.append("bbox_ymax", ymax);
      formData.append("category", category);

      return axios
        .post("http://127.0.0.1:8000/search_bbox/", formData)
        .then((response) => {
          const result = response.data;
          if (!result) throw new Error("앙상블 결과를 받지 못했습니다.");
          console.log(result);

          const imagePaths = result.similar_images || [];
          const p_scores = result.p_scores || [];

          if (!imagePaths.length || !p_scores.length) {
            throw new Error("앙상블 검색 결과가 없습니다.");
          }

          const fullUrls = imagePaths.map(
            (path, idx) =>{
              if (path.startsWith('http://') || path.startsWith('https://')) {
                return `${path}?v=${Date.now()}_${idx}`;
              } else {
                return `http://127.0.0.1:8000${path.replace(/\\/g, "/")}?v=${Date.now()}_${idx}`;
              }
          });

          setResultsByModel((prev) => ({
            ...prev,
            ensemble: {
              urls: fullUrls,
              p_scores: p_scores,
            },
          }));
        })
        .catch((error) => {
          console.error("Error processing ensemble:", error);
          let errorMessage = error.message;

          if (error.response) {
            errorMessage = `Server Error (${
              error.response.status
            }): ${JSON.stringify(error.response.data)}`;
          } else if (error.request) {
            errorMessage = "서버로부터 응답을 받지 못했습니다.";
          }

          setResultsByModel((prev) => ({
            ...prev,
            ensemble: {
              urls: [],
              p_scores: [],
              error: errorMessage,
            },
          }));
        });
    });
  };

  const handleDetection = async () => {
    const formData = new FormData();
    formData.append("file", selectedFile);

    return axios
      .post("http://127.0.0.1:8000/detect/", formData)
      .then((response) => {
        const result = response.data;
        if (!result || !result.detections)
          throw new Error("탐지 결과를 받지 못했습니다.");
        console.log(result);

        setDetections(result.detections);
      })
      .catch((error) => {
        console.error("❌ 탐지 실패:", error);
      });
  };

  const handleReset = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:8000/reset_all");
      console.log(response.data.message);
    } catch (error) {
      console.error("Error processing reset:", error);
    } finally {
      setSelectedFile(null);
      setPreviewURL(null);
      setCategory("");
      setSelectedModels([]);
      setUseEnsemble(false);
      setResultsByModel({});
      setDetections([]);
      setImageLoaded(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      {/* 왼쪽: 컨트롤 영역 */}
      <div
        style={{
          flex: "0 0 250px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "20px",
          borderRight: "1px solid #ccc",
        }}
      >
        <div>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            ref={fileInputRef}
            style={{ marginBottom: "20px" }}
          />

          <div
            style={{
              marginBottom: "20px",
              maxHeight: "200px",
              overflowY: "auto",
            }}
          >
            {["ensemble", ...modelOptions].map((model) => (
              <div key={model}>
                <label>
                  <input
                    type="checkbox"
                    value={model}
                    checked={
                      model === "ensemble"
                        ? useEnsemble
                        : selectedModels.includes(model)
                    }
                    onChange={
                      model === "ensemble"
                        ? (e) => setUseEnsemble(e.target.checked)
                        : handleModelSelection
                    }
                  />
                  {model}
                </label>
              </div>
            ))}
          </div>

          <button
            onClick={() => {
              handleSearch();
              handleDetection();
            }}
            disabled={
              !selectedFile || (selectedModels.length === 0 && !useEnsemble)
            }
          >
            이미지 검색
          </button>
          <button onClick={handleReset} style={{ marginTop: "10px" }}>
            리셋
          </button>
        </div>

        {previewURL && (
          <div
            style={{
              position: "relative",
              display: "inline-block",
              marginTop: "30px",
            }}
          >
            <img
              ref={imageRef}
              src={previewURL}
              alt="미리보기"
              style={{
                maxWidth: "100%",
                maxHeight: "300px",
                border: "1px solid #ccc",
                objectFit: "contain",
                display: "block",
              }}
              onLoad={() => setImageLoaded(true)}
            />

            {/* bbox 표시 */}
            {imageLoaded &&
              imageRef.current &&
              detections
                .map((d, idx) => {
                  const [x1, y1, x2, y2] = d.bbox;
                  const img = imageRef.current;

                  const area = (x2 - x1) * (y2 - y1);
                  return { ...d, area };
                })
                .sort((a, b) => b.area - a.area)
                .map((d, sortedIdx) => {
                  const [x1, y1, x2, y2] = d.bbox;
                  const img = imageRef.current;

                  const naturalWidth = img.naturalWidth;
                  const naturalHeight = img.naturalHeight;
                  const imgWidth = img.width;
                  const imgHeight = img.height;

                  const scaleX = imgWidth / naturalWidth;
                  const scaleY = imgHeight / naturalHeight;

                  const left = x1 * scaleX;
                  const top = y1 * scaleY;
                  const width = (x2 - x1) * scaleX;
                  const height = (y2 - y1) * scaleY;

                  return (
                    <div
                      key={sortedIdx}
                      style={{
                        position: "absolute",
                        left: `${left}px`,
                        top: `${top}px`,
                        width: `${width}px`,
                        height: `${height}px`,
                        border: "2px solid red",
                        boxSizing: "border-box",
                        cursor: "pointer",
                        backgroundColor: "transparent",
                        zIndex: 1000 + sortedIdx, // 작은 박스일수록 높은 zIndex
                        pointerEvents: "auto", // 이벤트 받기
                      }}
                      title={`${d.class} (${(d.confidence * 100).toFixed(1)}%)`}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor =
                          "rgba(255, 0, 0, 0.3)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = "transparent";
                      }}
                      onClick={() => {
                        handleSearchByBox(d.bbox, d.class);
                        console.log(d.class, "Click!!!")
                      }}
                    />
                  );
                })}
          </div>
        )}
      </div>

      {/* 오른쪽: 결과 영역 */}
      <div style={{ flex: 1, padding: "20px", overflowY: "auto" }}>
        {Object.keys(resultsByModel).length === 0 ? (
          <h3>처리된 결과 이미지가 없습니다.</h3>
        ) : (
          <>
            {/* Ensemble 결과 */}
            {resultsByModel["ensemble"] && (
              <div style={{ marginBottom: "20px" }}>
                <h4 style={{ margin: "5px 0" }}>🔍 Model: ensemble</h4>
                {resultsByModel["ensemble"].error ? (
                  <div style={{ color: "red", marginBottom: "10px" }}>
                    Error: {resultsByModel["ensemble"].error}
                  </div>
                ) : (
                  <div
                    style={{
                      display: "flex",
                      flexWrap: "wrap",
                      gap: "4px",
                      justifyContent: "flex-start",
                    }}
                  >
                    {resultsByModel["ensemble"].urls.map((url, index) => (
                      <div
                        key={url}
                        style={{
                          width: "calc(10% - 4px)",
                          textAlign: "center",
                          boxSizing: "border-box",
                        }}
                      >
                        <img
                          src={url}
                          alt={`결과 ${index + 1}`}
                          style={{
                            width: "100%",
                            height: "auto",
                            objectFit: "cover",
                            border: "1px solid #999",
                          }}
                        />
                        <p
                          style={{
                            fontSize: "11px",
                            margin: "2px 0",
                            fontWeight: "500",
                          }}
                        >
                          P Score: {resultsByModel["ensemble"].p_scores[index]?.toFixed(4)}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* 개별 모델 결과 표시 */}
            {Object.entries(resultsByModel)
              .filter(([modelName]) => modelName !== "ensemble")
              .map(([modelName, data]) => (
                <div key={modelName} style={{ marginBottom: "20px" }}>
                  <h4 style={{ margin: "5px 0" }}>🔍 Model: {modelName}</h4>
                  {data.error ? (
                    <div style={{ color: "red", marginBottom: "10px" }}>
                      Error: {data.error}
                    </div>
                  ) : (
                    <div
                      style={{
                        display: "flex",
                        flexWrap: "wrap",
                        gap: "4px",
                        justifyContent: "flex-start",
                      }}
                    >
                      {data.urls.map((url, index) => (
                        <div
                          key={url}
                          style={{
                            width: "calc(10% - 4px)",
                            textAlign: "center",
                            boxSizing: "border-box",
                          }}
                        >
                          <img
                            src={url}
                            alt={`결과 ${index + 1}`}
                            style={{
                              width: "100%",
                              height: "auto",
                              objectFit: "cover",
                              border: "1px solid #999",
                            }}
                          />
                          <p
                            style={{
                              fontSize: "11px",
                              margin: "2px 0",
                              fontWeight: "500",
                            }}
                          >
                            P Score: {data.p_scores[index]?.toFixed(4)}
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
