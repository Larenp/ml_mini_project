import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");

  const handleChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://127.0.0.1:5001/predict", formData);
      setResult(res.data.prediction);
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Blood Group Detection</h1>

      <input type="file" onChange={handleChange} />
      <br /><br />

      <button onClick={handleUpload}>Predict</button>

      <h2>Result: {result}</h2>
    </div>
  );
}

export default App;