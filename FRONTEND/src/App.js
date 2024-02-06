import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [question, setQuestion] = useState('What is life?');
  const [model, setModel] = useState('gemini-pro');
  const [response, setResponse] = useState('');
  const [error, setError] = useState('');
  const [image, setImage] = useState(null);
  const [pdf, setPdf] = useState(null);
  const [submittedQuestion, setSubmittedQuestion] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const formData = new FormData();
  
      // Display the submitted question
      setSubmittedQuestion(question);
  
      if (model === 'gemini-vision' && image) {
        formData.append('image', image);
        const uploadResponse = await axios.post('http://127.0.0.1:8000/upload_image/', formData);
        console.log(uploadResponse.data);
      } else if (model === 'pdf-gpt' && pdf) {
        formData.append('pdf', pdf);
        const uploadResponse = await axios.post('http://127.0.0.1:8000/upload_pdf/', formData);
        console.log(uploadResponse.data);
      }
  
      const generateResponse = await axios.get(`http://127.0.0.1:8000/generate_text_gemini/?model_name=${model}&question=${question}`);
      setResponse(generateResponse.data['AI Response']);
      setError('');
    } catch (error) {
      console.error('Error:', error);
      if (error.response) {
        setError(error.response.data.detail);
      } else {
        setError('An error occurred while processing your request. Please try again later.');
      }
    }
  };
  

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handlePdfChange = (e) => {
    setPdf(e.target.files[0]);
  };

  return (
    <div className="App">
      <h1>SYNAPSE</h1>
      <form onSubmit={handleSubmit}>
        <label htmlFor="question">Question:</label>
        <input
          type="text"
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          required
        />
        <label htmlFor="model">Select Model:</label>
        <select
          id="model"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="gemini-pro">Gemini Pro</option>
          <option value="gemini-vision">Gemini Vision</option>
          <option value="pdf-gpt">PDF GPT</option>
        </select>
        {model === 'gemini-vision' && (
          <>
            <label htmlFor="image">Upload Image:</label>
            <input
              type="file"
              id="image"
              accept="image/*"
              onChange={handleImageChange}
              required
            />
          </>
        )}
        {model === 'pdf-gpt' && (
          <>
            <label htmlFor="pdf">Upload PDF:</label>
            <input
              type="file"
              id="pdf"
              accept=".pdf"
              onChange={handlePdfChange}
              required
            />
          </>
        )}
        <button type="submit">Submit</button>
      </form>
      {submittedQuestion && (
        <div>
          <h2>Submitted Question:</h2>
          <p>{submittedQuestion}</p>
        </div>
      )}
      {error && (
        <div className="error">
          <p>{error}</p>
        </div>
      )}
      {response && (
        <div>
          <h2>AI Response:</h2>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
}

export default App;
