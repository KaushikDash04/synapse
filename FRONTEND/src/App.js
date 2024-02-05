import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [question, setQuestion] = useState('What is life?');
  const [model, setModel] = useState('gemini-pro');
  const [response, setResponse] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.get(`http://127.0.0.1:8000/generate_text_gemini/?model_name=${model}&question=${question}`);
      setResponse(res.data['AI Response']);
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

  return (
    <div className="App">
      <h1>AI Text Generation</h1>
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
        <button type="submit">Generate Text</button>
      </form>
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
