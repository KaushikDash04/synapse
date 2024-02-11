import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  Button,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  TextField,
} from "@mui/material";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import "./App.css"; // Import the CSS file for styling
import Sidebar from "./components/Sidebar";
import SendSharpIcon from "@mui/icons-material/SendSharp";
import KeyboardDoubleArrowUpSharpIcon from "@mui/icons-material/KeyboardDoubleArrowUpSharp";
import PsychologyRoundedIcon from "@mui/icons-material/PsychologyRounded";
import PsychologyAltRoundedIcon from "@mui/icons-material/PsychologyAltRounded";
import TypewriterAnimation from "./TypewriterAnimation"; // Import the TypewriterAnimation component
import TypingAnimation from "./TypingAnimation";

function App() {
  const [question, setQuestion] = useState("");
  const [model, setModel] = useState("gemini-pro");
  const [response, setResponse] = useState("");
  const [error, setError] = useState("");
  const [image, setImage] = useState(null);
  const [pdf, setPdf] = useState(null);
  const [imagePreview, setImagePreview] = useState(null); // State to store image preview
  const [pdfPreview, setPdfPreview] = useState(null); // State to store PDF preview
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false); // Loading state
  const chatBottomRef = useRef(null); // Reference to the bottom of the chat area

  useEffect(() => {
    // Scroll to the bottom of the chat area when chatHistory changes
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setQuestion("");
    try {
      // Add the user's question to chat history
      setChatHistory((prevChat) => [
        ...prevChat,
        { question: question, response: "", loading: true }, // Include loading state in chat history
      ]);

      // Set loading state to true while waiting for response
      setIsLoading(true);

      // Create FormData object
      const formData = new FormData();
      formData.append("question", question);
      formData.append("model_name", model);

      // Add image file to FormData if available and model is gemini-vision
      if (model === 'gemini-vision' && image) {
        formData.append('image', image);
        setImagePreview(URL.createObjectURL(image));
        const uploadResponse = await axios.post('http://127.0.0.1:8000/upload_image/', formData);
        console.log(uploadResponse.data);
      }

      // Add PDF file to FormData if available and model is pdf-gpt
      if (model === 'pdf-gpt' && pdf) {
        formData.append('pdf', pdf);
        setPdfPreview(URL.createObjectURL(pdf));
        const uploadResponse = await axios.post('http://127.0.0.1:8000/upload_pdf/', formData);
        console.log(uploadResponse.data);
      }

      // Make API call to get response
      const generateResponse = await axios.get(
        `http://127.0.0.1:8000/generate_text_gemini/?model_name=${model}&question=${question}`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      const newResponse = generateResponse.data["AI Response"];

      // Update chat history with the response
      setChatHistory((prevChat) => {
        const updatedChat = [...prevChat];
        const lastChatIndex = updatedChat.length - 1;
        updatedChat[lastChatIndex].response = newResponse;
        updatedChat[lastChatIndex].loading = false; // Set loading state to false after response is received
        return updatedChat;
      });

      // Set response and error states
      setResponse(newResponse);
      setError("");

      // Set loading state to false after response is received
      setIsLoading(false);
    } catch (error) {
      console.error("Error:", error);
      if (error.response) {
        setError(error.response.data.detail);
      } else {
        setError(
          "An error occurred while processing your request. Please try again later."
        );
      }

      // Set loading state to false if there's an error
      setIsLoading(false);
    }
  };

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handlePdfChange = (e) => {
    const file = e.target.files[0];
    setPdf(file);
    // Set PDF preview
    setPdfPreview(URL.createObjectURL(file));
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
              type="text"
              value={question}
              className="input"
              sx={{
                "& input": {
                  color: "white",
                },
                "& .MuiInput-underline:before": {
                  borderBottom: "none",
                },
                "& .MuiInput-underline:after": {
                  borderBottom: "none",
                },
              }}
              style={{
                width: "700px",
                backgroundColor: "#2d2d2d",
                paddingLeft: "30px",
                height: "40px",
                marginTop: "0px",
                borderRadius: "12px",
                color: "white",
                border: "0px",
              }}
              onChange={(e) => setQuestion(e.target.value)}
              required
              placeholder="Enter your question..."
            />

            {/* Submit button */}
            <Button
              className="send"
              type="submit"
              variant="contained"
              color="primary"
              style={{
                left: "-80px",
                backgroundColor: "transparent",
                border: "none",
                width: "1px",
                boxShadow: "none",
              }}
            >
              <KeyboardDoubleArrowUpSharpIcon />
            </Button>
          </form>
        </div>
      )}
      {pdfPreview && (
        <div>
          <h2>Uploaded PDF:</h2>
          <embed src={pdfPreview} type="application/pdf" width="50%" height="400px" />
        </div>
      )}
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
