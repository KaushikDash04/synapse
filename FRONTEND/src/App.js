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
        const uploadResponse = await axios.post('https://synapse-backend-urh6.onrender.com/upload_image/', formData);
        console.log(uploadResponse.data);
      }

      // Add PDF file to FormData if available and model is pdf-gpt
      if (model === 'pdf-gpt' && pdf) {
        formData.append('pdf', pdf);
        setPdfPreview(URL.createObjectURL(pdf));
        const uploadResponse = await axios.post('https://synapse-backend-urh6.onrender.com/upload_pdf/', formData);
        console.log(uploadResponse.data);
      }

      // Make API call to get response
      const generateResponse = await axios.get(
        `https://synapse-backend-urh6.onrender.com/generate_text_gemini/?model_name=${model}&question=${question}`,
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
    <>
      <div
        className="app-container"
        style={{ display: "flex", opacity: "50%" }}
      >
        {/* Sidebar component */}
        <Sidebar imagePreview={imagePreview} pdfPreview={pdfPreview} />

        {/* Chatbot container */}
        <div className="chatbot-container">
          <div className="chatbot-header"></div>
          <div
            className="chatbot-chat-area"
            style={{
              backgroundColor: "#171717",
              color: "#a7bfe8",
              marginLeft: "320px",
              padding: "50px",
              maxHeight: "550px",
              overflowY: "auto",
              width: "1092px",
            }}
          >
            {chatHistory.map((chat, index) => (
              <div key={index} className="chatbot-message">
                <p
                  style={{
                    textAlign: "right",
                    padding: "10px",
                    color: "violet",
                  }}
                >
                  <PsychologyAltRoundedIcon sx={{ fontSize: 40, mt: "20px" }} />
                  <strong>USER:</strong> <br />
                  {chat.question}
                </p>
                <p className="synapse-message">
                  <PsychologyRoundedIcon sx={{ fontSize: 40 }} />
                  <strong>SYNAPSE:</strong> <br />
                  {chat.loading ? (
                    <TypingAnimation />
                  ) : (
                    <TypewriterAnimation text={chat.response} />
                  )}
                </p>
              </div>
            ))}

            {/* Empty div to keep track of the bottom of the chat area */}
            <div ref={chatBottomRef}></div>
          </div>

          {/* Chat input form */}
          <form
            onSubmit={handleSubmit}
            className="chatbot-input-form"
            style={{
              backgroundColor: "transparent",
              position: "fixed",
              top: "-20px",
              left: "430px",
              width: "2000px",
              gap: "5px",
            }}
          >
            {/* Upload section */}
            <div className="upload-section">
              {(model === "gemini-vision" || model === "pdf-gpt") && (
                <>
                  <label
                    htmlFor={model === "gemini-vision" ? "image" : "pdf"}
                    className="upload-button"
                    style={{ backgroundColor: "#2d2d2d", borderRadius: "12px" }}
                  >
                    <UploadFileIcon sx={{ mt: "10px", bgcolor: "#2d2d2d" }} />
                  </label>
                  <input
                    type="file"
                    id={model === "gemini-vision" ? "image" : "pdf"}
                    accept={model === "gemini-vision" ? "image/*" : ".pdf"}
                    onChange={
                      model === "gemini-vision"
                        ? handleImageChange
                        : handlePdfChange
                    }
                    required
                    className="file-input"
                  />
                </>
              )}
            </div>

            {/* Model select form */}
            <div className="model-select-form">
              <label
                htmlFor="model-select"
                className="model-select-label"
              ></label>
              <select
                id="model-select"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="model-select"
                style={{
                  marginTop: "20px",
                  backgroundColor: "#2d2d2d",
                  border: "none",
                  borderRadius: "12px",
                  width: "150px",
                  color: "white",
                }}
              >
                <option value="gemini-pro">Gemini Pro</option>
                <option value="gemini-vision">Gemini Vision</option>
                <option value="pdf-gpt">PDF GPT</option>
              </select>
            </div>

            {/* Input field */}
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
      </div>
      <h5
        className="foot"
        style={{
          position: "fixed",
          color: "gray",
          marginLeft: "740px",
          marginTop: "570px",
        }}
      >
        Synapse can make mistakes. Consider checking important information.
      </h5>
      {/* Error message for mobile devices */}
      <div className="error-message">
        Sorry, this website is not compatible with mobile devices yet. Please view it on a desktop or laptop.
      </div>
    </>
  );
}

export default App;
