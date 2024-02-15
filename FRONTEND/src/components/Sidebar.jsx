import React, { useState } from "react"; // Import useState
import "./Sidebar.css"; // Import the CSS file for styling
import AutoModeRoundedIcon from '@mui/icons-material/AutoModeRounded';
import AddCircleOutlineSharpIcon from '@mui/icons-material/AddCircleOutlineSharp';

// const [imagePreview, setImagePreview] = useState('');
// const [pdfPreview, setPdfPreview] = useState('');

function Sidebar({ imagePreview, pdfPreview }) { // Receive both imagePreview and pdfPreview props

  function refreshPage() {
    window.location.reload();
}

  return (
    <div className="sidebar">
      {/* Sidebar content */}
      <div className="sidebar-content">
        <h1 href = "https://synapse-7fnqast6r-kaushikdash04s-projects.vercel.app">SYNAPSE     <AutoModeRoundedIcon/></h1>
         {/* Add any additional sidebar content here */}
         <button style={{width:"300px", borderRadius:"12px",display:"flex",marginTop:"20px"}}
         onClick={refreshPage}>
          <div className="new" style={{display:"flex", gap:"5px"}}><div style={{marginTop:"13px"}}><AddCircleOutlineSharpIcon/></div><h3>New Chat</h3></div></button>
        {imagePreview && (
          <div className="image">
{/*             <h2 color="white">Uploaded Image:</h2> */}
            <img src={imagePreview} alt="Uploaded" style={{ maxWidth: '100%', maxHeight: '200px' }} />
          </div>
        )}
        {pdfPreview && (
          <div>
{/*             <h2 color="white">Uploaded PDF:</h2> */}
            <embed src={pdfPreview} type="application/pdf" width="100%" height="400px" />
          </div>
        )}
      </div>
      <h3 className="developer">Designed and devloped by<br/>Kaushik and Hardeep.</h3>
    </div>
  );
}

export default Sidebar;
