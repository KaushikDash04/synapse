import React, { useState, useEffect } from "react";

const TypewriterAnimation = ({ text }) => {
  const [displayText, setDisplayText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const intervalId = setInterval(() => {
      if (currentIndex < text.length) {
        setDisplayText((prevText) => prevText + text[currentIndex]);
        setCurrentIndex((prevIndex) => prevIndex + 1);
      } else {
        clearInterval(intervalId);
      }
    }, 1); // Adjust speed here (milliseconds per character)
    
    return () => {
      clearInterval(intervalId);
    };
  }, [currentIndex, text]);

  return <span>{displayText}</span>;
};

export default TypewriterAnimation;
