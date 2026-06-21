document.getElementById("checkBtn").addEventListener("click", async () => {
  const text = document.getElementById("textInput").value.trim();
  const resultDiv = document.getElementById("result");
  
  if (!text) {
    alert("Please paste some text first.");
    return;
  }
  
  resultDiv.style.display = "block";
  resultDiv.innerText = "Analyzing...";
  resultDiv.className = "";
  
  try {
    const response = await fetch("https://sruti2006-satyameva-backend.hf.space/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    
    const data = await response.json();
    const confidencePct = Math.round(data.confidence * 100);
    
    resultDiv.innerText = `${data.prediction} (${confidencePct}% confidence)`;
    resultDiv.className = data.prediction.toLowerCase() === "real" ? "real" : "fake";
  } catch (err) {
    resultDiv.innerText = "Error contacting API.";
    resultDiv.style.color = "red";
  }
});
