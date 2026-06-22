// Add context menu when right-clicking highlighted text
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "checkFakeNews",
    title: "Check with Satyameva Detector",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "checkFakeNews" && info.selectionText) {
    console.log("Selected text to check:", info.selectionText);
    
    fetch("https://sruti2006-fake-news-detector-api.hf.space/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: info.selectionText })
    })
    .then(r => r.json())
    .then(data => {
      const confidence = Math.round(data.confidence * 100);
      
      // Send a notification with the prediction
      chrome.notifications.create({
        type: "basic",
        iconUrl: "icon.png", 
        title: `Satyameva Result: ${data.prediction}`,
        message: `Prediction: ${data.prediction} (${confidence}% confidence) via ${data.model_used}`,
        priority: 2
      });
    })
    .catch(err => {
      console.error("Inference call failed:", err);
    });
  }
});
