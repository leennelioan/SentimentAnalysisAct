// Function to predict sentiment
function predictSentiment() {
    const review = document.getElementById("review").value;
    const resultDiv = document.getElementById("result");
  
    // Check if review is empty
    if (!review.trim()) {
      resultDiv.textContent = "Please enter a review first.";
      return;
    }
  
    fetch("http://127.0.0.1:4123/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review: review }),
      mode: "cors",
    })
      .then((response) => response.json())
      .then((data) => {
        const sentiment = data.sentiment;
        // Get the accuracy from the response
        const accuracy = data.accuracy || 0;
  
        // Format the accuracy as a percentage
        const accuracyPercentage = (accuracy * 100).toFixed(2);
  
        // Create HTML for the result with sentiment and accuracy
        resultDiv.innerHTML = `
          <div class="sentiment-result">Predicted Sentiment: ${sentiment}</div>
          <div class="accuracy-container">
              <div class="accuracy-label">Confidence:</div>
              <div class="accuracy-bar-container">
                  <div class="accuracy-bar" style="width: ${accuracyPercentage}%"></div>
              </div>
              <div class="accuracy-percentage">${accuracyPercentage}%</div>
          </div>
        `;
  
        resultDiv.className = sentiment.toLowerCase();
      })
      .catch((error) => {
        console.error("Error:", error);
        resultDiv.textContent = "Error during prediction. Please try again.";
      });
  }
  
  // Add event listener for the Enter key
  document.addEventListener("DOMContentLoaded", function() {
    const reviewTextarea = document.getElementById("review");
    
    reviewTextarea.addEventListener("keydown", function(event) {
      // Check if the key pressed is Enter (key code 13)
      if (event.key === "Enter" && !event.shiftKey) {
        // Prevent the default action (new line in textarea)
        event.preventDefault();
        
        // Call the predictSentiment function
        predictSentiment();
      }
    });
  });