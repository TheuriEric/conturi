const chatButton = document.getElementById("chatButton")
const chatModal = document.getElementById("chatModal")
const closeChat = document.getElementById("closeChat")
const chatInput = document.getElementById("chatInput")
const sendButton = document.getElementById("sendButton")
const chatMessages = document.getElementById("chatMessages")
const careerForm = document.getElementById("careerForm")


// Open chat modal
chatButton.addEventListener("click", () => {
  chatModal.classList.add("active")
  chatInput.focus()
})

// Close chat modal
closeChat.addEventListener("click", () => {
  chatModal.classList.remove("active")
})

// Close modal when clicking outside
chatModal.addEventListener("click", (e) => {
  if (e.target === chatModal) {
    chatModal.classList.remove("active")
  }
})

// Send message
async function sendMessage() {
  const message = chatInput.value.trim()
  if (message === "") return

  // Add user message to chat
  const userMessageDiv = document.createElement("div")
  userMessageDiv.className = "message user"
  userMessageDiv.innerHTML = `<div class="message-content">${escapeHtml(message)}</div>`
  chatMessages.appendChild(userMessageDiv)

  chatInput.value = ""
  chatMessages.scrollTop = chatMessages.scrollHeight

  // Add "typing" indicator
  const typingDiv = document.createElement("div")
  typingDiv.className = "message bot"
  typingDiv.innerHTML = `<div class="message-content">...</div>`
  chatMessages.appendChild(typingDiv)
  chatMessages.scrollTop = chatMessages.scrollHeight

  try {
  const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query: message })
  });

  if (!response.ok) throw new Error("Network error");

  const rawResponseText = await response.text();
  console.log("=== RAW CREWAI RESPONSE ===");
  console.log("Raw response text:", rawResponseText);
  console.log("Raw response length:", rawResponseText.length);
  console.log("=== END RAW RESPONSE ===");

  const data = JSON.parse(rawResponseText);
  
  console.log("=== DEBUG: Checking response structure ===");
  console.log("Full parsed data object:", data);
  console.log("data.response exists:", !!data.response, "Value:", data.response);
  console.log("data.answer exists:", !!data.answer, "Value:", data.answer);
  console.log("data.output exists:", !!data.output, "Value:", data.output);
  
  if (data.output) {
    console.log("data.output is an object:", typeof data.output === 'object');
    console.log("data.output.display exists:", !!data.output.display);
    console.log("data.output.display type:", typeof data.output.display);
    console.log("data.output.display value:", data.output.display);
    
    console.log("Full data.output structure:", JSON.stringify(data.output, null, 2));
  } else {
    console.log("data.output is undefined or null");
  }
  console.log("=== END DEBUG ===");

  let botMessage;
  
  if (data.response) {
    botMessage = data.response;
    console.log("✅ Using general chat: data.response");
  }
  else if (data.answer) {
    botMessage = data.answer;
    console.log("✅ Using alternative chat: data.answer");
  }
  else if (data.output && data.output.display) {
    botMessage = data.output.display;
    console.log("✅ Using CrewAI: data.output.display");
  }
  else {
    botMessage = "No response available.";
    console.log("❌ No valid response format found");
  }

  console.log("Final bot message:", botMessage);

  const safeText = escapeHtml(botMessage);
  const formattedMessage = safeText.replace(/\n/g, '<br>');
  typingDiv.innerHTML = `<div class="message-content">${formattedMessage}</div>`
  
} catch (error) {
  console.error("Full error:", error);
  typingDiv.innerHTML = `<div class="message-content error">⚠️ Error: ${escapeHtml(error.message)}</div>`
}


  chatMessages.scrollTop = chatMessages.scrollHeight
}

sendButton.addEventListener("click", sendMessage)
chatInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    sendMessage()
  }
})

// Escape HTML to prevent XSS
function escapeHtml(text) {
  const div = document.createElement("div")
  div.textContent = text
  return div.innerHTML
}

// CTA buttons
document.querySelectorAll(".cta-button, .hero .btn-primary").forEach((button) => {
  button.addEventListener("click", () => {
    chatModal.classList.add("active")
    chatInput.focus()
  })
})

if (careerForm) {
  careerForm.addEventListener("submit", async (e) => {
    e.preventDefault()

    const firstName = document.getElementById("firstName").value
    const lastName = document.getElementById("lastName").value
    const email = document.getElementById("email").value
    const prompt = document.getElementById("prompt").value
    const emailFrequency = document.getElementById("emailFrequency").value

    console.log("Form submitted:", {
      firstName,
      lastName,
      email,
      prompt,
      emailFrequency,
    })

    try {
      // Show loading state
      const submitButton = careerForm.querySelector('button[type="submit"]')
      const originalText = submitButton.textContent
      submitButton.textContent = "Setting up your automation..."
      submitButton.disabled = true

      // Send to your FastAPI endpoint
      const response = await fetch("http://localhost:8000/automation", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          first_name: firstName,
          last_name: lastName,
          email: email,
          schedule: emailFrequency,
          prompt: prompt
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      
      console.log("Automation setup successful:", result)
      
      // Show success message
      alert(`✅ Perfect, ${firstName}! Your Synq Auto is now active. You'll receive ${emailFrequency} insights from Synq`)

      // Reset form
      careerForm.reset()
      
    } catch (error) {
      console.error("Automation setup failed:", error)
      alert(`❌ Sorry, we couldn't set up your automation right now. Please try again. Error: ${error.message}`)
    } finally {
      // Reset button state
      const submitButton = careerForm.querySelector('button[type="submit"]')
      submitButton.textContent = "Generate My Insights"
      submitButton.disabled = false
    }
  })
}
