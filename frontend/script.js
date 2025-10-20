const chatButton = document.getElementById("chatButton")
const chatModal = document.getElementById("chatModal")
const closeChat = document.getElementById("closeChat")
const chatInput = document.getElementById("chatInput")
const sendButton = document.getElementById("sendButton")
const chatMessages = document.getElementById("chatMessages")

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
    // 🔗 Call backend
    const response = await fetch("http://localhost:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Add JWT if using auth
        // "Authorization": `Bearer ${localStorage.getItem("token")}`
      },
      body: JSON.stringify({ query: message })
    })

    if (!response.ok) throw new Error("Network error")

    const data = await response.json()

    // Replace typing indicator with actual response
    typingDiv.innerHTML = `<div class="message-content">${escapeHtml(data.answer || data.response || "No response received.")}</div>`
  } catch (error) {
    typingDiv.innerHTML = `<div class="message-content error">⚠️ Error: ${error.message}</div>`
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
document.querySelectorAll(".btn-primary, .cta-button").forEach((button) => {
  button.addEventListener("click", () => {
    chatModal.classList.add("active")
    chatInput.focus()
  })
})
