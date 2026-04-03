const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const chatMessages = document.getElementById('chatMessages');
const newChatBtn = document.getElementById('newChatBtn');

newChatBtn.addEventListener('click', () => {
    // Simply reload the page to restart the session fully
    window.location.reload();
});

// We hold state on the frontend for simplicity!
let currentState = 'initial'; 
let currentMatchedSymptoms = [];

function addMessage(text, sender, isHTML = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${sender}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = sender === 'bot' ? 'AI' : 'You';
    
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    
    if (isHTML) {
        bubble.innerHTML = text;
    } else {
        bubble.textContent = text;
    }
    
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatMessages.appendChild(msgDiv);
    
    // Auto scroll
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return bubble;
}

function showTypingIndicator() {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message bot-message typing`;
    msgDiv.id = 'typingIndicator';
    msgDiv.innerHTML = `
        <div class="avatar">AI</div>
        <div class="bubble">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const t = document.getElementById('typingIndicator');
    if(t) t.remove();
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if(!text) return;
    
    addMessage(text, 'user');
    userInput.value = '';
    userInput.disabled = true;
    
    showTypingIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: 'initial',
                message: text
            })
        });
        
        const data = await response.json();
        removeTypingIndicator();
        
        if (data.status === 'error') {
            addMessage(data.message, 'bot');
            userInput.disabled = false;
            userInput.focus();
            return;
        }
        
        // Success! Progress state
        currentState = 'co_occurring';
        currentMatchedSymptoms = data.matched;
        
        const bubble = addMessage(data.message, 'bot', true);
        
        // Render suggestion chips
        if (data.suggestions && data.suggestions.length > 0) {
            const container = document.createElement('div');
            container.className = 'suggestions-container';
            
            const selectedSet = new Set();
            
            data.suggestions.forEach(sym => {
                const chip = document.createElement('div');
                chip.className = 'suggestion-chip';
                chip.textContent = sym;
                chip.onclick = () => {
                    chip.classList.toggle('selected');
                    if (selectedSet.has(sym)) selectedSet.delete(sym);
                    else selectedSet.add(sym);
                };
                container.appendChild(chip);
            });
            
            const actRow = document.createElement('div');
            actRow.className = 'action-row';
            
            const confirmBtn = document.createElement('button');
            confirmBtn.className = 'action-btn';
            confirmBtn.textContent = 'Submit Symptoms & Predict Diseases';
            confirmBtn.onclick = () => {
                const newSyms = Array.from(selectedSet);
                submitFinalPrediction(newSyms);
                // hide UI
                container.style.opacity = '0.5';
                container.style.pointerEvents = 'none';
            };
            
            const noneBtn = document.createElement('button');
            noneBtn.className = 'action-btn';
            noneBtn.style.background = 'rgba(255,255,255,0.1)';
            noneBtn.textContent = 'I don\'t have any of these';
            noneBtn.onclick = () => {
                submitFinalPrediction([]);
                container.style.opacity = '0.5';
                container.style.pointerEvents = 'none';
            };
            
            actRow.appendChild(confirmBtn);
            actRow.appendChild(noneBtn);
            container.appendChild(actRow);
            
            bubble.appendChild(container);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } else {
            // No suggestions found, immediately predict
            submitFinalPrediction([]);
        }
        
    } catch (e) {
        removeTypingIndicator();
        addMessage("Server communication error.", 'bot');
        userInput.disabled = false;
    }
});

async function submitFinalPrediction(addedSymptoms) {
    // combine initial + user clicked
    currentMatchedSymptoms = currentMatchedSymptoms.concat(addedSymptoms);
    
    if (addedSymptoms.length > 0) {
        addMessage(`I also have: ${addedSymptoms.join(', ')}`, 'user');
    } else {
        addMessage(`I don't have any of those.`, 'user');
    }
    
    showTypingIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: 'predict',
                symptoms: currentMatchedSymptoms
            })
        });
        
        const data = await response.json();
        removeTypingIndicator();
        
        if (data.status === 'success') {
            addMessage(data.message, 'bot', true);
            // Reset for a new chat
            currentState = 'initial';
            currentMatchedSymptoms = [];
            userInput.disabled = false;
            userInput.focus();
        } else {
            addMessage(data.message, 'bot');
        }
    } catch (e) {
        removeTypingIndicator();
        addMessage("Failed to predict diseases.", 'bot');
    }
}
