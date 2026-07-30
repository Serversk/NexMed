document.addEventListener('DOMContentLoaded', () => {

    // --- Theme Toggle Logic (Works on all pages) ---
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        const currentTheme = localStorage.getItem('theme');

        if (currentTheme) {
            document.body.setAttribute('data-theme', currentTheme);
            if (currentTheme === 'dark') {
                themeToggle.checked = true;
            }
        }

        themeToggle.addEventListener('change', () => {
            if (themeToggle.checked) {
                document.body.setAttribute('data-theme', 'dark');
                localStorage.setItem('theme', 'dark');
            } else {
                document.body.setAttribute('data-theme', 'light');
                localStorage.setItem('theme', 'light');
            }
        });
    }
    
    // --- Image Upload Logic (Only runs if uploader exists) ---
    const imageUploader = document.getElementById('image-uploader');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const uploadContent = document.querySelector('.upload-content');

    // Check if the image uploader element is on the page
    if (imageUploader) {
        fileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                displayImage(file);
            }
        });

        imageUploader.addEventListener('dragover', (event) => {
            event.preventDefault();
            imageUploader.classList.add('dragover');
        });

        imageUploader.addEventListener('dragleave', () => {
            imageUploader.classList.remove('dragover');
        });

        imageUploader.addEventListener('drop', (event) => {
            event.preventDefault();
            imageUploader.classList.remove('dragover');
            const file = event.dataTransfer.files[0];
            if (file) {
                displayImage(file);
            }
        });
// ADD THESE TWO FUNCTIONS

// This is the updated displayImage function
function displayImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const imagePreview = document.getElementById('image-preview');
        const uploadContent = document.querySelector('.upload-content');
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        uploadContent.style.display = 'none';
        imageUploader.style.border = 'none';
        imageUploader.style.backgroundImage = 'none';

        // This new line sends the image to your backend
        sendImageToBackend(file); 
    };
    reader.readAsDataURL(file);
}

// This new function handles the backend communication
function sendImageToBackend(file) {
    const chatBody = document.getElementById('chat-body');
    const formData = new FormData();
    formData.append('image', file); // <-- key must be 'image'

    const processingMessage = document.createElement('div');
    processingMessage.classList.add('chat-message', 'ai');
    processingMessage.textContent = 'Processing your image...';
    chatBody.appendChild(processingMessage);
    chatBody.scrollTop = chatBody.scrollHeight;

    fetch('http://127.0.0.1:5000/detect', { // <-- endpoint must be '/detect'
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        chatBody.removeChild(processingMessage);
        
        const aiMessage = document.createElement('div');
        aiMessage.classList.add('chat-message', 'ai');
        console.log('Response from backend:', data);
        if (data.analysis) { // <-- expects 'analysis' key
            aiMessage.textContent = data.analysis;
        } else if (data.prediction) { // fallback for old backend
            aiMessage.textContent = data.prediction;
        } else {
            aiMessage.textContent = 'An error occurred during analysis. Please try again.';
            console.error('Error from backend:', data.error);
        }
        
        chatBody.appendChild(aiMessage);
        chatBody.scrollTop = chatBody.scrollHeight;
    })
    .catch(error => {
        chatBody.removeChild(processingMessage);
        console.error('Error sending image to backend:', error);

        const errorMessage = document.createElement('div');
        errorMessage.classList.add('chat-message', 'ai');
        errorMessage.textContent = error;
        chatBody.appendChild(errorMessage);
        chatBody.scrollTop = chatBody.scrollHeight;
    });
}
    }

    // --- Chat Logic (Only runs if chat exists) ---
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const chatBody = document.getElementById('chat-body');

    // Check if the chat input element is on the page
    if (chatInput) {
        const sendMessage = () => {
            const messageText = chatInput.value.trim();
            if (messageText === '') return;

            const userMessage = document.createElement('div');
            userMessage.classList.add('chat-message', 'user');
            userMessage.textContent = messageText;
            chatBody.appendChild(userMessage);

            chatInput.value = '';
            chatBody.scrollTop = chatBody.scrollHeight;

            // AI placeholder response
            setTimeout(() => {
                const aiMessage = document.createElement('div');
                aiMessage.classList.add('chat-message', 'ai');
                aiMessage.textContent = "To use the model upload an image and your diagnosis will be provided within seconds";
                chatBody.appendChild(aiMessage);
                chatBody.scrollTop = chatBody.scrollHeight;
            }, 1000);
        };
        
        sendButton.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
});