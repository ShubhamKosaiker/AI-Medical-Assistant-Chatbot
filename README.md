# 🩺 AI Doctor Assistant

A multimodal AI-powered medical assistant that accepts **voice input + medical images**
and responds with a spoken diagnosis — like a real doctor consultation.

---

## 🎥 Live Demo
👉 **[Try it live on HuggingFace](https://huggingface.co/spaces/ShubhamKosaiker/AI-Medical-Assistant)**

---

## 🧠 How It Works

1. 🎤 User speaks their symptoms via microphone
2. 📝 Groq Whisper transcribes speech to text
3. 🖼️ LLaMA 4 Scout analyzes the text + uploaded image together
4. 🔊 gTTS converts the diagnosis to speech
5. 🌐 Gradio serves the full UI in the browser

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Speech to Text | Groq Whisper large-v3 |
| Vision + LLM | Meta LLaMA 4 Scout 17B |
| Text to Speech | gTTS |
| UI Framework | Gradio |
| Language | Python |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```
git clone https://github.com/ShubhamKosaiker/AI-Medical-Assistant-Chatbot.git
cd AI-Medical-Assistant-Chatbot
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Add your API keys
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

### 4. Run the app
```
python app.py
```

---

## 📁 Project Structure

```
AI_Doctor_Assistant/
├── app.py                  # Main UI application
├── brain_of_the_doctor.py  # LLM + vision logic
├── voice_of_the_patient.py # Speech to text (Groq Whisper)
├── voice_of_the_doctor.py  # Text to speech (gTTS)
├── .env                    # API keys (not uploaded)
├── .gitignore
└── README.md
```

---

## ⚠️ Disclaimer
This project is for **educational purposes only** and is not a substitute
for professional medical advice, diagnosis, or treatment.

---

## 👨‍💻 Built By
**Shubham Kosaiker** — [LinkedIn](https://www.linkedin.com/in/shubham-kosaiker-31054321b) · [GitHub](https://github.com/ShubhamKosaiker)
