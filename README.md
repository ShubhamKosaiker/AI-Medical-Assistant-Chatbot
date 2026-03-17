# 🩺 AI Medical Assistant Chatbot

A multimodal AI-powered medical assistant that accepts **voice input + medical images** 
and responds with a spoken diagnosis — like a real doctor consultation.

---

## 🎥 Demo
> Record a short screen recording and add the link or GIF here

---

## 🧠 How It Works

1. 🎤 User speaks their symptoms via microphone
2. 📝 Groq Whisper transcribes speech to text
3. 🖼️ LLaMA 4 Scout analyzes the text + uploaded image together
4. 🔊 ElevenLabs converts the diagnosis to natural speech
5. 🌐 Gradio serves the full UI in the browser

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Speech to Text | Groq Whisper large-v3 |
| Vision + LLM | Meta LLaMA 4 Scout 17B |
| Text to Speech | ElevenLabs |
| UI Framework | Gradio |
| Language | Python |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/ShubhamKosaiker/AI-Medical-Assistant-Chatbot.git
cd AI-Medical-Assistant-Chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API keys
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key
ELEVENLABS_API_KEYS=your_elevenlabs_api_key
```

### 4. Run the app
```bash
python gradio_app.py
```

Then open **http://127.0.0.1:7860** in your browser.

---

## 📁 Project Structure
```
AI_Medical_Assistant_Chatbot/
├── gradio_app.py          # Main UI application
├── brain_of_the_doctor.py # LLM + vision logic
├── voice_of_the_patient.py# Speech to text (Groq Whisper)
├── voice_of_the_doctor.py # Text to speech (ElevenLabs + gTTS)
├── .env                   # API keys (not uploaded)
├── .gitignore
└── README.md
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and is not a substitute 
for professional medical advice, diagnosis, or treatment.

---

## 👨‍💻 Built By

**Shubham Kosaiker** — [LinkedIn](www.linkedin.com/in/shubham-kosaiker-31054321b) · [GitHub](https://github.com/ShubhamKosaiker)