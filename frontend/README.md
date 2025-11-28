# MentorFlow v0.8 — AI PM Learning Space  
Interactive lesson engine · Podcast-style AI lectures · RAG document search

MentorFlow is an interactive learning environment designed to help users learn **AI Product Management** through a combination of structured lessons, narrated lectures, role-play conversations, and retrieval-augmented question-answering.

This repository contains the **frontend** (React + TypeScript + Vite).  
The backend is a separate FastAPI service that provides LLM responses, audio synthesis, RAG search, and lesson/lecture logic.

---

# 🌟 Features (v0.8)

### 🎧 Podcast-Style Lecture Mode (NEW in v0.8)
- Generates 230–320 word spoken-narrative lecture scripts  
- Splits them into multi-part segments (Part 1 / Part 2 / Part 3 / …)  
- Uses **ElevenLabs TTS** to produce natural mp3 audio  
- Auto-plays audio in the frontend  
- Commands:
  - `start lecture 1`
  - `next`
  - `stop lesson`

### 🎓 Lesson Flow (Interactive Learning)
- Structured multi-unit lessons  
- Includes Key Idea → Concept → Check Your Understanding  
- Auto-unlocks next chapters  
- Lesson state tracked per user  

### 🎭 Role-Play Mode
- Scenario-based AI PM role-playing  
- Multiple conversational turns  
- Final evaluation & scoring  

### 📄 RAG Mode (from v0.7)
- Upload TXT / PDF documents via **Admin** tab  
- Query against embedded knowledge chunks  
- Provides grounded answers with citations  

---

# 🛠 Development Setup

Frontend uses **React + TypeScript + Vite**.

## Install dependencies
```bash
npm install
Run local dev server
bash
Copy code
npm run dev
Build for production
bash
Copy code
npm run build
Preview production build
bash
Copy code
npm run preview
🔗 Backend API Configuration
MentorFlow frontend communicates with the backend at:

cpp
Copy code
http://127.0.0.1:8000
To change the backend URL:

javascript
Copy code
src/App.tsx → function useBackendBaseUrl()
Replace with your deployed backend URL:

ts
Copy code
function useBackendBaseUrl() {
  return "https://your-backend-url.com";
}
🔧 Required Environment Variables (Backend)
Set the following variables in your backend hosting platform (Railway / Render / Fly.io):

Key	Required	Description
OPENAI_API_KEY	Yes	OpenAI model inference
ELEVENLABS_API_KEY	Yes	Generating mp3 audio
ELEVENLABS_VOICE_ID	Yes	ElevenLabs voice ID (e.g. 21m00Tcm4TlvDq8ikWAM)

If ELEVENLABS_API_KEY is missing, backend will fallback to browser TTS.

📤 Deployment Guide
🌐 Frontend Deployment (Vercel / Netlify)
Run npm run build

Deploy the dist/ folder or connect your GitHub repo

Ensure backend URL is configured correctly in App.tsx

No environment variables needed for frontend

🚀 Backend Deployment (Railway / Render)
Deploy the FastAPI backend repo

Set required environment variables

Expose port 8000

Confirm logs show:

csharp
Copy code
[CFG] ELEVENLABS_API_KEY set: True
[CFG] ELEVENLABS_VOICE_ID: ...
🧪 Commands You Can Use (Learner Tab)
Command	Description
start lecture 1	Begin podcast-style lecture
start lesson 1	Guided, structured lesson
start roleplay	AI PM scenario simulation
next	Continue lecture
stop lesson	Exit lecture/lesson mode

🎨 UI Specifications (v0.8)
White card UI with soft shadows

Slate text hierarchy

Two-column layout (chat left, controls right)

MF brand avatar in header

Auto Read / RAG toggles

Clean system-style chat bubbles

ElevenLabs audio playback integrated

📁 Project Structure
bash
Copy code
src/
├── components/        # UI components (shadcn/ui)
├── lib/               # utilities (cn, helpers)
├── assets/            # icons / images
├── App.tsx            # main UI logic + chat + lecture mode + audio
├── main.tsx           # entry point
└── index.css          # global styles
Backend is separate and must be deployed independently.

📚 Backend Responsibilities (not included in this repo)
Lesson engine

Lecture generator

RAG document processing

User session tracking

ElevenLabs mp3 synthesis

Unlock progression system

❤️ Credits
MentorFlow v0.8
Built as an AI-powered learning engine for AI Product Management.
Frontend: React + TypeScript + Vite
Backend: FastAPI
Audio: ElevenLabs

📜 License
MIT License
Feel free to fork, modify, and build upon MentorFlow.