# voicebox - Current State Overview

**Last Updated:** January 25, 2026  
**Status:** ✅ MVP Core Features Working - Voice generation from Tauri app successful!

---

## 🎯 What We Have

### ✅ **Fully Implemented & Working**

#### **Backend (Python FastAPI)**
- **Voice Profile Management**
  - Create, read, update, delete profiles
  - Add multiple audio samples per profile
  - Multi-reference voice combination (combines multiple samples)
  - Profile storage in SQLite + file system (`data/profiles/`)
  
- **Voice Generation**
  - Qwen3-TTS model integration (1.7B and 0.6B support)
  - Automatic model downloading from HuggingFace Hub
  - Voice prompt caching for instant re-generation
  - Support for English and Chinese
  - Seed-based reproducibility
  - GPU/CPU/MPS device detection
  
- **Generation History**
  - Full CRUD operations
  - Search by text content
  - Filter by profile
  - Pagination support
  - Statistics endpoint
  - Audio file storage (`data/generations/`)
  
- **Audio Transcription**
  - Whisper integration for speech-to-text
  - Language detection/selection
  - Used for reference text extraction from samples
  
- **Database**
  - SQLite with SQLAlchemy ORM
  - Tables: `profiles`, `profile_samples`, `generations`, `projects` (ready for future)
  - Automatic schema initialization
  
- **API Endpoints**
  - RESTful API with FastAPI
  - OpenAPI schema generation
  - CORS enabled
  - Health check endpoint
  - File serving for audio files

#### **Frontend (React + TypeScript + Tauri)**
- **Voice Profile UI**
  - Profile list with cards
  - Create/edit profile dialog
  - Upload audio samples with transcription
  - Sample management (view/delete)
  - Profile detail view
  
- **Generation UI**
  - Form with profile selection
  - Text input (up to 5000 chars)
  - Language selection (en/zh)
  - Optional seed input
  - Loading states and error handling
  
- **History UI**
  - Table view with pagination
  - Search functionality
  - Play audio inline
  - Download audio files
  - Delete generations
  
- **Server Settings**
  - Connection form (local/remote mode)
  - Server status display
  - Health check integration
  
- **State Management**
  - React Query for server state
  - Zustand for client state (server URL, connection status)
  - Type-safe API client
  
- **UI Components**
  - shadcn/ui component library
  - Tailwind CSS styling
  - Responsive design
  - Toast notifications
  - Form validation with Zod

#### **Tauri Desktop App**
- **Rust Backend**
  - Sidecar management for Python server
  - Start/stop server commands
  - Remote mode support (0.0.0.0 binding)
  - Process lifecycle management
  
- **Build System**
  - Tauri v2 configuration
  - Platform-specific builds
  - Dev tools in debug mode

---

## 🏗️ Architecture

### **Project Structure**
```
voicebox/
├── app/                    # Shared React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   │   ├── VoiceProfiles/  ✅ Complete
│   │   │   ├── Generation/     ✅ Complete
│   │   │   ├── History/        ✅ Complete
│   │   │   ├── ServerSettings/ ✅ Complete
│   │   │   └── AudioStudio/    📦 Placeholder (future)
│   │   ├── lib/
│   │   │   ├── api/       # Type-safe API client ✅
│   │   │   ├── hooks/     # React Query hooks ✅
│   │   │   └── utils/     # Utilities ✅
│   │   └── stores/        # Zustand stores ✅
│
├── backend/               # Python FastAPI server
│   ├── main.py           # FastAPI app + routes ✅
│   ├── models.py         # Pydantic models ✅
│   ├── database.py       # SQLAlchemy ORM ✅
│   ├── profiles.py        # Profile management ✅
│   ├── history.py         # History management ✅
│   ├── tts.py            # Qwen3-TTS integration ✅
│   ├── transcribe.py     # Whisper integration ✅
│   ├── studio.py         # Audio studio (future)
│   └── utils/
│       ├── audio.py      # Audio processing ✅
│       ├── cache.py      # Voice prompt caching ✅
│       └── validation.py # Validation helpers ✅
│
├── tauri/                # Tauri desktop wrapper
│   ├── src/             # React entry point ✅
│   └── src-tauri/       # Rust backend ✅
│       └── src/main.rs  # Sidecar management ✅
│
├── data/                 # User data directory
│   ├── profiles/         # Profile audio samples
│   ├── generations/      # Generated audio files
│   ├── cache/            # Cached voice prompts
│   └── voicebox.db       # SQLite database
│
└── scripts/              # Build & generation scripts
    ├── generate-api.sh   # OpenAPI client generation
    └── build-server.sh   # Python binary build
```

### **Data Flow**

```
User Action (Tauri App)
    ↓
React Component (Form Submit)
    ↓
React Query Hook (useGeneration)
    ↓
API Client (apiClient.generateSpeech)
    ↓
HTTP Request → FastAPI Backend
    ↓
Backend Route Handler (/generate)
    ↓
Business Logic:
  1. Get profile from DB
  2. Create voice prompt (with caching)
  3. Generate audio with Qwen3-TTS
  4. Save audio file
  5. Create history entry
    ↓
Response (GenerationResponse)
    ↓
React Query Cache Update
    ↓
UI Refresh (History table updates)
```

### **Key Technologies**

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Desktop Framework** | Tauri v2 | Native desktop app wrapper |
| **Frontend Framework** | React 18 | UI components |
| **Language** | TypeScript | Type safety |
| **Styling** | Tailwind CSS | Utility-first CSS |
| **UI Components** | shadcn/ui | Component library |
| **State Management** | React Query + Zustand | Server & client state |
| **Form Handling** | React Hook Form + Zod | Form validation |
| **Backend Framework** | FastAPI | Async REST API |
| **Database** | SQLite + SQLAlchemy | Data persistence |
| **ML Models** | Qwen3-TTS + Whisper | Voice cloning + transcription |
| **Audio Processing** | librosa + soundfile | Audio I/O and processing |
| **Package Manager** | Bun | Fast JS/TS package management |
| **Build Tool** | Vite | Frontend bundling |

---

## 🔑 Key Features & Capabilities

### **1. Voice Profile System**
- **Multi-sample support**: Add multiple audio samples per profile
- **Automatic combination**: Multiple samples are combined for better quality
- **Voice prompt caching**: Re-use voice prompts for instant re-generation
- **Audio validation**: Ensures samples meet quality requirements

### **2. Generation Pipeline**
- **Lazy model loading**: Model loads on first use
- **Device detection**: Automatically uses GPU if available
- **Caching layer**: Voice prompts cached by audio hash + text
- **Error handling**: Graceful degradation and clear error messages

### **3. History & Search**
- **Full-text search**: Search generations by text content
- **Pagination**: Efficient loading of large histories
- **Audio playback**: Inline audio player
- **File management**: Download and delete operations

### **4. Server/Client Architecture**
- **Local mode**: Backend runs alongside Tauri app
- **Remote mode**: Connect to remote GPU machine
- **One-click server**: Start server from UI
- **Connection management**: Persistent server URL storage

---

## 📊 Database Schema

### **Tables**

```sql
-- Voice Profiles
profiles
  - id (PK, UUID)
  - name (unique)
  - description
  - language (en/zh)
  - created_at
  - updated_at

-- Profile Samples
profile_samples
  - id (PK, UUID)
  - profile_id (FK → profiles.id)
  - audio_path
  - reference_text

-- Generations
generations
  - id (PK, UUID)
  - profile_id (FK → profiles.id)
  - text
  - language
  - audio_path
  - duration (seconds)
  - seed (optional)
  - created_at

-- Projects (ready for future)
projects
  - id (PK, UUID)
  - name
  - data (JSON)
  - created_at
  - updated_at
```

---

## 🎨 UI Components Status

| Component | Status | Features |
|-----------|--------|----------|
| **ProfileList** | ✅ Complete | List, create, empty state |
| **ProfileCard** | ✅ Complete | Display profile info |
| **ProfileForm** | ✅ Complete | Create/edit dialog |
| **ProfileDetail** | ✅ Complete | View samples, add samples |
| **SampleUpload** | ✅ Complete | File upload + transcription |
| **GenerationForm** | ✅ Complete | Full generation form |
| **HistoryTable** | ✅ Complete | Table, search, pagination, play/download |
| **ConnectionForm** | ✅ Complete | Server URL input |
| **ServerStatus** | ✅ Complete | Health check display |
| **AudioStudio** | 📦 Placeholder | Timeline editor (future) |

---

## 🔌 API Endpoints

### **Profiles**
- `POST /profiles` - Create profile
- `GET /profiles` - List all profiles
- `GET /profiles/{id}` - Get profile
- `PUT /profiles/{id}` - Update profile
- `DELETE /profiles/{id}` - Delete profile
- `POST /profiles/{id}/samples` - Add sample
- `GET /profiles/{id}/samples` - List samples
- `DELETE /profiles/samples/{id}` - Delete sample

### **Generation**
- `POST /generate` - Generate speech

### **History**
- `GET /history` - List generations (with filters)
- `GET /history/{id}` - Get generation
- `DELETE /history/{id}` - Delete generation
- `GET /history/stats` - Get statistics

### **Transcription**
- `POST /transcribe` - Transcribe audio

### **Audio**
- `GET /audio/{id}` - Serve audio file

### **Health**
- `GET /health` - Health check with model status

### **Model Management**
- `POST /models/load` - Load TTS model
- `POST /models/unload` - Unload TTS model

---

## 🚀 What's Next (Planned Features)

### **Phase 2: Advanced Features**
- [ ] Multi-reference voice combination UI
- [ ] Batch generation (multiple variations)
- [ ] Advanced audio normalization
- [ ] Export options (MP3, OGG, etc.)
- [ ] M3GAN voice effect

### **Phase 3: Audio Studio**
- [ ] Timeline-based audio editor
- [ ] Word-level timestamps
- [ ] Project system (save/load sessions)
- [ ] Audio effects and filters
- [ ] Multi-track editing

### **Phase 4: Voice Design**
- [ ] Text-to-voice (no reference needed)
- [ ] Preset voices with style control
- [ ] Conversation mode (multi-speaker)
- [ ] Custom audio effects library

---

## 📝 Code Quality Standards

- ✅ **Type safety**: TypeScript strict mode, Pydantic models
- ✅ **Modular architecture**: No files over 500 lines
- ✅ **Error handling**: Comprehensive error messages
- ✅ **Caching**: Voice prompt caching for performance
- ✅ **Database**: SQLAlchemy ORM with proper relationships
- ✅ **API design**: RESTful with OpenAPI schema
- ✅ **UI/UX**: Responsive, accessible, loading states

---

## 🧪 Testing Status

- ✅ **Manual testing**: Voice generation working end-to-end
- 📦 **Unit tests**: Not yet implemented
- 📦 **Integration tests**: Not yet implemented
- 📦 **E2E tests**: Not yet implemented

---

## 📦 Dependencies

### **Backend**
- FastAPI - Web framework
- SQLAlchemy - ORM
- Pydantic - Validation
- Qwen3-TTS - Voice cloning model
- Whisper - Speech recognition
- librosa - Audio processing
- soundfile - Audio I/O
- PyTorch - ML framework

### **Frontend**
- React 18 - UI framework
- TypeScript - Type safety
- React Query - Server state
- Zustand - Client state
- React Hook Form - Forms
- Zod - Schema validation
- Tailwind CSS - Styling
- shadcn/ui - Components
- Lucide React - Icons

### **Desktop**
- Tauri v2 - Desktop framework
- Rust - System backend

---

## 🎯 Current Capabilities Summary

✅ **Working End-to-End:**
1. Create voice profiles with audio samples
2. Generate speech from text using cloned voices
3. View and manage generation history
4. Play and download generated audio
5. Search and filter history
6. Connect to local or remote backend
7. Automatic model downloading
8. Voice prompt caching for speed

🎉 **You just successfully generated voice from the Tauri app!**

---

## 🔍 Key Files Reference

### **Backend Core**
- `backend/main.py` - FastAPI app and routes
- `backend/tts.py` - Qwen3-TTS model wrapper
- `backend/profiles.py` - Profile business logic
- `backend/history.py` - History business logic
- `backend/database.py` - Database models

### **Frontend Core**
- `app/src/App.tsx` - Main app component
- `app/src/lib/api/client.ts` - API client
- `app/src/lib/hooks/` - React Query hooks
- `app/src/stores/` - Zustand stores

### **Tauri**
- `tauri/src-tauri/src/main.rs` - Rust backend
- `tauri/src/main.tsx` - React entry point

---

## 💡 Development Workflow

1. **Start backend**: `bun run dev:backend` (or via Tauri)
2. **Start frontend**: `bun run dev` (Tauri) or `bun run dev:web` (web)
3. **Generate API client**: `bun run generate:api` (after backend changes)
4. **Build server binary**: `bun run build:server` (for Tauri bundling)

---

**Ready to build more features! 🚀**
