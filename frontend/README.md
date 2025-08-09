# Agentic ASR Frontend Setup

This is a React-based frontend for the Agentic ASR system that provides a web interface for viewing transcriptions, summaries, translations, and chat interactions.

## Features

### 📊 Dashboard
- Overview of transcriptions and chat sessions
- Recent activity display
- Quick navigation to other sections

### 📝 Transcriptions
- View all transcription files
- Upload new audio files for transcription
- Browse transcription content
- File management and organization

### 💬 Chat Interface
- Interactive chat with AI agent
- Session management
- Tool call visualization
- Real-time conversation handling

### 🔍 Analysis Tools
- **Text Analysis**: Summary, keywords, and sentiment analysis
- **Text Correction**: AI-powered transcription error correction
- **Summarization**: Generate comprehensive summaries with key points and actions
- **Translation**: Translate transcriptions between languages

### 📚 Summaries & Translations
- Browse saved summaries and translations
- Filter by filename and language
- View detailed analysis results
- Export capabilities

### 🗂️ Session Management
- View all chat sessions
- Session details and history
- Delete old sessions
- Session metadata

## Prerequisites

Make sure you have the following installed:
- Node.js (version 16 or higher)
- npm or yarn
- Python backend running on port 8000

## Installation

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Install missing dependencies** (if needed):
   ```bash
   npm install date-fns
   ```

## Configuration

The frontend is configured to connect to the backend API at:
- Development: `http://localhost:8000`
- Production: Same domain as frontend

## Running the Frontend

### Development Mode
```bash
npm start
```
This runs the app in development mode. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

### Production Build
```bash
npm run build
```
This builds the app for production to the `build` folder.

## API Integration

The frontend integrates with the backend API endpoints:

- **Transcriptions**: `/transcriptions`, `/transcriptions/{filename}`, `/transcriptions/upload`
- **Analysis**: `/analyze`, `/correct`, `/summarize`, `/translate`
- **Chat**: `/chat`
- **Sessions**: `/sessions`
- **Summaries**: `/summaries`
- **Translations**: `/translations`

## Usage

1. **Start the backend server** (make sure it's running on port 8000)
2. **Start the frontend**:
   ```bash
   npm start
   ```
3. **Open your browser** to `http://localhost:3000`

### Basic Workflow

1. **Upload Audio**: Go to Transcriptions → Upload audio files
2. **View Results**: Browse transcriptions as they're processed
3. **Analyze Content**: Use Analysis tools for summaries, translations, etc.
4. **Chat with AI**: Ask questions about your transcriptions
5. **Review Results**: Check Summaries & Translations for saved analysis

## File Structure

```
frontend/src/
├── components/
│   └── Navbar.tsx          # Navigation component
├── pages/
│   ├── Dashboard.tsx       # Main dashboard
│   ├── Transcriptions.tsx  # File management
│   ├── Chat.tsx           # AI chat interface
│   ├── Analysis.tsx       # Analysis tools
│   ├── Sessions.tsx       # Session management
│   └── SummariesAndTranslations.tsx  # Results viewer
├── services/
│   └── api.ts             # API integration
├── App.tsx                # Main app component
└── index.tsx              # Entry point
```

## Troubleshooting

### Common Issues

1. **API Connection Issues**:
   - Ensure backend is running on port 8000
   - Check CORS settings in the backend
   - Verify API endpoints are accessible

2. **Build Issues**:
   - Run `npm install` to ensure all dependencies are installed
   - Check for TypeScript errors
   - Clear node_modules and reinstall if needed

3. **Upload Issues**:
   - Ensure backend upload directory exists and is writable
   - Check file size limits
   - Verify supported audio formats

### Development Tips

- Use browser developer tools to debug API calls
- Check the backend logs for processing status
- Monitor the console for React errors
- Use the Network tab to verify API responses

## Features in Detail

### Upload & Transcription
- Drag-and-drop file upload
- Support for multiple audio formats (WAV, MP3, M4A, OGG, FLAC)
- Real-time processing status
- Background transcription processing

### Analysis Tools
- **Text Analysis**: Automatic keyword extraction and sentiment analysis
- **Correction**: AI-powered grammar and transcription error correction
- **Summarization**: Generate summaries with different detail levels
- **Translation**: Multi-language translation support

### Chat Features
- Persistent conversation sessions
- Tool usage visualization
- Context-aware responses
- Session management

### Data Management
- Browse and filter results
- Export summaries and translations
- Session history
- File organization

This frontend provides a comprehensive interface for all your transcription and analysis needs!
