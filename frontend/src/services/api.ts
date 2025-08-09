import axios from 'axios';

const BASE_URL = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request/response interceptors for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export interface Transcription {
  filename: string;
  size: number;
  created_at: string;
  modified_at: string;
}

export interface TranscriptionContent {
  filename: string;
  content: string;
  size: number;
  created_at: string;
  modified_at: string;
}

export interface ChatMessage {
  message: string;
  session_id?: string;
}

export interface ChatResponse {
  response: string;
  session_id: string;
  tool_calls: Array<{
    id: string;
    name: string;
    arguments: Record<string, any>;
  }>;
}

export interface Session {
  session_id: string;
  created_at: string;
  message_count?: number;
  session_metadata?: string;
}

export interface AnalysisRequest {
  text: string;
  analysis_type: 'summary' | 'keywords' | 'sentiment';
}

export interface CorrectionRequest {
  text: string;
  context?: string;
  correction_level: 'light' | 'medium' | 'heavy';
}

export interface SummarizationRequest {
  filename: string;
  summary_type?: string;
  extract_actions?: boolean;
  extract_key_points?: boolean;
  max_summary_length?: number;
}

export interface TranslationRequest {
  filename: string;
  target_language: string;
  source_language?: string;
}

export const apiService = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // Transcriptions
  async getTranscriptions(): Promise<Transcription[]> {
    const response = await api.get('/transcriptions');
    return response.data;
  },

  async getTranscription(filename: string): Promise<TranscriptionContent> {
    const response = await api.get(`/transcriptions/${filename}`);
    return response.data;
  },

  async uploadAudioFile(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/transcriptions/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Analysis
  async analyzeText(request: AnalysisRequest): Promise<any> {
    const response = await api.post('/analyze', request);
    return response.data;
  },

  async correctText(request: CorrectionRequest): Promise<any> {
    const response = await api.post('/correct', request);
    return response.data;
  },

  async summarizeFile(request: SummarizationRequest): Promise<any> {
    const response = await api.post('/summarize', request);
    return response.data;
  },

  async translateFile(request: TranslationRequest): Promise<any> {
    const response = await api.post('/translate', request);
    return response.data;
  },

  // Chat
  async sendChatMessage(message: ChatMessage): Promise<ChatResponse> {
    const response = await api.post('/chat', message);
    return response.data;
  },

  // Sessions
  async getSessions(): Promise<Session[]> {
    const response = await api.get('/sessions');
    return response.data;
  },

  async deleteSession(sessionId: string): Promise<any> {
    const response = await api.delete(`/sessions/${sessionId}`);
    return response.data;
  },

  // Summaries and Translations
  async getSummaries(filename?: string, limit: number = 50): Promise<any[]> {
    const params = new URLSearchParams();
    if (filename) params.append('filename', filename);
    params.append('limit', limit.toString());
    
    const response = await api.get(`/summaries?${params.toString()}`);
    return response.data;
  },

  async getTranslations(filename?: string, targetLanguage?: string, limit: number = 50): Promise<any[]> {
    const params = new URLSearchParams();
    if (filename) params.append('filename', filename);
    if (targetLanguage) params.append('target_language', targetLanguage);
    params.append('limit', limit.toString());
    
    const response = await api.get(`/translations?${params.toString()}`);
    return response.data;
  },
};

export default apiService;
