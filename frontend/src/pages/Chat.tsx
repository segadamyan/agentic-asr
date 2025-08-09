import React, { useState, useEffect, useRef } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Chip,
  CircularProgress,
  Snackbar,
  Alert,
  Divider,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Send,
  Person,
  SmartToy,
  Clear,
  History,
  Settings,
} from '@mui/icons-material';
import { apiService, ChatMessage, ChatResponse } from '../services/api.ts';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  toolCalls?: Array<{
    id: string;
    name: string;
    arguments: Record<string, any>;
  }>;
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Auto-scroll to bottom when new messages arrive
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    // Focus input on component mount
    inputRef.current?.focus();
  }, []);

  const sendMessage = async () => {
    if (!currentMessage.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: currentMessage,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setLoading(true);

    try {
      const chatMessage: ChatMessage = {
        message: currentMessage,
        session_id: sessionId || undefined,
      };

      const response: ChatResponse = await apiService.sendChatMessage(chatMessage);
      
      // Update session ID if it's a new conversation
      if (!sessionId) {
        setSessionId(response.session_id);
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.response,
        timestamp: new Date(),
        toolCalls: response.tool_calls,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      setError('Failed to send message');
      console.error('Chat error:', err);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId(null);
    inputRef.current?.focus();
  };

  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const renderToolCalls = (toolCalls: Array<{ id: string; name: string; arguments: Record<string, any> }>) => {
    if (!toolCalls || toolCalls.length === 0) return null;

    return (
      <Box mt={1}>
        <Typography variant="caption" color="text.secondary" gutterBottom>
          Tools used:
        </Typography>
        <Box display="flex" flexWrap="wrap" gap={0.5}>
          {toolCalls.map((tool, index) => (
            <Chip
              key={index}
              label={tool.name.replace('_', ' ')}
              size="small"
              variant="outlined"
              color="primary"
            />
          ))}
        </Box>
      </Box>
    );
  };

  return (
    <Container maxWidth="lg">
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" gutterBottom>
          Chat with AI Agent
        </Typography>
        <Box>
          <IconButton onClick={() => setSettingsOpen(true)} title="Settings">
            <Settings />
          </IconButton>
          <IconButton onClick={clearChat} title="Clear Chat">
            <Clear />
          </IconButton>
        </Box>
      </Box>

      <Paper elevation={2} sx={{ height: '70vh', display: 'flex', flexDirection: 'column' }}>
        {/* Messages Area */}
        <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
          {messages.length === 0 ? (
            <Box 
              display="flex" 
              flexDirection="column" 
              alignItems="center" 
              justifyContent="center" 
              height="100%"
              textAlign="center"
            >
              <SmartToy sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Start a conversation
              </Typography>
              <Typography variant="body2" color="text.secondary" mb={3}>
                Ask me anything about transcriptions, summaries, or audio analysis
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1} justifyContent="center">
                {[
                  "What can you help me with?",
                  "Summarize my latest transcription",
                  "List all transcription files",
                  "Translate a transcription to English"
                ].map((suggestion, index) => (
                  <Chip
                    key={index}
                    label={suggestion}
                    variant="outlined"
                    clickable
                    onClick={() => setCurrentMessage(suggestion)}
                  />
                ))}
              </Box>
            </Box>
          ) : (
            <List>
              {messages.map((message, index) => (
                <ListItem key={message.id} sx={{ alignItems: 'flex-start', py: 1 }}>
                  <ListItemAvatar>
                    <Avatar sx={{ 
                      bgcolor: message.type === 'user' ? 'primary.main' : 'secondary.main' 
                    }}>
                      {message.type === 'user' ? <Person /> : <SmartToy />}
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="subtitle2">
                          {message.type === 'user' ? 'You' : 'AI Assistant'}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {formatTime(message.timestamp)}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      <Box mt={0.5}>
                        <Typography 
                          variant="body1" 
                          component="div" 
                          sx={{ whiteSpace: 'pre-wrap' }}
                        >
                          {message.content}
                        </Typography>
                        {message.toolCalls && renderToolCalls(message.toolCalls)}
                      </Box>
                    }
                  />
                </ListItem>
              ))}
              {loading && (
                <ListItem>
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: 'secondary.main' }}>
                      <SmartToy />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary="AI Assistant"
                    secondary={
                      <Box display="flex" alignItems="center" gap={1} mt={0.5}>
                        <CircularProgress size={16} />
                        <Typography variant="body2" color="text.secondary">
                          Thinking...
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
              )}
            </List>
          )}
          <div ref={messagesEndRef} />
        </Box>

        <Divider />

        {/* Input Area */}
        <Box p={2}>
          {sessionId && (
            <Typography variant="caption" color="text.secondary" gutterBottom>
              Session: {sessionId.slice(0, 8)}...
            </Typography>
          )}
          <Box display="flex" gap={1} alignItems="flex-end">
            <TextField
              ref={inputRef}
              fullWidth
              multiline
              maxRows={4}
              variant="outlined"
              placeholder="Type your message here... (Shift+Enter for new line)"
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading}
            />
            <Button
              variant="contained"
              endIcon={<Send />}
              onClick={sendMessage}
              disabled={!currentMessage.trim() || loading}
              sx={{ minWidth: 'auto', px: 2 }}
            >
              Send
            </Button>
          </Box>
        </Box>
      </Paper>

      {/* Settings Dialog */}
      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)}>
        <DialogTitle>Chat Settings</DialogTitle>
        <DialogContent>
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Current Session ID:
            </Typography>
            <Typography variant="body2" sx={{ fontFamily: 'monospace', mb: 2 }}>
              {sessionId || 'New session'}
            </Typography>
            
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Messages in this session:
            </Typography>
            <Typography variant="body2" gutterBottom>
              {messages.length}
            </Typography>

            <Typography variant="body2" color="text.secondary" gutterBottom>
              Available Commands:
            </Typography>
            <Box component="ul" sx={{ pl: 2, m: 0 }}>
              <li>Ask about transcriptions</li>
              <li>Request summaries and analysis</li>
              <li>Translate content</li>
              <li>Get help with audio processing</li>
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Error Snackbar */}
      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={() => setError(null)}
      >
        <Alert onClose={() => setError(null)} severity="error">
          {error}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default Chat;
