import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Box,
  Button,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  CircularProgress,
  Snackbar,
  Alert,
  Divider,
} from '@mui/material';
import {
  History,
  Delete,
  Chat,
  Schedule,
  Person,
  SmartToy,
  Visibility,
  DeleteForever,
} from '@mui/icons-material';
import { format } from 'date-fns';
import { apiService, Session } from '../services/api.ts';

interface SessionDetails {
  session_id: string;
  created_at: string;
  updated_at?: string;
  metadata?: any;
  messages?: Array<{
    id: string;
    role: string;
    content: string;
    timestamp: string;
    message_type?: string;
  }>;
}

const Sessions: React.FC = () => {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSession, setSelectedSession] = useState<SessionDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      setLoading(true);
      const data = await apiService.getSessions();
      setSessions(data);
    } catch (err) {
      setError('Failed to load sessions');
      console.error('Load sessions error:', err);
    } finally {
      setLoading(false);
    }
  };

  const viewSession = async (sessionId: string) => {
    try {
      // For now, we'll just show the basic session info
      // In a real implementation, you might want to add an API endpoint to get session details with messages
      const session = sessions.find(s => s.session_id === sessionId);
      if (session) {
        setSelectedSession({
          ...session,
          messages: [] // This would come from an API call in a real implementation
        });
        setViewDialogOpen(true);
      }
    } catch (err) {
      setError('Failed to load session details');
      console.error('View session error:', err);
    }
  };

  const confirmDeleteSession = (sessionId: string) => {
    setSessionToDelete(sessionId);
    setDeleteDialogOpen(true);
  };

  const deleteSession = async () => {
    if (!sessionToDelete) return;

    try {
      setDeleting(sessionToDelete);
      await apiService.deleteSession(sessionToDelete);
      setSessions(sessions.filter(s => s.session_id !== sessionToDelete));
      setSuccess('Session deleted successfully');
      setDeleteDialogOpen(false);
      setSessionToDelete(null);
    } catch (err) {
      setError('Failed to delete session');
      console.error('Delete session error:', err);
    } finally {
      setDeleting(null);
    }
  };

  const formatDate = (dateString: string): string => {
    try {
      return format(new Date(dateString), 'MMM dd, yyyy HH:mm');
    } catch {
      return 'Invalid date';
    }
  };

  const getSessionDuration = (createdAt: string, updatedAt?: string): string => {
    try {
      const start = new Date(createdAt);
      const end = updatedAt ? new Date(updatedAt) : new Date();
      const diffMs = end.getTime() - start.getTime();
      const diffMins = Math.floor(diffMs / (1000 * 60));
      
      if (diffMins < 60) {
        return `${diffMins} min`;
      } else {
        const hours = Math.floor(diffMins / 60);
        const mins = diffMins % 60;
        return `${hours}h ${mins}m`;
      }
    } catch {
      return 'Unknown';
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Chat Sessions
      </Typography>

      {sessions.length === 0 ? (
        <Card>
          <CardContent>
            <Box textAlign="center" py={4}>
              <History sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No chat sessions yet
              </Typography>
              <Typography variant="body2" color="text.secondary" mb={3}>
                Start a conversation to see your chat history here
              </Typography>
              <Button
                variant="contained"
                startIcon={<Chat />}
                href="/chat"
              >
                Start Chatting
              </Button>
            </Box>
          </CardContent>
        </Card>
      ) : (
        <Grid container spacing={3}>
          {sessions.map((session) => (
            <Grid item xs={12} md={6} lg={4} key={session.session_id}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    <Chat color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6" noWrap>
                      Session {session.session_id.slice(0, 8)}...
                    </Typography>
                  </Box>
                  
                  <Box mb={2}>
                    <Chip 
                      label={`${session.message_count || 0} messages`}
                      size="small" 
                      variant="outlined"
                      color="primary"
                    />
                  </Box>

                  <Box display="flex" alignItems="center" mb={1}>
                    <Schedule fontSize="small" sx={{ mr: 0.5 }} />
                    <Typography variant="body2" color="text.secondary">
                      Started: {formatDate(session.created_at)}
                    </Typography>
                  </Box>

                  {session.session_metadata && (
                    <Box display="flex" alignItems="center" mb={1}>
                      <Typography variant="body2" color="text.secondary">
                        Duration: {getSessionDuration(session.created_at, session.session_metadata)}
                      </Typography>
                    </Box>
                  )}

                  <Typography variant="caption" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                    ID: {session.session_id}
                  </Typography>
                </CardContent>

                <CardActions>
                  <Button
                    size="small"
                    startIcon={<Visibility />}
                    onClick={() => viewSession(session.session_id)}
                  >
                    View
                  </Button>
                  <IconButton
                    size="small"
                    color="error"
                    onClick={() => confirmDeleteSession(session.session_id)}
                    disabled={deleting === session.session_id}
                  >
                    {deleting === session.session_id ? (
                      <CircularProgress size={16} />
                    ) : (
                      <Delete />
                    )}
                  </IconButton>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* View Session Dialog */}
      <Dialog 
        open={viewDialogOpen} 
        onClose={() => setViewDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Session Details
        </DialogTitle>
        <DialogContent>
          {selectedSession && (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Session ID: {selectedSession.session_id}
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Created: {formatDate(selectedSession.created_at)}
              </Typography>
              {selectedSession.updated_at && (
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Last Updated: {formatDate(selectedSession.updated_at)}
                </Typography>
              )}

              <Divider sx={{ my: 2 }} />

              <Typography variant="h6" gutterBottom>
                Messages
              </Typography>

              {selectedSession.messages && selectedSession.messages.length > 0 ? (
                <List>
                  {selectedSession.messages.map((message, index) => (
                    <ListItem key={index} alignItems="flex-start">
                      <ListItemIcon>
                        {message.role === 'user' ? <Person /> : <SmartToy />}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box display="flex" alignItems="center" gap={1}>
                            <Typography variant="subtitle2">
                              {message.role === 'user' ? 'You' : 'AI Assistant'}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {formatDate(message.timestamp)}
                            </Typography>
                          </Box>
                        }
                        secondary={
                          <Typography variant="body2" sx={{ mt: 0.5, whiteSpace: 'pre-wrap' }}>
                            {message.content}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Box textAlign="center" py={2}>
                  <Typography variant="body2" color="text.secondary">
                    No messages available for this session
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Message details are not yet implemented in the API
                  </Typography>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>
          <Box display="flex" alignItems="center">
            <DeleteForever color="error" sx={{ mr: 1 }} />
            Delete Session
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this chat session? This action cannot be undone.
          </Typography>
          {sessionToDelete && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1, fontFamily: 'monospace' }}>
              Session ID: {sessionToDelete}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={deleteSession}
            color="error"
            variant="contained"
            disabled={!!deleting}
          >
            {deleting ? <CircularProgress size={20} /> : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbars */}
      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={() => setError(null)}
      >
        <Alert onClose={() => setError(null)} severity="error">
          {error}
        </Alert>
      </Snackbar>

      <Snackbar 
        open={!!success} 
        autoHideDuration={6000} 
        onClose={() => setSuccess(null)}
      >
        <Alert onClose={() => setSuccess(null)} severity="success">
          {success}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default Sessions;
