import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  Transcribe,
  Chat,
  Analytics,
  History,
  Description,
  Schedule,
} from '@mui/icons-material';
import { apiService } from '../services/api.ts';

interface DashboardStats {
  totalTranscriptions: number;
  totalSessions: number;
  recentTranscriptions: Array<{
    filename: string;
    created_at: string;
    size: number;
  }>;
  recentSessions: Array<{
    session_id: string;
    created_at: string;
    message_count?: number;
  }>;
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const [transcriptions, sessions] = await Promise.all([
        apiService.getTranscriptions(),
        apiService.getSessions(),
      ]);

      setStats({
        totalTranscriptions: transcriptions.length,
        totalSessions: sessions.length,
        recentTranscriptions: transcriptions.slice(0, 5),
        recentSessions: sessions.slice(0, 5),
      });
    } catch (err) {
      setError('Failed to load dashboard data');
      console.error('Dashboard error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const formatDate = (dateString: string): string => {
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return 'Invalid date';
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg">
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <Typography>Loading dashboard...</Typography>
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg">
        <Box className="error">
          <Typography>{error}</Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Stats Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Transcribe color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Transcriptions</Typography>
              </Box>
              <Typography variant="h3" color="primary">
                {stats?.totalTranscriptions || 0}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Total files transcribed
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Chat color="secondary" sx={{ mr: 1 }} />
                <Typography variant="h6">Sessions</Typography>
              </Box>
              <Typography variant="h3" color="secondary">
                {stats?.totalSessions || 0}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Chat conversations
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Analytics color="success" sx={{ mr: 1 }} />
                <Typography variant="h6">Analysis</Typography>
              </Box>
              <Typography variant="h3" color="success.main">
                {stats?.recentTranscriptions.length || 0}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Recent analyses
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <History color="info" sx={{ mr: 1 }} />
                <Typography variant="h6">History</Typography>
              </Box>
              <Typography variant="h3" color="info.main">
                {stats?.recentSessions.length || 0}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Active sessions
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Transcriptions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Transcriptions
              </Typography>
              {stats?.recentTranscriptions.length === 0 ? (
                <Typography variant="body2" color="textSecondary">
                  No transcriptions yet
                </Typography>
              ) : (
                <List dense>
                  {stats?.recentTranscriptions.map((transcription, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <Description color="primary" />
                      </ListItemIcon>
                      <ListItemText
                        primary={transcription.filename}
                        secondary={
                          <Box display="flex" alignItems="center" gap={1}>
                            <Chip 
                              label={formatFileSize(transcription.size)} 
                              size="small" 
                              variant="outlined" 
                            />
                            <Box display="flex" alignItems="center">
                              <Schedule fontSize="small" sx={{ mr: 0.5 }} />
                              {formatDate(transcription.created_at)}
                            </Box>
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Sessions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Chat Sessions
              </Typography>
              {stats?.recentSessions.length === 0 ? (
                <Typography variant="body2" color="textSecondary">
                  No chat sessions yet
                </Typography>
              ) : (
                <List dense>
                  {stats?.recentSessions.map((session, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <Chat color="secondary" />
                      </ListItemIcon>
                      <ListItemText
                        primary={`Session ${session.session_id.slice(0, 8)}...`}
                        secondary={
                          <Box display="flex" alignItems="center" gap={1}>
                            <Chip 
                              label={`${session.message_count || 0} messages`} 
                              size="small" 
                              variant="outlined" 
                              color="secondary"
                            />
                            <Box display="flex" alignItems="center">
                              <Schedule fontSize="small" sx={{ mr: 0.5 }} />
                              {formatDate(session.created_at)}
                            </Box>
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
