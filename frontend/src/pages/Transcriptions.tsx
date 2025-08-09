import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Snackbar,
  Alert,
  CircularProgress,
  Fab,
} from '@mui/material';
import {
  Description,
  Upload,
  Visibility,
  Schedule,
  Storage,
  Add,
  Close,
  CloudUpload,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { apiService, Transcription, TranscriptionContent } from '../services/api.ts';

const Transcriptions: React.FC = () => {
  const [transcriptions, setTranscriptions] = useState<Transcription[]>([]);
  const [selectedTranscription, setSelectedTranscription] = useState<TranscriptionContent | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);

  useEffect(() => {
    loadTranscriptions();
  }, []);

  const loadTranscriptions = async () => {
    try {
      setLoading(true);
      const data = await apiService.getTranscriptions();
      setTranscriptions(data);
    } catch (err) {
      setError('Failed to load transcriptions');
      console.error('Load transcriptions error:', err);
    } finally {
      setLoading(false);
    }
  };

  const viewTranscription = async (filename: string) => {
    try {
      const content = await apiService.getTranscription(filename);
      setSelectedTranscription(content);
      setViewDialogOpen(true);
    } catch (err) {
      setError('Failed to load transcription content');
      console.error('View transcription error:', err);
    }
  };

  const onDrop = async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    
    // Check file type
    const validTypes = ['audio/wav', 'audio/mp3', 'audio/m4a', 'audio/ogg', 'audio/flac'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|m4a|ogg|flac)$/i)) {
      setError('Please upload a valid audio file (WAV, MP3, M4A, OGG, FLAC)');
      return;
    }

    try {
      setUploading(true);
      const result = await apiService.uploadAudioFile(file);
      setSuccess(`File uploaded successfully: ${result.filename}. Transcription started.`);
      setUploadDialogOpen(false);
      // Refresh the list after a delay to allow processing
      setTimeout(() => {
        loadTranscriptions();
      }, 2000);
    } catch (err) {
      setError('Failed to upload file');
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
    },
    multiple: false
  });

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
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg">
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" gutterBottom>
          Transcriptions
        </Typography>
        <Button
          variant="contained"
          startIcon={<Upload />}
          onClick={() => setUploadDialogOpen(true)}
        >
          Upload Audio
        </Button>
      </Box>

      {transcriptions.length === 0 ? (
        <Card>
          <CardContent>
            <Box textAlign="center" py={4}>
              <Description sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary" gutterBottom>
                No transcriptions yet
              </Typography>
              <Typography variant="body2" color="text.secondary" mb={3}>
                Upload an audio file to get started with transcription
              </Typography>
              <Button
                variant="contained"
                startIcon={<Upload />}
                onClick={() => setUploadDialogOpen(true)}
              >
                Upload Your First Audio File
              </Button>
            </Box>
          </CardContent>
        </Card>
      ) : (
        <Grid container spacing={3}>
          {transcriptions.map((transcription, index) => (
            <Grid item xs={12} md={6} lg={4} key={index}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    <Description color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6" noWrap>
                      {transcription.filename}
                    </Typography>
                  </Box>
                  
                  <Box mb={2}>
                    <Chip 
                      label={formatFileSize(transcription.size)} 
                      size="small" 
                      variant="outlined"
                      icon={<Storage />}
                    />
                  </Box>

                  <Box display="flex" alignItems="center" mb={2}>
                    <Schedule fontSize="small" sx={{ mr: 0.5 }} />
                    <Typography variant="body2" color="text.secondary">
                      {formatDate(transcription.created_at)}
                    </Typography>
                  </Box>

                  <Button
                    variant="outlined"
                    startIcon={<Visibility />}
                    onClick={() => viewTranscription(transcription.filename)}
                    fullWidth
                  >
                    View Content
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Upload Dialog */}
      <Dialog 
        open={uploadDialogOpen} 
        onClose={() => setUploadDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Upload Audio File
          <IconButton
            onClick={() => setUploadDialogOpen(false)}
            sx={{ position: 'absolute', right: 8, top: 8 }}
          >
            <Close />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          <Box
            {...getRootProps()}
            sx={{
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.300',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              cursor: 'pointer',
              backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
              '&:hover': {
                backgroundColor: 'action.hover',
              },
            }}
          >
            <input {...getInputProps()} />
            <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            {uploading ? (
              <Box>
                <CircularProgress sx={{ mb: 2 }} />
                <Typography>Uploading and processing...</Typography>
              </Box>
            ) : (
              <Box>
                <Typography variant="h6" gutterBottom>
                  {isDragActive ? 'Drop the file here' : 'Drag & drop an audio file here'}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  or click to select a file
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Supported formats: WAV, MP3, M4A, OGG, FLAC
                </Typography>
              </Box>
            )}
          </Box>
        </DialogContent>
      </Dialog>

      {/* View Dialog */}
      <Dialog 
        open={viewDialogOpen} 
        onClose={() => setViewDialogOpen(false)}
        maxWidth="md"
        fullWidth
        scroll="paper"
      >
        <DialogTitle>
          {selectedTranscription?.filename}
          <IconButton
            onClick={() => setViewDialogOpen(false)}
            sx={{ position: 'absolute', right: 8, top: 8 }}
          >
            <Close />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers>
          {selectedTranscription && (
            <Box>
              <Box mb={2}>
                <Typography variant="body2" color="text.secondary">
                  Size: {formatFileSize(selectedTranscription.size)} • 
                  Created: {formatDate(selectedTranscription.created_at)}
                </Typography>
              </Box>
              <TextField
                multiline
                rows={20}
                fullWidth
                value={selectedTranscription.content}
                variant="outlined"
                InputProps={{
                  readOnly: true,
                }}
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setViewDialogOpen(false)}>Close</Button>
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

export default Transcriptions;
