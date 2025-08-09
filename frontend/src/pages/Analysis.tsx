import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tabs,
  Tab,
  Paper,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  CircularProgress,
  Snackbar,
  Alert,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Analytics,
  TextFields,
  Translate,
  Summarize,
  Psychology,
  ExpandMore,
  CheckCircle,
  Error,
  Info,
} from '@mui/icons-material';
import { 
  apiService, 
  AnalysisRequest, 
  CorrectionRequest, 
  SummarizationRequest, 
  TranslationRequest,
  Transcription 
} from '../services/api.ts';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analysis-tabpanel-${index}`}
      aria-labelledby={`analysis-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const Analysis: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [transcriptions, setTranscriptions] = useState<Transcription[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Text Analysis state
  const [analysisText, setAnalysisText] = useState('');
  const [analysisType, setAnalysisType] = useState<'summary' | 'keywords' | 'sentiment'>('summary');
  const [analysisResult, setAnalysisResult] = useState<any>(null);

  // Text Correction state
  const [correctionText, setCorrectionText] = useState('');
  const [correctionContext, setCorrectionContext] = useState('');
  const [correctionLevel, setCorrectionLevel] = useState<'light' | 'medium' | 'heavy'>('medium');
  const [correctionResult, setCorrectionResult] = useState<any>(null);

  // Summarization state
  const [summaryFilename, setSummaryFilename] = useState('');
  const [summaryType, setSummaryType] = useState('comprehensive');
  const [extractActions, setExtractActions] = useState(true);
  const [extractKeyPoints, setExtractKeyPoints] = useState(true);
  const [maxSummaryLength, setMaxSummaryLength] = useState(500);
  const [summaryResult, setSummaryResult] = useState<any>(null);

  // Translation state
  const [translationFilename, setTranslationFilename] = useState('');
  const [targetLanguage, setTargetLanguage] = useState('en');
  const [sourceLanguage, setSourceLanguage] = useState('');
  const [translationResult, setTranslationResult] = useState<any>(null);

  useEffect(() => {
    loadTranscriptions();
  }, []);

  const loadTranscriptions = async () => {
    try {
      const data = await apiService.getTranscriptions();
      setTranscriptions(data);
    } catch (err) {
      console.error('Load transcriptions error:', err);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const analyzeText = async () => {
    if (!analysisText.trim()) {
      setError('Please enter text to analyze');
      return;
    }

    setLoading(true);
    try {
      const request: AnalysisRequest = {
        text: analysisText,
        analysis_type: analysisType,
      };
      const result = await apiService.analyzeText(request);
      setAnalysisResult(result);
      setSuccess('Text analysis completed');
    } catch (err) {
      setError('Failed to analyze text');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const correctText = async () => {
    if (!correctionText.trim()) {
      setError('Please enter text to correct');
      return;
    }

    setLoading(true);
    try {
      const request: CorrectionRequest = {
        text: correctionText,
        context: correctionContext,
        correction_level: correctionLevel,
      };
      const result = await apiService.correctText(request);
      setCorrectionResult(result);
      setSuccess('Text correction completed');
    } catch (err) {
      setError('Failed to correct text');
      console.error('Correction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const summarizeFile = async () => {
    if (!summaryFilename) {
      setError('Please select a file to summarize');
      return;
    }

    setLoading(true);
    try {
      const request: SummarizationRequest = {
        filename: summaryFilename,
        summary_type: summaryType,
        extract_actions: extractActions,
        extract_key_points: extractKeyPoints,
        max_summary_length: maxSummaryLength,
      };
      const result = await apiService.summarizeFile(request);
      setSummaryResult(result);
      setSuccess('File summarization completed');
    } catch (err) {
      setError('Failed to summarize file');
      console.error('Summarization error:', err);
    } finally {
      setLoading(false);
    }
  };

  const translateFile = async () => {
    if (!translationFilename) {
      setError('Please select a file to translate');
      return;
    }

    setLoading(true);
    try {
      const request: TranslationRequest = {
        filename: translationFilename,
        target_language: targetLanguage,
        source_language: sourceLanguage || undefined,
      };
      const result = await apiService.translateFile(request);
      setTranslationResult(result);
      setSuccess('File translation completed');
    } catch (err) {
      setError('Failed to translate file');
      console.error('Translation error:', err);
    } finally {
      setLoading(false);
    }
  };

  const renderAnalysisResult = (result: any) => {
    if (!result) return null;

    if (result.analysis_type === 'summary') {
      return (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Summary Result
            </Typography>
            <Typography variant="body1" paragraph>
              {result.result}
            </Typography>
            <Box display="flex" gap={1}>
              <Chip label={`Original: ${result.original_length} chars`} size="small" />
              <Chip label={`Summary: ${result.summary_length} chars`} size="small" />
              <Chip label={`Compression: ${(result.compression_ratio * 100).toFixed(1)}%`} size="small" />
            </Box>
          </CardContent>
        </Card>
      );
    }

    if (result.analysis_type === 'keywords') {
      return (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Keywords Result
            </Typography>
            <Box display="flex" flexWrap="wrap" gap={1} mb={2}>
              {result.keywords.map((keyword: any, index: number) => (
                <Chip 
                  key={index} 
                  label={`${keyword.word} (${keyword.frequency})`}
                  color="primary"
                  variant="outlined"
                />
              ))}
            </Box>
            <Box display="flex" gap={1}>
              <Chip label={`Total words: ${result.total_words}`} size="small" />
              <Chip label={`Unique words: ${result.unique_words}`} size="small" />
            </Box>
          </CardContent>
        </Card>
      );
    }

    if (result.analysis_type === 'sentiment') {
      return (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Sentiment Analysis Result
            </Typography>
            <Box display="flex" alignItems="center" gap={2} mb={2}>
              <Chip 
                label={result.sentiment.toUpperCase()} 
                color={
                  result.sentiment === 'positive' ? 'success' : 
                  result.sentiment === 'negative' ? 'error' : 'default'
                }
              />
              <Typography variant="body2">
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </Typography>
            </Box>
            <Box display="flex" gap={1}>
              <Chip label={`Positive words: ${result.positive_words_found}`} color="success" size="small" />
              <Chip label={`Negative words: ${result.negative_words_found}`} color="error" size="small" />
            </Box>
          </CardContent>
        </Card>
      );
    }

    return null;
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Analysis Tools
      </Typography>

      <Paper elevation={2}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="analysis tabs">
          <Tab icon={<Psychology />} label="Text Analysis" />
          <Tab icon={<TextFields />} label="Text Correction" />
          <Tab icon={<Summarize />} label="Summarization" />
          <Tab icon={<Translate />} label="Translation" />
        </Tabs>

        {/* Text Analysis Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Analyze Text
              </Typography>
              
              <TextField
                fullWidth
                multiline
                rows={8}
                label="Text to analyze"
                value={analysisText}
                onChange={(e) => setAnalysisText(e.target.value)}
                margin="normal"
              />

              <FormControl fullWidth margin="normal">
                <InputLabel>Analysis Type</InputLabel>
                <Select
                  value={analysisType}
                  label="Analysis Type"
                  onChange={(e) => setAnalysisType(e.target.value as any)}
                >
                  <MenuItem value="summary">Summary</MenuItem>
                  <MenuItem value="keywords">Keywords</MenuItem>
                  <MenuItem value="sentiment">Sentiment</MenuItem>
                </Select>
              </FormControl>

              <Button
                variant="contained"
                onClick={analyzeText}
                disabled={loading || !analysisText.trim()}
                fullWidth
                sx={{ mt: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'Analyze Text'}
              </Button>
            </Grid>

            <Grid item xs={12} md={6}>
              {renderAnalysisResult(analysisResult)}
            </Grid>
          </Grid>
        </TabPanel>

        {/* Text Correction Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Correct Text
              </Typography>
              
              <TextField
                fullWidth
                multiline
                rows={6}
                label="Text to correct"
                value={correctionText}
                onChange={(e) => setCorrectionText(e.target.value)}
                margin="normal"
              />

              <TextField
                fullWidth
                multiline
                rows={3}
                label="Context (optional)"
                value={correctionContext}
                onChange={(e) => setCorrectionContext(e.target.value)}
                margin="normal"
                placeholder="Provide context about the content..."
              />

              <FormControl fullWidth margin="normal">
                <InputLabel>Correction Level</InputLabel>
                <Select
                  value={correctionLevel}
                  label="Correction Level"
                  onChange={(e) => setCorrectionLevel(e.target.value as any)}
                >
                  <MenuItem value="light">Light - Minor fixes only</MenuItem>
                  <MenuItem value="medium">Medium - Standard corrections</MenuItem>
                  <MenuItem value="heavy">Heavy - Comprehensive editing</MenuItem>
                </Select>
              </FormControl>

              <Button
                variant="contained"
                onClick={correctText}
                disabled={loading || !correctionText.trim()}
                fullWidth
                sx={{ mt: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'Correct Text'}
              </Button>
            </Grid>

            <Grid item xs={12} md={6}>
              {correctionResult && (
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Corrected Text
                    </Typography>
                    <TextField
                      fullWidth
                      multiline
                      rows={12}
                      value={correctionResult.corrected_text || correctionResult.result || 'No correction available'}
                      variant="outlined"
                      InputProps={{ readOnly: true }}
                    />
                  </CardContent>
                </Card>
              )}
            </Grid>
          </Grid>
        </TabPanel>

        {/* Summarization Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Summarize File
              </Typography>
              
              <FormControl fullWidth margin="normal">
                <InputLabel>Select File</InputLabel>
                <Select
                  value={summaryFilename}
                  label="Select File"
                  onChange={(e) => setSummaryFilename(e.target.value)}
                >
                  {transcriptions.map((transcription) => (
                    <MenuItem key={transcription.filename} value={transcription.filename}>
                      {transcription.filename}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl fullWidth margin="normal">
                <InputLabel>Summary Type</InputLabel>
                <Select
                  value={summaryType}
                  label="Summary Type"
                  onChange={(e) => setSummaryType(e.target.value)}
                >
                  <MenuItem value="brief">Brief</MenuItem>
                  <MenuItem value="detailed">Detailed</MenuItem>
                  <MenuItem value="comprehensive">Comprehensive</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                type="number"
                label="Max Summary Length"
                value={maxSummaryLength}
                onChange={(e) => setMaxSummaryLength(Number(e.target.value))}
                margin="normal"
              />

              <Button
                variant="contained"
                onClick={summarizeFile}
                disabled={loading || !summaryFilename}
                fullWidth
                sx={{ mt: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'Summarize File'}
              </Button>
            </Grid>

            <Grid item xs={12} md={6}>
              {summaryResult && (
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Summary Result
                    </Typography>
                    <TextField
                      fullWidth
                      multiline
                      rows={8}
                      value={summaryResult.summary}
                      variant="outlined"
                      InputProps={{ readOnly: true }}
                      sx={{ mb: 2 }}
                    />
                    
                    {summaryResult.key_points && summaryResult.key_points.length > 0 && (
                      <Accordion>
                        <AccordionSummary expandIcon={<ExpandMore />}>
                          <Typography>Key Points ({summaryResult.key_points.length})</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <List dense>
                            {summaryResult.key_points.map((point: string, index: number) => (
                              <ListItem key={index}>
                                <ListItemIcon>
                                  <CheckCircle color="primary" fontSize="small" />
                                </ListItemIcon>
                                <ListItemText primary={point} />
                              </ListItem>
                            ))}
                          </List>
                        </AccordionDetails>
                      </Accordion>
                    )}

                    {summaryResult.actions && summaryResult.actions.length > 0 && (
                      <Accordion>
                        <AccordionSummary expandIcon={<ExpandMore />}>
                          <Typography>Actions ({summaryResult.actions.length})</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <List dense>
                            {summaryResult.actions.map((action: string, index: number) => (
                              <ListItem key={index}>
                                <ListItemIcon>
                                  <Info color="secondary" fontSize="small" />
                                </ListItemIcon>
                                <ListItemText primary={action} />
                              </ListItem>
                            ))}
                          </List>
                        </AccordionDetails>
                      </Accordion>
                    )}
                  </CardContent>
                </Card>
              )}
            </Grid>
          </Grid>
        </TabPanel>

        {/* Translation Tab */}
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Translate File
              </Typography>
              
              <FormControl fullWidth margin="normal">
                <InputLabel>Select File</InputLabel>
                <Select
                  value={translationFilename}
                  label="Select File"
                  onChange={(e) => setTranslationFilename(e.target.value)}
                >
                  {transcriptions.map((transcription) => (
                    <MenuItem key={transcription.filename} value={transcription.filename}>
                      {transcription.filename}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl fullWidth margin="normal">
                <InputLabel>Target Language</InputLabel>
                <Select
                  value={targetLanguage}
                  label="Target Language"
                  onChange={(e) => setTargetLanguage(e.target.value)}
                >
                  <MenuItem value="en">English</MenuItem>
                  <MenuItem value="es">Spanish</MenuItem>
                  <MenuItem value="fr">French</MenuItem>
                  <MenuItem value="de">German</MenuItem>
                  <MenuItem value="it">Italian</MenuItem>
                  <MenuItem value="pt">Portuguese</MenuItem>
                  <MenuItem value="ru">Russian</MenuItem>
                  <MenuItem value="ja">Japanese</MenuItem>
                  <MenuItem value="ko">Korean</MenuItem>
                  <MenuItem value="zh">Chinese</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                label="Source Language (optional)"
                value={sourceLanguage}
                onChange={(e) => setSourceLanguage(e.target.value)}
                margin="normal"
                placeholder="Leave empty for auto-detection"
              />

              <Button
                variant="contained"
                onClick={translateFile}
                disabled={loading || !translationFilename}
                fullWidth
                sx={{ mt: 2 }}
              >
                {loading ? <CircularProgress size={24} /> : 'Translate File'}
              </Button>
            </Grid>

            <Grid item xs={12} md={6}>
              {translationResult && (
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Translation Result
                    </Typography>
                    <TextField
                      fullWidth
                      multiline
                      rows={12}
                      value={translationResult.translated_text || translationResult.result || 'No translation available'}
                      variant="outlined"
                      InputProps={{ readOnly: true }}
                    />
                  </CardContent>
                </Card>
              )}
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>

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

export default Analysis;
