import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  CircularProgress,
  Snackbar,
  Alert,
} from '@mui/material';
import {
  Summarize,
  Translate,
  ExpandMore,
  CheckCircle,
  Info,
  Schedule,
  Description,
  Language,
  Refresh,
} from '@mui/icons-material';
import { apiService } from '../services/api.ts';

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
      id={`summaries-tabpanel-${index}`}
      aria-labelledby={`summaries-tab-${index}`}
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

const SummariesAndTranslations: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [summaries, setSummaries] = useState<any[]>([]);
  const [translations, setTranslations] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Filters
  const [summaryFilenameFilter, setSummaryFilenameFilter] = useState('');
  const [translationFilenameFilter, setTranslationFilenameFilter] = useState('');
  const [targetLanguageFilter, setTargetLanguageFilter] = useState('');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [summariesData, translationsData] = await Promise.all([
        apiService.getSummaries(),
        apiService.getTranslations(),
      ]);
      setSummaries(summariesData);
      setTranslations(translationsData);
    } catch (err) {
      setError('Failed to load data');
      console.error('Load data error:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadFilteredSummaries = async () => {
    setLoading(true);
    try {
      const data = await apiService.getSummaries(summaryFilenameFilter || undefined);
      setSummaries(data);
    } catch (err) {
      setError('Failed to load summaries');
      console.error('Load summaries error:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadFilteredTranslations = async () => {
    setLoading(true);
    try {
      const data = await apiService.getTranslations(
        translationFilenameFilter || undefined,
        targetLanguageFilter || undefined
      );
      setTranslations(data);
    } catch (err) {
      setError('Failed to load translations');
      console.error('Load translations error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const formatDate = (dateString: string): string => {
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return 'Invalid date';
    }
  };

  const renderSummaryCard = (summary: any, index: number) => (
    <Grid item xs={12} md={6} lg={4} key={index}>
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" mb={2}>
            <Summarize color="primary" sx={{ mr: 1 }} />
            <Typography variant="h6" noWrap>
              {summary.filename}
            </Typography>
          </Box>
          
          <Box mb={2}>
            <Chip 
              label={summary.summary_type} 
              size="small" 
              variant="outlined"
              color="primary"
            />
          </Box>

          <Box display="flex" alignItems="center" mb={2}>
            <Schedule fontSize="small" sx={{ mr: 0.5 }} />
            <Typography variant="body2" color="text.secondary">
              {formatDate(summary.created_at)}
            </Typography>
          </Box>

          <Typography variant="body2" sx={{ mb: 2, height: '3em', overflow: 'hidden' }}>
            {summary.summary}
          </Typography>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="subtitle2">View Details</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <TextField
                fullWidth
                multiline
                rows={6}
                value={summary.summary}
                variant="outlined"
                InputProps={{ readOnly: true }}
                sx={{ mb: 2 }}
              />
              
              {summary.key_points && summary.key_points.length > 0 && (
                <Box mb={2}>
                  <Typography variant="subtitle2" gutterBottom>
                    Key Points ({summary.key_points.length})
                  </Typography>
                  <List dense>
                    {summary.key_points.map((point: string, idx: number) => (
                      <ListItem key={idx}>
                        <ListItemIcon>
                          <CheckCircle color="primary" fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={point} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}

              {summary.actions && summary.actions.length > 0 && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Actions ({summary.actions.length})
                  </Typography>
                  <List dense>
                    {summary.actions.map((action: string, idx: number) => (
                      <ListItem key={idx}>
                        <ListItemIcon>
                          <Info color="secondary" fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={action} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </AccordionDetails>
          </Accordion>
        </CardContent>
      </Card>
    </Grid>
  );

  const renderTranslationCard = (translation: any, index: number) => (
    <Grid item xs={12} md={6} lg={4} key={index}>
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" mb={2}>
            <Translate color="secondary" sx={{ mr: 1 }} />
            <Typography variant="h6" noWrap>
              {translation.filename}
            </Typography>
          </Box>
          
          <Box mb={2} display="flex" gap={1}>
            <Chip 
              label={`From: ${translation.source_language}`} 
              size="small" 
              variant="outlined"
              color="default"
            />
            <Chip 
              label={`To: ${translation.target_language}`} 
              size="small" 
              variant="outlined"
              color="secondary"
            />
          </Box>

          <Box display="flex" alignItems="center" mb={2}>
            <Schedule fontSize="small" sx={{ mr: 0.5 }} />
            <Typography variant="body2" color="text.secondary">
              {formatDate(translation.created_at)}
            </Typography>
          </Box>

          <Typography variant="body2" sx={{ mb: 2, height: '3em', overflow: 'hidden' }}>
            {translation.translated_text}
          </Typography>

          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="subtitle2">View Translation</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Original Text ({translation.source_language})
                </Typography>
                <TextField
                  fullWidth
                  multiline
                  rows={4}
                  value={translation.original_text}
                  variant="outlined"
                  InputProps={{ readOnly: true }}
                  sx={{ mb: 2 }}
                />
                
                <Typography variant="subtitle2" gutterBottom>
                  Translated Text ({translation.target_language})
                </Typography>
                <TextField
                  fullWidth
                  multiline
                  rows={4}
                  value={translation.translated_text}
                  variant="outlined"
                  InputProps={{ readOnly: true }}
                />
              </Box>
            </AccordionDetails>
          </Accordion>
        </CardContent>
      </Card>
    </Grid>
  );

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Summaries & Translations
      </Typography>

      <Card elevation={2}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="summaries tabs">
          <Tab icon={<Summarize />} label={`Summaries (${summaries.length})`} />
          <Tab icon={<Translate />} label={`Translations (${translations.length})`} />
        </Tabs>

        {/* Summaries Tab */}
        <TabPanel value={tabValue} index={0}>
          <Box mb={3}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Filter by filename"
                  value={summaryFilenameFilter}
                  onChange={(e) => setSummaryFilenameFilter(e.target.value)}
                  placeholder="Enter filename to filter..."
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <Button
                  variant="outlined"
                  onClick={loadFilteredSummaries}
                  disabled={loading}
                  fullWidth
                >
                  {loading ? <CircularProgress size={20} /> : 'Filter'}
                </Button>
              </Grid>
              <Grid item xs={12} md={3}>
                <Button
                  variant="outlined"
                  startIcon={<Refresh />}
                  onClick={loadData}
                  disabled={loading}
                  fullWidth
                >
                  Refresh
                </Button>
              </Grid>
            </Grid>
          </Box>

          {summaries.length === 0 ? (
            <Card>
              <CardContent>
                <Box textAlign="center" py={4}>
                  <Summarize sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    No summaries found
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Summaries will appear here after you use the analysis tools
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          ) : (
            <Grid container spacing={3}>
              {summaries.map(renderSummaryCard)}
            </Grid>
          )}
        </TabPanel>

        {/* Translations Tab */}
        <TabPanel value={tabValue} index={1}>
          <Box mb={3}>
            <Grid container spacing={2} alignItems="center">
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Filter by filename"
                  value={translationFilenameFilter}
                  onChange={(e) => setTranslationFilenameFilter(e.target.value)}
                  placeholder="Enter filename to filter..."
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth>
                  <InputLabel>Target Language</InputLabel>
                  <Select
                    value={targetLanguageFilter}
                    label="Target Language"
                    onChange={(e) => setTargetLanguageFilter(e.target.value)}
                  >
                    <MenuItem value="">All Languages</MenuItem>
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
              </Grid>
              <Grid item xs={12} md={2}>
                <Button
                  variant="outlined"
                  onClick={loadFilteredTranslations}
                  disabled={loading}
                  fullWidth
                >
                  {loading ? <CircularProgress size={20} /> : 'Filter'}
                </Button>
              </Grid>
              <Grid item xs={12} md={2}>
                <Button
                  variant="outlined"
                  startIcon={<Refresh />}
                  onClick={loadData}
                  disabled={loading}
                  fullWidth
                >
                  Refresh
                </Button>
              </Grid>
            </Grid>
          </Box>

          {translations.length === 0 ? (
            <Card>
              <CardContent>
                <Box textAlign="center" py={4}>
                  <Translate sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" color="text.secondary" gutterBottom>
                    No translations found
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Translations will appear here after you use the translation tools
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          ) : (
            <Grid container spacing={3}>
              {translations.map(renderTranslationCard)}
            </Grid>
          )}
        </TabPanel>
      </Card>

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

export default SummariesAndTranslations;
