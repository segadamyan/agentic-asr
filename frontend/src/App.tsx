import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';

import Navbar from './components/Navbar.tsx';
import Dashboard from './pages/Dashboard.tsx';
import Transcriptions from './pages/Transcriptions.tsx';
import Chat from './pages/Chat.tsx';
import Analysis from './pages/Analysis.tsx';
import Sessions from './pages/Sessions.tsx';
import SummariesAndTranslations from './pages/SummariesAndTranslations.tsx';
import MarkdownDemo from './pages/MarkdownDemo.tsx';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          borderRadius: '8px',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: '6px',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router basename="/agentic-asr">
        <Box className="app">
          <Navbar />
          <Box className="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/transcriptions" element={<Transcriptions />} />
              <Route path="/chat" element={<Chat />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/sessions" element={<Sessions />} />
              <Route path="/summaries" element={<SummariesAndTranslations />} />
              <Route path="/markdown-demo" element={<MarkdownDemo />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
