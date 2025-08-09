import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
} from '@mui/material';
import {
  Home,
  Transcribe,
  Chat,
  Analytics,
  History,
  Storage,
} from '@mui/icons-material';

const Navbar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <Home /> },
    { path: '/transcriptions', label: 'Transcriptions', icon: <Transcribe /> },
    { path: '/chat', label: 'Chat', icon: <Chat /> },
    { path: '/analysis', label: 'Analysis', icon: <Analytics /> },
    { path: '/summaries', label: 'Summaries', icon: <Storage /> },
    { path: '/sessions', label: 'Sessions', icon: <History /> },
  ];

  return (
    <AppBar position="static" elevation={2}>
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 0, mr: 3 }}>
          Agentic ASR
        </Typography>
        
        <Box sx={{ flexGrow: 1, display: 'flex', gap: 1 }}>
          {navItems.map((item) => (
            <Button
              key={item.path}
              color="inherit"
              onClick={() => navigate(item.path)}
              startIcon={item.icon}
              sx={{
                backgroundColor: location.pathname === item.path ? 'rgba(255,255,255,0.1)' : 'transparent',
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.2)',
                },
              }}
            >
              {item.label}
            </Button>
          ))}
        </Box>

        <Typography variant="body2" sx={{ ml: 2 }}>
          Intelligent Speech Recognition
        </Typography>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
