import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { 
  Typography, 
  Paper, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow,
  Link,
  Box,
  Divider,
} from '@mui/material';

interface MarkdownRendererProps {
  content: string;
  variant?: 'body1' | 'body2';
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ 
  content, 
  variant = 'body1' 
}) => {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        // Headings
        h1: ({ children }) => (
          <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 'bold', mt: 2, mb: 1 }}>
            {children}
          </Typography>
        ),
        h2: ({ children }) => (
          <Typography variant="h5" component="h2" gutterBottom sx={{ fontWeight: 'bold', mt: 2, mb: 1 }}>
            {children}
          </Typography>
        ),
        h3: ({ children }) => (
          <Typography variant="h6" component="h3" gutterBottom sx={{ fontWeight: 'bold', mt: 1.5, mb: 1 }}>
            {children}
          </Typography>
        ),
        h4: ({ children }) => (
          <Typography variant="subtitle1" component="h4" gutterBottom sx={{ fontWeight: 'bold', mt: 1.5, mb: 0.5 }}>
            {children}
          </Typography>
        ),
        h5: ({ children }) => (
          <Typography variant="subtitle2" component="h5" gutterBottom sx={{ fontWeight: 'bold', mt: 1, mb: 0.5 }}>
            {children}
          </Typography>
        ),
        h6: ({ children }) => (
          <Typography variant="subtitle2" component="h6" gutterBottom sx={{ fontWeight: 'bold', mt: 1, mb: 0.5 }}>
            {children}
          </Typography>
        ),
        
        // Paragraphs
        p: ({ children }) => (
          <Typography variant={variant} component="p" gutterBottom sx={{ mb: 1 }}>
            {children}
          </Typography>
        ),
        
        // Links
        a: ({ href, children }) => (
          <Link href={href} target="_blank" rel="noopener noreferrer" color="primary">
            {children}
          </Link>
        ),
        
        // Code blocks
        code: ({ className, children, ...props }: any) => {
          const match = /language-(\w+)/.exec(className || '');
          const language = match ? match[1] : '';
          const isInline = !className;
          
          if (!isInline && language) {
            return (
              <Paper 
                elevation={1} 
                sx={{ 
                  my: 2, 
                  p: 2, 
                  backgroundColor: '#1e1e1e',
                  borderRadius: 2,
                  overflow: 'auto'
                }}
              >
                <Typography
                  variant="caption"
                  sx={{ 
                    color: '#888', 
                    mb: 1, 
                    display: 'block',
                    textTransform: 'uppercase',
                    fontSize: '0.7rem'
                  }}
                >
                  {language}
                </Typography>
                <Typography
                  component="pre"
                  sx={{
                    fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
                    fontSize: '14px',
                    lineHeight: 1.5,
                    color: '#d4d4d4',
                    margin: 0,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  {String(children).replace(/\n$/, '')}
                </Typography>
              </Paper>
            );
          }
          
          return (
            <Box
              component="code"
              sx={{
                backgroundColor: 'grey.100',
                color: 'error.main',
                px: 0.5,
                py: 0.25,
                borderRadius: 1,
                fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
                fontSize: '0.9em',
                display: 'inline',
              }}
            >
              {children}
            </Box>
          );
        },
        
        // Blockquotes
        blockquote: ({ children }) => (
          <Paper 
            elevation={0} 
            sx={{ 
              borderLeft: 4, 
              borderColor: 'primary.main', 
              backgroundColor: 'grey.50',
              p: 2, 
              my: 2,
              fontStyle: 'italic'
            }}
          >
            {children}
          </Paper>
        ),
        
        // Lists
        ul: ({ children }) => (
          <Box component="ul" sx={{ pl: 2, py: 0, mb: 1 }}>
            {children}
          </Box>
        ),
        ol: ({ children }) => (
          <Box component="ol" sx={{ pl: 2, py: 0, mb: 1 }}>
            {children}
          </Box>
        ),
        li: ({ children }) => (
          <Box component="li" sx={{ mb: 0.5 }}>
            {children}
          </Box>
        ),
        
        // Tables
        table: ({ children }) => (
          <TableContainer component={Paper} elevation={1} sx={{ my: 2 }}>
            <Table size="small">
              {children}
            </Table>
          </TableContainer>
        ),
        thead: ({ children }) => (
          <TableHead>
            {children}
          </TableHead>
        ),
        tbody: ({ children }) => (
          <TableBody>
            {children}
          </TableBody>
        ),
        tr: ({ children }) => (
          <TableRow>
            {children}
          </TableRow>
        ),
        td: ({ children }) => (
          <TableCell>
            {children}
          </TableCell>
        ),
        th: ({ children }) => (
          <TableCell sx={{ fontWeight: 'bold' }}>
            {children}
          </TableCell>
        ),
        
        // Horizontal rule
        hr: () => <Divider sx={{ my: 3 }} />,
        
        // Strong/Bold
        strong: ({ children }) => (
          <Typography component="strong" sx={{ fontWeight: 'bold' }}>
            {children}
          </Typography>
        ),
        
        // Emphasis/Italic
        em: ({ children }) => (
          <Typography component="em" sx={{ fontStyle: 'italic' }}>
            {children}
          </Typography>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

export default MarkdownRenderer;