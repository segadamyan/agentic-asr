import React, { useState } from 'react';
import { 
  Container, 
  Typography, 
  Paper, 
  Box, 
  Button,
  Divider 
} from '@mui/material';
import MarkdownRenderer from '../components/MarkdownRenderer.tsx';

const MarkdownDemo: React.FC = () => {
  const [currentExample, setCurrentExample] = useState(0);

  const examples = [
    {
      title: "Basic Formatting",
      content: `# Heading 1
## Heading 2
### Heading 3

This is a paragraph with **bold text** and *italic text*.

Here's some \`inline code\` and a [link](https://example.com).

> This is a blockquote with important information.

---

Here's a horizontal rule above.`
    },
    {
      title: "Lists and Code Blocks",
      content: `## Lists

### Unordered List:
- Item 1
- Item 2
  - Nested item
  - Another nested item
- Item 3

### Ordered List:
1. First item
2. Second item
3. Third item

## Code Block:

\`\`\`python
def greet(name):
    """Greet someone with a friendly message."""
    return f"Hello, {name}! Welcome to the chat."

# Example usage
message = greet("User")
print(message)
\`\`\`

\`\`\`javascript
const processResponse = (response) => {
  return response.split('\\n').map(line => line.trim()).filter(Boolean);
};
\`\`\``
    },
    {
      title: "Tables",
      content: `## Data Table

| Feature | Status | Description |
|---------|--------|-------------|
| Markdown Rendering | ✅ Complete | Full markdown support with syntax highlighting |
| Code Blocks | ✅ Complete | Python, JavaScript, and other languages |
| Tables | ✅ Complete | Responsive tables with Material-UI styling |
| Lists | ✅ Complete | Ordered and unordered lists |
| Links | ✅ Complete | External links open in new tab |

## Simple Table

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |`
    },
    {
      title: "Mixed Content",
      content: `# AI Chat Response Example

Thank you for your question! Here's a comprehensive answer:

## Analysis Results

I've analyzed your transcription files and found the following:

### Summary Statistics:
- **Total files**: 25
- **Average duration**: 15.3 minutes
- **Languages detected**: Armenian (80%), English (20%)

### Key Findings:

1. **Audio Quality**: Most files have good audio quality
2. **Speaker Identification**: 
   - Single speaker: 18 files
   - Multiple speakers: 7 files
3. **Content Categories**:
   - Meetings: 12 files
   - Interviews: 8 files
   - Presentations: 5 files

## Code Example

Here's how you can process the transcriptions:

\`\`\`python
from agentic_asr import TranscriptionProcessor

processor = TranscriptionProcessor()
results = processor.analyze_batch('data/transcriptions/')

for file, analysis in results.items():
    print(f"File: {file}")
    print(f"Duration: {analysis['duration']}")
    print(f"Word count: {analysis['word_count']}")
    print("---")
\`\`\`

> **Note**: For better accuracy, ensure audio files are in WAV format with 16kHz sample rate.

Would you like me to dive deeper into any specific aspect?`
    }
  ];

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Markdown Rendering Demo
      </Typography>
      
      <Typography variant="body1" color="text.secondary" gutterBottom>
        This demonstrates how AI responses will be rendered with proper markdown formatting.
      </Typography>

      <Box display="flex" gap={1} mb={3} flexWrap="wrap">
        {examples.map((example, index) => (
          <Button
            key={index}
            variant={currentExample === index ? "contained" : "outlined"}
            onClick={() => setCurrentExample(index)}
            size="small"
          >
            {example.title}
          </Button>
        ))}
      </Box>

      <Paper elevation={2} sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          {examples[currentExample].title}
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <MarkdownRenderer content={examples[currentExample].content} />
      </Paper>

      <Box mt={3}>
        <Typography variant="h6" gutterBottom>
          Raw Markdown Source:
        </Typography>
        <Paper elevation={1} sx={{ p: 2, backgroundColor: 'grey.50' }}>
          <Typography 
            component="pre" 
            variant="body2" 
            sx={{ 
              fontFamily: 'monospace', 
              whiteSpace: 'pre-wrap',
              fontSize: '12px'
            }}
          >
            {examples[currentExample].content}
          </Typography>
        </Paper>
      </Box>
    </Container>
  );
};

export default MarkdownDemo;