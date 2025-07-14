# YouTube Audio Downloader

This script allows you to download audio from YouTube videos and convert them to MP3 format.

## Prerequisites

1. **Install ffmpeg** (required for audio conversion):
   ```bash
   # macOS (using Homebrew)
   brew install ffmpeg
   
   # Alternative: using MacPorts
   sudo port install ffmpeg
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r ../requirements.txt
   ```

## Usage Examples

### Download a single video
```bash
python data_source.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Download multiple videos
```bash
python data_source.py --urls "https://youtu.be/VIDEO1" "https://youtu.be/VIDEO2" "https://youtu.be/VIDEO3"
```

### Download from a file containing URLs
```bash
python data_source.py --file sample_urls.txt
```

### Download entire playlist
```bash
python data_source.py --playlist "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

### Specify custom output directory
```bash
python data_source.py --url "https://youtu.be/VIDEO_ID" --output my_downloads
```

### Get video information without downloading
```bash
python data_source.py --info "https://www.youtube.com/watch?v=VIDEO_ID"
```

### List all downloaded MP3 files
```bash
python data_source.py --list
```

## File Format for Batch Downloads

Create a text file (like `sample_urls.txt`) with one YouTube URL per line:
```
https://www.youtube.com/watch?v=VIDEO_ID_1
https://youtu.be/VIDEO_ID_2
https://www.youtube.com/watch?v=VIDEO_ID_3
# Comments start with # and are ignored
```

## Output

- Downloaded files are saved as MP3 format in the `downloads` directory (or custom directory specified with `--output`)
- File names are based on the video title
- Audio quality is set to 192 kbps by default

## Troubleshooting

1. **"yt_dlp not found" error**: Install yt-dlp with `pip install yt-dlp`
2. **"ffmpeg not found" error**: Install ffmpeg (see Prerequisites above)
3. **Download fails**: Check if the URL is valid and the video is available
4. **Permission errors**: Make sure you have write permissions in the output directory

## Features

- Support for single URLs, multiple URLs, and batch processing
- Playlist support
- Video information extraction
- Progress tracking for batch downloads
- Error handling and summary reporting
- Custom output directory support
- MP3 conversion with configurable quality
