"""
YouTube Audio Downloader Script

This script downloads audio from YouTube videos and converts them to MP3 format.
It supports single URLs, multiple URLs, and batch processing from a file.

Requirements:
- yt-dlp
- ffmpeg (for audio conversion)

Usage:
    python data_source.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
    python data_source.py --urls url1 url2 url3
    python data_source.py --file urls.txt
    python data_source.py --playlist "https://www.youtube.com/playlist?list=PLAYLIST_ID"
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Optional
import yt_dlp
from urllib.parse import urlparse, parse_qs


class YouTubeAudioDownloader:
    def __init__(self, output_dir: str = "downloads", quality: str = "best"):
        """
        Initialize the YouTube audio downloader.
        
        Args:
            output_dir: Directory to save downloaded files
            quality: Audio quality preference ('best', 'worst', or specific format)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.quality = quality
        
        # yt-dlp options for audio download
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'extractaudio': True,
            'audioformat': 'mp3',
            'embed_subs': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
        }
    
    def is_valid_youtube_url(self, url: str) -> bool:
        """Check if the URL is a valid YouTube URL."""
        parsed = urlparse(url)
        if parsed.netloc in ['www.youtube.com', 'youtube.com', 'youtu.be', 'm.youtube.com']:
            return True
        return False
    
    def get_video_info(self, url: str) -> Optional[dict]:
        """Get video information without downloading."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'url': url
                }
        except Exception as e:
            print(f"Error getting info for {url}: {str(e)}")
            return None
    
    def download_audio(self, url: str) -> bool:
        """
        Download audio from a single YouTube URL.
        
        Args:
            url: YouTube URL to download
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_valid_youtube_url(url):
            print(f"Invalid YouTube URL: {url}")
            return False
        
        try:
            print(f"Downloading audio from: {url}")
            
            # Get video info first
            info = self.get_video_info(url)
            if info:
                print(f"Title: {info['title']}")
                print(f"Duration: {info['duration']} seconds")
                print(f"Uploader: {info['uploader']}")
            
            # Download the audio
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([url])
            
            print(f"Successfully downloaded audio from: {url}")
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False
    
    def download_multiple(self, urls: List[str]) -> dict:
        """
        Download audio from multiple YouTube URLs.
        
        Args:
            urls: List of YouTube URLs to download
            
        Returns:
            dict: Summary of successful and failed downloads
        """
        successful = []
        failed = []
        
        print(f"Starting batch download of {len(urls)} URLs...")
        
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Processing: {url}")
            
            if self.download_audio(url):
                successful.append(url)
            else:
                failed.append(url)
        
        summary = {
            'total': len(urls),
            'successful': len(successful),
            'failed': len(failed),
            'successful_urls': successful,
            'failed_urls': failed
        }
        
        self._print_summary(summary)
        return summary
    
    def download_playlist(self, playlist_url: str) -> dict:
        """
        Download all videos from a YouTube playlist.
        
        Args:
            playlist_url: YouTube playlist URL
            
        Returns:
            dict: Summary of downloads
        """
        try:
            print(f"Extracting playlist: {playlist_url}")
            
            # Extract playlist info
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)
                
            if 'entries' not in playlist_info:
                print("No videos found in playlist")
                return {'total': 0, 'successful': 0, 'failed': 0}
            
            urls = []
            for entry in playlist_info['entries']:
                if entry:
                    urls.append(entry['webpage_url'])
            
            print(f"Found {len(urls)} videos in playlist")
            return self.download_multiple(urls)
            
        except Exception as e:
            print(f"Error processing playlist: {str(e)}")
            return {'total': 0, 'successful': 0, 'failed': 1}
    
    def download_from_file(self, file_path: str) -> dict:
        """
        Download audio from URLs listed in a text file.
        
        Args:
            file_path: Path to file containing URLs (one per line)
            
        Returns:
            dict: Summary of downloads
        """
        try:
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if not urls:
                print("No URLs found in file")
                return {'total': 0, 'successful': 0, 'failed': 0}
            
            print(f"Found {len(urls)} URLs in file: {file_path}")
            return self.download_multiple(urls)
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return {'total': 0, 'successful': 0, 'failed': 1}
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return {'total': 0, 'successful': 0, 'failed': 1}
    
    def _print_summary(self, summary: dict):
        """Print download summary."""
        print("\n" + "="*50)
        print("DOWNLOAD SUMMARY")
        print("="*50)
        print(f"Total URLs: {summary['total']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        
        if summary['failed'] > 0:
            print("\nFailed URLs:")
            for url in summary['failed_urls']:
                print(f"  - {url}")
    
    def list_downloads(self) -> List[str]:
        """List all downloaded MP3 files."""
        mp3_files = list(self.output_dir.glob("*.mp3"))
        return [str(f) for f in mp3_files]


def main():
    parser = argparse.ArgumentParser(
        description="Download audio from YouTube videos as MP3 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s --urls "https://youtu.be/VIDEO1" "https://youtu.be/VIDEO2"
  %(prog)s --file urls.txt
  %(prog)s --playlist "https://www.youtube.com/playlist?list=PLxxxxxx"
  %(prog)s --list
        """
    )
    
    # URL input options
    parser.add_argument('--url', type=str, help='Single YouTube URL to download')
    parser.add_argument('--urls', nargs='+', help='Multiple YouTube URLs to download')
    parser.add_argument('--file', type=str, help='Text file containing URLs (one per line)')
    parser.add_argument('--playlist', type=str, help='YouTube playlist URL to download')
    
    # Configuration options
    parser.add_argument('--output', '-o', type=str, default='data/downloads',
                       help='Output directory for downloaded files (default: downloads)')
    parser.add_argument('--quality', '-q', type=str, default='best',
                       choices=['best', 'worst'], help='Audio quality preference')
    
    # Utility options
    parser.add_argument('--list', action='store_true', help='List all downloaded MP3 files')
    parser.add_argument('--info', type=str, help='Get video information without downloading')
    
    args = parser.parse_args()
    
    # Create downloader instance
    downloader = YouTubeAudioDownloader(output_dir=args.output, quality=args.quality)
    
    # Handle different operations
    if args.list:
        files = downloader.list_downloads()
        if files:
            print(f"Found {len(files)} MP3 files:")
            for file in files:
                print(f"  - {os.path.basename(file)}")
        else:
            print("No MP3 files found in downloads directory")
        return
    
    if args.info:
        info = downloader.get_video_info(args.info)
        if info:
            print(json.dumps(info, indent=2))
        return
    
    # Download operations
    if args.url:
        downloader.download_audio(args.url)
    elif args.urls:
        downloader.download_multiple(args.urls)
    elif args.file:
        downloader.download_from_file(args.file)
    elif args.playlist:
        downloader.download_playlist(args.playlist)
    else:
        parser.print_help()
        print("\nError: Please provide at least one URL source (--url, --urls, --file, or --playlist)")
        sys.exit(1)


if __name__ == "__main__":
    main()