import os
from yt_dlp import YoutubeDL

# Hardcoded values
URL = "https://www.youtube.com/watch?v=-8zyEwAa50Q"
OUTPUT_PATH = r"C:\Users\danil\Downloads\Atualizado\videos"
FILE_NAME = "caixamercado.mp4"
FFMPEG_PATH = r"C:\Users\danil\Downloads\Atualizado\FFmpegV\bin\ffmpeg.exe"

def main() -> None:
    try:
        # Ensure output directory exists
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        # Configure yt-dlp options
        ydl_opts = {
            'outtmpl': os.path.join(OUTPUT_PATH, FILE_NAME),  # Output path and filename
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Flexible format selection
            'merge_output_format': 'mp4',  # Ensure output is MP4
            'noplaylist': True,  # Download single video
            'ffmpeg_location': FFMPEG_PATH,  # Specify FFmpeg path
        }

        # Download video
        with YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading video from: {URL}")
            ydl.download([URL])
            print(f"Video downloaded successfully to: {os.path.join(OUTPUT_PATH, FILE_NAME)}")

    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        print("Troubleshooting steps:")
        print("1. Ensure FFmpeg is installed at:", FFMPEG_PATH)
        print("   Run 'C:\\Users\\danil\\Downloads\\Atualizado\\FFmpeg\\bin\\ffmpeg.exe -version' to verify.")
        print("2. Update yt-dlp: 'pip install --upgrade yt-dlp'")
        print("3. Check available formats: 'yt-dlp --list-formats <URL>'")
        print("4. Verify the video is accessible in a browser (not restricted).")
        print("5. Try a different URL, e.g., 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'.")

if __name__ == "__main__":
    main()