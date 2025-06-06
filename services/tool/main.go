package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/kkdai/youtube/v2"
)

// ProgressWriter tracks download progress
type ProgressWriter struct {
	Total      int64
	Downloaded int64
	Prefix     string
}

func (pw *ProgressWriter) Write(p []byte) (int, error) {
	n := len(p)
	pw.Downloaded += int64(n)
	percentage := float64(pw.Downloaded) / float64(pw.Total) * 100
	fmt.Printf("\r%s %.2f%% complete", pw.Prefix, percentage)
	return n, nil
}

func sanitizeFilename(filename string) string {
	invalidChars := []string{"\\", "/", ":", "*", "?", "\"", "<", ">", "|"}
	for _, char := range invalidChars {
		filename = strings.ReplaceAll(filename, char, "_")
	}
	return filename
}

func main() {
	id := flag.String("id", "", "YouTube video ID (required)")
	outputDir := flag.String("o", "./", "Output directory (default is current directory)")
	flag.Parse()

	if *id == "" {
		fmt.Println("❌ You must provide a video ID using --id")
		os.Exit(1)
	}

	// Ensure output directory exists and has trailing slash
	if !strings.HasSuffix(*outputDir, "/") && !strings.HasSuffix(*outputDir, "\\") {
		*outputDir += "/"
	}
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		fmt.Printf("❌ Failed to create output directory: %v\n", err)
		os.Exit(1)
	}

	client := youtube.Client{}
	videoURL := "https://www.youtube.com/watch?v=" + *id

	// Get video info
	video, err := client.GetVideo(videoURL)
	if err != nil {
		fmt.Printf("❌ Failed to get video info: %v\n", err)
		os.Exit(1)
	}

	safeTitle := sanitizeFilename(video.Title)
	fmt.Printf("Downloading: %s\n", safeTitle)

	// Find the best video-only format (1080p)
	var bestVideoFormat *youtube.Format
	for _, f := range video.Formats {
		if strings.Contains(f.MimeType, "video/mp4") && !strings.Contains(f.MimeType, "audio") && f.Height == 1080 {
			bestVideoFormat = &f
			break
		}
	}

	// Fallback to best video available if 1080p not found
	if bestVideoFormat == nil {
		for _, f := range video.Formats {
			if strings.Contains(f.MimeType, "video/mp4") && !strings.Contains(f.MimeType, "audio") {
				if bestVideoFormat == nil || f.Height > bestVideoFormat.Height {
					bestVideoFormat = &f
				}
			}
		}
	}

	// Find the best audio-only format
	var bestAudioFormat *youtube.Format
	for _, f := range video.Formats {
		if strings.Contains(f.MimeType, "audio/mp4") {
			if bestAudioFormat == nil || f.Bitrate > bestAudioFormat.Bitrate {
				bestAudioFormat = &f
			}
		}
	}

	// Fallback to any audio format if no audio/mp4 found
	if bestAudioFormat == nil {
		for _, f := range video.Formats {
			if strings.Contains(f.MimeType, "audio") {
				bestAudioFormat = &f
				break
			}
		}
	}

	if bestVideoFormat == nil || bestAudioFormat == nil {
		fmt.Println("❌ Could not find suitable video or audio formats")
		os.Exit(1)
	}

	// Download video stream
	videoFileName := filepath.Join(*outputDir, safeTitle+"_video.mp4")
	if err := downloadStream(client, video, bestVideoFormat, videoFileName, "Downloading video"); err != nil {
		fmt.Printf("❌ Failed to download video: %v\n", err)
		os.Exit(1)
	}

	// Download audio stream
	audioFileName := filepath.Join(*outputDir, safeTitle+"_audio.m4a")
	if err := downloadStream(client, video, bestAudioFormat, audioFileName, "Downloading audio"); err != nil {
		fmt.Printf("❌ Failed to download audio: %v\n", err)
		os.Exit(1)
	}

	// Merge video and audio
	finalFileName := filepath.Join(*outputDir, safeTitle+".mp4")
	fmt.Println("\nMerging video and audio...")
	if err := mergeVideoAudio(videoFileName, audioFileName, finalFileName); err != nil {
		fmt.Printf("❌ Failed to merge video and audio: %v\n", err)
		os.Exit(1)
	}

	// Clean up temporary files
	os.Remove(videoFileName)
	os.Remove(audioFileName)

	fmt.Printf("\n✅ Download complete: %s\n", finalFileName)
}

func downloadStream(client youtube.Client, video *youtube.Video, format *youtube.Format, filename, prefix string) error {
	stream, _, err := client.GetStream(video, format)
	if err != nil {
		return err
	}
	defer stream.Close()

	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	progress := &ProgressWriter{
		Total:  format.ContentLength,
		Prefix: prefix,
	}

	_, err = io.Copy(file, io.TeeReader(stream, progress))
	return err
}

func mergeVideoAudio(videoFile, audioFile, outputFile string) error {
	cmd := exec.Command("ffmpeg",
		"-i", videoFile,
		"-i", audioFile,
		"-c:v", "copy",
		"-c:a", "copy",
		"-shortest",
		outputFile,
		"-y", // Overwrite without asking
	)

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	return cmd.Run()
}