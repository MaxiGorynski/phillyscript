import os
import pandas as pd
import speech_recognition as sr
from pathlib import Path
from pydub import AudioSegment


def convert_to_wav(audio_path):
    """
    Convert audio file to WAV format if it's not already WAV.
    Returns path to WAV file.
    """
    print(f"Converting {audio_path} to WAV format...")
    audio_path = Path(audio_path)
    if audio_path.suffix.lower() == '.wav':
        return str(audio_path)

    # Create a temporary WAV file
    wav_path = audio_path.with_suffix('.wav')

    # Convert audio to WAV
    if audio_path.suffix.lower() == '.mp3':
        audio = AudioSegment.from_mp3(str(audio_path))
    elif audio_path.suffix.lower() == '.flac':
        audio = AudioSegment.from_file(str(audio_path), format="flac")
    elif audio_path.suffix.lower() == '.m4a':
        audio = AudioSegment.from_file(str(audio_path), format="m4a")
    else:
        raise ValueError(f"Unsupported audio format: {audio_path.suffix}")

    audio.export(wav_path, format="wav")
    print(f"Successfully converted to {wav_path}")
    return str(wav_path)


def transcribe_audio(audio_path):
    """
    Transcribe an audio file to text using speech recognition.
    """
    # Convert to WAV if needed
    wav_path = convert_to_wav(audio_path)

    print("Starting transcription...")
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        print("Reading audio file...")
        audio = recognizer.record(source)
        try:
            print("Sending to speech recognition service...")
            transcript = recognizer.recognize_google(audio)
            print(f"Transcription received. Length: {len(transcript)} characters")
            print(f"Transcript: {transcript}")
            return transcript.lower()  # Convert to lowercase for easier matching
        except sr.UnknownValueError:
            print(f"Could not understand audio in {audio_path}")
            return ""
        except sr.RequestError as e:
            print(f"Error with speech recognition service for {audio_path}; {e}")
            return ""
        finally:
            # Clean up temporary WAV file if it was converted
            if wav_path != str(audio_path):
                Path(wav_path).unlink(missing_ok=True)
                print("Cleaned up temporary WAV file")


def process_transcript(transcript):
    """
    Process transcript text to extract features and comments.
    Returns a list of dictionaries containing features and their comments.
    """
    print("\nProcessing transcript...")
    results = []

    # Split transcript at "new feature" occurrences
    parts = transcript.split("new feature")

    # Skip the first split if it's empty (transcript starts with "new feature")
    parts = [p for p in parts if p.strip()]

    print(f"Found {len(parts)} feature sections")

    for part in parts:
        # Split each part at "comment"
        feature_comment = part.split("comment")

        if len(feature_comment) >= 2:
            feature = feature_comment[0].strip()
            comment = feature_comment[1].strip()

            # If there are multiple "comment" splits, join the rest
            if len(feature_comment) > 2:
                comment = " comment ".join(feature_comment[1:]).strip()

            print(f"Found feature: {feature}")
            print(f"Found comment: {comment}")

            results.append({
                "Feature": feature,
                "Comment": comment
            })
        else:
            # If no "comment" found, treat everything as feature
            feature = feature_comment[0].strip()
            print(f"Found feature without comment: {feature}")
            results.append({
                "Feature": feature,
                "Comment": ""
            })

    print(f"\nProcessed {len(results)} feature-comment pairs")
    return results


def main():
    # Set up the directory paths
    audio_dir = Path("/Users/supriyarai/Code/PhillyScript/audio_files")
    transcript_dir = Path("/Users/supriyarai/Code/PhillyScript/transcripts")

    # Create transcripts directory if it doesn't exist
    transcript_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nScanning directory: {audio_dir}")
    print(f"Saving transcripts to: {transcript_dir}")

    # Process each audio file in the directory
    for audio_file in audio_dir.glob("*"):
        # Skip non-audio files
        if not audio_file.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
            continue

        print(f"\nProcessing {audio_file.name}...")

        try:
            # Transcribe the audio
            transcript = transcribe_audio(str(audio_file))

            if transcript:
                # Process the transcript
                results = process_transcript(transcript)

                if results:
                    # Create DataFrame and save to CSV
                    df = pd.DataFrame(results)

                    # Generate output filename based on input audio filename
                    output_filename = f"{audio_file.stem}_transcript.csv"
                    output_path = transcript_dir / output_filename

                    df.to_csv(output_path, index=False)
                    print(f"\nResults saved to {output_path}")
                    print("\nExtracted features and comments:")
                    print(df)
                else:
                    print(f"No features found in {audio_file.name}")
            else:
                print("No transcript was generated for this file")

        except Exception as e:
            print(f"Error processing {audio_file.name}: {str(e)}")
            continue


if __name__ == "__main__":
    main()