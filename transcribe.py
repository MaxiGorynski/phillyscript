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
    Process transcript text to extract room, attribute, features and comments.
    Returns a list of dictionaries containing these elements.
    """
    print("\nProcessing transcript...")
    results = []

    # Initialize current state
    current_room = ""
    current_attribute = ""
    current_feature = ""
    current_mode = None  # Tracks what we're currently collecting (room, attribute, feature, or comment)

    # Valid attributes list
    valid_attributes = [
        "doors", "floors", "skirting boards", "walls",
        "cornicing", "ceilings", "fixtures and fittings", "furniture"
    ]

    # Split transcript into words for easier processing
    words = transcript.split()
    i = 0
    buffer = ""  # To collect text between markers

    while i < len(words):
        # Detect markers that indicate a mode change
        if i < len(words) - 1 and words[i] == "new" and words[i + 1] == "room":
            # Save any previous data before changing modes
            if current_mode == "comment" and buffer and current_feature:
                results.append({
                    "Room": current_room,
                    "Attribute": current_attribute,
                    "Feature": current_feature,
                    "Comment": buffer.strip()
                })
                print(f"Added comment: {buffer.strip()}")
                buffer = ""

            # Change to room collection mode
            current_mode = "room"
            buffer = ""  # Reset buffer for new room
            i += 2  # Skip "new room"
            continue

        elif i < len(words) - 1 and words[i] == "new" and words[i + 1] == "attribute":
            # Save room data if changing from room mode
            if current_mode == "room" and buffer:
                current_room = buffer.strip()
                print(f"Set current room to: {current_room}")

            # Change to attribute collection mode
            current_mode = "attribute"
            buffer = ""  # Reset buffer for new attribute
            i += 2  # Skip "new attribute"
            continue

        elif i < len(words) - 1 and words[i] == "new" and words[i + 1] == "feature":
            # Save attribute data if changing from attribute mode
            if current_mode == "attribute" and buffer:
                # Normalize and validate attribute
                normalized_buffer = buffer.strip().lower()
                if any(attr == normalized_buffer for attr in valid_attributes):
                    current_attribute = normalized_buffer.title()  # Capitalize properly
                    print(f"Set current attribute to: {current_attribute}")
                else:
                    closest_match = min(valid_attributes, key=lambda x:
                    sum(1 for a, b in zip(x, normalized_buffer) if a != b))
                    current_attribute = closest_match.title()  # Use closest match
                    print(f"Invalid attribute '{normalized_buffer}', using closest match: {current_attribute}")

            # Change to feature collection mode
            current_mode = "feature"
            current_feature = ""  # Reset feature for new one
            buffer = ""  # Reset buffer for new feature
            i += 2  # Skip "new feature"
            continue

        elif words[i] == "comment":
            # Save feature data if changing from feature mode
            if current_mode == "feature" and buffer:
                current_feature = buffer.strip()
                print(f"Set current feature to: {current_feature}")

            # If we're already in comment mode, this means it's a new comment
            # Save the previous comment first
            if current_mode == "comment" and buffer and current_feature:
                results.append({
                    "Room": current_room,
                    "Attribute": current_attribute,
                    "Feature": current_feature,
                    "Comment": buffer.strip()
                })
                print(f"Added comment: {buffer.strip()}")

            # Change to comment collection mode
            current_mode = "comment"
            buffer = ""  # Reset buffer for new comment
            i += 1  # Skip "comment"
            continue

        # Add current word to the buffer if we're in a collection mode
        if current_mode:
            buffer += words[i] + " "

        # Move to next word
        i += 1

    # Handle any remaining data in the buffer
    if current_mode == "comment" and buffer and current_feature:
        results.append({
            "Room": current_room,
            "Attribute": current_attribute,
            "Feature": current_feature,
            "Comment": buffer.strip()
        })
        print(f"Added final comment: {buffer.strip()}")

    print(f"\nProcessed {len(results)} entries")
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
                    print("\nExtracted room, attribute, feature, and comments:")
                    print(df)
                else:
                    print(f"No data found in {audio_file.name}")
            else:
                print("No transcript was generated for this file")

        except Exception as e:
            print(f"Error processing {audio_file.name}: {str(e)}")
            continue


if __name__ == "__main__":
    main()