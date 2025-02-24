from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
import os
import speech_recognition as sr
from pathlib import Path
from pydub import AudioSegment
import pandas as pd
import uuid

app = Flask(__name__)

# Add this configuration
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Create directories for uploads and transcripts
UPLOAD_FOLDER = Path('temp_uploads')
TRANSCRIPT_FOLDER = Path('temp_transcripts')
UPLOAD_FOLDER.mkdir(exist_ok=True)
TRANSCRIPT_FOLDER.mkdir(exist_ok=True)


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

    # Convert audio to WAV based on file extension
    file_ext = audio_path.suffix.lower()

    try:
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(str(audio_path))
        elif file_ext == '.flac':
            audio = AudioSegment.from_file(str(audio_path), format="flac")
        elif file_ext == '.m4a':
            audio = AudioSegment.from_file(str(audio_path), format="m4a")
        elif file_ext == '.aac':
            audio = AudioSegment.from_file(str(audio_path), format="aac")
        else:
            # Try to automatically detect format
            audio = AudioSegment.from_file(str(audio_path))

        audio.export(wav_path, format="wav")
        print(f"Successfully converted to {wav_path}")
        return str(wav_path)
    except Exception as e:
        print(f"Error converting {audio_path}: {str(e)}")
        # If conversion fails, try to return the original path
        # Speech recognition might still work for some formats
        return str(audio_path)


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
    current_feature = None
    is_first_comment = True

    # Split transcript into words for easier processing
    words = transcript.split()
    i = 0

    while i < len(words):
        # Check for "new feature"
        if i < len(words) - 1 and words[i] == "new" and words[i + 1] == "feature":
            # Start collecting new feature
            current_feature = ""
            is_first_comment = True
            i += 2  # Skip "new feature"

            # Collect feature text until "comment" or another "new feature"
            while i < len(words):
                if words[i] == "comment":
                    break
                if i < len(words) - 1 and words[i] == "new" and words[i + 1] == "feature":
                    i -= 1  # Back up to process new feature
                    break
                current_feature += words[i] + " "
                i += 1

            current_feature = current_feature.strip()
            print(f"Found new feature: {current_feature}")

        # Check for "comment"
        elif words[i] == "comment":
            i += 1  # Skip "comment"
            comment_text = ""

            # Collect comment text until another "comment" or "new feature"
            while i < len(words):
                if words[i] == "comment":
                    break
                if i < len(words) - 1 and words[i] == "new" and words[i + 1] == "feature":
                    i -= 1  # Back up to process new feature
                    break
                comment_text += words[i] + " "
                i += 1

            comment_text = comment_text.strip()
            print(f"Found comment: {comment_text}")

            # Add to results if we have a feature
            if current_feature:
                results.append({
                    "Feature": current_feature if is_first_comment else " ",  # Empty space for subsequent comments
                    "Comment": comment_text
                })
                print(
                    f"Added row - Feature: {'[First]' if is_first_comment else '[Subsequent]'}, Comment: {comment_text}")
                is_first_comment = False

        else:
            i += 1

    print(f"\nProcessed {len(results)} feature-comment pairs")
    return results


def process_audio_file(file_path, original_filename):
    """Process audio file and create a CSV."""
    # Transcribe the audio
    transcript = transcribe_audio(file_path)

    if not transcript:
        return None, "Failed to transcribe audio"

    # Process the transcript
    results = process_transcript(transcript)

    if not results:
        return None, "No features found in the transcript"

    # Create DataFrame
    df = pd.DataFrame(results)

    # Generate output filename based on input audio filename
    base_filename = Path(original_filename).stem
    output_filename = f"{base_filename}_transcript.csv"
    output_path = TRANSCRIPT_FOLDER / output_filename

    # Save to CSV
    df.to_csv(output_path, index=False)

    return output_path, None


# Add a basic test route
@app.route('/test')
def test():
    return "Server is working!"


@app.route('/basic')
def basic():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PhillyScript Test</title>
        <style>
            body { font-family: sans-serif; padding: 20px; }
            h1 { color: #4a6fa5; }
        </style>
    </head>
    <body>
        <h1>PhillyScript - Test Page</h1>
        <p>If you can see this page, your Flask server is working correctly.</p>
    </body>
    </html>
    """


@app.route('/')
def index():
    try:
        # Try both options
        if os.path.exists('templates/transcription_page.html'):
            return render_template('transcription_page.html')
        elif os.path.exists('static/transcription_page.html'):
            return send_from_directory('static', 'transcription_page.html')
        else:
            # Return a basic HTML if template is not found
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>PhillyScript</title>
                <style>
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 20px;
                        background-color: #f5f5f5;
                        color: #333;
                    }
                    .container {
                        max-width: 800px;
                        margin: 2rem auto;
                        padding: 2rem;
                        background: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    h1 {
                        color: #4a6fa5;
                    }
                    .drop-area {
                        border: 3px dashed #4a6fa5;
                        border-radius: 12px;
                        padding: 4rem 2rem;
                        text-align: center;
                        margin: 2rem 0;
                        cursor: pointer;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>PhillyScript</h1>
                    <p>Welcome to PhillyScript - Your audio feature extractor</p>

                    <div class="drop-area" id="drop-area">
                        <h2>Drag & Drop MP3 File</h2>
                        <p>Or click to select file</p>
                        <input type="file" id="fileInput" accept=".mp3" style="display: none;">
                    </div>

                    <p>Note: Template file not found. This is a basic fallback interface.</p>
                </div>

                <script>
                    // Basic drag-and-drop functionality
                    const dropArea = document.getElementById('drop-area');
                    const fileInput = document.getElementById('fileInput');

                    dropArea.addEventListener('click', () => fileInput.click());
                    dropArea.addEventListener('dragover', (e) => {
                        e.preventDefault();
                        dropArea.style.backgroundColor = '#f0f0f0';
                    });
                    dropArea.addEventListener('dragleave', () => {
                        dropArea.style.backgroundColor = '';
                    });
                    dropArea.addEventListener('drop', (e) => {
                        e.preventDefault();
                        dropArea.style.backgroundColor = '';
                        if (e.dataTransfer.files.length) {
                            alert('File upload functionality is enabled in the full version');
                        }
                    });
                </script>
            </body>
            </html>
            """
    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audioFile' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})

    file = request.files['audioFile']

    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})

    # Check if file has an allowed extension
    allowed_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac']
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        return jsonify({
            'status': 'error',
            'message': 'Unsupported file type. Please upload an MP3, WAV, FLAC, M4A, or AAC file.'
        })

    # Generate a unique ID for this upload
    process_id = str(uuid.uuid4())

    # Create temp file path
    temp_file_path = UPLOAD_FOLDER / f"{process_id}_{file.filename}"

    # Save the file
    file.save(temp_file_path)

    # Return process ID so client can check progress
    return jsonify({
        'status': 'success',
        'processId': process_id,
        'message': 'File uploaded successfully. Processing started.'
    })


@app.route('/process/<process_id>', methods=['POST'])
def process_file(process_id):
    # Find the uploaded file
    uploaded_files = list(UPLOAD_FOLDER.glob(f"{process_id}_*"))

    if not uploaded_files:
        return jsonify({'status': 'error', 'message': 'No file found for this process ID'})

    file_path = uploaded_files[0]
    original_filename = file_path.name.replace(f"{process_id}_", "")

    try:
        # Process the file
        output_path, error = process_audio_file(file_path, original_filename)

        if error:
            return jsonify({'status': 'error', 'message': error})

        # Return success with the output filename
        return jsonify({
            'status': 'success',
            'outputFilename': output_path.name,
            'message': 'Processing complete'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    finally:
        # Clean up uploaded file
        try:
            file_path.unlink(missing_ok=True)
        except:
            pass


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = TRANSCRIPT_FOLDER / filename
    if not file_path.exists():
        return jsonify({'status': 'error', 'message': 'File not found'})

    # Return the file for download
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible from other devices
    print("Starting PhillyScript server...")
    print("Available routes:")
    print("  - http://127.0.0.1:5001/ (main interface)")
    print("  - http://127.0.0.1:5001/test (server test)")
    print("  - http://127.0.0.1:5001/basic (basic HTML test)")
    app.run(debug=True, host='0.0.0.0', port=5001)