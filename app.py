from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
import os
import speech_recognition as sr
from pathlib import Path
from pydub import AudioSegment
import pandas as pd
import uuid
import cv2
import numpy as np
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Add this configuration
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Create directories for uploads and transcripts
UPLOAD_FOLDER = Path('temp_uploads')
TRANSCRIPT_FOLDER = Path('temp_transcripts')
TRANSCRIPT_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER = Path('static/results')
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(exist_ok=True, parents=True)


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
        # Serve the index.html page as the root
        return render_template('index.html')
    except Exception as e:
        return f"Error: {str(e)}"

# Add a separate route for the transcription page
@app.route('/transcribe')
def transcribe():
    try:
        return render_template('transcription_page.html')
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


@app.route('/diff_check')
def diff_check():
    """Render the image difference checker page"""
    return render_template('diff_check.html')


@app.route('/diff_result/<result_id>')
def diff_result(result_id):
    """Render the results page for image comparison"""
    return render_template('diff_result.html')


@app.route('/api/diff_result/<result_id>')
def get_diff_result(result_id):
    """API endpoint to get the image comparison results"""
    # Check if result exists
    original_path = RESULT_FOLDER / f"{result_id}_original.jpg"
    diff_path = RESULT_FOLDER / f"{result_id}_diff.jpg"

    if not original_path.exists() or not diff_path.exists():
        return jsonify({
            'status': 'error',
            'message': 'Result not found'
        })

    # Read the metadata file for difference count
    metadata_path = RESULT_FOLDER / f"{result_id}_metadata.txt"
    difference_count = 0
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = f.read().strip()
            try:
                difference_count = int(metadata)
            except:
                difference_count = 0

    # Return paths and metadata
    return jsonify({
        'status': 'success',
        'originalImageUrl': f"/static/results/{result_id}_original.jpg",
        'diffImageUrl': f"/static/results/{result_id}_diff.jpg",
        'differenceCount': difference_count
    })


@app.route('/compare_images', methods=['POST'])
def compare_images():
    """Endpoint to handle image comparison uploads and processing"""
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'Both images are required'
        })

    file1 = request.files['image1']
    file2 = request.files['image2']

    if file1.filename == '' or file2.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        })

    # Generate a unique ID for this comparison
    result_id = str(uuid.uuid4())

    # Save uploaded files temporarily
    temp_path1 = UPLOAD_FOLDER / f"{result_id}_1{Path(secure_filename(file1.filename)).suffix}"
    temp_path2 = UPLOAD_FOLDER / f"{result_id}_2{Path(secure_filename(file2.filename)).suffix}"

    file1.save(temp_path1)
    file2.save(temp_path2)

    # Process images and find differences
    try:
        difference_count = process_image_difference(
            str(temp_path1),
            str(temp_path2),
            str(RESULT_FOLDER / f"{result_id}_original.jpg"),
            str(RESULT_FOLDER / f"{result_id}_diff.jpg")
        )

        # Save metadata
        with open(RESULT_FOLDER / f"{result_id}_metadata.txt", 'w') as f:
            f.write(str(difference_count))

        # Clean up temporary files
        temp_path1.unlink(missing_ok=True)
        temp_path2.unlink(missing_ok=True)

        return jsonify({
            'status': 'success',
            'resultId': result_id,
            'message': 'Images compared successfully'
        })

    except Exception as e:
        # Clean up temporary files
        temp_path1.unlink(missing_ok=True)
        temp_path2.unlink(missing_ok=True)

        return jsonify({
            'status': 'error',
            'message': f'Error processing images: {str(e)}'
        })


def process_image_difference(image1_path, image2_path, output_original_path, output_diff_path):
    """
    Process two images to find and highlight differences

    Args:
        image1_path: Path to the reference image
        image2_path: Path to the comparison image
        output_original_path: Path where to save the original image
        output_diff_path: Path where to save the diff image with highlighted differences

    Returns:
        The number of differences found
    """
    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Check if images were loaded correctly
    if img1 is None or img2 is None:
        raise ValueError("Failed to load one or both images")

    # Resize images to match if they have different dimensions
    if img1.shape != img2.shape:
        # Resize the second image to match the first
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between grayscale images
    diff = cv2.absdiff(gray1, gray2)

    # Apply threshold to highlight differences
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours of differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size (remove very small differences that might be noise)
    min_area = 50  # Minimum area in pixels
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Create a copy of the second image to draw the differences on
    img_diff = img2.copy()

    # Draw rectangles around detected differences
    for contour in significant_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_diff, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the images
    cv2.imwrite(output_original_path, img1)
    cv2.imwrite(output_diff_path, img_diff)

    return len(significant_contours)

if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible from other devices
    print("Starting PhillyScript server...")
    print("Available routes:")
    print("  - http://127.0.0.1:5001/ (main interface)")
    print("  - http://127.0.0.1:5001/test (server test)")
    print("  - http://127.0.0.1:5001/basic (basic HTML test)")
    app.run(debug=True, host='0.0.0.0', port=5001)