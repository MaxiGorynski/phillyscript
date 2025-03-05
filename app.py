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
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import difflib
import os
from werkzeug.utils import secure_filename
import uuid
import re
from docx import Document
from io import BytesIO
from datetime import datetime
from flask_login import LoginManager, login_required, current_user
from models import db, User
from auth import auth as auth_blueprint

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-goes-here'  # Change this to a random secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

#Initialise Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register the auth blueprint
app.register_blueprint(auth_blueprint)

# Create all database tables
with app.app_context():
    db.create_all()

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


def transcribe_audio_optimized(audio_path):
    """
    Optimized version of transcribe_audio function with improved file path handling.
    """
    try:
        print(f"Converting {audio_path} to WAV format...")
        audio_path = Path(audio_path)

        # Verify file exists
        if not audio_path.exists():
            print(f"Error: File does not exist: {audio_path}")
            return ""

        # Skip conversion for WAV files
        if audio_path.suffix.lower() == '.wav':
            wav_path = str(audio_path)
        else:
            # Create a temporary WAV file with optimized parameters
            wav_path = audio_path.with_suffix('.wav')

            # Convert audio to WAV with optimized parameters
            try:
                if audio_path.suffix.lower() == '.mp3':
                    audio = AudioSegment.from_mp3(str(audio_path))
                elif audio_path.suffix.lower() == '.flac':
                    audio = AudioSegment.from_file(str(audio_path), format="flac")
                elif audio_path.suffix.lower() == '.m4a':
                    audio = AudioSegment.from_file(str(audio_path), format="m4a")
                else:
                    audio = AudioSegment.from_file(str(audio_path))

                # Optimize audio for speech recognition
                # Downsampling to 16kHz, mono, 16-bit which is optimal for most speech recognition
                audio = audio.set_frame_rate(16000).set_channels(1)

                # Export with optimized settings
                audio.export(wav_path, format="wav", parameters=["-q:a", "0"])
                print(f"Successfully converted to {wav_path}")
            except Exception as e:
                print(f"Error converting audio: {e}")
                return ""

        # Verify WAV file exists and has content
        wav_path_obj = Path(wav_path)
        if not wav_path_obj.exists():
            print(f"Error: WAV file does not exist: {wav_path}")
            return ""

        if wav_path_obj.stat().st_size == 0:
            print(f"Error: WAV file is empty: {wav_path}")
            return ""

        print("Starting transcription...")
        recognizer = sr.Recognizer()

        # Set recognition parameters for better performance
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8

        with sr.AudioFile(str(wav_path)) as source:
            print("Reading audio file...")
            # Adjust for ambient noise to improve accuracy
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Record the audio with optimized parameters
            audio = recognizer.record(source)

            print("Sending to speech recognition service...")
            transcript = recognizer.recognize_google(audio, language="en-US")
            print(f"Transcription received. Length: {len(transcript)} characters")
            print(f"Transcript: {transcript[:200]}..." if len(transcript) > 200 else f"Transcript: {transcript}")
            return transcript.lower()
            return transcript.lower()
    except sr.UnknownValueError:
        print(f"Could not understand audio in {audio_path}")
        return ""
    except sr.RequestError as e:
        print(f"Error with speech recognition service for {audio_path}; {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error processing audio: {e}")
        return ""
    finally:
        # Clean up temporary WAV file if it was converted
        try:
            if 'wav_path' in locals() and wav_path != str(audio_path) and Path(wav_path).exists():
                Path(wav_path).unlink(missing_ok=True)
                print("Cleaned up temporary WAV file")
        except Exception as e:
            print(f"Error cleaning up temporary file: {e}")


def format_text(text):
    """
    Apply advanced formatting rules to make transcribed text more professional.
    Uses heuristics to detect sentence boundaries and format accordingly.

    Args:
        text: Raw text string from transcription

    Returns:
        Formatted text with proper capitalization and punctuation
    """
    if not text:
        return text

    # Preprocessing - clean up common transcription artifacts
    text = text.strip()

    # Replace multiple spaces with a single space
    text = ' '.join(text.split())

    # Sentence boundary detection
    # Add periods where they're likely missing between sentences

    # Common indicators of sentence boundaries in speech
    boundary_indicators = [
        ' and then ', ' afterwards ', ' after that ', ' next ', ' following that ',
        ' subsequently ', ' consequently ', ' as a result ', ' therefore ',
        ' thus ', ' hence ', ' so ', ' finally ', ' lastly ', ' in conclusion ',
        ' to sum up ', ' in summary ', ' additionally ', ' moreover ', ' furthermore ',
        ' in addition ', ' also ', ' besides ', ' however ', ' nevertheless ',
        ' nonetheless ', ' still ', ' yet ', ' conversely ', ' on the other hand ',
        ' on the contrary ', ' in contrast ', ' instead ', ' alternatively ',
        ' otherwise ', ' rather ', ' whereas ', ' while ', ' though ', ' although '
    ]

    # Add periods before these indicators if there's no punctuation
    for indicator in boundary_indicators:
        text = text.replace(indicator, f'. {indicator.strip()}')

    # Replace common conjunctions at the start of a sentence that might indicate a boundary
    conjunctions = [' but ', ' or ', ' nor ', ' for ', ' so ']
    for conj in conjunctions:
        text = text.replace(conj, f'. {conj.strip()} ')

    # Split text into sentences based on existing punctuation and the indicators we added
    # This regex will split on ., !, ? followed by a space or end of string
    import re
    sentences = re.split(r'([.!?])\s+', text)

    # Rejoin sentences with proper spacing and formatting
    formatted_text = ''
    i = 0
    while i < len(sentences):
        if i < len(sentences) - 1 and sentences[i + 1] in '.!?':
            # This is a sentence + its ending punctuation
            sentence = sentences[i]
            punct = sentences[i + 1]

            # Capitalize first letter of sentence if it's not already
            if sentence and sentence[0].islower():
                sentence = sentence[0].upper() + sentence[1:]

            formatted_text += sentence + punct + ' '
            i += 2
        else:
            # This might be the last sentence or one without punctuation
            sentence = sentences[i]

            # Capitalize first letter of sentence if it's not already
            if sentence and sentence[0].islower():
                sentence = sentence[0].upper() + sentence[1:]

            # Add period if there's no ending punctuation
            if sentence and not sentence.endswith(('.', '!', '?')):
                sentence += '.'

            formatted_text += sentence + ' '
            i += 1

    # Final cleanup and formatting
    formatted_text = formatted_text.strip()

    # Fix common capitalization issues
    formatted_text = re.sub(r'\bi\b', 'I', formatted_text)  # Capitalize standalone 'i'
    formatted_text = re.sub(r'\bi\'m\b', 'I\'m', formatted_text, flags=re.IGNORECASE)
    formatted_text = re.sub(r'\bi\'ll\b', 'I\'ll', formatted_text, flags=re.IGNORECASE)
    formatted_text = re.sub(r'\bi\'ve\b', 'I\'ve', formatted_text, flags=re.IGNORECASE)
    formatted_text = re.sub(r'\bi\'d\b', 'I\'d', formatted_text, flags=re.IGNORECASE)

    # Capitalize proper nouns (common ones in your context)
    proper_nouns = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                    'september', 'october', 'november', 'december']

    for noun in proper_nouns:
        formatted_text = re.sub(r'\b' + noun + r'\b', noun.capitalize(),
                                formatted_text, flags=re.IGNORECASE)

    # Fix spacing around punctuation
    formatted_text = re.sub(r'\s+([.,;:!?])', r'\1', formatted_text)

    # Ensure consistent spacing after punctuation
    formatted_text = re.sub(r'([.,;:!?])(\S)', r'\1 \2', formatted_text)

    # Remove double periods
    formatted_text = formatted_text.replace('..', '.')

    # Ensure the text ends with proper punctuation
    if not formatted_text[-1] in '.!?':
        formatted_text += '.'

    return formatted_text


def process_transcript(transcript):
    """
    Process transcript text to extract features and comments.
    Returns a list of dictionaries containing features and comments.
    Also detects 'Tenant Responsibility' mentions in comments.
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
            # Apply text formatting to the feature
            current_feature = format_text(current_feature)
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

            # Check for "Tenant Responsibility" in the comment
            is_tenant_responsibility = False
            # Convert to lowercase for case-insensitive matching
            comment_lower = comment_text.lower()
            if "tenant responsibility" in comment_lower:
                is_tenant_responsibility = True
                print("Tenant Responsibility detected in comment")

            # Apply text formatting to the comment
            comment_text = format_text(comment_text)
            print(f"Found comment: {comment_text}")

            # Add to results if we have a feature
            if current_feature:
                results.append({
                    "Feature": current_feature if is_first_comment else " ",  # Empty space for subsequent comments
                    "Comment": comment_text,
                    "Tenant Responsibility (TR)": "✓" if is_tenant_responsibility else ""
                })
                print(f"Added row - Feature: {'[First]' if is_first_comment else '[Subsequent]'}, " +
                      f"Comment: {comment_text}, TR: {'Yes' if is_tenant_responsibility else 'No'}")
                is_first_comment = False

        else:
            i += 1

    print(f"\nProcessed {len(results)} feature-comment pairs")
    return results


def process_audio_file(file_path, original_filename):
    """Process audio file and create a CSV."""
    # Transcribe the audio
    transcript = transcribe_audio_optimized(file_path)

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
@login_required
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


# Update the process_file route to use multithreading
@app.route('/process/<process_id>', methods=['POST'])
def process_file(process_id):
    # Find the uploaded file
    uploaded_files = list(UPLOAD_FOLDER.glob(f"{process_id}_*"))

    if not uploaded_files:
        return jsonify({'status': 'error', 'message': 'No file found for this process ID'})

    file_path = uploaded_files[0]
    original_filename = file_path.name.replace(f"{process_id}_", "")

    try:
        # Use a thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Use the optimized transcription function
            transcript_future = executor.submit(transcribe_audio_optimized, str(file_path))

            # Wait for transcription to complete
            transcript = transcript_future.result()

            if not transcript:
                return jsonify({'status': 'error', 'message': 'Failed to transcribe audio'})

            # Process the transcript
            results = process_transcript(transcript)

            if not results:
                return jsonify({'status': 'error', 'message': 'No features found in the transcript'})

            # Create DataFrame
            df = pd.DataFrame(results)

            # Generate output filename based on input audio filename
            base_filename = Path(original_filename).stem
            output_filename = f"{base_filename}_transcript.csv"
            output_path = TRANSCRIPT_FOLDER / output_filename

            # Save to CSV
            df.to_csv(output_path, index=False)

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
@login_required
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


@app.route('/finalise_report')
@login_required
def finalise_report():
    """Render the finalise report page"""
    return render_template('finalise_report.html')


@app.route('/compare_text', methods=['POST'])
@login_required
def compare_text():
    """Endpoint to handle text comparison uploads and processing"""
    if 'original' not in request.files or 'comparison' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'Both files are required'
        })

    original_file = request.files['original']
    comparison_file = request.files['comparison']

    if original_file.filename == '' or comparison_file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        })

    # Generate a unique ID for this comparison
    result_id = str(uuid.uuid4())

    # Save uploaded files temporarily
    original_path = UPLOAD_FOLDER / f"{result_id}_original{Path(secure_filename(original_file.filename)).suffix}"
    comparison_path = UPLOAD_FOLDER / f"{result_id}_comparison{Path(secure_filename(comparison_file.filename)).suffix}"

    original_file.save(original_path)
    comparison_file.save(comparison_path)

    # Process files and find differences
    try:
        # Extract text from files
        original_text = extract_text_from_file(original_path)
        comparison_text = extract_text_from_file(comparison_path)

        if not original_text or not comparison_text:
            raise ValueError("Could not extract text from one or both files")

        # Compare texts and generate HTML with differences highlighted
        result_html = generate_diff_html(original_text, comparison_text)

        # Save the result
        result_path = RESULT_FOLDER / f"{result_id}_diff.html"
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(result_html)

        # Clean up temporary files
        original_path.unlink(missing_ok=True)
        comparison_path.unlink(missing_ok=True)

        return jsonify({
            'status': 'success',
            'resultUrl': f"/view_comparison/{result_id}",  # Changed URL to point to a new route
            'message': 'Files compared successfully'
        })

    except Exception as e:
        # Clean up temporary files
        original_path.unlink(missing_ok=True)
        comparison_path.unlink(missing_ok=True)

        return jsonify({
            'status': 'error',
            'message': f'Error processing files: {str(e)}'
        })


@app.route('/view_comparison/<result_id>')
def view_comparison(result_id):
    """Serve the comparison result directly"""
    result_path = RESULT_FOLDER / f"{result_id}_diff.html"

    if not result_path.exists():
        return "Comparison result not found", 404

    with open(result_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    return html_content

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report():
    """Endpoint to handle report generation from CSV"""
    use_latest = request.form.get('useLatest') == 'true'

    try:
        csv_path = None

        if use_latest:
            # Find the most recent CSV file in the transcripts folder
            transcript_dir = Path('temp_transcripts')
            if not transcript_dir.exists():
                return jsonify({
                    'status': 'error',
                    'message': 'No transcript directory found'
                })

            csv_files = list(transcript_dir.glob('*.csv'))
            if not csv_files:
                return jsonify({
                    'status': 'error',
                    'message': 'No CSV files found in the transcript directory'
                })

            # Sort by modification time, newest first
            csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            csv_path = csv_files[0]
        else:
            if 'csv' not in request.files:
                return jsonify({
                    'status': 'error',
                    'message': 'No CSV file provided'
                })

            csv_file = request.files['csv']
            if csv_file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'No file selected'
                })

            # Generate a unique ID for this report
            result_id = str(uuid.uuid4())

            # Save uploaded file temporarily
            csv_path = UPLOAD_FOLDER / f"{result_id}_{secure_filename(csv_file.filename)}"
            csv_file.save(csv_path)

        # Generate report from CSV
        result_path = generate_docx_report(csv_path)

        # Clean up temporary file if it was uploaded
        if not use_latest:
            csv_path.unlink(missing_ok=True)

        return jsonify({
            'status': 'success',
            'resultUrl': f"/download_report/{result_path.name}",
            'message': 'Report generated successfully'
        })

    except Exception as e:
        # Clean up temporary file if it exists and was uploaded
        if csv_path and not use_latest and csv_path.exists():
            csv_path.unlink(missing_ok=True)

        return jsonify({
            'status': 'error',
            'message': f'Error generating report: {str(e)}'
        })


@app.route('/download_report/<filename>')
def download_report(filename):
    """Serve the generated report for download"""
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)


def extract_text_from_file(file_path):
    """Extract text content from various file formats"""
    file_ext = file_path.suffix.lower()

    try:
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        elif file_ext in ['.doc', '.docx']:
            # Use the properly imported Document class
            from docx import Document
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])

        elif file_ext == '.pdf':
            # Note: This would require PyPDF2 or another PDF library
            # For simplicity, we'll just return an error for now
            raise ValueError("PDF extraction not implemented")

        elif file_ext == '.rtf':
            # RTF parsing is complex, you'd need a library like striprtf
            raise ValueError("RTF extraction not implemented")

        else:
            # Try to read as plain text by default
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    except Exception as e:
        raise ValueError(f"Could not extract text from file: {str(e)}")


def generate_diff_html(original_text, comparison_text):
    """
    Generate HTML highlighting differences between two texts
    """
    # Split texts into lines
    original_lines = original_text.splitlines()
    comparison_lines = comparison_text.splitlines()

    # Get differences
    differ = difflib.HtmlDiff(wrapcolumn=80)
    diff_html = differ.make_file(original_lines, comparison_lines,
                                 'Original Text', 'Comparison Text',
                                 context=True, numlines=3)

    # Enhance the HTML with our own styling
    styled_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Comparison Results</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 20px;
                line-height: 1.5;
            }}
            .diff-header {{
                background-color: #4a6fa5;
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .diff-header h1 {{
                margin: 0;
                font-size: 24px;
            }}
            table.diff {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
                font-family: monospace;
            }}
            .diff td {{
                padding: 5px;
                border: 1px solid #ddd;
                vertical-align: top;
            }}
            .diff span.diff_add {{
                background-color: #e6ffe6;
                color: green;
                font-weight: bold;
            }}
            .diff span.diff_sub {{
                background-color: #ffe6e6;
                color: red;
                text-decoration: line-through;
            }}
            .diff span.diff_chg {{
                background-color: #fff5cc;
                color: #996600;
                font-weight: bold;
            }}
            .diff th {{
                background-color: #f0f0f0;
                padding: 5px;
                border: 1px solid #ddd;
                text-align: center;
            }}
            .legend {{
                margin-top: 20px;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }}
            .legend ul {{
                padding-left: 20px;
            }}
            .back-button {{
                padding: 8px 15px;
                background-color: #4a6fa5;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
            }}
            .back-button:hover {{
                background-color: #375d8a;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="diff-header">
            <h1>Text Comparison Results</h1>
            <button onclick="goBack()" class="back-button">Back to Report Options</button>
        </div>

        {diff_html}

        <div class="legend">
            <h3>Legend:</h3>
            <ul>
                <li><span style="color: green; font-weight: bold;">Added content</span> - Content that appears in the comparison text but not in the original</li>
                <li><span style="color: red; text-decoration: line-through;">Removed content</span> - Content that appears in the original text but not in the comparison</li>
                <li><span style="color: #996600; font-weight: bold;">Changed content</span> - Content that has been modified between the texts</li>
            </ul>
        </div>

        <div class="footer">
            <button onclick="goBack()" class="back-button">Back to Report Options</button>
        </div>

        <script>
        function goBack() {{
            window.location.href = window.location.origin + '/finalise_report';
        }}
        </script>
    </body>
    </html>
    """

    return styled_html


def generate_docx_report(csv_path):
    """
    Generate a formatted Word document from CSV data
    """
    import pandas as pd
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from datetime import datetime  # Add this import inside the function as well for extra safety

    # Read CSV data
    df = pd.read_csv(csv_path)

    # Create a new Document
    doc = Document()

    # Set document properties
    doc.core_properties.title = "Property Inspection Report"
    doc.core_properties.author = "PhillyScript"

    # Add a title
    title = doc.add_heading('Property Inspection Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add date
    date_paragraph = doc.add_paragraph()
    date_run = date_paragraph.add_run(f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
    date_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add a line break
    doc.add_paragraph()

    # Add an introduction
    doc.add_heading('Introduction', level=1)
    intro_text = ("This report documents the condition of the property and identifies any issues "
                  "that require attention. Items marked with 'TR' indicate tenant responsibility.")
    doc.add_paragraph(intro_text)

    # Add a line break
    doc.add_paragraph()

    # Add section for features and comments
    doc.add_heading('Inspection Details', level=1)

    # Track current feature for grouping comments
    current_feature = None

    # Process each row in the CSV
    for index, row in df.iterrows():
        feature = row['Feature'].strip() if 'Feature' in row and pd.notna(row['Feature']) and row[
            'Feature'].strip() else None
        comment = row['Comment'].strip() if 'Comment' in row and pd.notna(row['Comment']) else ""

        # Check for tenant responsibility
        is_tenant_responsibility = False
        if 'Tenant Responsibility (TR)' in row and pd.notna(row['Tenant Responsibility (TR)']):
            is_tenant_responsibility = bool(row['Tenant Responsibility (TR)'])

        # If this is a new feature, add a subheading
        if feature and feature != " ":
            current_feature = feature
            doc.add_heading(feature, level=2)

        # Add the comment as a paragraph
        if comment:
            p = doc.add_paragraph()
            p.style = 'List Bullet'
            comment_run = p.add_run(comment)

            # If tenant responsibility, add indicator and style
            if is_tenant_responsibility:
                tr_run = p.add_run(" [TR]")
                tr_run.bold = True
                tr_run.font.color.rgb = RGBColor(255, 0, 0)  # Red color

    # Add summary section
    doc.add_heading('Summary', level=1)
    doc.add_paragraph("This report provides a comprehensive overview of the property's condition. "
                      "Please address any issues identified in this report promptly.")

    # Save the document
    result_id = str(uuid.uuid4())
    result_path = RESULT_FOLDER / f"{result_id}_inspection_report.docx"
    doc.save(result_path)

    return result_path

if __name__ == '__main__':
    # Determine optimal number of threads
    num_threads = min(multiprocessing.cpu_count() * 2, 8)  # Use 2x CPU cores, max 8

    print(f"Starting PhillyScript server with {num_threads} worker threads...")
    print("Available routes:")
    print("  - http://127.0.0.1:5001/ (main interface)")
    print("  - http://127.0.0.1:5001/transcribe (transcription page)")
    print("  - http://127.0.0.1:5001/diff_check (image comparison)")

    # Set threaded mode and optimal worker count
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)