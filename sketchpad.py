# Add these imports at the top of your application.py
from io import BytesIO
import logging
import shutil


# Update your route for generating enhanced reports
@application.route('/api/generate_enhanced_report', methods=['POST'])
@login_required
def generate_enhanced_report():
    """API endpoint to generate a final report with property details and room images"""
    try:
        # Get form data
        report_type = request.form.get('reportType', 'full')
        csv_id = request.form.get('csvId')
        address = request.form.get('address')
        inspection_date = request.form.get('inspectionDate')
        on_behalf_of = request.form.get('onBehalfOf')

        if not csv_id or not address or not inspection_date or not on_behalf_of:
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields'
            })

        # Find the CSV file
        csv_path = None
        if csv_id == 'latest':
            # Find the most recent CSV file
            transcript_dir = Path('/tmp/temp_transcripts')
            if not transcript_dir.exists():
                return jsonify({
                    'status': 'error',
                    'message': 'No transcript directory found'
                })

            csv_files = list(transcript_dir.glob('*.csv'))
            if not csv_files:
                return jsonify({
                    'status': 'error',
                    'message': 'No CSV files found'
                })

            # Sort by modification time, newest first
            csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            csv_path = csv_files[0]
        else:
            # Find the specific CSV file
            csv_path = UPLOAD_FOLDER / f"{csv_id}.csv"
            if not csv_path.exists():
                return jsonify({
                    'status': 'error',
                    'message': 'CSV file not found'
                })

        # Process uploaded images for each room
        # Process uploaded images for each room
        room_images = {}

        # Debug: print all keys in request.files
        logging.info(f"Form keys: {list(request.form.keys())}")
        logging.info(f"File keys: {list(request.files.keys())}")

        # Check for image files in the request
        for key in request.files.keys():
            if key.startswith('roomImages['):
                # Extract room name from the input name format: roomImages[Room Name]
                room_name = key[11:-1].lower()  # Extract what's between 'roomImages[' and ']'
                logging.info(f"Processing images for room: {room_name}")

                # Initialize room in the dictionary if needed
                if room_name not in room_images:
                    room_images[room_name] = []

                # Get all files for this room
                files = request.files.getlist(key)
                logging.info(f"Number of files for {room_name}: {len(files)}")

                # Save each image file
                for file in files:
                    if file and file.filename:
                        # Generate a unique filename
                        img_id = str(uuid.uuid4())
                        img_ext = Path(secure_filename(file.filename)).suffix
                        img_path = UPLOAD_FOLDER / f"{img_id}{img_ext}"

                        # Save the image
                        file.save(img_path)
                        room_images[room_name].append(str(img_path))

                        # Log the save
                        logging.info(f"Saved image for {room_name}: {img_path} (from {file.filename})")

        # Debug log the final image count per room
        for room, images in room_images.items():
            logging.info(f"Room {room}: {len(images)} images collected")

        # Generate the enhanced report
        result_path = generate_enhanced_docx_report(
            csv_path,
            report_type,
            address,
            inspection_date,
            on_behalf_of,
            room_images
        )

        # Create a filename based on address and date
        formatted_date = datetime.strptime(inspection_date, '%Y-%m-%d').strftime('%d-%m-%Y')
        report_name = f"{address.replace(' ', '_')}_{formatted_date}_{report_type}_report.docx"

        return jsonify({
            'status': 'success',
            'reportUrl': f"/download_report/{result_path.name}",
            'reportName': report_name,
            'message': f'{report_type.capitalize()} report generated successfully'
        })

    except Exception as e:
        logging.error(f"Error generating report: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Error generating report: {str(e)}'
        })


# Update the room image handling section in your generate_enhanced_docx_report function

# Inside your generate_enhanced_docx_report function, replace the room image section with this:

# Add room images if available
if room in room_images and room_images[room]:
    # Add subheading for images
    img_heading = doc.add_heading('Room Images', level=2)

    # Count the images for this room
    num_images = len(room_images[room])

    # Determine appropriate layout based on number of images
    images_per_row = 2  # Default to 2 images per row
    if num_images <= 3:
        images_per_row = 1  # Single column for 1-3 images (larger images)
    elif num_images <= 6:
        images_per_row = 2  # Two columns for 4-6 images
    else:
        images_per_row = 3  # Three columns for 7+ images (smaller images)

    # Calculate image width based on number per row
    image_width = Inches(7.0 / images_per_row)  # Adjust for page width (assuming 8.5" with margins)

    # Add images in rows
    for i in range(0, num_images, images_per_row):
        # Create a table for this row of images
        img_table = doc.add_table(rows=1, cols=images_per_row)
        img_table.alignment = WD_TABLE_ALIGNMENT.CENTER

        # Remove cell borders for better appearance
        for cell in img_table.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(1)  # Minimal text size

            # Set cell borders to white (effectively invisible)
            for border in ['top', 'right', 'bottom', 'left']:
                setattr(cell._element.tcPr, f'tc{border.capitalize()}', None)

        # Add images to the table cells
        for j in range(images_per_row):
            if i + j < num_images:
                img_path = room_images[room][i + j]
                cell = img_table.cell(0, j)

                try:
                    # Center content in cell
                    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

                    # Try to add the image to the cell
                    img_run = cell.paragraphs[0].add_run()

                    # Add the image with calculated width
                    img_run.add_picture(img_path, width=image_width)

                    # Extract just the filename for caption
                    img_filename = os.path.basename(img_path)

                    # Add a small caption under the image
                    caption = cell.add_paragraph(f"Image {i + j + 1}")
                    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    caption.style = 'Caption'

                    # Make caption smaller and gray
                    for run in caption.runs:
                        run.font.size = Pt(8)
                        run.font.color.rgb = RGBColor(100, 100, 100)

                except Exception as e:
                    # If adding the image fails, add a placeholder text
                    cell.text = f"[Image {i + j + 1} could not be displayed]"
                    logging.error(f"Error adding image to report: {str(e)}")

        # Add some space after the table
        doc.add_paragraph()