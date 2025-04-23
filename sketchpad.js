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
                    caption = cell.add_paragraph(f"Image {i+j+1}")
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