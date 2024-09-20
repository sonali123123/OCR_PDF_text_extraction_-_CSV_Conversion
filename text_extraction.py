import os  # For file and directory operations
import cv2  # For image processing
import numpy as np  # For array operations
import tensorflow as tf  # For machine learning operations
import pytesseract  # For OCR
import imutils  # For image manipulation
from pdf2image import convert_from_path  # For converting PDF pages to images
from paddleocr import PaddleOCR  # For OCR using PaddleOCR
import pandas as pd  # For data manipulation
import csv  # For CSV file operations
import streamlit as st  # For web application

# Initialize PaddleOCR with English language
ocr = PaddleOCR(lang='en')



# Function to convert PDF to images
def pdf_to_images(pdf_path, output_dir, dpi=500):
    pages = convert_from_path(pdf_path, dpi)  # Convert PDF to images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist
    image_paths = []  # List to store paths of images
    image_counter = 1  # Counter for naming images
    for page in pages:
        filename = os.path.join(output_dir, f"page_{image_counter}.jpg")  # Define filename for each image
        page.save(filename, "JPEG")  # Save image
        image_paths.append(filename)  # Add image path to the list
        image_counter += 1  # Increment counter
    return image_paths  # Return list of image paths


# Function to rotate image if needed and save it
def rotate_and_save_image_if_needed(image_path, output_path):
    image = cv2.imread(image_path)  # Read image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB
    results = pytesseract.image_to_osd(rgb, output_type=pytesseract.Output.DICT)  # Detect orientation and script
    rotation_angle = int(results["rotate"])  # Get rotation angle

    if rotation_angle != 0:
        rotated = imutils.rotate_bound(image, angle=rotation_angle)  # Rotate image
        cv2.imwrite(output_path, rotated)  # Save rotated image
        print(f"Rotated image saved to: {output_path}")  # Print message
    else:
        cv2.imwrite(output_path, image)  # Save image without rotation
        print(f"Image was already in correct orientation, saved without rotation to: {output_path}")  # Print message


# Function to perform OCR on image and save result to CSV
def ocr_to_csv(image_path, csv_file_path):
    image_cv = cv2.imread(image_path)  # Read image
    if image_cv is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")  # Raise error if image not found
    image_height = image_cv.shape[0]  # Get image height
    image_width = image_cv.shape[1]  # Get image width
    output = ocr.ocr(image_path)[0]  # Perform OCR
    boxes = [line[0] for line in output]  # Extract bounding boxes
    texts = [line[1][0] for line in output]  # Extract texts
    probabilities = [line[1][1] for line in output]  # Extract probabilities

    image_boxes = image_cv.copy()  # Copy image for drawing boxes
    for box, text in zip(boxes, texts):
        cv2.rectangle(image_boxes, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), (int(box[2][1]))), (0, 0, 255), 1)  # Draw rectangle
        cv2.putText(image_boxes, text, (int(box[0][0]), int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (222, 0, 0), 1)  # Draw text

    horiz_boxes = []  # List to store horizontal boxes
    vert_boxes = []  # List to store vertical boxes
    for box in boxes:
        x_h, x_v = 0, int(box[0][0])
        y_h, y_v = int(box[0][1]), 0
        width_h, width_v = image_width, int(box[2][0] - box[0][0])
        height_h, height_v = int(box[2][1] - box[0][1]), image_height
        horiz_boxes.append([x_h, y_h, x_h + width_h, y_h + height_h])  # Add horizontal box
        vert_boxes.append([x_v, y_v, x_v + width_v, y_v + height_v])  # Add vertical box

    horiz_out = tf.image.non_max_suppression(
        horiz_boxes,
        probabilities,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )
    horiz_lines = np.sort(np.array(horiz_out))  # Sort horizontal lines
    vert_out = tf.image.non_max_suppression(
        vert_boxes,
        probabilities,
        max_output_size=1000,
        iou_threshold=0.1,
        score_threshold=float('-inf'),
        name=None
    )
    vert_lines = np.sort(np.array(vert_out))  # Sort vertical lines

    out_array = [["" for _ in range(len(vert_lines))] for _ in range(len(horiz_lines))]  # Initialize output array
    unordered_boxes = []
    for i in vert_lines:
        unordered_boxes.append(vert_boxes[i][0])
    ordered_boxes = np.argsort(unordered_boxes)  # Order boxes

    def intersection(box_1, box_2):
        return [box_2[0], box_1[1], box_2[2], box_1[3]]  # Define intersection of boxes

    def iou(box_1, box_2):
        x_1 = max(box_1[0], box_2[0])
        y_1 = max(box_1[1], box_2[1])
        x_2 = min(box_1[2], box_2[2])
        y_2 = min(box_1[3], box_2[3])
        inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1, 0)))  # Calculate intersection area
        if inter == 0:
            return 0
        box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))  # Calculate area of box 1
        box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))  # Calculate area of box 2
        return inter / float(box_1_area + box_2_area - inter)  # Calculate Intersection over Union

    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]])  # Get resultant box
            for b in range(len(boxes)):
                the_box = [boxes[b][0][0], boxes[b][0][1], boxes[b][2][0], boxes[b][2][1]]
                if iou(resultant, the_box) > 0.1:
                    out_array[i][j] = texts[b]  # Add text to output array

    out_array = np.array(out_array)  # Convert to numpy array
    out_array = align_columns(out_array)  # Align columns
    out_array = remove_incomplete_duplicates(out_array)  # Remove incomplete duplicates
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)  # Write to CSV file
        writer.writerows(out_array)
    print(f"Data successfully saved to {csv_file_path}")  # Print message

# Function to align columns in the array
def align_columns(out_array):
    df = pd.DataFrame(out_array)  # Convert to DataFrame
    for i in range(1, df.shape[1]):
        for j in range(df.shape[0]):
            if df.iloc[j, i] and not df.iloc[j, i - 1]:
                df.iloc[j, i - 1] = df.iloc[j, i]
                df.iloc[j, i] = ""  # Align columns
    return df.values  # Return aligned array

# Function to remove incomplete duplicates
def remove_incomplete_duplicates(out_array):
    df = pd.DataFrame(out_array)  # Convert to DataFrame
    df_cleaned = df.copy()  # Copy DataFrame
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df_cleaned = df_cleaned.drop(columns=[col])  # Drop incomplete columns
    return df_cleaned.values  # Return cleaned array

# Function to process DataFrame
def process_dataframe(df):
    def remove_duplicates_in_row(row):
        return row.drop_duplicates(keep='first')  # Remove duplicates in a row
    df_processed = df.apply(remove_duplicates_in_row, axis=1)  # Apply to each row
    return df_processed  # Return processed DataFrame

# Function to convert CSV to XLSX
def csv_to_xlsx(csv_file_path, xlsx_file_path):
    df = pd.read_csv(csv_file_path, header=None)  # Read CSV file
    df = process_dataframe(df)  # Process DataFrame
    df.to_excel(xlsx_file_path, index=False, header=False)  # Write to XLSX file
    print(f"CSV data successfully converted to XLSX format at {xlsx_file_path}")  # Print message

# Special cleaning task function
def special_cleaning_task(csv_file_path):
    df = pd.read_csv(csv_file_path, header=None)  # Read CSV file
    df.columns = ["0", "1", "2", "3", "4", "5"]  # Set column names
    df = df.drop(df.index[15:19])  # Drop specific rows
    df = df.drop(df.columns[[4, 5]], axis=1)  # Drop specific columns
    cleaned_csv_path = csv_file_path.replace(".csv", "_cleaned.csv")  # Define cleaned CSV path
    df.to_csv(cleaned_csv_path, index=False)  # Save cleaned CSV
    print(f"Special cleaned CSV saved at {cleaned_csv_path}")  # Print message
    return cleaned_csv_path  # Return cleaned CSV path

# Streamlit web application title
st.title("PDF to CSV and XLSX Converter")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    pdf_path = os.path.join("uploads", uploaded_file.name)  # Define PDF path
    output_dir = "output_images"  # Define output directory
    os.makedirs("uploads", exist_ok=True)  # Create uploads directory if not exists

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Save uploaded file

    st.success("File uploaded successfully")  # Show success message

    # Convert PDF to images
    image_paths = pdf_to_images(pdf_path, output_dir)  # Get image paths
    st.write(f"Saved images: {image_paths}")  # Display image paths

    csv_file_paths = []
    for image_path in image_paths:
        rotated_image_path = image_path.replace(".jpg", "_rotated.jpg")  # Define rotated image path
        rotate_and_save_image_if_needed(image_path, rotated_image_path)  # Rotate and save image
        csv_file_path = rotated_image_path.replace(".jpg", ".csv")  # Define CSV file path
        ocr_to_csv(rotated_image_path, csv_file_path)  # Perform OCR and save to CSV
        csv_file_paths.append(csv_file_path)  # Add CSV file path to the list
        st.write(f"CSV file saved: {csv_file_path}")  # Display CSV file path

    # Convert CSV to XLSX
    xlsx_file_path = "output.xlsx"  # Define XLSX file path
    with pd.ExcelWriter(xlsx_file_path) as writer:
        for csv_file_path in csv_file_paths:
            df = pd.read_csv(csv_file_path, header=None)  # Read CSV file
            df = process_dataframe(df)  # Process DataFrame
            sheet_name = os.path.basename(csv_file_path).replace(".csv", "")  # Define sheet name
            df.to_excel(writer, index=False, header=False, sheet_name=sheet_name)  # Write to XLSX
    st.write(f"Data successfully saved to {xlsx_file_path}")  # Display XLSX file path

    st.success("PDF conversion to CSV and XLSX completed.")  # Show success message
    st.download_button(
        label="Download XLSX file",
        data=open(xlsx_file_path, "rb").read(),  # Read XLSX file
        file_name="output.xlsx",  # Define file name
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # Define MIME type
    )  # Add download button
