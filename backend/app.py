from fastapi import FastAPI, UploadFile, File, HTTPException
import google.generativeai as genai
import PyPDF2
import os
import fitz  # PyMuPDF
import uvicorn
import os
import base64
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import json
from typing import Dict, Any
# from Database.db import Database

# Initialize FastAPI app
app = FastAPI(
    title="Bill Processing API",
    description="API for processing bills from PDF and extracting details using Gemini AI"
)

# Set up logging configuration
logging.basicConfig(
    filename=f'bill_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def pdf_to_images(pdf_path, output_folder, zoom=3):
    """
    Converts each page of a PDF to an image and saves them in a folder.

    Args:
        pdf_path (str): Path to the input PDF file
        output_folder (str): Path to the output folder for images
        zoom (int): Zoom factor to increase image resolution (default: 3)
    """
    try:
        logging.info(f"Starting PDF to image conversion for file: {pdf_path}")
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Open the PDF file
        pdf_document = fitz.open(pdf_path)

        print(f"Processing {len(pdf_document)} pages...")
        logging.info(f"Processing {len(pdf_document)} pages")

        # Iterate over each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)

            # Create matrix for higher resolution (zoom factor)
            mat = fitz.Matrix(zoom, zoom)

            # Get pixel map for the page
            pix = page.get_pixmap(matrix=mat)

            # Create output path
            output_path = os.path.join(output_folder, f"page_{page_num + 1}.jpg")

            # Save the image
            pix.save(output_path, "jpeg")  # Quality 0-100
            logging.info(f"Processed and saved page {page_num + 1} to {output_path}")

        pdf_document.close()
        logging.info("PDF to image conversion completed successfully")

    except Exception as e:
        logging.error(f"Error in pdf_to_images: {str(e)}")
        print(f"Error: {e}")

def extract_bill_details_gemini(output_folder):
    load_dotenv()
    try:
        logging.info("Starting bill details extraction using Gemini")
        
        # Initialize list to store results
        all_results = []
        
        # Configure Gemini API
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            logging.error("GOOGLE_API_KEY environment variable not set")
            raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
            
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logging.info("Gemini API configured successfully")

        prompt = """From the following context: fill the given template fields.
            Template : What is the ["Date",
                "Seller",
                "Seller Gst No.",
                "Client",
                "Invoice Number / Bill No.",
                "Transport",
                "LR. No.",
                "Clinet Gst No.",
                "City Name / Area of the Client",
                "Items or Desciption of Goods Details",
                "Quantity"
                "Rate of Goods",
                "Less / Discount"
                "Total Gst Amount",
                "Total Amount"]?
            Add all the goods that are present in the table with comma seperator. All add the rate of goods with comma seperator.
            The output should be in json foramt.
            Provide only the field name and its answer, if the answer is not present in the context answer "NAN" for that field."""

        # Get all image files from the output folder
        image_files = [f for f in os.listdir(output_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images = len(image_files)
        logging.info(f"Found {total_images} images to process")

        for image_file in sorted(image_files, key=lambda x: int(x.split('_')[1].split('.')[0])):
            image_path = os.path.join(output_folder, image_file)
            page_num = int(image_file.split('_')[1].split('.')[0])  # Extract page number from filename
            logging.info(f"Processing {image_file} (Page {page_num})")

            try:
                # Read and process image in chunks to handle large files
                with open(image_path, 'rb') as image_file:
                    image_content = image_file.read()
                
                # Process with Gemini
                response = model.generate_content([
                    {
                        'mime_type': 'image/jpeg', 
                        'data': base64.b64encode(image_content).decode('utf-8')
                    }, 
                    prompt
                ])
                logging.info(f"Successfully processed page {page_num}")
                
                # Parse JSON response and add page number
                try:
                    cleaned_response = response.text.replace("```json", "").replace("```", "").strip()
                    result_dict = json.loads(cleaned_response)
                    result_dict['Page_Number'] = page_num
                    all_results.append(result_dict)
                    logging.info(f"Successfully parsed JSON for page {page_num}")
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON for page {page_num}: {str(e)}")
                    continue
                
                # Add adaptive sleep based on API limits
                time.sleep(2)  # Reduced sleep time for better performance

            except Exception as e:
                logging.error(f"Error processing {image_file}: {str(e)}")
                continue

        if not all_results:
            logging.warning("No results were successfully processed")
            return pd.DataFrame()

        # Create DataFrame from all results
        df = pd.DataFrame(all_results)
        
        # Save DataFrame to CSV with timestamp
        output_file = f"bill_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")
        
        return [df, output_file]

    except Exception as e:
        logging.error(f"Error in extract_bill_details_gemini: {str(e)}")
        raise

@app.post("/extract_bill_data", response_model=Dict[str, Any])
async def extract_bill_data(pdf_file: UploadFile = File(...)):
    """
    API endpoint to process bills from PDF and extract details
    Returns JSON with extracted bill details
    """
    try:
        logging.info("="*50)
        logging.info("Starting bill processing request")
        logging.info(f"Request file: {pdf_file.filename}")
        
        if not pdf_file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
            
        # Save the uploaded PDF temporarily
        temp_pdf_path = f"temp_{int(time.time())}.pdf"
        output_folder = f"pdf_images_{int(time.time())}"
        
        # Save uploaded file
        content = await pdf_file.read()
        with open(temp_pdf_path, "wb") as f:
            f.write(content)
        logging.info(f"PDF saved temporarily at: {temp_pdf_path}")
        
        # Convert PDF to images
        pdf_to_images(temp_pdf_path, output_folder)
        logging.info("PDF to images conversion completed")
        
        # Extract bill details
        df, output_file = extract_bill_details_gemini(output_folder)
        
        logging.info(f"Bill details extracted. DataFrame shape: {df.shape}")
        
        # Clean up temporary files
        os.remove(temp_pdf_path)
        files_removed = 0
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
            files_removed += 1
        os.rmdir(output_folder)
        logging.info(f"Cleanup completed. Removed {files_removed} files and directory")
        
        # Convert DataFrame to JSON
        result = df.to_dict(orient='records')
        
        # Initialize database connection
        # db = Database()
        
        # Insert the data into Supabase
        # db.insert_multiple_bills(result)
        
        logging.info("Successfully processed bill and inserted into database")
        return {"success": True, "data": result, "output_file": output_file}
        
    except Exception as e:
        error_msg = f"Error processing bill: {str(e)}"
        logging.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)



"""
if __name__ == "__main__":
    try:
        logging.info("Starting bill processing application")
        
        # Configuration
        pdf_path = "D:/Automation/contents/Scanned_Bills.pdf"
        output_folder = "pdf_images"
        zoom_factor = 3
        
        logging.info(f"Configuration - PDF Path: {pdf_path}, Output Folder: {output_folder}, Zoom Factor: {zoom_factor}")

        # Run the conversion
        pdf_to_images(pdf_path, output_folder, zoom_factor)
        df = extract_bill_details_gemini()
        print("DataFrame shape:", df.shape)
        logging.info("Application completed successfully")

    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        raise
"""