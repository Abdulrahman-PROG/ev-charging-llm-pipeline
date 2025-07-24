#!/usr/bin/env python3
"""
Test script for the complete EV data collection pipeline
Including both PDF processing and web scraping
"""

# Simple imports - no Spark dependencies
import pdfplumber
import pandas as pd
import os
import json
import re
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime

# Create output folder for all generated files
output_folder = "output_data"
os.makedirs(output_folder, exist_ok=True)
print(f"Output folder created: {output_folder}")

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def process_pdfs(pdf_folder):
    """Process all PDFs in a folder"""
    data = []
    
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Processing PDF: {filename}")
            
            text = extract_pdf_text(pdf_path)
            if text:
                data.append({
                    'source': 'pdf',
                    'filename': filename,
                    'text': text,
                    'length': len(text),
                    'timestamp': datetime.now().isoformat()
                })
    
    return data

def scrape_webpage(url):
    """Scrape text content from a webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def is_relevant_content(text, url):
    """Check if content is relevant to EV charging"""
    ev_keywords = [
        'charging station', 'electric vehicle', 'ev charging', 'chargepoint',
        'supercharger', 'fast charging', 'charging network', 'charging infrastructure'
    ]
    
    text_lower = text.lower()
    url_lower = url.lower()
    
    # Check for keywords in text
    keyword_count = sum(1 for keyword in ev_keywords if keyword in text_lower)
    
    # Check for keywords in URL
    url_keyword_count = sum(1 for keyword in ev_keywords if keyword.replace(' ', '') in url_lower)
    
    # Content is relevant if it has keywords or URL indicates relevance
    return keyword_count >= 1 or url_keyword_count >= 1

def scrape_websites(urls):
    """Scrape multiple websites for EV charging content"""
    data = []
    
    for url in urls:
        print(f"Scraping: {url}")
        text = scrape_webpage(url)
        
        if text and len(text) > 200 and is_relevant_content(text, url):
            data.append({
                'source': 'web',
                'url': url,
                'text': text,
                'length': len(text),
                'timestamp': datetime.now().isoformat()
            })
            print(f"  ✓ Collected {len(text):,} characters")
        else:
            print(f"  ✗ Skipped (not relevant or too short)")
        
        # Be respectful to websites
        time.sleep(2)
    
    return data

def split_text(text, chunk_size=500):
    """Split text into smaller chunks"""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_qa_pairs(text_chunks, source_info):
    """Generate simple Q&A pairs from text chunks"""
    qa_pairs = []
    
    for chunk in text_chunks:
        if len(chunk) > 100:  # Only use substantial chunks
            # Simple question generation based on content
            if "charging station" in chunk.lower():
                question = "What information is available about charging stations?"
                qa_pairs.append({
                    "instruction": question,
                    "input": "",
                    "output": chunk,
                    "source": source_info
                })
            
            if "electric vehicle" in chunk.lower():
                question = "Tell me about electric vehicles and charging."
                qa_pairs.append({
                    "instruction": question,
                    "input": "",
                    "output": chunk,
                    "source": source_info
                })
            
            if "fast charging" in chunk.lower() or "dc charging" in chunk.lower():
                question = "How does fast charging work for electric vehicles?"
                qa_pairs.append({
                    "instruction": question,
                    "input": "",
                    "output": chunk,
                    "source": source_info
                })
    
    return qa_pairs

if __name__ == "__main__":
    # Define URLs for web scraping
    ev_urls = [
        "https://afdc.energy.gov/fuels/electricity_charging_home.html",
        "https://afdc.energy.gov/fuels/electricity_charging_public.html",
        "https://www.energy.gov/eere/electricvehicles/charging-home",
        "https://www.energy.gov/eere/electricvehicles/electric-vehicle-charging-infrastructure"
    ]

    # Collect data from PDFs
    print("=== Processing PDF Files ===")
    pdf_folder = "pdfs"
    pdf_data = process_pdfs(pdf_folder)

    # Collect data from websites
    print("\n=== Web Scraping ===")
    web_data = scrape_websites(ev_urls)

    # Combine all data
    all_data = pdf_data + web_data

    # Display summary
    print(f"\n=== Collection Summary ===")
    print(f"PDF files processed: {len(pdf_data)}")
    print(f"Web pages scraped: {len(web_data)}")
    print(f"Total data sources: {len(all_data)}")

    for item in all_data:
        if item['source'] == 'pdf':
            print(f"- PDF: {item['filename']}: {item['length']:,} characters")
        else:
            print(f"- Web: {item['url']}: {item['length']:,} characters")

    # Save raw data to CSV in output folder
    df = pd.DataFrame(all_data)
    csv_path = os.path.join(output_folder, 'ev_raw_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nRaw data saved to {csv_path}")

    # Display data summary
    print("\nData summary:")
    print(df.groupby('source').agg({
        'length': ['count', 'sum', 'mean']
    }).round(0))

    # Generate training data from all sources
    print("\n=== Generating Training Data ===")
    all_qa_pairs = []

    for item in all_data:
        source_info = item['filename'] if item['source'] == 'pdf' else item['url']
        print(f"Processing: {source_info}")
        
        # Split text into chunks
        chunks = split_text(item['text'])
        
        # Generate Q&A pairs
        qa_pairs = generate_qa_pairs(chunks, source_info)
        all_qa_pairs.extend(qa_pairs)
        
        print(f"  Generated {len(qa_pairs)} Q&A pairs")

    print(f"\nTotal training examples: {len(all_qa_pairs)}")

    # Save training data to JSON in output folder
    training_path = os.path.join(output_folder, 'ev_training_data.json')
    with open(training_path, 'w', encoding='utf-8') as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_qa_pairs)} training examples to {training_path}")

    # Save training data in Alpaca format (without source info for training)
    alpaca_data = [{
        "instruction": item["instruction"],
        "input": item["input"],
        "output": item["output"]
    } for item in all_qa_pairs]

    alpaca_path = os.path.join(output_folder, 'ev_training_alpaca.json')
    with open(alpaca_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    print(f"Saved Alpaca format training data to {alpaca_path}")

    # Show first example
    if all_qa_pairs:
        print(f"\nFirst example:")
        print(f"Question: {all_qa_pairs[0]['instruction']}")
        print(f"Answer: {all_qa_pairs[0]['output'][:200]}...")
        print(f"Source: {all_qa_pairs[0]['source']}")

    print(f"\n=== Pipeline Complete ===")
    print(f"All output files saved in: {output_folder}/")
    print(f"- ev_raw_data.csv: Raw extracted data")
    print(f"- ev_training_data.json: Training data with source info")
    print(f"- ev_training_alpaca.json: Training data in Alpaca format")

