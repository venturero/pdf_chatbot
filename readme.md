# MultiPDF Chat App

> This project is cloned from [Ask Multiple PDFs](https://github.com/alejandro-ao/ask-multiple-pdfs) and enhanced to support PDFs with images for question and answer tasks. Future improvements include refining HTML styling for better readability and adding functionality for Excel and Word files. Entrepreneurship sample files will also be uploaded to this repository for demonstration purposes.
### Updates
> In version 1.2 Excel files can also be uploaded for Q&A and visual is better with changes in HTML.
> In the future some files can be uploaded by default.
> Conversation can be only about Entrepreneurship subject.
> Chat histor might be improved.
> Users can get answer to their questions without selecting or processing files. 

## Introduction
The MultiPDF Chat App is a Python application that lets you interact with multiple PDF documents. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the documents' content, leveraging a language model to deliver precise answers. Note that the application will only respond to questions pertinent to the loaded PDFs.

## How It Works

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

The application processes your questions through the following steps:

1. **PDF Loading**: Reads multiple PDF documents and extracts text and image content.
2. **Text Chunking**: Divides extracted text and images into manageable chunks for processing.
3. **Language Model**: Utilizes a language model to generate vector representations (embeddings) of the chunks.
4. **Similarity Matching**: Compares your question to the chunks and identifies the most semantically similar ones.
5. **Response Generation**: Passes selected chunks to the language model, which generates a response based on relevant PDF content.

## Dependencies and Installation

To install the MultiPDF Chat App, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
