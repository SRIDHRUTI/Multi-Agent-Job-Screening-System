import PyPDF2
import docx
from pathlib import Path
from typing import Dict, Optional
import re


class DocumentParser:
    """Parse PDF and DOCX documents"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """Extract text based on file extension"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == '.pdf':
            return DocumentParser.extract_text_from_pdf(file_path)
        elif extension in ['.docx', '.doc']:
            return DocumentParser.extract_text_from_docx(file_path)
        elif extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    @staticmethod
    def extract_cv_info(cv_text: str) -> Dict[str, Optional[str]]:
        """Extract basic info from CV text"""
        info = {
            "name": None,
            "email": None,
            "phone": None
        }
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, cv_text)
        if email_match:
            info["email"] = email_match.group(0)
        
        # Extract phone
        phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}'
        phone_match = re.search(phone_pattern, cv_text)
        if phone_match:
            info["phone"] = phone_match.group(0)
        
        # Try to extract name (first few words before email/contact)
        lines = cv_text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line.split()) <= 4 and not any(char.isdigit() for char in line):
                info["name"] = line
                break
        
        return info


# Global instance
doc_parser = DocumentParser()