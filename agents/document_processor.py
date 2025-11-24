from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils.config import Config
from utils.pdf_parser import doc_parser


class DocumentProcessorAgent:
    """Agent for processing and chunking documents"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_job_description(self, file_path: str) -> Dict:
        """Process and analyze job description"""
        # Extract text
        text = doc_parser.extract_text(file_path)
        
        # Generate summary using LLM
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert recruiter. Analyze the job description and provide:
1. A concise summary (2-3 sentences)
2. Key required skills and qualifications
3. Nice-to-have skills
4. Role level (Entry/Mid/Senior)

Format as JSON with keys: summary, required_skills, preferred_skills, level"""),
            ("human", "Job Description:\n{text}")
        ])
        
        chain = summary_prompt | self.llm
        response = chain.invoke({"text": text})
        
        # Chunk the document
        chunks = self.text_splitter.split_text(text)
        
        return {
            "text": text,
            "chunks": chunks,
            "analysis": response.content,
            "num_chunks": len(chunks)
        }
    
    def process_cv(self, file_path: str) -> Dict:
        """Process and analyze CV"""
        # Extract text
        text = doc_parser.extract_text(file_path)
        
        # Extract basic info
        info = doc_parser.extract_cv_info(text)
        
        # Generate summary using LLM
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert recruiter. Analyze the CV and provide:
1. Candidate's key skills and expertise
2. Years of experience
3. Education background
4. Notable achievements

Format as JSON with keys: skills, experience_years, education, achievements"""),
            ("human", "CV Text:\n{text}")
        ])
        
        chain = summary_prompt | self.llm
        response = chain.invoke({"text": text})
        
        # Chunk the document
        chunks = self.text_splitter.split_text(text)
        
        return {
            "text": text,
            "chunks": chunks,
            "info": info,
            "analysis": response.content,
            "num_chunks": len(chunks)
        }
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into smaller pieces"""
        return self.text_splitter.split_text(text)


# Global instance
doc_processor = DocumentProcessorAgent()