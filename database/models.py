from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class JobDescription(Base):
    """Job Description model"""
    __tablename__ = "job_descriptions"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    company = Column(String(200))
    description = Column(Text, nullable=False)
    requirements = Column(Text)
    summary = Column(Text)  # AI-generated summary
    file_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    candidates = relationship("Candidate", back_populates="job")
    
    def __repr__(self):
        return f"<JobDescription(id={self.id}, title='{self.title}')>"


class Candidate(Base):
    """Candidate model"""
    __tablename__ = "candidates"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), nullable=False)
    phone = Column(String(50))
    cv_text = Column(Text, nullable=False)
    file_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    job_id = Column(Integer, ForeignKey("job_descriptions.id"))
    
    # Relationships
    job = relationship("JobDescription", back_populates="candidates")
    match_result = relationship("MatchResult", back_populates="candidate", uselist=False)
    
    def __repr__(self):
        return f"<Candidate(id={self.id}, name='{self.name}')>"


class MatchResult(Base):
    """Match result between candidate and job"""
    __tablename__ = "match_results"
    
    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), unique=True)
    match_score = Column(Float, nullable=False)
    strengths = Column(Text)  # JSON string
    gaps = Column(Text)  # JSON string
    reasoning = Column(Text)
    is_shortlisted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    candidate = relationship("Candidate", back_populates="match_result")
    interview = relationship("Interview", back_populates="match_result", uselist=False)
    
    def __repr__(self):
        return f"<MatchResult(candidate_id={self.candidate_id}, score={self.match_score})>"


class Interview(Base):
    """Interview scheduling information"""
    __tablename__ = "interviews"
    
    id = Column(Integer, primary_key=True)
    match_result_id = Column(Integer, ForeignKey("match_results.id"), unique=True)
    scheduled_date = Column(DateTime)
    status = Column(String(50), default="pending")  # pending, invited, confirmed, completed
    invite_sent = Column(Boolean, default=False)
    invite_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    match_result = relationship("MatchResult", back_populates="interview")
    
    def __repr__(self):
        return f"<Interview(id={self.id}, status='{self.status}')>"
