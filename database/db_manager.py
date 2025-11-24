from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional
import json
from contextlib import contextmanager

from .models import Base, JobDescription, Candidate, MatchResult, Interview
from utils.config import Config


class DatabaseManager:
    """Manages database operations"""
    
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URL, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    # Job Description Operations
    def create_job_description(self, title: str, company: str, description: str, 
                              requirements: str, summary: str, file_path: str) -> JobDescription:
        """Create a new job description"""
        with self.get_session() as session:
            jd = JobDescription(
                title=title,
                company=company,
                description=description,
                requirements=requirements,
                summary=summary,
                file_path=file_path
            )
            session.add(jd)
            session.flush()
            session.refresh(jd)
            return jd
    
    def get_job_description(self, job_id: int) -> Optional[JobDescription]:
        """Get job description by ID"""
        with self.get_session() as session:
            return session.query(JobDescription).filter_by(id=job_id).first()
    
    def get_all_job_descriptions(self) -> List[JobDescription]:
        """Get all job descriptions"""
        with self.get_session() as session:
            return session.query(JobDescription).all()
    
    # Candidate Operations
    def create_candidate(self, name: str, email: str, phone: str, cv_text: str, 
                        file_path: str, job_id: int) -> Candidate:
        """Create a new candidate"""
        with self.get_session() as session:
            candidate = Candidate(
                name=name,
                email=email,
                phone=phone,
                cv_text=cv_text,
                file_path=file_path,
                job_id=job_id
            )
            session.add(candidate)
            session.flush()
            session.refresh(candidate)
            return candidate
    
    def get_candidates_for_job(self, job_id: int) -> List[Candidate]:
        """Get all candidates for a job"""
        with self.get_session() as session:
            return session.query(Candidate).filter_by(job_id=job_id).all()
    
    # Match Result Operations
    def create_match_result(self, candidate_id: int, match_score: float, 
                           strengths: List[str], gaps: List[str], 
                           reasoning: str, is_shortlisted: bool) -> MatchResult:
        """Create a match result"""
        with self.get_session() as session:
            match = MatchResult(
                candidate_id=candidate_id,
                match_score=match_score,
                strengths=json.dumps(strengths),
                gaps=json.dumps(gaps),
                reasoning=reasoning,
                is_shortlisted=is_shortlisted
            )
            session.add(match)
            session.flush()
            session.refresh(match)
            return match
    
    def get_shortlisted_candidates(self, job_id: int) -> List[tuple]:
        """Get shortlisted candidates with their match results"""
        with self.get_session() as session:
            results = session.query(Candidate, MatchResult).join(
                MatchResult, Candidate.id == MatchResult.candidate_id
            ).filter(
                Candidate.job_id == job_id,
                MatchResult.is_shortlisted == True
            ).order_by(MatchResult.match_score.desc()).all()
            
            return results
    
    # Interview Operations
    def create_interview(self, match_result_id: int, invite_message: str) -> Interview:
        """Create an interview record"""
        with self.get_session() as session:
            interview = Interview(
                match_result_id=match_result_id,
                invite_message=invite_message,
                invite_sent=True,
                status="invited"
            )
            session.add(interview)
            session.flush()
            session.refresh(interview)
            return interview
    
    def get_pending_interviews(self, job_id: int) -> List[tuple]:
        """Get pending interviews for a job"""
        with self.get_session() as session:
            results = session.query(Candidate, Interview).join(
                MatchResult, Candidate.id == MatchResult.candidate_id
            ).join(
                Interview, MatchResult.id == Interview.match_result_id
            ).filter(
                Candidate.job_id == job_id,
                Interview.status == "invited"
            ).all()
            
            return results


# Global instance
db_manager = DatabaseManager()