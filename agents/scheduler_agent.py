from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils.config import Config
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class SchedulerAgent:
    """Agent for scheduling interviews and sending invitations"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.7,
            openai_api_key=Config.OPENAI_API_KEY
        )
    
    def generate_interview_invite(self, candidate_name: str, candidate_email: str,
                                 job_title: str, company: str, 
                                 match_score: float, strengths: List[str]) -> Dict:
        """Generate personalized interview invitation"""
        
        invite_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional HR coordinator. Create a warm, personalized 
interview invitation email that:
1. Congratulates the candidate on being shortlisted
2. Mentions 1-2 specific strengths that stood out
3. Provides interview details
4. Maintains professional yet friendly tone

Keep it concise (150-200 words)."""),
            ("human", """Create an interview invitation for:

Candidate: {candidate_name}
Email: {candidate_email}
Position: {job_title}
Company: {company}
Match Score: {match_score}%
Key Strengths: {strengths}

Generate the email body (no subject line):""")
        ])
        
        chain = invite_prompt | self.llm
        response = chain.invoke({
            "candidate_name": candidate_name,
            "candidate_email": candidate_email,
            "job_title": job_title,
            "company": company,
            "match_score": match_score,
            "strengths": ", ".join(strengths[:2])
        })
        
        email_body = response.content
        
        return {
            "to": candidate_email,
            "subject": f"Interview Invitation - {job_title} at {company}",
            "body": email_body,
            "status": "generated"
        }
    
    def send_email(self, to_email: str, subject: str, body: str) -> bool:
        """Send email invitation (demo implementation)"""
        
        # In production, use actual SMTP settings
        # For demo, we'll just simulate sending
        
        try:
            if not Config.EMAIL_PASSWORD:
                # Demo mode - just log
                print(f"\n{'='*60}")
                print(f"DEMO: Email would be sent to: {to_email}")
                print(f"Subject: {subject}")
                print(f"Body:\n{body}")
                print(f"{'='*60}\n")
                return True
            
            # Actual email sending (if credentials provided)
            msg = MIMEMultipart()
            msg['From'] = Config.EMAIL_FROM
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT) as server:
                server.starttls()
                server.login(Config.EMAIL_FROM, Config.EMAIL_PASSWORD)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False
    
    def schedule_interviews(self, shortlisted_candidates: List[Dict], 
                          job_title: str, company: str) -> List[Dict]:
        """Generate and send interview invitations for shortlisted candidates"""
        results = []
        
        for candidate in shortlisted_candidates:
            invite = self.generate_interview_invite(
                candidate_name=candidate['name'],
                candidate_email=candidate['email'],
                job_title=job_title,
                company=company,
                match_score=candidate['match_score'],
                strengths=candidate['strengths']
            )
            
            # Send email
            sent = self.send_email(invite['to'], invite['subject'], invite['body'])
            
            results.append({
                **candidate,
                "invite_sent": sent,
                "invite_subject": invite['subject'],
                "invite_body": invite['body'],
                "sent_at": datetime.utcnow().isoformat() if sent else None
            })
        
        return results


# Global instance
scheduler_agent = SchedulerAgent()