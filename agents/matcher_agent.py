from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils.config import Config
from agents.embedding_agent import embedding_agent
import json


class MatcherAgent:
    """Agent for matching candidates to job descriptions"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.3,
            openai_api_key=Config.OPENAI_API_KEY
        )
    
    def calculate_match_score(self, candidate_id: int, candidate_name: str,
                             jd_summary: str, job_id: int) -> Dict:
        """Calculate match score between candidate and job"""
        
        # Get CV context from ChromaDB
        cv_context = embedding_agent.get_cv_context(candidate_id)
        
        # Get relevant JD chunks for comparison
        jd_chunks = embedding_agent.search_similar_jd_chunks(cv_context[:500], job_id, n_results=3)
        jd_context = "\n\n".join([chunk['document'] for chunk in jd_chunks])
        
        # Create matching prompt
        matching_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert recruiter evaluating candidate-job fit.
            
Analyze the candidate's CV against the job requirements and provide:
1. Match Score (0-100): Overall fit percentage
2. Key Strengths: What makes this candidate a good fit (list 3-5 points)
3. Skill Gaps: Areas where candidate may need development (list 2-4 points)
4. Reasoning: Brief explanation of the score

Consider:
- Technical skills alignment
- Experience level and relevance
- Education and certifications
- Project experience
- Cultural and role fit indicators

Return ONLY a valid JSON object with this exact structure:
{{
    "match_score": <number between 0-100>,
    "strengths": ["strength1", "strength2", "strength3"],
    "gaps": ["gap1", "gap2"],
    "reasoning": "brief explanation",
    "recommendation": "strong_match|good_match|potential_match|poor_match"
}}"""),
            ("human", """Job Description Summary:
{jd_summary}

Relevant Job Requirements:
{jd_context}

Candidate: {candidate_name}
CV Details:
{cv_context}

Provide your evaluation:""")
        ])
        
        chain = matching_prompt | self.llm
        response = chain.invoke({
            "jd_summary": jd_summary,
            "jd_context": jd_context,
            "candidate_name": candidate_name,
            "cv_context": cv_context
        })
        
        # Parse response
        try:
            result = json.loads(response.content)
            
            # Ensure all required fields exist
            match_data = {
                "match_score": float(result.get("match_score", 0)),
                "strengths": result.get("strengths", []),
                "gaps": result.get("gaps", []),
                "reasoning": result.get("reasoning", ""),
                "recommendation": result.get("recommendation", "poor_match"),
                "is_shortlisted": float(result.get("match_score", 0)) >= Config.MIN_MATCH_SCORE
            }
            
            return match_data
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "match_score": 50.0,
                "strengths": ["Unable to parse detailed analysis"],
                "gaps": ["Requires manual review"],
                "reasoning": response.content[:200],
                "recommendation": "potential_match",
                "is_shortlisted": False
            }
    
    def batch_match_candidates(self, candidates: List[Dict], jd_summary: str, 
                              job_id: int) -> List[Dict]:
        """Match multiple candidates and return sorted results"""
        results = []
        
        for candidate in candidates:
            match_result = self.calculate_match_score(
                candidate_id=candidate['id'],
                candidate_name=candidate['name'],
                jd_summary=jd_summary,
                job_id=job_id
            )
            
            results.append({
                **candidate,
                **match_result
            })
        
        # Sort by match score descending
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        return results


# Global instance
matcher_agent = MatcherAgent()