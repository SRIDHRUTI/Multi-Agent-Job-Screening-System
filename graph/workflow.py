from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
import operator

from agents.document_processor import doc_processor
from agents.embedding_agent import embedding_agent
from agents.matcher_agent import matcher_agent
from agents.scheduler_agent import scheduler_agent
from database.db_manager import db_manager


class WorkflowState(TypedDict):
    """State definition for the workflow"""
    job_id: int
    job_title: str
    company: str
    jd_file_path: str
    jd_text: str
    jd_summary: str
    jd_chunks: List[str]
    cv_file_paths: List[str]
    candidates: Annotated[List[Dict], operator.add]
    match_results: Annotated[List[Dict], operator.add]
    shortlisted: List[Dict]
    invites_sent: List[Dict]
    status: str
    error: str


def process_job_description_node(state: WorkflowState) -> WorkflowState:
    """Node 1: Process and chunk job description"""
    try:
        print(f"📄 Processing Job Description: {state['jd_file_path']}")
        
        # Process JD
        jd_data = doc_processor.process_job_description(state['jd_file_path'])
        
        # Store in database
        job = db_manager.create_job_description(
            title=state['job_title'],
            company=state['company'],
            description=jd_data['text'],
            requirements=jd_data['text'],
            summary=jd_data['analysis'],
            file_path=state['jd_file_path']
        )
        
        print(f"✅ Job Description processed: {jd_data['num_chunks']} chunks")
        
        return {
            **state,
            "job_id": job.id,
            "jd_text": jd_data['text'],
            "jd_summary": jd_data['analysis'],
            "jd_chunks": jd_data['chunks'],
            "status": "jd_processed"
        }
    except Exception as e:
        return {**state, "status": "error", "error": f"JD Processing: {str(e)}"}


def index_job_description_node(state: WorkflowState) -> WorkflowState:
    """Node 2: Generate embeddings and index JD in ChromaDB"""
    try:
        print(f"🔍 Indexing Job Description embeddings...")
        
        # Index in ChromaDB
        num_indexed = embedding_agent.index_job_description(
            job_id=state['job_id'],
            chunks=state['jd_chunks'],
            metadata={
                "job_id": state['job_id'],
                "job_title": state['job_title'],
                "company": state['company']
            }
        )
        
        print(f"✅ Indexed {num_indexed} JD chunks in ChromaDB")
        
        return {**state, "status": "jd_indexed"}
    except Exception as e:
        return {**state, "status": "error", "error": f"JD Indexing: {str(e)}"}


def process_cvs_node(state: WorkflowState) -> WorkflowState:
    """Node 3: Process and chunk all CVs"""
    try:
        print(f"📋 Processing {len(state['cv_file_paths'])} CVs...")
        
        candidates = []
        
        for cv_path in state['cv_file_paths']:
            try:
                # Process CV
                cv_data = doc_processor.process_cv(cv_path)
                
                # Extract info
                info = cv_data['info']
                name = info.get('name') or f"Candidate_{len(candidates)+1}"
                email = info.get('email') or f"candidate{len(candidates)+1}@example.com"
                phone = info.get('phone') or ""
                
                # Store in database
                candidate = db_manager.create_candidate(
                    name=name,
                    email=email,
                    phone=phone,
                    cv_text=cv_data['text'],
                    file_path=cv_path,
                    job_id=state['job_id']
                )
                
                candidates.append({
                    "id": candidate.id,
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "cv_text": cv_data['text'],
                    "chunks": cv_data['chunks'],
                    "analysis": cv_data['analysis']
                })
                
                print(f"  ✅ Processed CV: {name}")
                
            except Exception as e:
                print(f"  ⚠️  Failed to process {cv_path}: {str(e)}")
                continue
        
        print(f"✅ Processed {len(candidates)} candidates")
        
        return {**state, "candidates": candidates, "status": "cvs_processed"}
    except Exception as e:
        return {**state, "status": "error", "error": f"CV Processing: {str(e)}"}


def index_cvs_node(state: WorkflowState) -> WorkflowState:
    """Node 4: Generate embeddings and index CVs in ChromaDB"""
    try:
        print(f"🔍 Indexing CV embeddings...")
        
        for candidate in state['candidates']:
            embedding_agent.index_cv(
                candidate_id=candidate['id'],
                chunks=candidate['chunks'],
                metadata={
                    "candidate_id": candidate['id'],
                    "candidate_name": candidate['name'],
                    "job_id": state['job_id']
                }
            )
        
        print(f"✅ Indexed {len(state['candidates'])} CVs in ChromaDB")
        
        return {**state, "status": "cvs_indexed"}
    except Exception as e:
        return {**state, "status": "error", "error": f"CV Indexing: {str(e)}"}


def match_candidates_node(state: WorkflowState) -> WorkflowState:
    """Node 5: Calculate match scores using RAG"""
    try:
        print(f"🎯 Matching candidates against job requirements...")
        
        match_results = []
        
        for candidate in state['candidates']:
            print(f"  Evaluating: {candidate['name']}...")
            
            match_data = matcher_agent.calculate_match_score(
                candidate_id=candidate['id'],
                candidate_name=candidate['name'],
                jd_summary=state['jd_summary'],
                job_id=state['job_id']
            )
            
            # Store in database
            db_manager.create_match_result(
                candidate_id=candidate['id'],
                match_score=match_data['match_score'],
                strengths=match_data['strengths'],
                gaps=match_data['gaps'],
                reasoning=match_data['reasoning'],
                is_shortlisted=match_data['is_shortlisted']
            )
            
            match_results.append({
                **candidate,
                **match_data
            })
            
            print(f"    Score: {match_data['match_score']:.1f}% - {match_data['recommendation']}")
        
        # Sort by score
        match_results.sort(key=lambda x: x['match_score'], reverse=True)
        
        print(f"✅ Matched {len(match_results)} candidates")
        
        return {**state, "match_results": match_results, "status": "matched"}
    except Exception as e:
        return {**state, "status": "error", "error": f"Matching: {str(e)}"}


def shortlist_candidates_node(state: WorkflowState) -> WorkflowState:
    """Node 6: Shortlist top candidates"""
    try:
        print(f"📊 Shortlisting candidates...")
        
        # Filter shortlisted
        shortlisted = [
            c for c in state['match_results'] 
            if c['is_shortlisted']
        ][:10]  # Top 10
        
        print(f"✅ Shortlisted {len(shortlisted)} candidates (score >= {matcher_agent.llm.temperature})")
        
        return {**state, "shortlisted": shortlisted, "status": "shortlisted"}
    except Exception as e:
        return {**state, "status": "error", "error": f"Shortlisting: {str(e)}"}


def send_invites_node(state: WorkflowState) -> WorkflowState:
    """Node 7: Generate and send interview invitations"""
    try:
        if not state['shortlisted']:
            print("⚠️  No candidates to invite")
            return {**state, "invites_sent": [], "status": "completed"}
        
        print(f"📧 Sending interview invitations...")
        
        # Generate and send invites
        invites = scheduler_agent.schedule_interviews(
            shortlisted_candidates=state['shortlisted'],
            job_title=state['job_title'],
            company=state['company']
        )
        
        # Store in database
        for invite in invites:
            if invite['invite_sent']:
                # Get match result ID
                candidate = db_manager.get_candidates_for_job(state['job_id'])
                match_result = None
                for c in candidate:
                    if c.id == invite['id']:
                        if c.match_result:
                            match_result = c.match_result
                            break
                
                if match_result:
                    db_manager.create_interview(
                        match_result_id=match_result.id,
                        invite_message=invite['invite_body']
                    )
        
        print(f"✅ Sent {len(invites)} interview invitations")
        
        return {**state, "invites_sent": invites, "status": "completed"}
    except Exception as e:
        return {**state, "status": "error", "error": f"Sending Invites: {str(e)}"}


def should_continue(state: WorkflowState) -> str:
    """Determine next step based on status"""
    if state.get('status') == 'error':
        return END
    return "continue"


# Build the workflow graph
def create_screening_workflow():
    """Create the LangGraph workflow"""
    
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("process_jd", process_job_description_node)
    workflow.add_node("index_jd", index_job_description_node)
    workflow.add_node("process_cvs", process_cvs_node)
    workflow.add_node("index_cvs", index_cvs_node)
    workflow.add_node("match_candidates", match_candidates_node)
    workflow.add_node("shortlist", shortlist_candidates_node)
    workflow.add_node("send_invites", send_invites_node)
    
    # Define edges
    workflow.set_entry_point("process_jd")
    workflow.add_edge("process_jd", "index_jd")
    workflow.add_edge("index_jd", "process_cvs")
    workflow.add_edge("process_cvs", "index_cvs")
    workflow.add_edge("index_cvs", "match_candidates")
    workflow.add_edge("match_candidates", "shortlist")
    workflow.add_edge("shortlist", "send_invites")
    workflow.add_edge("send_invites", END)
    
    return workflow.compile()


# Global workflow instance
screening_workflow = create_screening_workflow()
