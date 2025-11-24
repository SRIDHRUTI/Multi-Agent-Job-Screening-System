import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import json

from graph.workflow import screening_workflow
from database.db_manager import db_manager
from utils.config import Config

# Page config
st.set_page_config(
    page_title="AI Job Screening System",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .candidate-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .score-high {
        color: #10b981;
        font-weight: bold;
    }
    .score-medium {
        color: #f59e0b;
        font-weight: bold;
    }
    .score-low {
        color: #ef4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'screening_results' not in st.session_state:
    st.session_state.screening_results = None
if 'workflow_running' not in st.session_state:
    st.session_state.workflow_running = False

def save_uploaded_file(uploaded_file, directory):
    """Save uploaded file and return path"""
    file_path = directory / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)

def get_score_class(score):
    """Get CSS class based on score"""
    if score >= 75:
        return "score-high"
    elif score >= 60:
        return "score-medium"
    else:
        return "score-low"

# Header
st.markdown('<p class="main-header">🎯 AI Job Screening System</p>', unsafe_allow_html=True)
st.markdown("**Multi-Agent Recruitment Automation with LangGraph & RAG**")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(f"**Model:** {Config.LLM_MODEL}")
    st.info(f"**Embeddings:** {Config.EMBEDDING_MODEL}")
    st.info(f"**Min Score:** {Config.MIN_MATCH_SCORE}%")
    
    st.divider()
    
    st.header("📊 System Stats")
    all_jobs = db_manager.get_all_job_descriptions()
    st.metric("Total Jobs", len(all_jobs))
    
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.checkbox("Confirm deletion"):
            # Clear database and ChromaDB
            os.system(f"rm -f {Config.DATABASE_URL.replace('sqlite:///', '')}")
            os.system(f"rm -rf {Config.CHROMA_PERSIST_DIR}")
            st.success("All data cleared!")
            st.rerun()

# Main content
tab1, tab2, tab3 = st.tabs(["🚀 New Screening", "📋 View Results", "📖 About"])

with tab1:
    st.header("Start New Screening Process")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Job Description")
        job_title = st.text_input("Job Title *", placeholder="e.g., Senior Python Developer")
        company_name = st.text_input("Company Name *", placeholder="e.g., Tech Corp")
        jd_file = st.file_uploader("Upload Job Description *", type=['pdf', 'docx', 'txt'])
    
    with col2:
        st.subheader("👥 Candidate CVs")
        cv_files = st.file_uploader(
            "Upload Candidate CVs *", 
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )
        
        if cv_files:
            st.success(f"✅ {len(cv_files)} CVs uploaded")
    
    st.divider()
    
    # Run workflow
    if st.button("🚀 Start Screening Process", type="primary", disabled=st.session_state.workflow_running):
        if not job_title or not company_name or not jd_file or not cv_files:
            st.error("❌ Please fill all required fields")
        else:
            try:
                st.session_state.workflow_running = True
                
                # Save files
                jd_path = save_uploaded_file(jd_file, Config.UPLOAD_DIR)
                cv_paths = [save_uploaded_file(cv, Config.UPLOAD_DIR) for cv in cv_files]
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize workflow state
                initial_state = {
                    "job_id": 0,
                    "job_title": job_title,
                    "company": company_name,
                    "jd_file_path": jd_path,
                    "jd_text": "",
                    "jd_summary": "",
                    "jd_chunks": [],
                    "cv_file_paths": cv_paths,
                    "candidates": [],
                    "match_results": [],
                    "shortlisted": [],
                    "invites_sent": [],
                    "status": "initialized",
                    "error": ""
                }
                
                # Run workflow with progress updates
                status_text.text("📄 Processing Job Description...")
                progress_bar.progress(10)
                
                result = screening_workflow.invoke(initial_state)
                
                progress_bar.progress(100)
                status_text.text("✅ Screening Complete!")
                
                # Store results
                st.session_state.screening_results = result
                st.session_state.workflow_running = False
                
                # Show results
                st.success("🎉 Screening process completed successfully!")
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📋 CVs Processed</h3>
                        <h2>{len(result['candidates'])}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>✅ Shortlisted</h3>
                        <h2>{len(result['shortlisted'])}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    avg_score = sum(c['match_score'] for c in result['match_results']) / len(result['match_results'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 Avg Score</h3>
                        <h2>{avg_score:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📧 Invites Sent</h3>
                        <h2>{len(result['invites_sent'])}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.workflow_running = False

with tab2:
    st.header("Screening Results")
    
    if st.session_state.screening_results:
        results = st.session_state.screening_results
        
        # Display all candidates
        st.subheader("📊 All Candidates")
        
        for i, candidate in enumerate(results['match_results']):
            score = candidate['match_score']
            score_class = get_score_class(score)
            
            with st.expander(f"{'✅' if candidate['is_shortlisted'] else '❌'} {candidate['name']} - {score:.1f}%"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Email:** {candidate['email']}")
                    st.markdown(f"**Phone:** {candidate['phone']}")
                    st.markdown(f"**Match Score:** <span class='{score_class}'>{score:.1f}%</span>", unsafe_allow_html=True)
                    st.markdown(f"**Recommendation:** {candidate['recommendation'].replace('_', ' ').title()}")
                
                with col2:
                    if candidate['is_shortlisted']:
                        st.success("✅ SHORTLISTED")
                    else:
                        st.error("❌ Not Shortlisted")
                
                st.markdown("**💪 Strengths:**")
                for strength in candidate['strengths']:
                    st.markdown(f"- {strength}")
                
                st.markdown("**⚠️ Skill Gaps:**")
                for gap in candidate['gaps']:
                    st.markdown(f"- {gap}")
                
                st.markdown("**💭 Reasoning:**")
                st.info(candidate['reasoning'])
        
        # Display shortlisted candidates
        if results['shortlisted']:
            st.divider()
            st.subheader("🌟 Shortlisted Candidates")
            
            for candidate in results['shortlisted']:
                st.markdown(f"""
                <div class="candidate-card">
                    <h3>✅ {candidate['name']} - <span class="score-high">{candidate['match_score']:.1f}%</span></h3>
                    <p><strong>Email:</strong> {candidate['email']}</p>
                    <p><strong>Top Strengths:</strong> {', '.join(candidate['strengths'][:3])}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Display invites
        if results['invites_sent']:
            st.divider()
            st.subheader("📧 Interview Invitations")
            
            for invite in results['invites_sent']:
                with st.expander(f"📩 {invite['name']} - {'✅ Sent' if invite['invite_sent'] else '❌ Failed'}"):
                    st.markdown(f"**To:** {invite['email']}")
                    st.markdown(f"**Subject:** {invite['invite_subject']}")
                    st.markdown("**Message:**")
                    st.text_area("", invite['invite_body'], height=200, key=f"invite_{invite['id']}")
                    if invite.get('sent_at'):
                        st.caption(f"Sent at: {invite['sent_at']}")
    
    else:
        st.info("No screening results yet. Run a screening process in the 'New Screening' tab.")

with tab3:
    st.header("About This System")
    
    st.markdown("""
    ### 🎯 Multi-Agent Job Screening System
    
    This system automates the recruitment process using AI agents powered by LangGraph, RAG, and ChromaDB.
    
    #### 🔄 Workflow Process:
    
    1. **Document Processing Agent** 📄
       - Ingests and parses job descriptions and CVs
       - Chunks documents intelligently for better processing
    
    2. **Embedding Agent** 🔍
       - Generates embeddings using OpenAI's embedding model
       - Stores in ChromaDB for efficient similarity search
    
    3. **Matcher Agent** 🎯
       - Uses RAG to match candidates against job requirements
       - Calculates match scores with detailed reasoning
       - Identifies strengths and skill gaps
    
    4. **Scheduler Agent** 📧
       - Generates personalized interview invitations
       - Sends automated emails to shortlisted candidates
    
    #### 🛠️ Technologies Used:
    - **LangGraph**: Multi-agent workflow orchestration
    - **LangChain**: LLM integration and RAG pipeline
    - **ChromaDB**: Vector database for embeddings
    - **SQLite**: Relational data storage
    - **Streamlit**: Interactive web interface
    - **OpenAI GPT-4**: Language model for analysis
    
    #### 📊 Key Features:
    - ✅ Automated CV parsing and analysis
    - ✅ Semantic matching using RAG
    - ✅ Intelligent candidate scoring
    - ✅ Automated interview scheduling
    - ✅ Personalized communication
    - ✅ Complete audit trail
    
    ---
    
    **Created with ❤️ using LangGraph & AI**
    """)

# Footer
st.divider()
st.caption("© 2025 AI Job Screening System | Powered by LangGraph, OpenAI, and ChromaDB")