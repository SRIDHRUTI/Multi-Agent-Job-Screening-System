JD Screening

A lightweight AI-powered system that screens candidate CVs against a Job Description (JD) using document parsing, embeddings, and an LLM-based matching workflow. This simplified README keeps only the essential components needed to understand, run, and extend the project.

Overview

JD Screening automates candidate evaluation by:

Extracting text from the JD and CVs

Creating embeddings and storing them in a vector database

Retrieving and comparing candidate skills with the JD

Using an LLM to score and summarize candidate fit

The project includes a minimal Streamlit UI for uploading files and visualizing results.

Core Features

JD & CV Uploads (PDF/Text)

Text Extraction & Cleaning

Chunking + Embedding using ChromaDB

LLM-Based Matching (score + strengths + weaknesses)

Shortlisting based on score

Basic Interview Scheduling (placeholder)

Tech Stack

Python 3.10+

Streamlit – UI

LangChain / LangGraph – workflow orchestration

OpenAI LLMs – matching logic

ChromaDB – vector store

SQLAlchemy – database persistence

Project Structure

JD_Screening/
├─ app.py # Streamlit frontend
├─ graph/workflow.py # Screening workflow
├─ agents/ # Document, embedding, matching, scheduling agents
├─ utils/ # Config + PDF/text parsing
├─ database/ # SQLAlchemy models & DB utilities
└─ requirements.txt

Setup & Run

Install dependencies:

pip install -r requirements.txt

Add environment variables via .env:

OPENAI_API_KEY=your_key
DATABASE_URL=sqlite:///job_screening.db
LLM_MODEL=gpt-4o-mini

Run the app:
streamlit run app.py
