from flask import Flask, request, jsonify
from flask_cors import CORS
import concurrent.futures
import hashlib
import pickle
import os
import requests
import re 
from datetime import datetime
import io
import json

# --- Dependency Imports ---
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    import PyPDF2
except ImportError:
    print("Dependencies not found. Please run: pip install -r requirements.txt")
    SentenceTransformer = None; np = None; faiss = None; PyPDF2 = None

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Caching Setup ---
CACHE_DIR = ".rag_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

class AIGrader:
    """Handles the final grading, now capable of using a teacher-approved rubric."""
    def __init__(self, strictness_level='Be balanced.', course_material=None, additional_instructions=None, rubric=None):
        self.strictness_level = strictness_level or ""
        self.additional_instructions = additional_instructions or ""
        self.rubric = rubric or None # New rubric property
        self.ollama_url = "http://localhost:11434/api/generate"
        self.faiss_index = None
        self.text_chunks = []
        self.rag_model = None
        if course_material and faiss and SentenceTransformer:
            self.rag_model = SentenceTransformer('all-MiniLM-L6-v2')
            self._index_course_material(course_material)
        else:
            print("--> AI Grader Initialized without RAG.")

    def _index_course_material(self, text_content):
        content_hash = hashlib.md5(text_content.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{content_hash}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                self.text_chunks, self.faiss_index = cached_data['chunks'], faiss.deserialize_index(cached_data['index'])
            return
        chunks = [p.strip() for p in text_content.split('\n\n') if p.strip()]
        if not chunks: return
        self.text_chunks = chunks
        embeddings = self.rag_model.encode(chunks, show_progress_bar=True)
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        with open(cache_file, "wb") as f:
            pickle.dump({'chunks': self.text_chunks, 'index': faiss.serialize_index(self.faiss_index)}, f)

    def _retrieve_relevant_context(self, question, top_k=3):
        if not self.faiss_index or not self.rag_model: return ""
        question_embedding = self.rag_model.encode([question])
        _, indices = self.faiss_index.search(question_embedding, top_k)
        return "\n---\n".join([self.text_chunks[i] for i in indices[0]])

    def _parse_ollama_response(self, response_text, max_score):
        try:
            score, feedback = 0, ""
            score_match = re.search(r'score\D*(\d+)', response_text, re.IGNORECASE)
            if score_match: score = int(score_match.group(1))
            feedback_match = re.search(r'FEEDBACK:(.*)', response_text, re.IGNORECASE | re.DOTALL)
            if feedback_match: feedback = feedback_match.group(1).strip()
            if not feedback and response_text.strip(): feedback = response_text.strip()
            return max(0, min(score, max_score)), feedback
        except Exception: return 0, f"Error parsing response. Raw: {response_text}"

    def _grade_with_ollama(self, question, student_answer, question_type, max_score):
        if not student_answer or not student_answer.strip():
             return 0, "This is a container question with no answer to grade."
        
        relevant_context = self._retrieve_relevant_context(question)
        
        # --- NEW: Rubric Injection into Prompt ---
        rubric_section = ""
        if self.rubric and isinstance(self.rubric, dict) and self.rubric.get('criteria'):
            rubric_text = json.dumps(self.rubric['criteria'], indent=2)
            rubric_section = f"""**Primary Grading Rubric:** You MUST use the following rubric to determine the score. The criteria listed here are the most important factor for grading.\n{rubric_text}\n"""

        # The rest of the prompt logic (overriding command vs. strictness) remains
        if self.additional_instructions.strip():
            prompt = f"""You are an AI assistant following a command.
**Primary Directive:** Execute this command precisely: "{self.additional_instructions}"
**Required Output Format:** SCORE: [number] FEEDBACK: [feedback]
---
**Context:** Question: {question}, Student's Answer: "{student_answer}", Max Score: {max_score}
---
Execute the command."""
        else:
            strictness_section = f'3. **Grading Strictness:** "{self.strictness_level}"' if self.strictness_level.strip() else ""
            context_section = f"**Reference Material:**\n{relevant_context}\n---" if relevant_context else ""
            prompt = f"""You are an expert AI exam grader.
**Required Output Format:** SCORE: [number] FEEDBACK: [feedback]
---
**Core Grading Instructions:**
{rubric_section}
1. **Question:** {question}
2. **Maximum Score:** {max_score}
{strictness_section}
{context_section}
---
**Student's Answer:**
"{student_answer}"
---
Provide your evaluation."""
            
        payload = {"model": "gemma", "prompt": prompt, "stream": False}
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=90)
            response.raise_for_status()
            return self._parse_ollama_response(response.json().get("response", ""), max_score)
        except requests.exceptions.RequestException as e:
            return 0, f"Error communicating with Ollama: {e}"

    def grade_exam(self, student_exam):
        def flatten_exam(exam_parts):
            flat_list = []
            for part in exam_parts:
                flat_list.append(part)
                if 'parts' in part and part['parts']: flat_list.extend(flatten_exam(part['parts']))
            return flat_list

        flat_exam_list = flatten_exam(student_exam)
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_question = {executor.submit(self._grade_with_ollama, item['question'], item['answer'], item.get('type', 'short_question'), item.get('points', 0)): item for item in flat_exam_list}
            for future in concurrent.futures.as_completed(future_to_question):
                item = future_to_question[future]
                question = item['question']
                score, feedback = future.result()
                results[question] = {'score': score, 'feedback': feedback, 'student_answer': item['answer'], 'max_score': item.get('points', 0)}
        return results

def call_ai_parser(prompt):
    """Generic function to call the AI and parse its JSON response."""
    payload = {"model": "gemma", "prompt": prompt, "stream": False, "format": "json"}
    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
    response.raise_for_status()
    ai_response_text = response.json().get("response", "")
    try:
        # The AI sometimes returns a string that contains a JSON object, not a pure JSON object.
        # This robustly finds and parses the first valid JSON object in the string.
        json_match = re.search(r'\[.*\]', ai_response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(ai_response_text)
    except json.JSONDecodeError:
        print(f"--> [ERROR] AI returned invalid JSON: {ai_response_text}")
        return None

# --- NEW ENDPOINT FOR RUBRIC GENERATION ---
@app.route('/generate_rubric', methods=['POST'])
def generate_rubric():
    print(f"\n{'='*50}\n--- Received RUBRIC GENERATION request at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{'='*50}")
    data = request.json
    exam_questions = data.get('student_exam', [])
    
    if not exam_questions:
        return jsonify({"error": "No exam questions provided to generate a rubric."}), 400

    question_summary = "\n".join([f"- {q['question']} ({q['points']} marks)" for q in exam_questions if q.get('question')])

    prompt = f"""You are an expert in pedagogy and educational assessment. Your task is to generate 3 distinct, high-quality grading rubrics for the provided exam questions.

**Primary Directive:** Your entire response MUST be a single JSON array containing exactly 3 rubric objects.

**JSON Schema for each rubric object:**
{{
  "title": "A creative and descriptive title for the rubric (e.g., 'Analytical Skills Rubric')",
  "criteria": [
    {{
      "Criterion": "The name of a grading criterion (e.g., 'Clarity and Cohesion')",
      "Excellent": "A description of what constitutes top performance for this criterion.",
      "Good": "A description of average or good performance.",
      "Needs Improvement": "A description of poor performance or what is lacking."
    }}
  ]
}}

**Exam Questions to Analyze:**
---
{question_summary}
---

Now, generate the JSON array containing the 3 rubrics.
"""
    try:
        rubrics = call_ai_parser(prompt)
        if rubrics and isinstance(rubrics, list) and len(rubrics) > 0:
            print(f"--> [RUBRIC] Successfully generated {len(rubrics)} rubric options.")
            return jsonify(rubrics)
        else:
            raise ValueError("AI did not return a valid list of rubrics.")
    except Exception as e:
        print(f"--> [FATAL ERROR] An unexpected error occurred during rubric generation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/grade', methods=['POST'])
def grade():
    print(f"\n{'='*50}\n--- Received GRADING request at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{'='*50}")
    data = request.json
    try:
        grader = AIGrader(
            strictness_level=data.get('strictness_level', ''),
            course_material=data.get('course_material', ''),
            additional_instructions=data.get('additional_instructions', ''),
            rubric=data.get('rubric', None) # Pass the rubric to the grader
        )
        results = grader.grade_exam(data['student_exam'])
        print(f"--- Grading complete. Sending response back to client. ---")
        return jsonify(results)
    except Exception as e:
        print(f"--> [FATAL ERROR] An unexpected error occurred in the /grade endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask AI Grading Server with Rubric Generation...")
    app.run(host='0.0.0.0', port=5000)

