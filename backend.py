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
import math

# --- Dependency Imports ---
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    import PyPDF2
except ImportError:
    print("Dependencies not found. Please run: pip install -r requirements.txt")
    SentenceTransformer, np, faiss, PyPDF2 = None, None, None, None

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)
CACHE_DIR = ".rag_cache"
if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)

class AIGrader:
    # ... (AIGrader class is unchanged from the previous version) ...
    def __init__(self, strictness_level=None, course_material=None, additional_instructions=None):
        self.strictness_level = strictness_level or ""
        self.additional_instructions = additional_instructions or ""
        self.ollama_url = "http://localhost:11434/api/generate"
        self.faiss_index, self.text_chunks, self.rag_model = None, [], None
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
        except Exception as e: 
            print(f"--> [ERROR] Parsing Ollama response failed: {e}")
            return 0, f"Error parsing response. Raw: {response_text}"

    def _grade_with_ollama(self, question_data):
        question = question_data.get('question', '')
        student_answer = question_data.get('answer', '')
        max_score = question_data.get('points', 0)
        rubric = question_data.get('rubric', None)

        if not student_answer: return 0, "This is a container question with no answer to grade."
        
        relevant_context = self._retrieve_relevant_context(question)
        
        rubric_section = ""
        if rubric and isinstance(rubric, list):
            try:
                rubric_text = json.dumps(rubric, indent=2)
                rubric_section = f"""**Primary Grading Rubric:** You MUST use the following rubric for this specific question to determine the score.\n{rubric_text}\n"""
            except TypeError:
                rubric_section = ""

        if self.additional_instructions:
            prompt = f"""**Command:** "{self.additional_instructions}"\n**Format:** SCORE: [number] FEEDBACK: [feedback]\n**Context:** Q: {question}, A: "{student_answer}", Max Score: {max_score}\nExecute."""
        else:
            strictness_section = f'3. **Grading Strictness:** "{self.strictness_level}"' if self.strictness_level else ""
            context_section = f"**Reference Material:**\n{relevant_context}\n---" if relevant_context else ""
            prompt = f"""You are an expert AI exam grader.
**Format:** SCORE: [number] FEEDBACK: [feedback]
---
**Instructions:**
{rubric_section}
1. **Question:** {question}
2. **Maximum Score:** {max_score}
{strictness_section}
{context_section}
---
**Student's Answer:**
"{student_answer}"
---
Evaluate and provide score."""
            
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
            future_to_question = {executor.submit(self._grade_with_ollama, item): item for item in flat_exam_list}
            for future in concurrent.futures.as_completed(future_to_question):
                item = future_to_question[future]
                question = item['question']
                score, feedback = future.result()
                results[question] = {'score': score, 'feedback': feedback, 'student_answer': item['answer'], 'max_score': item.get('points', 0)}
        return results

def call_ai_parser_for_rubric(prompt):
    """Specialized parser for the rubric endpoint."""
    payload = {"model": "gemma", "prompt": prompt, "stream": False, "format": "json"}
    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
    response.raise_for_status()
    ai_response_text = response.json().get("response", "")
    try:
        # The AI is now asked to return an object like {"rubric_criteria": [...]}, which is more robust
        parsed_json = json.loads(ai_response_text)
        if isinstance(parsed_json, dict) and "rubric_criteria" in parsed_json and isinstance(parsed_json["rubric_criteria"], list):
            return parsed_json["rubric_criteria"]
        else:
            print(f"--> [ERROR] AI returned valid JSON, but it was not in the expected format: {ai_response_text}")
            return None
    except json.JSONDecodeError:
        print(f"--> [ERROR] AI returned invalid JSON: {ai_response_text}")
        return None

def _calculate_mark_ranges(total_marks, num_columns):
    if num_columns <= 0: return []
    points_per_level = total_marks / num_columns
    ranges = []
    level_names = {2: ["Excellent", "Needs Improvement"], 3: ["Excellent", "Good", "Needs Improvement"], 4: ["Excellent", "Good", "Satisfactory", "Needs Improvement"], 5: ["Excellent", "Very Good", "Good", "Satisfactory", "Needs Improvement"]}
    names = level_names.get(num_columns, [f"Level {i+1}" for i in range(num_columns)])
    upper_bound = total_marks
    for i in range(num_columns):
        lower_bound = math.ceil(total_marks - (i + 1) * points_per_level)
        if i == num_columns - 1: lower_bound = 0
        range_str = f"{upper_bound} Marks" if lower_bound == upper_bound else f"{lower_bound}-{upper_bound} Marks"
        ranges.append(f"{range_str} ({names[i]})")
        upper_bound = lower_bound - 1
    return ranges

@app.route('/generate_question_rubric', methods=['POST'])
def generate_question_rubric():
    print(f"\n{'='*50}\n--- Received PER-QUESTION RUBRIC request at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{'='*50}")
    data = request.json
    question, total_marks, num_columns = data.get('question'), data.get('points'), data.get('columns', 3)

    if not question or not total_marks: return jsonify({"error": "Missing question text or total marks."}), 400

    mark_ranges = _calculate_mark_ranges(total_marks, num_columns)
    print(f"--> [RUBRIC] Calculated Mark Ranges: {mark_ranges}")

    # --- FINAL, MOST ROBUST PROMPT ---
    # Asks for a JSON object containing the list, which is more reliable.
    prompt = f"""You are an expert in pedagogy. Your task is to generate the criteria for a pre-defined grading rubric.
Your entire response MUST be a single, valid JSON object. Do not include any introductory text, explanations, or markdown. Your response must start with '{{' and end with '}}'.

The JSON object must have one key: "rubric_criteria". The value of this key must be an array of objects.

**Instructions for the "rubric_criteria" array:**
1.  Analyze the question: "{question}" (Worth {total_marks} marks).
2.  The performance levels have been calculated for you. They are: {json.dumps(mark_ranges)}
3.  The first object in the array must define the columns. It must have a "Criterion" key with the value "Description", and the other keys must be the exact mark range headers provided above.
4.  Create at least two subsequent objects describing criteria (e.g., "Clarity", "Correctness"). For each criterion, write a description for each mark range.

Generate the JSON object now.
"""
    try:
        rubric_data = call_ai_parser_for_rubric(prompt)
        if rubric_data:
            print(f"--> [RUBRIC] Successfully generated rubric for question: '{question[:50]}...'")
            return jsonify(rubric_data)
        else:
            raise ValueError("AI did not return the expected JSON object structure.")
    except Exception as e:
        print(f"--> [FATAL ERROR] An unexpected error occurred during rubric generation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/grade', methods=['POST'])
def grade():
    print(f"\n{'='*50}\n--- Received GRADING request at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{'='*50}")
    data = request.json
    try:
        grader = AIGrader(strictness_level=data.get('strictness_level'), course_material=data.get('course_material', ''), additional_instructions=data.get('additional_instructions', ''))
        results = grader.grade_exam(data['student_exam'])
        print(f"--- Grading complete. Sending response back to client. ---")
        return jsonify(results)
    except Exception as e:
        print(f"--> [FATAL ERROR] An unexpected error occurred in the /grade endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask AI Grading Server with Per-Question Rubric Generation...")
    app.run(host='0.0.0.0', port=5000)

