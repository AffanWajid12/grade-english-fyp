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
        # ... (This function is unchanged)
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
        # ... (This function is unchanged)
        if not self.faiss_index or not self.rag_model: return ""
        question_embedding = self.rag_model.encode([question])
        _, indices = self.faiss_index.search(question_embedding, top_k)
        return "\n---\n".join([self.text_chunks[i] for i in indices[0]])

    def _get_score_from_ai(self, question, student_answer, max_score, rubric_section, context_section):
        """AI Call #1: Get only the numerical score."""
        prompt = f"""You are a scoring AI. Your only job is to return a single integer score.
**Instructions:**
{rubric_section}
- **Question:** {question}
- **Maximum Score:** {max_score}
- **Student's Answer:** "{student_answer}"
- **Reference Material:** "{context_section}"
Based on all the information, what is the score from 0 to {max_score}? Return only the number.
"""
        payload = {"model": "gemma", "prompt": prompt, "stream": False}
        response = requests.post(self.ollama_url, json=payload, timeout=90)
        response.raise_for_status()
        response_text = response.json().get("response", "0")
        # Extract the first number found in the response for robustness
        score_match = re.search(r'\d+', response_text)
        score = int(score_match.group(0)) if score_match else 0
        return max(0, min(score, max_score))

    def _get_feedback_from_ai(self, question, student_answer, max_score, assigned_score, rubric_section, context_section):
        """AI Call #2: Get only the formatted text feedback."""
        prompt = f"""You are an expert teacher providing feedback. Your only job is to generate a formatted text explanation.
The final score of {assigned_score}/{max_score} has already been decided.

Your feedback string MUST be formatted with the following markdown headings, each on a new line:
- **Positive Points:** (List the specific points the student answered correctly)
- **Areas for Improvement:** (List the mistakes or missed points that led to the score of {assigned_score})
- **Summary:** (A concise, one-paragraph summary of why the student received {assigned_score}/{max_score})

**Context:**
{rubric_section}
- **Question:** {question}
- **Student's Answer:** "{student_answer}"
- **Reference Material:** "{context_section}"

Generate the formatted feedback string now.
"""
        payload = {"model": "gemma", "prompt": prompt, "stream": False}
        response = requests.post(self.ollama_url, json=payload, timeout=90)
        response.raise_for_status()
        return response.json().get("response", "Feedback could not be generated.")

    def _grade_with_ollama(self, question_data):
        question = question_data.get('question', '')
        student_answer = question_data.get('answer', '')
        max_score = question_data.get('points', 0)
        rubric = question_data.get('rubric', None)

        if not student_answer: return 0, "This is a container question with no answer to grade."
        
        relevant_context = self._retrieve_relevant_context(question)
        
        rubric_section = ""
        if rubric and isinstance(rubric, list):
            rubric_text = json.dumps(rubric, indent=2)
            rubric_section = f"""**Primary Grading Rubric:**\n{rubric_text}\n"""
        
        try:
            # --- TWO-STEP PROCESS ---
            # Step 1: Get the score
            print(f"--> [GRADING-AI-1] Getting score for: '{question[:50]}...'")
            assigned_score = self._get_score_from_ai(question, student_answer, max_score, rubric_section, relevant_context)
            
            # Step 2: Get the feedback based on the score
            print(f"--> [GRADING-AI-2] Getting feedback for score {assigned_score}/{max_score}...")
            feedback_text = self._get_feedback_from_ai(question, student_answer, max_score, assigned_score, rubric_section, relevant_context)
            
            # Step 3: Manually assemble the final result
            return assigned_score, feedback_text

        except requests.exceptions.RequestException as e:
            return 0, f"Error communicating with Ollama: {e}"
        except Exception as e:
            return 0, f"An unexpected error occurred during grading: {e}"

    def grade_exam(self, student_exam):
        # ... (This function is unchanged)
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
    # ... (This function is unchanged)
    payload = {"model": "gemma", "prompt": prompt, "stream": False, "format": "json"}
    response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
    response.raise_for_status()
    ai_response_text = response.json().get("response", "")
    try:
        parsed_json = json.loads(ai_response_text)
        if isinstance(parsed_json, dict) and "rubric_criteria" in parsed_json and isinstance(parsed_json["rubric_criteria"], list):
            return parsed_json["rubric_criteria"]
        else:
            return None
    except json.JSONDecodeError:
        return None

def _calculate_mark_ranges(total_marks, num_columns):
    # ... (This function is unchanged)
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
    # ... (This function is unchanged)
    data = request.json
    question, total_marks, num_columns = data.get('question'), data.get('points'), data.get('columns', 3)
    if not question or not total_marks: return jsonify({"error": "Missing question text or total marks."}), 400
    mark_ranges = _calculate_mark_ranges(total_marks, num_columns)
    prompt = f"""Your entire response MUST be a single, valid JSON object with one key: "rubric_criteria". The value must be an array of objects.
**Instructions for the "rubric_criteria" array:**
1.  Analyze the question: "{question}" (Worth {total_marks} marks).
2.  The performance levels are: {json.dumps(mark_ranges)}
3.  The first object must define the columns: a "Criterion" key with the value "Description", and other keys being the exact mark range headers.
4.  Create subsequent objects for criteria like "Clarity", "Correctness".
Generate the JSON object now."""
    try:
        rubric_data = call_ai_parser_for_rubric(prompt)
        if rubric_data:
            return jsonify(rubric_data)
        else:
            raise ValueError("AI did not return the expected JSON object structure.")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/grade', methods=['POST'])
def grade():
    # ... (This function is unchanged)
    data = request.json
    try:
        grader = AIGrader(strictness_level=data.get('strictness_level'), course_material=data.get('course_material', ''), additional_instructions=data.get('additional_instructions', ''))
        results = grader.grade_exam(data['student_exam'])
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask AI Grading Server with Two-Step Grading Pipeline...")
    app.run(host='0.0.0.0', port=5000)

