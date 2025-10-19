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
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


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
        content_hash = hashlib.md5(text_content.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{content_hash}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                self.text_chunks, self.faiss_index = cached_data['chunks'], faiss.deserialize_index(cached_data['index'])
            return
        chunks = [p.strip() for p in text_content.split('\n\n') if p.strip()]
        if not chunks:
            return
        self.text_chunks = chunks
        embeddings = self.rag_model.encode(chunks, show_progress_bar=True)
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        with open(cache_file, "wb") as f:
            pickle.dump({'chunks': self.text_chunks, 'index': faiss.serialize_index(self.faiss_index)}, f)

    def _retrieve_relevant_context(self, question, top_k=3):
        if not self.faiss_index or not self.rag_model:
            return ""
        question_embedding = self.rag_model.encode([question])
        _, indices = self.faiss_index.search(question_embedding, top_k)
        return "\n---\n".join([self.text_chunks[i] for i in indices[0]])

    def _get_score_from_ai(self, question, student_answer, max_score, rubric_section, context_section, additional_instructions=None, strictness_text=None):
        """
        Ask the LLM for a base integer score, then deterministically adjust the score
        according to strictness_text (keyword->multiplier). The prompt is explicit about
        priorities (rubric > teacher instructions > strictness).
        """
        # Build the instruction block in a clear priority order
        priority_block = "Priority (apply in this order):\n1) Primary Grading Rubric (if provided) — MUST be followed first.\n2) Teacher's Additional Instructions — use these to emphasize or de-emphasize criteria.\n3) Strictness Level — adjust final numeric score accordingly.\n\n"
        teacher_block = ""
        if additional_instructions:
            teacher_block = f"**Teacher Additional Instructions:**\n{additional_instructions}\n\n"

        rubric_block = ""
        if rubric_section:
            rubric_block = f"**Primary Grading Rubric:**\n{rubric_section}\n\n"

        # Compose the scoring prompt (very explicit)
        prompt = (
            "You are a precise scoring AI. Your ONLY job is to return a single integer score.\n"
            + priority_block
            + rubric_block
            + teacher_block
            + f"- Question: {question}\n"
            + f"- Maximum Score: {max_score}\n"
            + f'- Student Answer: "{student_answer}"\n'
            + f'- Reference Material (if any): "{context_section}"\n\n'
            + "IMPORTANT:\n"
            + "1) Use the rubric first if available — match rubric criteria and mark ranges.\n"
            + "2) Then apply teacher additional instructions (if present) to adjust emphasis or award/penalize specific behavior.\n"
            + "3) Finally, apply the strictness level only to slightly adjust the final numeric score (the system may apply a small make-up or deduction but keep the result within bounds).\n"
            + "Return only the integer number (0 .. {max_score}). No text, no reasoning, no punctuation.\n"
        )

        payload = {"model": "gemma", "prompt": prompt, "stream": False}
        response = requests.post(self.ollama_url, json=payload, timeout=90)
        response.raise_for_status()
        response_text = response.json().get("response", "").strip()
        # extract base integer
        m = re.search(r'-?\d+', response_text)
        base_score = int(m.group(0)) if m else 0
        base_score = max(0, min(base_score, max_score))

        # Apply deterministic strictness multiplier
        multiplier = self._strictness_multiplier(strictness_text or "")
        adjusted = int(round(base_score * multiplier))

        # enforce bounds
        adjusted = max(0, min(adjusted, max_score))

        # Debugging trace (prints to server logs)
        print(f"[SCORE] question='{question[:50]}...' base_score={base_score} multiplier={multiplier} adjusted={adjusted} (max={max_score})")

        return adjusted


    def _get_feedback_from_ai(self, question, student_answer, max_score, assigned_score, rubric_section, context_section, additional_instructions=None, strictness_text=None):
        # Compose the feedback prompt with explicit context and priority
        priority_block = "Priority (apply in this order):\n1) Primary Grading Rubric (if provided) — MUST be used to justify points.\n2) Teacher's Additional Instructions — use to tailor feedback.\n3) Strictness Level — mention if the strictness influenced score.\n\n"
        teacher_block = f"**Teacher Additional Instructions:**\n{additional_instructions}\n\n" if additional_instructions else ""
        rubric_block = f"**Primary Grading Rubric:**\n{rubric_section}\n\n" if rubric_section else ""

        prompt = (
            "You are an expert teacher writing feedback. Produce feedback with the exact headings below.\n"
            + priority_block
            + rubric_block
            + teacher_block
            + f"- Question: {question}\n"
            + f'- Student Answer: "{student_answer}"\n'
            + f"- Maximum Score: {max_score}\n"
            + f"- Assigned Score: {assigned_score}\n"
            + f'- Reference Material (if any): "{context_section}"\n\n'
            + "Your feedback MUST use these markdown headings EXACTLY:\n"
            + "- **Positive Points:**\n"
            + "- **Areas for Improvement:**\n"
            + "- **Summary:**\n"
            + "\nBe concise and reference the rubric criteria when possible. If strictness influenced scoring, note this briefly in the Summary.\n"
        )

        payload = {"model": "gemma", "prompt": prompt, "stream": False}
        response = requests.post(self.ollama_url, json=payload, timeout=90)
        response.raise_for_status()
        return response.json().get("response", "Feedback could not be generated.")


    def _grade_with_ollama(self, question_data):
        question, student_answer, max_score, rubric = (question_data.get(k) for k in ['question', 'answer', 'points', 'rubric'])
        if not student_answer:
            return 0, "This is a container question with no answer to grade."

        relevant_context = self._retrieve_relevant_context(question)
        rubric_section = f"**Primary Grading Rubric:**\n{json.dumps(rubric, indent=2)}\n" if rubric and isinstance(rubric, list) else ""

        try:
            print(f"--> [GRADING-AI-1] Getting score for: '{question[:50]}...'")
            assigned_score = self._get_score_from_ai(
                question, student_answer, max_score,
                rubric_section, relevant_context,
                additional_instructions=self.additional_instructions,
                strictness_text=self.strictness_level
            )


            print(f"--> [GRADING-AI-2] Getting feedback for score {assigned_score}/{max_score}...")
            feedback_text = self._get_feedback_from_ai(
                question, student_answer, max_score, assigned_score,
                rubric_section, relevant_context,
                additional_instructions=self.additional_instructions,
                strictness_text=self.strictness_level
            )

            return assigned_score, feedback_text
        except requests.exceptions.RequestException as e:
            return 0, f"Error communicating with Ollama: {e}"
        except Exception as e:
            return 0, f"An unexpected error occurred during grading: {e}"

    def grade_exam(self, student_exam):
        """
        Grade only leaf questions (questions without 'parts').
        Returns a mapping keyed by position-based IDs like "q-0-1-2" to avoid collisions
        and to ensure parent/container questions are NOT graded directly.
        """
        # Collect leaf tasks with their position path
        tasks = []  # list of (path_list, item)
        def walk(parts, path_prefix):
            for idx, part in enumerate(parts):
                cur_path = path_prefix + [idx]
                # if there are sub-parts, recurse (container)
                if part.get('parts'):
                    walk(part['parts'], cur_path)
                else:
                    tasks.append((cur_path, part))
        walk(student_exam or [], [])

        results = {}
        # Grade only leaf items concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_task = {executor.submit(self._grade_with_ollama, item): (path, item) for (path, item) in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                path, item = future_to_task[future]
                try:
                    score, feedback = future.result()
                except Exception as e:
                    score, feedback = 0, f"Error during grading: {e}"

                # Build stable path key like 'q-0-1-2'
                path_key = "q-" + "-".join(map(str, path))
                results[path_key] = {
                    "score": int(score),
                    "feedback": feedback,
                    "student_answer": item.get('answer', ''),
                    "max_score": int(item.get('points', 0)),
                    "question": item.get('question', ''),
                    "path": path
                }

        return results

    def _strictness_multiplier(self, strictness_text):
        """
        Map teacher's strictness instruction text to a deterministic multiplier.
        Multipliers >1 lenient (raise score), <1 strict (lower score).
        The values are conservative so we don't wildly change scores.
        """
        if not strictness_text:
            return 1.0
        t = strictness_text.lower()
        # priority map (try most specific phrases first)
        if "unforgiving" in t or "extremely strict" in t or "extremely unforgiving" in t:
            return 0.80
        if "extremely lenient" in t:
            return 1.15
        if "very strict" in t or "be very strict" in t:
            return 0.85
        if "very lenient" in t:
            return 1.10
        if "strict" in t and "slightly" in t:
            return 0.97
        if "slightly strict" in t:
            return 0.97
        if "slightly lenient" in t:
            return 1.03
        if "lenient" in t and "very" not in t and "extremely" not in t:
            return 1.05
        if "balanced" in t or "fair" in t or "be balanced" in t:
            return 1.0
        # fallback: look for "lenient" or "strict"
        if "lenient" in t:
            return 1.05
        if "strict" in t:
            return 0.95
        return 1.0


# ------------------ Helper to call Ollama ------------------
def _call_ollama(prompt, timeout=60, model="gemma", extra_kwargs=None):
    payload = {"model": model, "prompt": prompt, "stream": False}
    if extra_kwargs:
        payload.update(extra_kwargs)
    resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "")


def _generate_criteria_from_ai(question, num_columns):
    prompt = f"""
You are an expert teacher. Given the question below, produce a short list of rubric criterion NAMES.
Return ONLY a comma-separated list of criterion names.

Question: "{question}"
"""
    try:
        ai_text = _call_ollama(prompt, timeout=60)
        first_line = ai_text.splitlines()[0].strip()
        criteria = [c.strip() for c in first_line.split(",") if c.strip()]
        if not criteria:
            raise ValueError("AI returned no criteria.")
        return criteria
    except Exception as e:
        raise RuntimeError(f"Error generating criteria from AI: {e}")


def _calculate_mark_ranges(total_marks, num_columns):
    if num_columns <= 0:
        return []
    points_per_level = total_marks / num_columns
    ranges = []
    names = {
        2: ["Excellent", "Needs Improvement"],
        3: ["Excellent", "Good", "Needs Improvement"],
        4: ["Excellent", "Good", "Satisfactory", "Needs Improvement"],
        5: ["Excellent", "Very Good", "Good", "Satisfactory", "Needs Improvement"],
    }.get(num_columns, [f"Level {i+1}" for i in range(num_columns)])
    upper_bound = total_marks
    for i in range(num_columns):
        lower_bound = math.ceil(total_marks - (i + 1) * points_per_level)
        if i == num_columns - 1:
            lower_bound = 0
        range_str = f"{upper_bound} Marks" if lower_bound == upper_bound else f"{lower_bound}-{upper_bound} Marks"
        ranges.append(f"{range_str} ({names[i]})")
        upper_bound = lower_bound - 1
    return ranges

# -------------------------
# Rubric helper + bulk endpoints
# -------------------------

def _generate_rubric_matrix(question, total_marks, num_columns=3, provided_criteria=None, timeout_per_call=60):
    """Generate rubric matrix using either provided criteria or AI-generated criteria."""
    # 🧩 FIX: Normalize provided criteria into a list
    if isinstance(provided_criteria, str):
        provided_criteria = [c.strip() for c in provided_criteria.split(",") if c.strip()]

    if provided_criteria and isinstance(provided_criteria, list) and any(str(c).strip() for c in provided_criteria):
        criteria_list = [str(c).strip() for c in provided_criteria if str(c).strip()]
    else:
        criteria_list = _generate_criteria_from_ai(question, num_columns)

    mark_ranges = _calculate_mark_ranges(total_marks, num_columns)
    rubric_matrix = []

    for criterion in criteria_list:
        cell_descriptions = {}
        for mark_range in mark_ranges:
            prompt = f"""
You are an expert teacher creating grading rubrics.

Question: "{question}"
Criterion: "{criterion}"
Performance level: "{mark_range}"

Describe briefly what student performance at this level looks like for this criterion.
Return only plain text.
"""
            try:
                cell_text = _call_ollama(prompt, timeout=timeout_per_call).strip()
                cell_text = " ".join([line.strip() for line in cell_text.splitlines() if line.strip()])
                if not cell_text:
                    cell_text = f"(No description generated for {mark_range})"
            except Exception as e:
                cell_text = f"Error generating description: {e}"
            cell_descriptions[mark_range] = cell_text

        row = {"Criterion": criterion}
        row.update(cell_descriptions)
        rubric_matrix.append(row)

    return rubric_matrix


@app.route("/generate_question_rubric", methods=["POST"])
def generate_question_rubric():
    """Generate rubric for one question, respecting teacher criteria if given."""
    data = request.json or {}
    question = data.get("question")
    total_marks = data.get("points")
    num_columns = int(data.get("columns", 3))
    provided_criteria = data.get("criteria")
    path = data.get("path")

    if not question or total_marks is None:
        return jsonify({"error": "Missing question or marks"}), 400

    # 🧩 FIX: convert string to list if needed
    if isinstance(provided_criteria, str):
        provided_criteria = [c.strip() for c in provided_criteria.split(",") if c.strip()]

    try:
        rubric_matrix = _generate_rubric_matrix(
            question, total_marks, num_columns, provided_criteria
        )
    except Exception as e:
        return jsonify({"error": f"Failed to generate rubric: {e}"}), 500

    path_key = None
    if path is not None:
        path_key = "q-" + "-".join(map(str, path)) if isinstance(path, list) else str(path)

    payload = {"rubric": rubric_matrix}
    if path_key:
        payload["path_key"] = path_key
    return jsonify(payload), 200


@app.route("/generate_rubrics_bulk", methods=["POST"])
def generate_rubrics_bulk():
    """Generate rubrics for all leaf questions, using teacher criteria when provided.
    Supports per-question 'columns' field (q['columns']) as override of the global 'columns'.
    """
    data = request.json or {}
    student_exam = data.get("student_exam", [])
    global_num_columns = int(data.get("columns", 3))
    force = bool(data.get("force", False))
    timeout_per_call = int(data.get("timeout_per_call", 60))
    max_workers = int(data.get("max_workers", 6))

    # collect leaf questions with their position path
    leaves = []
    def walk(parts, prefix):
        for idx, part in enumerate(parts):
            cur_path = prefix + [idx]
            if part.get("parts"):
                walk(part["parts"], cur_path)
            else:
                leaves.append((cur_path, part))
    walk(student_exam or [], [])

    if not leaves:
        return jsonify({"generated": {}, "skipped": []}), 200

    generated = {}
    skipped = []

    # helper to process a single leaf
    def _process_leaf(entry):
        path, q = entry
        path_key = "q-" + "-".join(map(str, path))

        # If rubric exists and force==False, skip
        if q.get("rubric") and not force:
            return ("skip", path_key, None)

        # Determine per-question columns if provided, otherwise use global
        leaf_columns = global_num_columns
        try:
            # q.get('columns') may be int or string — normalize safely
            if q.get("columns") is not None:
                leaf_columns = int(q.get("columns"))
        except Exception:
            leaf_columns = global_num_columns

        # Accept either 'criteria' or 'suggested_criteria' from frontend
        provided_criteria = q.get("criteria")
        if not provided_criteria:
            provided_criteria = q.get("suggested_criteria")

        # If provided_criteria is a comma-separated string, convert to list
        if isinstance(provided_criteria, str):
            provided_criteria = [c.strip() for c in provided_criteria.split(",") if c.strip()]

        try:
            rubric = _generate_rubric_matrix(
                q.get("question", ""),
                int(q.get("points", 0) or 0),
                int(leaf_columns),
                provided_criteria=provided_criteria,
                timeout_per_call=timeout_per_call
            )
            return ("ok", path_key, rubric)
        except Exception as e:
            return ("error", path_key, str(e))

    # Run in parallel but with a safe limit
    workers = min(max_workers, max(1, len(leaves)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process_leaf, entry): entry for entry in leaves}
        for fut in concurrent.futures.as_completed(futures):
            status, pk, payload = fut.result()
            if status == "ok":
                generated[pk] = payload
            elif status == "skip":
                skipped.append(pk)
            else:
                # error -> include an error entry under generated with error string
                generated[pk] = {"error": payload}

    return jsonify({"generated": generated, "skipped": skipped}), 200



@app.route('/delete_rubric', methods=['POST'])
def delete_rubric():
    """
    Delete a rubric for a specific question. Accepts:
      - { "path_key": "q-0-1" }
      - or { "path": [0,1] }

    Returns { "deleted": "<path_key>" } or 404 if not found.
    Note: this endpoint does NOT mutate your original exam storage — it returns the path_key so frontend
    can update its `exam_questions` state. If you want server-side persistence, adapt to your DB.
    """
    data = request.json or {}
    path_key = data.get('path_key')
    path = data.get('path')

    if not path_key and path is None:
        return jsonify({"error": "Provide 'path_key' or 'path'."}), 400

    if path_key is None and isinstance(path, list):
        path_key = "q-" + "-".join(map(str, path))

    # In this simple implementation we just return the path_key so the frontend can clear it locally.
    # If you have server-side exam storage, delete the rubric there and return a status.
    return jsonify({"deleted": path_key}), 200



@app.route('/grade', methods=['POST'])
def grade():
    print(f"\n{'=' * 50}\n--- Received GRADING request at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n{'=' * 50}")
    data = request.json
    try:
        grader = AIGrader(
            strictness_level=data.get('strictness_level'),
            course_material=data.get('course_material', ''),
            additional_instructions=data.get('additional_instructions', '')
        )
        results = grader.grade_exam(data['student_exam'])
        print(f"--- Grading complete. Sending response back to client. ---")
        return jsonify(results)
    except Exception as e:
        print(f"--> [FATAL ERROR] An unexpected error occurred in the /grade endpoint: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Starting Flask AI Grading Server with Teacher-Guided Rubrics...")
    app.run(host='0.0.0.0', port=5000)
