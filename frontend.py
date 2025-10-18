import streamlit as st
import requests
import PyPDF2
import io
from pptx import Presentation
import json

# --- Configuration for Strictness Levels ---
STRICTNESS_LEVELS = {
    "None": "",
    1: "Be extremely lenient. Award points for any attempt, even if it's mostly incorrect. Focus on effort.",
    2: "Be very lenient. Award partial credit generously. The student's answer only needs to hint at the correct concept.",
    3: "Be lenient. The student's answer can be partially incorrect or incomplete and still get most of the points.",
    4: "Be slightly lenient. Minor inaccuracies are acceptable. The core concept should be generally correct.",
    5: "Be balanced. A fair mix of strictness and leniency. The answer should be correct but can have small errors.",
    6: "Be slightly strict. The answer must be mostly correct. Deduct points for minor errors or omissions.",
    7: "Be strict. The answer must be correct and well-explained. Deduct points for any inaccuracies.",
    8: "Be very strict. The answer must be precise and fully correct. Small mistakes should result in significant point deductions.",
    9: "Be extremely strict. The answer must be perfect, matching the rubric's intent exactly. No partial credit unless flawless.",
    10: "Be unforgiving. The answer must be absolutely perfect in every detail. Any deviation from the ideal answer results in zero points."
}

# --- Backend API URLs ---
BACKEND_URL_GRADE = "http://127.0.0.1:5000/grade"

def extract_text_from_files(uploaded_files):
    """Extracts text from various file types for RAG context."""
    full_text = []
    if not uploaded_files:
        return ""
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type == "text/plain":
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                full_text.append(stringio.read())
            elif uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    full_text.append(page.extract_text())
            elif uploaded_file.type in [
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/vnd.ms-powerpoint"
            ]:
                prs = Presentation(io.BytesIO(uploaded_file.getvalue()))
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            full_text.append(shape.text)
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
    return "\n".join(full_text)

st.set_page_config(layout="wide", page_title="AI Grading Assistant")
st.title("📝 AI Grading Assistant v9.0 (Manual Entry)")
st.markdown("Manually enter questions, answers, and sub-parts to create a detailed exam structure for AI grading.")

# --- Session State Initialization ---
if 'exam_questions' not in st.session_state:
    st.session_state.exam_questions = []

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Grader Configuration")
    strictness_options = list(STRICTNESS_LEVELS.keys())
    strictness_level_selection = st.selectbox("Select Strictness Level", options=strictness_options, index=0)
    strictness_instruction = STRICTNESS_LEVELS[strictness_level_selection]
    if strictness_instruction: st.info(f"**AI Instruction:** \"{strictness_instruction}\"")
    additional_instructions = st.text_area("Overriding Command (Optional)", height=150, placeholder="e.g., 'Give full marks for any attempt.'")
    uploaded_files = st.file_uploader("Upload Course Materials (Optional)", accept_multiple_files=True, type=['pdf', 'txt', 'pptx'])

# --- Main Area for Exam Input ---
st.header("📄 Student's Exam Paper")
st.markdown("Use the buttons below to build the student's exam, including questions with nested sub-parts.")

# --- RECURSIVE UI RENDERING ---
def delete_question_part(path):
    """Deletes a question/part at a specific path in the session state."""
    data = st.session_state.exam_questions
    for index in path[:-1]: data = data[index]['parts']
    data.pop(path[-1])

def add_sub_part(path):
    """Adds a new sub-part to a question."""
    data = st.session_state.exam_questions
    target_list = data
    for index in path: target_list = target_list[index]['parts']
    target_list.append({"question": "", "answer": "", "points": 1, "type": "short_question", "parts": []})

def display_questions(questions_list, path_prefix=[]):
    """Recursively displays UI for questions and their sub-parts."""
    for i, item in enumerate(questions_list):
        current_path = path_prefix + [i]
        unique_key = "-".join(map(str, current_path))
        with st.container(border=True):
            cols1 = st.columns([12, 2, 2])
            item['question'] = cols1[0].text_input("Question Text", value=item['question'], key=f"q_{unique_key}")
            cols1[1].button("➕ Sub-part", key=f"add_{unique_key}", on_click=add_sub_part, args=(current_path,))
            cols1[2].button("❌ Delete", key=f"del_{unique_key}", on_click=delete_question_part, args=(current_path,))
            cols2 = st.columns([3, 1])
            item['type'] = cols2[0].selectbox("Type", ('mcq', 'fill_in_the_blanks', 'short_question', 'long_question'), index=2, key=f"type_{unique_key}")
            points_value = max(1, item.get('points', 10))
            item['points'] = cols2[1].number_input("Max Points", min_value=1, value=points_value, key=f"pts_{unique_key}")
            
            # A question is a container if it has sub-parts.
            is_container = 'parts' in item and len(item['parts']) > 0
            if not is_container:
                item['answer'] = st.text_area("Student's Answer", value=item['answer'], key=f"ans_{unique_key}", height=100)
            else:
                # For container questions, the answer is irrelevant, so we ensure it's empty.
                item['answer'] = ""
                st.info("This is a container question. Gradeable answers are in the sub-parts below.")

            if 'parts' in item and item['parts']:
                display_questions(item['parts'], path_prefix=current_path)

def add_top_level_question():
    st.session_state.exam_questions.append({"question": "", "answer": "", "points": 10, "type": "short_question", "parts": []})

def clear_all_questions():
    st.session_state.exam_questions = []
    if 'results' in st.session_state: del st.session_state.results

# --- Initial UI Display & Controls ---
display_questions(st.session_state.exam_questions)

st.markdown("---")
bottom_cols = st.columns([1, 1, 3])
bottom_cols[0].button("➕ Add Top-Level Question", on_click=add_top_level_question, use_container_width=True)
bottom_cols[1].button("🧹 Clear All Questions", on_click=clear_all_questions, use_container_width=True)

# --- Grading Logic ---
if st.button("🚀 Grade Exam", type="primary", use_container_width=True):
    if not st.session_state.exam_questions: st.warning("Please add at least one question before grading.")
    else:
        with st.spinner("Sending exam to the grading server..."):
            course_material_text = extract_text_from_files(uploaded_files)
            payload = {"strictness_level": strictness_instruction, "course_material": course_material_text, "additional_instructions": additional_instructions, "student_exam": st.session_state.exam_questions}
            try:
                response = requests.post(BACKEND_URL_GRADE, json=payload)
                response.raise_for_status()
                st.session_state.results = response.json()
                st.success("Grading complete! See results below.")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the grading server: {e}")

# --- Display Results ---
if 'results' in st.session_state:
    st.header("📊 Grading Results")
    results_data = st.session_state.results

    def calculate_intelligent_total(questions_list):
        """Recursively calculates the total score, only counting leaf nodes."""
        total_score, max_score = 0, 0
        for item in questions_list:
            is_container = 'parts' in item and len(item['parts']) > 0
            if not is_container:
                result = results_data.get(item['question'])
                if result: total_score += result.get('score', 0)
                max_score += item.get('points', 0)
            else:
                sub_total, sub_max = calculate_intelligent_total(item['parts'])
                total_score += sub_total
                max_score += sub_max
        return total_score, max_score

    def display_nested_results(questions_list):
        """Recursively displays results, only showing expanders for graded questions."""
        for item in questions_list:
            result = results_data.get(item['question'])
            # Only show results for questions that were actually graded (i.e., not containers)
            if result and result.get('student_answer'):
                with st.expander(f"**{item['question']}** - Score: {result.get('score', 'N/A')} / {result.get('max_score', 'N/A')}"):
                    st.markdown("**Student's Answer:**"); st.info(result['student_answer'])
                    st.markdown("**AI Feedback:**"); st.success(result['feedback'])
            # Recurse into sub-parts to display their results
            if 'parts' in item and item['parts']: display_nested_results(item['parts'])
    
    final_score, max_possible_score = calculate_intelligent_total(st.session_state.exam_questions)
    
    display_nested_results(st.session_state.exam_questions)
    
    st.subheader(f"🏆 Final Score: {final_score} / {max_possible_score}")
    
    if st.button("Clear Results"):
        del st.session_state.results
        st.rerun()

