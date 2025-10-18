import streamlit as st
import requests
import PyPDF2
import io
from pptx import Presentation
import json

# --- Configuration ---
STRICTNESS_LEVELS = {
    1: "Be extremely lenient. Award points for any attempt.",
    2: "Be very lenient. Award partial credit generously.",
    3: "Be lenient. The answer can be partially incorrect.",
    4: "Be slightly lenient. Minor inaccuracies are acceptable.",
    5: "Be balanced. A fair mix of strictness and leniency.",
    6: "Be slightly strict. The answer must be mostly correct.",
    7: "Be strict. Deduct points for any inaccuracies.",
    8: "Be very strict. The answer must be precise and fully correct.",
    9: "Be extremely strict. The answer must be perfect.",
    10: "Be unforgiving. Any deviation results in zero points."
}
# --- FIX: Corrected the URL format ---
BACKEND_URL_GRADE = "http://127.0.0.1:5000/grade"
BACKEND_URL_RUBRIC = "http://127.0.0.1:5000/generate_question_rubric"

def extract_text_from_files(uploaded_files):
    full_text = []
    if not uploaded_files: return ""
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type == "text/plain":
                full_text.append(io.StringIO(uploaded_file.getvalue().decode("utf-8")).read())
            elif uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                full_text.extend([page.extract_text() for page in pdf_reader.pages])
            elif "presentationml" in uploaded_file.type:
                prs = Presentation(io.BytesIO(uploaded_file.getvalue()))
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"): full_text.append(shape.text)
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
    return "\n".join(full_text)

st.set_page_config(layout="wide", page_title="AI Grading Assistant")
st.title("📝 AI Grading Assistant v11.1 (Rubrics & Slider)")

if 'exam_questions' not in st.session_state: st.session_state.exam_questions = []

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Grader Configuration")
    
    use_strictness = st.toggle("Enable Strictness Level", value=True)
    strictness_instruction = ""
    if use_strictness:
        strictness_level_num = st.slider("Select Strictness Level (1-10)", 1, 10, 6)
        strictness_instruction = STRICTNESS_LEVELS[strictness_level_num]
        st.info(f"**AI Instruction:** \"{strictness_instruction}\"")
    
    additional_instructions = st.text_area("Overriding Command (Optional)", height=150, help="This will override the strictness level.")
    uploaded_files = st.file_uploader("Upload Course Materials (Optional)", accept_multiple_files=True, type=['pdf', 'txt', 'pptx'])

st.header("📄 Student's Exam Paper")

def delete_question_part(path):
    data = st.session_state.exam_questions
    for index in path[:-1]: data = data[index]['parts']
    data.pop(path[-1])

def add_sub_part(path):
    data = st.session_state.exam_questions
    target_list = data
    for index in path: target_list = target_list[index]['parts']
    target_list.append({"question": "", "answer": "", "points": 1, "type": "short_question", "parts": [], "rubric": None})

def generate_rubric_for_question(path):
    data = st.session_state.exam_questions
    item = data
    for index in path:
        item = item[index] if isinstance(item, list) else item['parts'][index]
    question_text, points, columns = item.get('question', ''), item.get('points', 0), item.get('rubric_cols', 3)
    if not question_text or not points:
        st.warning("Please provide question text and points before generating a rubric.")
        return
    with st.spinner(f"Generating rubric..."):
        try:
            payload = {"question": question_text, "points": points, "columns": columns}
            response = requests.post(BACKEND_URL_RUBRIC, json=payload)
            response.raise_for_status()
            item['rubric'] = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to generate rubric: {e}")

def display_questions(questions_list, path_prefix=[]):
    for i, item in enumerate(questions_list):
        current_path = path_prefix + [i]
        unique_key = "-".join(map(str, current_path))
        with st.container(border=True):
            cols1 = st.columns([12, 2, 2])
            item['question'] = cols1[0].text_input("Question Text", value=item.get('question', ''), key=f"q_{unique_key}")
            cols1[1].button("➕ Sub-part", key=f"add_{unique_key}", on_click=add_sub_part, args=(current_path,))
            cols1[2].button("❌ Delete", key=f"del_{unique_key}", on_click=delete_question_part, args=(current_path,))
            cols2 = st.columns([3, 1])
            item['type'] = cols2[0].selectbox("Type", ('mcq', 'fill_in_the_blanks', 'short_question', 'long_question'), index=2, key=f"type_{unique_key}")
            item['points'] = cols2[1].number_input("Max Points", min_value=1, value=max(1, item.get('points', 10)), key=f"pts_{unique_key}")
            is_container = 'parts' in item and len(item.get('parts', [])) > 0
            if not is_container:
                item['answer'] = st.text_area("Student's Answer", value=item.get('answer', ''), key=f"ans_{unique_key}", height=100)
                st.markdown("###### Rubric Generation")
                rubric_cols_ui = st.columns([2, 3])
                item['rubric_cols'] = rubric_cols_ui[0].number_input("Rubric Columns", min_value=2, max_value=5, value=3, key=f"rub_cols_{unique_key}")
                rubric_cols_ui[1].button("Generate AI Rubric", key=f"gen_rub_{unique_key}", on_click=generate_rubric_for_question, args=(current_path,))
                if item.get('rubric'):
                    st.markdown("###### Editable Rubric")
                    item['rubric'] = st.data_editor(item['rubric'], num_rows="dynamic", use_container_width=True, key=f"edit_rub_{unique_key}")
            else:
                item['answer'] = ""
                st.info("This is a container question. Gradeable answers are in the sub-parts below.")
            if 'parts' in item and item.get('parts'):
                display_questions(item['parts'], path_prefix=current_path)

def add_top_level_question():
    st.session_state.exam_questions.append({"question": "", "answer": "", "points": 10, "type": "short_question", "parts": [], "rubric": None})

def clear_all_questions():
    st.session_state.exam_questions = []
    if 'results' in st.session_state: del st.session_state.results

display_questions(st.session_state.exam_questions)
st.markdown("---")
bottom_cols = st.columns([1, 1, 3])
bottom_cols[0].button("➕ Add Top-Level Question", on_click=add_top_level_question)
bottom_cols[1].button("🧹 Clear All Questions", on_click=clear_all_questions)

if st.button("🚀 Grade Exam", type="primary", use_container_width=True):
    if not st.session_state.exam_questions: st.warning("Please add at least one question.")
    else:
        with st.spinner("Sending exam to the grading server..."):
            course_material_text = extract_text_from_files(uploaded_files)
            payload = {
                "strictness_level": strictness_instruction,
                "course_material": course_material_text,
                "additional_instructions": additional_instructions,
                "student_exam": st.session_state.exam_questions
            }
            try:
                response = requests.post(BACKEND_URL_GRADE, json=payload)
                response.raise_for_status()
                st.session_state.results = response.json()
                st.success("Grading complete!")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the grading server: {e}")

if 'results' in st.session_state:
    st.header("📊 Grading Results")
    results_data = st.session_state.results
    def calculate_intelligent_total(questions_list):
        total_score, max_score = 0, 0
        for item in questions_list:
            is_container = 'parts' in item and len(item.get('parts', [])) > 0
            if not is_container:
                result = results_data.get(item.get('question', ''))
                if result: total_score += result.get('score', 0)
                max_score += item.get('points', 0)
            else:
                sub_total, sub_max = calculate_intelligent_total(item.get('parts', []))
                total_score += sub_total; max_score += sub_max
        return total_score, max_score
    def display_nested_results(questions_list):
        for item in questions_list:
            question_text = item.get('question', '')
            result = results_data.get(question_text)
            if result and result.get('student_answer'):
                with st.expander(f"**{question_text}** - Score: {result.get('score', 'N/A')} / {result.get('max_score', 'N/A')}"):
                    st.markdown("**Student's Answer:**"); st.info(result['student_answer'])
                    st.markdown("**AI Feedback:**"); st.success(result['feedback'])
            if 'parts' in item and item.get('parts'):
                display_nested_results(item['parts'])
    if "error" in results_data:
        st.error(f"An error occurred on the backend: {results_data['error']}")
    else:
        final_score, max_possible_score = calculate_intelligent_total(st.session_state.exam_questions)
        display_nested_results(st.session_state.exam_questions)
        st.subheader(f"🏆 Final Score: {final_score} / {max_possible_score}")
        if st.button("Clear Results"):
            del st.session_state.results
            st.rerun()

