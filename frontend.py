import streamlit as st
import requests
import PyPDF2
import io
from pptx import Presentation

# ================== CONFIGURATION ==================
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

BACKEND_URL_GRADE = "http://127.0.0.1:5000/grade"
BACKEND_URL_RUBRIC = "http://127.0.0.1:5000/generate_question_rubric"

# ================== HELPERS ==================
def extract_text_from_files(uploaded_files):
    """Extract readable text from PDF, TXT, and PPTX files."""
    full_text = []
    if not uploaded_files:
        return ""
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type == "text/plain":
                full_text.append(uploaded_file.getvalue().decode("utf-8"))
            elif uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text.append(text)
            elif "presentationml" in uploaded_file.type:
                prs = Presentation(io.BytesIO(uploaded_file.getvalue()))
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            full_text.append(shape.text)
        except Exception as e:
            st.error(f"⚠️ Error processing file {uploaded_file.name}: {e}")
    return "\n".join(full_text)

# ================== PAGE SETUP ==================
st.set_page_config(layout="wide", page_title="AI Grading Assistant")
st.title("📝 AI Grading Assistant v12.2")

if "exam_questions" not in st.session_state:
    st.session_state.exam_questions = []

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ Configuration")

    use_strictness = st.toggle("Enable Strictness Level", value=True)
    strictness_instruction = ""
    if use_strictness:
        level = st.slider("Strictness Level", 1, 10, 6)
        strictness_instruction = STRICTNESS_LEVELS[level]
        st.info(f"**AI Instruction:** {strictness_instruction}")

    additional_instructions = st.text_area(
        "Custom AI Command (Optional)",
        height=150,
        help="Overrides strictness level if filled."
    )

    uploaded_files = st.file_uploader(
        "Upload Course Materials (Optional)",
        type=["pdf", "txt", "pptx"],
        accept_multiple_files=True
    )

# ================== QUESTION MANAGEMENT ==================
def delete_question_part(path):
    data = st.session_state.exam_questions
    for idx in path[:-1]:
        data = data[idx]["parts"]
    data.pop(path[-1])

def add_sub_part(path):
    data = st.session_state.exam_questions
    for idx in path:
        data = data[idx]["parts"]
    data.append({
        "question": "",
        "answer": "",
        "points": 1,
        "type": "short_question",
        "parts": [],
        "rubric": None
    })

def generate_rubric_for_question(path):
    """Generate rubric for a specific question."""
    data = st.session_state.exam_questions
    item = data
    for idx in path:
        item = item[idx] if isinstance(item, list) else item["parts"][idx]

    question_text = item.get("question", "")
    points = item.get("points", 0)
    cols = item.get("rubric_cols", 3)
    suggested_criteria = item.get("suggested_criteria", "").strip() or "auto"

    if not question_text or not points:
        st.warning("Please provide question text and points before generating a rubric.")
        return

    with st.spinner("Generating rubric using AI..."):
        try:
            payload = {
                "question": question_text,
                "points": points,
                "columns": cols,
                "criteria": [],  # AI will handle it now
                "suggested_criteria": suggested_criteria
            }
            response = requests.post(BACKEND_URL_RUBRIC, json=payload)
            response.raise_for_status()
            item["rubric"] = response.json()
            st.success("Rubric generated successfully ✅")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to generate rubric: {e}")

# ================== DISPLAY QUESTIONS ==================
def display_questions(questions, path_prefix=[]):
    for i, item in enumerate(questions):
        path = path_prefix + [i]
        key_prefix = "-".join(map(str, path))

        with st.container(border=True):
            cols1 = st.columns([10, 2, 2])
            item["question"] = cols1[0].text_input("Question", value=item.get("question", ""), key=f"q_{key_prefix}")
            cols1[1].button("➕ Sub-part", key=f"add_{key_prefix}", on_click=add_sub_part, args=(path,))
            cols1[2].button("❌ Delete", key=f"del_{key_prefix}", on_click=delete_question_part, args=(path,))

            cols2 = st.columns([3, 1])
            item["type"] = cols2[0].selectbox(
                "Type",
                ("mcq", "fill_in_the_blanks", "short_question", "long_question"),
                index=2,
                key=f"type_{key_prefix}"
            )
            item["points"] = cols2[1].number_input(
                "Max Points", min_value=1, value=max(1, item.get("points", 10)), key=f"pts_{key_prefix}"
            )

            if not item.get("parts"):
                item["answer"] = st.text_area("Student Answer", value=item.get("answer", ""), key=f"ans_{key_prefix}", height=100)
                st.markdown("**Rubric (Optional)**")
                r1, r2 = st.columns(2)

                # Rubric columns (still needed)
                item["rubric_cols"] = r1.number_input("Columns", 2, 5, 3, key=f"rub_cols_{key_prefix}")

                # Criteria input (AI or manual)
                item["suggested_criteria"] = st.text_input(
                    "Suggest Criteria (comma-separated or type 'auto' for AI generation)",
                    key=f"sug_crit_{key_prefix}",
                    help="Leave empty for default criteria or type 'auto' to let AI generate them."
                )

                st.button("🧠 Generate AI Rubric", key=f"rub_{key_prefix}", on_click=generate_rubric_for_question, args=(path,))

                if item.get("rubric"):
                    st.markdown("**Editable Rubric Table:**")
                    item["rubric"] = st.data_editor(
                        item["rubric"],
                        num_rows="dynamic",
                        use_container_width=True,
                        key=f"edit_rub_{key_prefix}"
                    )
            else:
                st.info("This is a parent question. Gradeable answers are in its sub-parts.")
                display_questions(item["parts"], path_prefix=path)


# ================== ADD / CLEAR BUTTONS ==================
st.markdown("---")
cols = st.columns([1, 1, 3])
cols[0].button("➕ Add Top-Level Question", on_click=lambda: st.session_state.exam_questions.append({
    "question": "",
    "answer": "",
    "points": 10,
    "type": "short_question",
    "parts": [],
    "rubric": None
}))
cols[1].button("🧹 Clear All", on_click=lambda: st.session_state.pop("exam_questions"))

display_questions(st.session_state.exam_questions)

# ================== GRADE BUTTON ==================
st.markdown("---")
if st.button("🚀 Grade Exam", type="primary", use_container_width=True):
    if not st.session_state.exam_questions:
        st.warning("Please add at least one question before grading.")
    else:
        with st.spinner("Grading in progress..."):
            try:
                payload = {
                    "strictness_level": strictness_instruction,
                    "course_material": extract_text_from_files(uploaded_files),
                    "additional_instructions": additional_instructions,
                    "student_exam": st.session_state.exam_questions
                }
                response = requests.post(BACKEND_URL_GRADE, json=payload)
                response.raise_for_status()
                st.session_state.results = response.json()
                st.success("✅ Grading complete!")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")

# ================== RESULTS DISPLAY ==================
if "results" in st.session_state:
    st.header("📊 Grading Results")
    results = st.session_state.results

    def calc_total(questions):
        total, max_total = 0, 0
        for q in questions:
            if q.get("parts"):
                t, m = calc_total(q["parts"])
                total += t
                max_total += m
            else:
                result = results.get(q.get("question", ""), {})
                total += result.get("score", 0)
                max_total += q.get("points", 0)
        return total, max_total

    def show_results(questions):
        for q in questions:
            res = results.get(q.get("question", ""), {})
            if res:
                with st.expander(f"{q.get('question')} — {res.get('score', 0)}/{res.get('max_score', q.get('points', 0))}"):
                    st.markdown("**Student Answer:**")
                    st.info(res.get("student_answer", "N/A"))
                    st.markdown("**AI Feedback:**")
                    st.success(res.get("feedback", "No feedback provided."))
            if q.get("parts"):
                show_results(q["parts"])

    total, max_total = calc_total(st.session_state.exam_questions)
    st.subheader(f"🏆 Final Score: {total} / {max_total}")
    show_results(st.session_state.exam_questions)

    if st.button("🗑️ Clear Results"):
        del st.session_state.results
        st.rerun()
