import pytest
import json
from unittest.mock import MagicMock, patch
import sys
import os

# Add the current directory to sys.path so we can import backend
sys.path.append(os.getcwd())

from backend import app, AIGrader, _calculate_mark_ranges

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def grader():
    return AIGrader()

# --- Unit Tests for Helper Functions ---

def test_calculate_mark_ranges_3_columns():
    """Test mark range calculation for 3 columns (default)."""
    ranges = _calculate_mark_ranges(10, 3)
    assert len(ranges) == 3
    # Expected: 10/3 = 3.33. 
    # 1: ceil(10 - 3.33) = 7 -> 7-10
    # 2: ceil(10 - 6.66) = 4 -> 4-6
    # 3: 0 -> 0-3
    assert "7-10 Marks" in ranges[0]
    assert "4-6 Marks" in ranges[1]
    assert "0-3 Marks" in ranges[2]

def test_calculate_mark_ranges_single_point():
    """Test mark range calculation for a small number of points."""
    ranges = _calculate_mark_ranges(2, 2)
    assert len(ranges) == 2
    # 2/2 = 1
    # 1: ceil(2-1) = 1 -> 1-2
    # 2: 0 -> 0
    assert "1-2 Marks" in ranges[0]
    assert "0 Marks" in ranges[1]

def test_strictness_multiplier():
    """Test the strictness multiplier logic."""
    grader = AIGrader()
    assert grader._strictness_multiplier("Balanced") == pytest.approx(1.0)
    assert grader._strictness_multiplier("Be extremely lenient") == pytest.approx(1.15)
    assert grader._strictness_multiplier("Be unforgiving") == pytest.approx(0.80)
    assert grader._strictness_multiplier("Unknown text") == pytest.approx(1.0)
    assert grader._strictness_multiplier(None) == pytest.approx(1.0)

# --- Unit Tests for AIGrader Class ---

def test_grader_initialization():
    """Test that AIGrader initializes correctly."""
    grader = AIGrader(strictness_level="Strict", additional_instructions="No cheating")
    assert grader.strictness_level == "Strict"
    assert grader.additional_instructions == "No cheating"
    assert grader.text_chunks == []

@patch('backend.requests.post')
def test_grade_exam_mocked(mock_post, grader):
    """Test grade_exam with mocked LLM response."""
    # Mock response for score
    mock_score_response = MagicMock()
    mock_score_response.json.return_value = {"response": "7"}
    mock_score_response.raise_for_status.return_value = None
    
    # Mock response for feedback
    mock_feedback_response = MagicMock()
    mock_feedback_response.json.return_value = {"response": "**Positive Points:**\nGood job."}
    
    # We need to handle multiple calls to requests.post (one for score, one for feedback)
    mock_post.side_effect = [mock_score_response, mock_feedback_response]

    student_exam = [
        {
            "question": "What is AI?",
            "answer": "Artificial Intelligence",
            "points": 10,
            "rubric": []
        }
    ]

    results = grader.grade_exam(student_exam)
    
    # Check structure of results
    key = "q-0"
    assert key in results
    assert results[key]['score'] == 7
    assert "Good job" in results[key]['feedback']
    assert results[key]['max_score'] == 10

@patch('backend.requests.post')
def test_grade_exam_nested_mocked(mock_post, grader):
    """Test grade_exam with nested questions."""
    # Mock responses: 2 calls per leaf question. We have 1 leaf here.
    mock_score = MagicMock()
    mock_score.json.return_value = {"response": "5"}
    mock_feedback = MagicMock()
    mock_feedback.json.return_value = {"response": "Feedback"}
    mock_post.side_effect = [mock_score, mock_feedback]

    student_exam = [
        {
            "question": "Parent",
            "parts": [
                {
                    "question": "Child",
                    "answer": "Ans",
                    "points": 5
                }
            ]
        }
    ]

    results = grader.grade_exam(student_exam)
    
    # Path should be q-0-0
    assert "q-0-0" in results
    assert results["q-0-0"]['score'] == 5

# --- Unit Tests for Flask Routes ---

@patch('backend._generate_rubric_matrix')
def test_generate_question_rubric_route(mock_gen_matrix, client):
    """Test the /generate_question_rubric endpoint."""
    mock_gen_matrix.return_value = [{"Criterion": "Test", "Level 1": "Desc"}]
    
    payload = {
        "question": "Test Q",
        "points": 10,
        "path": [0]
    }
    
    response = client.post('/generate_question_rubric', json=payload)
    assert response.status_code == 200
    data = response.json
    assert "rubric" in data
    assert data["path_key"] == "q-0"

def test_delete_rubric_route(client):
    """Test the /delete_rubric endpoint."""
    payload = {"path_key": "q-0-1"}
    response = client.post('/delete_rubric', json=payload)
    assert response.status_code == 200
    assert response.json["deleted"] == "q-0-1"

@patch('backend.AIGrader.grade_exam')
def test_grade_route(mock_grade_exam, client):
    """Test the /grade endpoint."""
    mock_grade_exam.return_value = {"q-0": {"score": 10}}
    
    payload = {
        "student_exam": [],
        "strictness_level": "Balanced"
    }
    
    response = client.post('/grade', json=payload)
    assert response.status_code == 200

if __name__ == "__main__":
    print("Test backend imported successfully")

