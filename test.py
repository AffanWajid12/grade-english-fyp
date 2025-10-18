import unittest
from backend import AIGrader

class TestAIGrader(unittest.TestCase):
    """
    Unit tests for the AIGrader backend class.
    These tests create dummy data to validate the grading logic.
    """

    def setUp(self):
        """Set up a common test environment before each test."""
        self.rubric = {
            "What is 2+2?": {"answer": "A", "points": 5},
            "Python is a ___ language.": {"answer": "programming", "points": 5},
            "What is Python?": {"answer": "high-level programming language", "points": 10}
        }
        self.course_material = "Python is a high-level programming language used for web development and data science."

        # Initialize graders with different strictness levels
        self.low_strict_grader = AIGrader('low', self.rubric, self.course_material)
        self.medium_strict_grader = AIGrader('medium', self.rubric, self.course_material)
        self.high_strict_grader = AIGrader('high', self.rubric, self.course_material)

    def test_grade_mcq_correct(self):
        """Test a correct MCQ answer."""
        exam = [{"question": "What is 2+2?", "type": "mcq", "answer": "A"}]
        result = self.medium_strict_grader.grade_exam(exam)
        self.assertEqual(result["What is 2+2?"]['score'], 5)

    def test_grade_mcq_incorrect(self):
        """Test an incorrect MCQ answer."""
        exam = [{"question": "What is 2+2?", "type": "mcq", "answer": "B"}]
        result = self.medium_strict_grader.grade_exam(exam)
        self.assertEqual(result["What is 2+2?"]['score'], 0)

    def test_grade_fill_in_the_blanks_correct(self):
        """Test a correct fill-in-the-blanks answer."""
        exam = [{"question": "Python is a ___ language.", "type": "fill_in_the_blanks", "answer": "programming"}]
        result = self.medium_strict_grader.grade_exam(exam)
        self.assertEqual(result["Python is a ___ language."]['score'], 5)

    def test_grade_short_question_high_strictness_fail(self):
        """Test a short answer that fails high strictness but might pass lower."""
        exam = [{"question": "What is Python?", "type": "short_question", "answer": "a language"}]
        result = self.high_strict_grader.grade_exam(exam)
        self.assertEqual(result["What is Python?"]['score'], 0)

    def test_grade_short_question_low_strictness_pass(self):
        """Test a short answer that passes with low strictness."""
        exam = [{"question": "What is Python?", "type": "short_question", "answer": "a programming language"}]
        result = self.low_strict_grader.grade_exam(exam)
        # Should get partial or full credit with low strictness
        self.assertGreater(result["What is Python?"]['score'], 0)

    def test_grade_short_question_perfect_answer(self):
        """Test a perfect short answer that should pass all levels."""
        exam = [{"question": "What is Python?", "type": "short_question", "answer": "Python is a high-level programming language"}]
        result = self.high_strict_grader.grade_exam(exam)
        self.assertEqual(result["What is Python?"]['score'], 10)

    def test_question_not_in_rubric(self):
        """Test handling of a question that is not in the rubric."""
        exam = [{"question": "What is Java?", "type": "short_question", "answer": "Another language"}]
        result = self.medium_strict_grader.grade_exam(exam)
        self.assertIn("Error: This question was not found", result["What is Java?"]['feedback'])
        self.assertEqual(result["What is Java?"]['score'], 0)

    def test_empty_answer(self):
        """Test that an empty answer receives a score of 0."""
        exam = [{"question": "What is 2+2?", "type": "mcq", "answer": " "}]
        result = self.medium_strict_grader.grade_exam(exam)
        self.assertEqual(result["What is 2+2?"]['score'], 0)
        self.assertIn("No answer provided", result["What is 2+2?"]['feedback'])

if __name__ == '__main__':
    unittest.main()
