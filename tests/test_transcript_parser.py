"""
Unit Tests for Transcript Parser
"""

import pytest
import json
from src.preprocessing.transcript_parser import TranscriptParser, ParsedTranscript, Turn


class TestTranscriptParser:
    """Tests for TranscriptParser class."""
    
    @pytest.fixture
    def parser(self):
        return TranscriptParser()
    
    @pytest.fixture
    def sample_structured_content(self):
        """Sample structured content like real earnings calls."""
        return [
            {'speaker': 'Operator', 'text': 'Good morning and welcome to the Q4 earnings call.'},
            {'speaker': 'John Smith, CEO', 'text': 'Thank you operator. I am pleased to report strong results. Our AI initiatives are driving growth.'},
            {'speaker': 'Jane Doe, CFO', 'text': 'Revenue increased 15% year over year. We continue to invest in machine learning capabilities.'},
            {'speaker': 'Operator', 'text': 'We will now begin our question and answer session.'},
            {'speaker': 'Mike Johnson, Goldman Sachs', 'text': 'Can you elaborate on your AI strategy? What is the expected ROI?'},
            {'speaker': 'John Smith, CEO', 'text': 'Great question. Our AI investments are focused on customer experience and automation.'},
            {'speaker': 'Sarah Lee, Morgan Stanley', 'text': 'How is competition affecting margins?'},
            {'speaker': 'Jane Doe, CFO', 'text': 'We are seeing some pressure but our efficiency programs offset this.'}
        ]
    
    def test_parse_structured_content_list(self, parser, sample_structured_content):
        """Test parsing when content is already a list."""
        result = parser.parse_structured_content(sample_structured_content)
        assert len(result) == 8
        assert result[0]['speaker'] == 'Operator'
    
    def test_parse_structured_content_json_string(self, parser, sample_structured_content):
        """Test parsing when content is JSON string."""
        json_str = json.dumps(sample_structured_content)
        result = parser.parse_structured_content(json_str)
        assert len(result) == 8
    
    def test_classify_role_operator(self, parser):
        """Test operator role classification."""
        assert parser.classify_role('Operator') == 'operator'
        assert parser.classify_role('Conference Operator') == 'operator'
    
    def test_classify_role_management(self, parser):
        """Test management role classification."""
        assert parser.classify_role('John Smith, CEO') == 'management'
        assert parser.classify_role('Jane Doe, Chief Financial Officer') == 'management'
        assert parser.classify_role('Bob Jones, President') == 'management'
    
    def test_classify_role_analyst(self, parser):
        """Test analyst role classification."""
        assert parser.classify_role('Mike Johnson, Goldman Sachs') == 'analyst'
        assert parser.classify_role('Sarah Lee, Morgan Stanley Research') == 'analyst'
    
    def test_find_qa_start_index(self, parser, sample_structured_content):
        """Test Q&A section detection."""
        qa_idx = parser.find_qa_start_index(sample_structured_content)
        assert qa_idx == 3  # Operator announces Q&A at index 3
    
    def test_parse_full_transcript(self, parser, sample_structured_content):
        """Test full transcript parsing."""
        result = parser.parse(
            structured_content=sample_structured_content,
            ticker='TEST',
            date='2024-01-15',
            quarter=4,
            year=2024
        )
        
        assert isinstance(result, ParsedTranscript)
        assert result.ticker == 'TEST'
        assert result.quarter == 4
        assert result.year == 2024
        
        # Check speech/qa split
        assert len(result.speech_turns) == 3  # Operator + CEO + CFO
        assert len(result.qa_turns) == 5  # Operator + 2 questions + 2 answers
        
        assert result.speech_word_count > 0
        assert result.qa_word_count > 0
    
    def test_is_question(self, parser):
        """Test question detection."""
        assert parser.is_question("What is your AI strategy?") == True
        assert parser.is_question("How will this affect margins?") == True
        assert parser.is_question("Can you elaborate on that?") == True
        assert parser.is_question("Revenue increased 15%.") == False
    
    def test_empty_content(self, parser):
        """Test handling of empty content."""
        result = parser.parse(
            structured_content=[],
            ticker='TEST',
            date='2024-01-15',
            quarter=1,
            year=2024
        )
        
        assert result.speech_text == ""
        assert result.qa_text == ""
        assert result.speech_word_count == 0
    
    def test_to_dict(self, parser, sample_structured_content):
        """Test serialization to dictionary."""
        result = parser.parse(
            structured_content=sample_structured_content,
            ticker='TEST',
            date='2024-01-15',
            quarter=4,
            year=2024
        )
        
        d = result.to_dict()
        
        assert 'ticker' in d
        assert 'speech_text' in d
        assert 'qa_turns' in d
        assert isinstance(d['speech_turns'], list)


class TestTurn:
    """Tests for Turn dataclass."""
    
    def test_turn_creation(self):
        turn = Turn(
            speaker="John Smith",
            role="management",
            text="This is a test.",
            is_question=False
        )
        
        assert turn.speaker == "John Smith"
        assert turn.role == "management"
        assert turn.is_question == False
    
    def test_turn_to_dict(self):
        turn = Turn(
            speaker="Analyst",
            role="analyst",
            text="What about AI?",
            is_question=True
        )
        
        d = turn.to_dict()
        assert d['is_question'] == True
        assert d['role'] == 'analyst'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
