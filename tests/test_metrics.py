"""
Unit Tests for Metrics Computation
"""

import pytest
import pandas as pd
import numpy as np
from src.metrics.ai_intensity import compute_section_intensity, compute_document_intensity
from src.metrics.initiation_score import extract_qa_exchanges, compute_initiation_scores, QAExchange
from src.baselines.keyword_detector import AIKeywordDetector


class TestAIKeywordDetector:
    """Tests for keyword detection baseline."""
    
    @pytest.fixture
    def detector(self):
        return AIKeywordDetector()
    
    def test_detect_ai_keywords(self, detector):
        """Test basic AI keyword detection."""
        text = "We are investing in artificial intelligence and machine learning."
        matches = detector.detect(text)
        
        assert len(matches) >= 2
        keywords = [m.keyword.lower() for m in matches]
        assert 'artificial intelligence' in keywords
        assert 'machine learning' in keywords
    
    def test_detect_generative_ai(self, detector):
        """Test generative AI keyword detection."""
        text = "Our ChatGPT integration and generative AI tools are transformative."
        matches = detector.detect(text)
        
        keywords = [m.keyword.lower() for m in matches]
        assert 'chatgpt' in keywords or 'generative ai' in keywords
    
    def test_is_ai_related(self, detector):
        """Test binary AI detection."""
        assert detector.is_ai_related("We use artificial intelligence.") == True
        assert detector.is_ai_related("Revenue grew 10%.") == False
    
    def test_count_matches(self, detector):
        """Test category counting."""
        text = "Our AI and machine learning models improve automation."
        counts = detector.count_matches(text)
        
        assert counts['total'] >= 2
        assert 'core_ai' in counts
        assert 'automation' in counts
    
    def test_get_ai_score(self, detector):
        """Test AI score computation."""
        text = "AI AI AI machine learning deep learning"
        score = detector.get_ai_score(text, normalize=True)
        
        # Should have high score (many AI terms relative to word count)
        assert score > 0
    
    def test_exclusion_patterns(self, detector):
        """Test that false positives are excluded."""
        # "aid" should not match as "AI"
        text = "We provide financial aid to students."
        matches = detector.detect(text)
        
        keywords = [m.keyword.lower() for m in matches]
        assert 'ai' not in keywords or len(matches) == 0


class TestAIIntensity:
    """Tests for AI intensity metric computation."""
    
    @pytest.fixture
    def sample_sentences_df(self):
        """Create sample sentence data."""
        data = {
            'doc_id': ['TEST_2024Q1'] * 10,
            'section': ['speech'] * 5 + ['qa'] * 5,
            'text': [
                'We are investing in AI.',
                'Revenue increased.',
                'Machine learning is key.',
                'Our strategy is clear.',
                'Automation drives growth.',
                'What about AI?',
                'We use deep learning.',
                'How are margins?',
                'They are stable.',
                'Any AI plans?'
            ],
            'ml_is_ai': [1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
            'ml_ai_prob': [0.9, 0.1, 0.85, 0.2, 0.75, 0.88, 0.92, 0.15, 0.1, 0.8],
            'kw_is_ai': [1, 0, 1, 0, 1, 1, 1, 0, 0, 1]
        }
        return pd.DataFrame(data)
    
    def test_compute_section_intensity(self, sample_sentences_df):
        """Test section-level intensity computation."""
        result = compute_section_intensity(sample_sentences_df)
        
        assert len(result) == 2  # speech + qa
        
        speech_row = result[result['section'] == 'speech'].iloc[0]
        assert speech_row['total_sentences'] == 5
        assert speech_row['ml_ai_sentences'] == 3
        assert speech_row['ml_ai_ratio'] == 0.6
    
    def test_compute_document_intensity(self, sample_sentences_df):
        """Test document-level intensity aggregation."""
        section_metrics = compute_section_intensity(sample_sentences_df)
        doc_metrics = compute_document_intensity(section_metrics)
        
        assert len(doc_metrics) == 1
        assert doc_metrics['doc_id'].iloc[0] == 'TEST_2024Q1'
        assert 'speech_ml_ai_ratio' in doc_metrics.columns
        assert 'qa_ml_ai_ratio' in doc_metrics.columns


class TestInitiationScore:
    """Tests for AI initiation score computation."""
    
    @pytest.fixture
    def sample_qa_sentences(self):
        """Create sample Q&A sentence data."""
        data = {
            'doc_id': ['TEST_2024Q1'] * 8,
            'section': ['qa'] * 8,
            'turn_idx': [0, 1, 2, 3, 4, 5, 6, 7],
            'role': ['operator', 'analyst', 'management', 'analyst', 'management', 'analyst', 'management', 'management'],
            'speaker': ['Operator', 'Analyst1', 'CEO', 'Analyst2', 'CFO', 'Analyst3', 'CEO', 'CFO'],
            'text': [
                'Questions please.',
                'What about AI strategy?',
                'We are focused on AI.',
                'How are margins?',
                'Margins are stable.',
                'Can you discuss ML?',
                'Machine learning is key.',
                'And automation too.'
            ],
            'ml_is_ai': [0, 1, 1, 0, 0, 1, 1, 1],
            'ml_ai_prob': [0.1, 0.9, 0.85, 0.2, 0.15, 0.88, 0.92, 0.8]
        }
        return pd.DataFrame(data)
    
    def test_extract_qa_exchanges(self, sample_qa_sentences):
        """Test Q&A exchange extraction."""
        exchanges = extract_qa_exchanges(sample_qa_sentences)
        
        # Should have 3 exchanges (3 analyst questions)
        assert len(exchanges) == 3
        
        # First exchange: analyst asks about AI, management answers with AI
        assert exchanges[0].question_is_ai == True
        assert exchanges[0].answer_is_ai == True
    
    def test_compute_initiation_scores(self, sample_qa_sentences):
        """Test initiation score computation."""
        exchanges = extract_qa_exchanges(sample_qa_sentences)
        scores = compute_initiation_scores(exchanges)
        
        assert len(scores) == 1
        assert 'ai_initiation_score' in scores.columns
        assert 'analyst_initiated_ratio' in scores.columns
        assert 'management_pivot_ratio' in scores.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
