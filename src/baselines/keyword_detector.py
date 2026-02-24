"""
Keyword Detector Module

Dictionary-based detector for AI topic detection using keyword matching.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor


@dataclass
class KeywordMatch:
    """Represents a keyword match in text."""
    keyword: str
    category: str
    start: int
    end: int
    context: str  # Surrounding text


class AIKeywordDetector:
    """
    Dictionary-based detector for AI-related content.
    Uses curated keyword lists and regex patterns.
    """
    
    # AI-related keywords organized by category
    KEYWORD_DICT = {
        'core_ai': [
            'artificial intelligence', 'ai', 'machine learning', 'ml',
            'deep learning', 'neural network', 'neural net'
        ],
        'generative_ai': [
            'generative ai', 'genai', 'gen ai', 'chatgpt', 'gpt',
            'large language model', 'llm', 'llms',
            'copilot', 'co-pilot', 'gemini', 'claude', 'bard',
            'openai', 'anthropic', 'midjourney', 'dall-e', 'stable diffusion',
            'generative model', 'foundation model', 'transformer model', 'deepseek'
        ],
        'ml_techniques': [
            'natural language processing', 'nlp', 'computer vision',
            'predictive analytics', 'predictive model', 'prediction model',
            'recommendation engine', 'recommendation system',
            'classification model', 'regression model',
            'clustering', 'anomaly detection', 'sentiment analysis',
            'image recognition', 'speech recognition', 'voice recognition',
            'reinforcement learning', 'supervised learning', 'unsupervised learning'
        ],
        'automation': [
            'automation', 'automate', 'automated', 'automating',
            'robotic process automation', 'rpa',
            'intelligent automation', 'hyperautomation',
            'workflow automation', 'process automation'
        ],
        'data_analytics': [
            'data analytics', 'advanced analytics', 'big data',
            'data science', 'data-driven', 'algorithm', 'algorithmic'
        ],
        'ai_infrastructure': [
            'gpu', 'gpus', 'nvidia', 'cuda', 'tensor',
            'cloud computing', 'edge computing', 'ai chip',
            'ai infrastructure', 'compute capacity', 'training data', 'data center'
        ],
        'ai_applications': [
            'chatbot', 'virtual assistant', 'digital assistant',
            'smart assistant', 'conversational ai',
            'ai-powered', 'ai-driven', 'ai-enabled', 'ai-based',
            'machine intelligence', 'cognitive computing'
        ]
    }
    
    # Exclusion patterns (to avoid false positives)
    EXCLUSION_PATTERNS = [
        r'\bair\b',  # "air" not "AI"
        r'\baid\b',  # "aid" not "AI"
        r'\baim\b',  # "aim" not "AI"  
        r'\bailey\b',  # "bailey" not "AI"
        r'email',
        r'detail',
        r'retail',
        r'maintain',
        r'ertain',
        r'contain',
    ]
    
    def __init__(self, case_sensitive: bool = False):
        """
        Args:
            case_sensitive: Whether to use case-sensitive matching
        """
        self.case_sensitive = case_sensitive
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.patterns = {}
        
        for category, keywords in self.KEYWORD_DICT.items():
            patterns = []
            for kw in keywords:
                # Create word boundary pattern
                # Handle special cases like "AI" which is short
                if len(kw) <= 3:
                    # For short terms, require word boundaries
                    pattern = r'\b' + re.escape(kw) + r'\b'
                else:
                    # For longer terms, be more flexible
                    pattern = r'\b' + re.escape(kw) + r's?\b'  # Allow plural
                
                flags = 0 if self.case_sensitive else re.IGNORECASE
                patterns.append(re.compile(pattern, flags))
            
            self.patterns[category] = patterns
        
        # Compile exclusion patterns
        self.exclusions = [re.compile(p, re.IGNORECASE) for p in self.EXCLUSION_PATTERNS]
    
    def _is_excluded(self, text: str, match_start: int, match_end: int) -> bool:
        """Check if match should be excluded (false positive)."""
        # Get context around match
        context_start = max(0, match_start - 10)
        context_end = min(len(text), match_end + 10)
        context = text[context_start:context_end].lower()
        
        for excl in self.exclusions:
            if excl.search(context):
                return True
        return False
    
    def detect(self, text: str) -> List[KeywordMatch]:
        """
        Detect AI-related keywords in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of KeywordMatch objects
        """
        if not text:
            return []
        
        matches = []
        
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Check exclusions
                    if self._is_excluded(text, match.start(), match.end()):
                        continue
                    
                    # Get context (50 chars before and after)
                    ctx_start = max(0, match.start() - 50)
                    ctx_end = min(len(text), match.end() + 50)
                    context = text[ctx_start:ctx_end]
                    
                    matches.append(KeywordMatch(
                        keyword=match.group(),
                        category=category,
                        start=match.start(),
                        end=match.end(),
                        context=context
                    ))
        
        return matches
    
    def is_ai_related(self, text: str) -> bool:
        """
        Check if text contains any AI-related content.
        
        Args:
            text: Text to check
            
        Returns:
            True if AI-related keywords found
        """
        return len(self.detect(text)) > 0
    
    def count_matches(self, text: str) -> Dict[str, int]:
        """
        Count AI keyword matches by category.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of category -> count
        """
        matches = self.detect(text)
        counts = {cat: 0 for cat in self.KEYWORD_DICT.keys()}
        counts['total'] = 0
        
        for m in matches:
            counts[m.category] += 1
            counts['total'] += 1
        
        return counts
    
    def get_ai_score(self, text: str, normalize: bool = True) -> float:
        """
        Get AI intensity score for text.
        
        Args:
            text: Text to analyze
            normalize: If True, normalize by word count
            
        Returns:
            AI intensity score
        """
        matches = self.detect(text)
        count = len(matches)
        
        if normalize and text:
            word_count = len(text.split())
            if word_count > 0:
                return count / word_count * 100  # Per 100 words
        
        return float(count)


def _process_texts_chunk(texts: List[str]) -> List[Dict[str, int | float | bool]]:
    detector = AIKeywordDetector()
    results = []
    for text in texts:
        matches = detector.detect(text)
        counts = detector.count_matches(text)
        result = {
            'kw_is_ai': len(matches) > 0,
            'kw_match_count': len(matches),
            'kw_ai_score': detector.get_ai_score(text),
            **{f'kw_{cat}_count': counts[cat] for cat in detector.KEYWORD_DICT.keys()}
        }
        results.append(result)
    return results


def compute_keyword_metrics(
    sentences_df: pd.DataFrame,
    text_col: str = 'text',
    doc_id_col: str = 'doc_id',
    section_col: str = 'section',
    num_workers: Optional[int] = None,
    chunk_size: int = 2000
) -> pd.DataFrame:
    """
    Compute keyword-based AI metrics for sentences.
    
    Args:
        sentences_df: DataFrame with sentence data
        text_col: Column containing text
        doc_id_col: Column for document ID
        section_col: Column for section (speech/qa)
        
    Returns:
        DataFrame with added AI detection columns
    """
    print("Detecting AI keywords in sentences...")

    texts = sentences_df[text_col].fillna("").astype(str).tolist()
    total = len(texts)
    if total == 0:
        result_df = pd.DataFrame(columns=['kw_is_ai', 'kw_match_count', 'kw_ai_score'])
        return pd.concat([sentences_df.reset_index(drop=True), result_df], axis=1)

    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)

    if num_workers <= 1 or total < chunk_size:
        detector = AIKeywordDetector()
        results = []
        for text in tqdm(texts, total=total):
            matches = detector.detect(text)
            counts = detector.count_matches(text)
            result = {
                'kw_is_ai': len(matches) > 0,
                'kw_match_count': len(matches),
                'kw_ai_score': detector.get_ai_score(text),
                **{f'kw_{cat}_count': counts[cat] for cat in detector.KEYWORD_DICT.keys()}
            }
            results.append(result)
    else:
        print(f"Using multiprocessing with {num_workers} workers (chunk_size={chunk_size})")
        chunks = [texts[i:i + chunk_size] for i in range(0, total, chunk_size)]
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for chunk_res in tqdm(ex.map(_process_texts_chunk, chunks), total=len(chunks)):
                results.extend(chunk_res)

    result_df = pd.DataFrame(results)
    return pd.concat([sentences_df.reset_index(drop=True), result_df], axis=1)


def compute_document_metrics(
    sentences_df: pd.DataFrame,
    doc_id_col: str = 'doc_id',
    section_col: str = 'section'
) -> pd.DataFrame:
    """
    Aggregate keyword metrics at document level.
    
    Args:
        sentences_df: DataFrame with sentence-level keyword metrics
        
    Returns:
        DataFrame with document-level metrics
    """
    # Group by document and section
    agg_funcs = {
        'kw_is_ai': ['sum', 'mean'],
        'kw_match_count': 'sum',
        'kw_ai_score': 'mean'
    }
    
    results = []
    
    for doc_id in sentences_df[doc_id_col].unique():
        doc_df = sentences_df[sentences_df[doc_id_col] == doc_id]
        
        doc_result = {'doc_id': doc_id}
        
        for section in ['speech', 'qa']:
            section_df = doc_df[doc_df[section_col] == section]
            
            if len(section_df) > 0:
                doc_result[f'{section}_total_sentences'] = len(section_df)
                doc_result[f'{section}_ai_sentences'] = section_df['kw_is_ai'].sum()
                doc_result[f'{section}_ai_ratio'] = section_df['kw_is_ai'].mean()
                doc_result[f'{section}_total_matches'] = section_df['kw_match_count'].sum()
                doc_result[f'{section}_avg_ai_score'] = section_df['kw_ai_score'].mean()
            else:
                doc_result[f'{section}_total_sentences'] = 0
                doc_result[f'{section}_ai_sentences'] = 0
                doc_result[f'{section}_ai_ratio'] = 0.0
                doc_result[f'{section}_total_matches'] = 0
                doc_result[f'{section}_avg_ai_score'] = 0.0
        
        results.append(doc_result)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Demo usage
    detector = AIKeywordDetector()
    
    test_texts = [
        "We are investing heavily in artificial intelligence and machine learning capabilities.",
        "Our ChatGPT integration has been transformative for customer service.",
        "Revenue increased by 15% this quarter.",
        "The automation of our processes using AI has reduced costs significantly.",
        "We see generative AI as a key growth driver going forward."
    ]
    
    print("=== Keyword Detection Demo ===\n")
    for text in test_texts:
        matches = detector.detect(text)
        is_ai = detector.is_ai_related(text)
        score = detector.get_ai_score(text)
        
        print(f"Text: {text[:60]}...")
        print(f"  AI-related: {is_ai}")
        print(f"  AI Score: {score:.2f}")
        print(f"  Matches: {[m.keyword for m in matches]}")
        print()
