"""
Transcript Parser Module

Parses earnings call transcripts to split into Speech (prepared remarks) 
and Q&A sections, with turn-level extraction for Q&A.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
from tqdm import tqdm


@dataclass
class Turn:
    """Represents a single turn in the Q&A session."""
    speaker: str
    role: str  # 'analyst', 'management', 'operator', 'unknown'
    text: str
    is_question: bool = False
    
    def to_dict(self) -> dict:
        return {
            'speaker': self.speaker,
            'role': self.role,
            'text': self.text,
            'is_question': self.is_question
        }


@dataclass 
class ParsedTranscript:
    """Container for a parsed earnings call transcript."""
    # Metadata
    ticker: str
    date: str
    quarter: int
    year: int
    
    # Split sections
    speech_text: str = ""
    qa_text: str = ""
    
    # Detailed turns
    speech_turns: List[Turn] = field(default_factory=list)
    qa_turns: List[Turn] = field(default_factory=list)
    
    # Statistics
    speech_word_count: int = 0
    qa_word_count: int = 0
    num_qa_exchanges: int = 0
    
    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'date': self.date,
            'quarter': self.quarter,
            'year': self.year,
            'speech_text': self.speech_text,
            'qa_text': self.qa_text,
            'speech_turns': [t.to_dict() for t in self.speech_turns],
            'qa_turns': [t.to_dict() for t in self.qa_turns],
            'speech_word_count': self.speech_word_count,
            'qa_word_count': self.qa_word_count,
            'num_qa_exchanges': self.num_qa_exchanges
        }


class TranscriptParser:
    """
    Parser for S&P 500 earnings call transcripts.
    
    Handles the `structured_content` field which contains speaker-attributed
    text in JSON format.
    """
    
    # Patterns to identify Q&A section start
    QA_START_PATTERNS = [
        r"question.{0,10}answer",
        r"q\s*&\s*a\s+session",
        r"q\s*&\s*a\s+portion",
        r"open.*(?:for|to).*questions",
        r"take.*questions",
        r"first question",
        r"over to.*(?:operator|questions)",
    ]
    
    # Patterns to identify analysts (questioners)
    ANALYST_PATTERNS = [
        r"analyst",
        r"research",
        r"securities",
        r"capital",
        r"partners",
        r"advisors",
        r"associates",
        # Common sell-side / brokerage firm identifiers (non-exhaustive)
        r"goldman",
        r"sachs",
        r"morgan stanley",
        r"jpmorgan",
        r"jp morgan",
        r"j\.p\. morgan",
        r"bank of america",
        r"bofa",
        r"barclays",
        r"citigroup",
        r"citi",
        r"deutsche",
        r"ubs",
        r"wells fargo",
    ]
    
    # Management/executive role keywords
    MANAGEMENT_KEYWORDS = [
        "ceo", "cfo", "coo", "cto", "cio", "president", "chairman",
        "chief", "executive", "officer", "director", "vp", "vice president",
        "head of", "senior vice", "evp", "svp", "treasurer", "controller"
    ]
    
    def __init__(self):
        self.qa_pattern = re.compile(
            "|".join(self.QA_START_PATTERNS), 
            re.IGNORECASE
        )
    
    def parse_structured_content(self, content: str) -> List[Dict]:
        """
        Parse the structured_content field.
        
        Args:
            content: JSON string or list of speaker turns
            
        Returns:
            List of turn dictionaries with 'speaker' and 'text' keys
        """
        if isinstance(content, str):
            try:
                # Try parsing as JSON
                if content.startswith('['):
                    return json.loads(content)
                else:
                    # Single text block, no structure
                    return [{'speaker': 'Unknown', 'text': content}]
            except json.JSONDecodeError:
                return [{'speaker': 'Unknown', 'text': content}]
        elif isinstance(content, list):
            return content
        elif isinstance(content, tuple):
            return list(content)
        else:
            # PyArrow can materialize list columns as numpy arrays per cell
            try:
                import numpy as np

                if isinstance(content, np.ndarray):
                    return content.tolist()
            except Exception:
                pass
            return []
    
    def classify_role(self, speaker: str, text: str = "") -> str:
        """
        Classify speaker role based on name and context.
        
        Args:
            speaker: Speaker name/title
            text: Optional text to help classify
            
        Returns:
            Role string: 'management', 'analyst', 'operator', 'unknown'
        """
        speaker_lower = speaker.lower()
        
        # Operator is clear
        if 'operator' in speaker_lower:
            return 'operator'
        
        # Check for management keywords
        for kw in self.MANAGEMENT_KEYWORDS:
            if kw in speaker_lower:
                return 'management'
        
        # Check for analyst patterns
        for pattern in self.ANALYST_PATTERNS:
            if pattern in speaker_lower:
                return 'analyst'
        
        # If text contains a question, likely analyst
        if text and ('?' in text[:200] or text.strip().lower().startswith(('can ', 'could ', 'what ', 'how ', 'why ', 'when ', 'is ', 'are ', 'do ', 'does '))):
            return 'analyst'
        
        return 'unknown'
    
    def find_qa_start_index(self, turns: List[Dict]) -> int:
        """
        Find the index where Q&A section begins.
        
        Returns:
            Index of first Q&A turn, or len(turns) if no Q&A found
        """
        for i, turn in enumerate(turns):
            text = turn.get('text', '')
            speaker = turn.get('speaker', '')
            
            # Operator announcing Q&A
            if 'operator' in speaker.lower():
                if self.qa_pattern.search(text):
                    return i
            
            # Check text for Q&A patterns
            if self.qa_pattern.search(text[:500] if len(text) > 500 else text):
                return i
        
        # Heuristic: if no clear Q&A marker, look for first question
        for i, turn in enumerate(turns):
            text = turn.get('text', '').strip()
            # First non-management speaker asking a question
            if '?' in text[:300]:
                role = self.classify_role(turn.get('speaker', ''), text)
                if role == 'analyst':
                    return max(0, i - 1)  # Include operator intro
        
        return len(turns)  # No Q&A found
    
    def is_question(self, text: str) -> bool:
        """Determine if text contains a question."""
        # Simple heuristics
        text_start = text[:500] if len(text) > 500 else text
        
        # Contains question mark
        if '?' in text_start:
            return True
        
        # Starts with question words
        first_words = text.strip().lower()[:50]
        question_starters = ['can ', 'could ', 'what ', 'how ', 'why ', 'when ', 
                            'where ', 'is ', 'are ', 'do ', 'does ', 'would ', 
                            'will ', 'should ', 'may ', 'might ']
        for starter in question_starters:
            if first_words.startswith(starter):
                return True
        
        return False
    
    def parse(
        self, 
        structured_content,
        ticker: str,
        date: str,
        quarter: int,
        year: int
    ) -> ParsedTranscript:
        """
        Parse a single transcript into Speech and Q&A sections.
        
        Args:
            structured_content: The structured_content field (JSON or list)
            ticker: Stock ticker symbol
            date: Earnings call date
            quarter: Fiscal quarter
            year: Fiscal year
            
        Returns:
            ParsedTranscript object
        """
        result = ParsedTranscript(
            ticker=ticker,
            date=str(date),
            quarter=quarter,
            year=year
        )
        
        # Parse structured content
        turns = self.parse_structured_content(structured_content)
        if not turns:
            return result
        
        # Find Q&A boundary
        qa_start = self.find_qa_start_index(turns)
        
        # Split into speech and Q&A
        speech_turns_raw = turns[:qa_start]
        qa_turns_raw = turns[qa_start:]
        
        # Process speech turns
        speech_texts = []
        for turn in speech_turns_raw:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            role = self.classify_role(speaker, text)
            
            result.speech_turns.append(Turn(
                speaker=speaker,
                role=role,
                text=text,
                is_question=False
            ))
            speech_texts.append(text)
        
        result.speech_text = "\n\n".join(speech_texts)
        result.speech_word_count = len(result.speech_text.split())
        
        # Process Q&A turns
        qa_texts = []
        num_exchanges = 0
        for turn in qa_turns_raw:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            role = self.classify_role(speaker, text)
            is_q = self.is_question(text) and role in ('analyst', 'unknown')
            
            if is_q:
                num_exchanges += 1
            
            result.qa_turns.append(Turn(
                speaker=speaker,
                role=role,
                text=text,
                is_question=is_q
            ))
            qa_texts.append(text)
        
        result.qa_text = "\n\n".join(qa_texts)
        result.qa_word_count = len(result.qa_text.split())
        result.num_qa_exchanges = num_exchanges
        
        return result
    
    def parse_dataframe(
        self, 
        df: pd.DataFrame,
        structured_content_col: str = 'structured_content',
        ticker_col: str = 'ticker',
        date_col: str = 'date',
        quarter_col: str = 'quarter',
        year_col: str = 'year',
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Parse all transcripts in a DataFrame.
        
        Args:
            df: DataFrame with transcript data
            structured_content_col: Column name for structured content
            ticker_col: Column name for ticker
            date_col: Column name for date
            quarter_col: Column name for quarter
            year_col: Column name for year
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with parsed transcript information
        """
        results = []
        iterator = tqdm(df.iterrows(), total=len(df), desc="Parsing transcripts") if show_progress else df.iterrows()
        
        for idx, row in iterator:
            try:
                parsed = self.parse(
                    structured_content=row[structured_content_col],
                    ticker=row.get(ticker_col, 'UNK'),
                    date=row.get(date_col, ''),
                    quarter=row.get(quarter_col, 0),
                    year=row.get(year_col, 0)
                )
                results.append(parsed.to_dict())
            except Exception as e:
                print(f"Error parsing row {idx}: {e}")
                results.append({
                    'ticker': row.get(ticker_col, 'UNK'),
                    'date': str(row.get(date_col, '')),
                    'quarter': row.get(quarter_col, 0),
                    'year': row.get(year_col, 0),
                    'error': str(e)
                })
        
        return pd.DataFrame(results)


def process_dataset(
    input_path: str,
    output_path: str,
    sample_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Process the full dataset and save parsed transcripts.
    
    Args:
        input_path: Path to input parquet/csv file
        output_path: Path to save output parquet file
        sample_n: If set, only process this many rows (for testing)
        
    Returns:
        DataFrame with parsed data
    """
    print(f"Loading data from {input_path}...")
    
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    if sample_n:
        df = df.head(sample_n)
        print(f"Processing sample of {sample_n} rows")
    
    print(f"Total rows to process: {len(df)}")
    
    # Initialize parser
    parser = TranscriptParser()
    
    # Parse all transcripts
    parsed_df = parser.parse_dataframe(df)
    
    # Save results
    print(f"Saving parsed data to {output_path}...")
    parsed_df.to_parquet(output_path, index=False)
    
    # Print summary
    print("\n=== Parsing Summary ===")
    print(f"Total transcripts: {len(parsed_df)}")
    print(f"Avg speech words: {parsed_df['speech_word_count'].mean():.0f}")
    print(f"Avg Q&A words: {parsed_df['qa_word_count'].mean():.0f}")
    print(f"Avg Q&A exchanges: {parsed_df['num_qa_exchanges'].mean():.1f}")
    
    return parsed_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse earnings call transcripts")
    parser.add_argument("--input", default="final_dataset.parquet", help="Input file path")
    parser.add_argument("--output", default="outputs/features/parsed_transcripts.parquet", help="Output file path")
    parser.add_argument("--sample", type=int, default=None, help="Number of samples for testing")
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.sample)
