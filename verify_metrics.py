import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
import sys
import os

from src.baselines.keyword_detector import AIKeywordDetector
from src.metrics.initiation_score import _compute_speaker_ai_activity, _determine_initiation_type

def test_ai_sentence():
    print("Evaluating AI sentence audit...")
    df = pd.read_csv('data/human_annotation/ai_sentence_audit__double.csv')
    detector = AIKeywordDetector()
    preds = []
    for text in df['text']:
        # keyword_detector logic
        preds.append(int(detector.is_ai_related(str(text))))
    
    acc = accuracy_score(df['adjudicated_is_ai_true'], preds)
    kappa = cohen_kappa_score(df['adjudicated_is_ai_true'], preds)
    print(f"AI sentence audit: Accuracy {acc:.4f}, Kappa {kappa:.4f}")

def test_initiation():
    print("Evaluating Initiation exchanges audit...")
    df = pd.read_csv('data/human_annotation/initiation_audit_exchanges__double.csv')
    detector = AIKeywordDetector()
    preds = []
    for _, row in df.iterrows():
        q_text = str(row['question_text'])
        a_text = str(row['answer_text'])
        
        q_is_ai = detector.is_ai_related(q_text)
        a_is_ai = detector.is_ai_related(a_text)
        
        # We also need strong vs weak signal counts if Initiation Score uses them, 
        # but let's see how _compute_speaker_ai_activity is implemented.
        try:
            # Maybe the refactored code just takes texts?
            init_type = _determine_initiation_type(q_is_ai, a_is_ai, q_text, a_text)
            preds.append(init_type)
        except Exception as e:
            # If function signature is different
            preds.append("error")
            print("Error computing initiation type:", e)
            break
            
    if "error" not in preds:
        acc = accuracy_score(df['adjudicated_initiation_type_true'], preds)
        kappa = cohen_kappa_score(df['adjudicated_initiation_type_true'], preds)
        print(f"Initiation exchanges audit: Accuracy {acc:.4f}, Kappa {kappa:.4f}")

if __name__ == '__main__':
    test_ai_sentence()
    try:
        test_initiation()
    except Exception as e:
        print("Could not test initiation:", e)
