import json
import os
import spacy
from nltk import ngrams

class AspectExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        possible_paths = [
            "aspect_candidates.json",
            os.path.join("data", "processed", "aspect_candidates.json"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aspect_candidates.json")
        ]
        
        for path in possible_paths:
            try:
                with open(path) as f:
                    self.aspect_candidates = json.load(f)
                    print(f"Successfully loaded aspects from {path}")
                    return
            except FileNotFoundError:
                continue
        
        print("WARNING: Could not find aspect_candidates.json - creating empty default")
        self.aspect_candidates = {
            "laptop": ["screen", "battery", "keyboard", "price", "performance"],
            "restaurant": ["food", "service", "ambience", "price", "menu"]
        }
    
    def extract_aspects(self, sentence, domain):
        """Extract non-overlapping aspects, prioritizing longer n-grams."""
        doc = self.nlp(sentence)
        tokens = [token.text.lower() for token in doc]
        aspects_found = set()
        covered_indices = set()  # Track indices already part of an aspect
        
        # Check n-grams from longest (3-grams) to shortest (1-grams)
        for n in range(3, 0, -1):
            for i in range(len(tokens) - n + 1):
                if any(idx in covered_indices for idx in range(i, i + n)):
                    continue
                
                candidate = " ".join(tokens[i:i + n])
                if candidate in self.aspect_candidates.get(domain, []):
                    aspects_found.add(candidate)
                    covered_indices.update(range(i, i + n))
        
        if not aspects_found:
            aspects_found.add("general")
            
        return list(aspects_found)