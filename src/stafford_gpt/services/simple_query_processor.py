"""
Simple query processor that works directly with program_variant_ids.
No need for complex specialization logic - each program is already unique.
"""

from typing import Dict, List, Any, Optional


class SimpleQueryProcessor:
    """Simple processor that matches user queries directly to program_variant_ids."""

    def __init__(self):
        pass

    def find_matching_programs(self, question: str, history: str, available_program_ids: List[str]) -> List[str]:
        """
        Find program_variant_ids that match the user's question and conversation history.
        Completely dynamic - no hardcoded terms.
        """
        search_text = (question + " " + history).lower()
        matches = []

        for program_id in available_program_ids:
            score = 0
            program_lower = program_id.lower()

            # Direct matching against program_variant_id components
            parts = program_id.split('_')
            if len(parts) >= 3:
                university = parts[0]
                program_type = parts[1]
                program_detail = parts[2]

                # Score based on exact matches of actual components
                if university in search_text:
                    score += 3
                if program_type.replace('-', ' ') in search_text or program_type in search_text:
                    score += 3
                if any(part in search_text for part in program_detail.split('-')):
                    score += 2

                # Also check individual words from program_type and program_detail
                for word in program_type.split('-'):
                    if word and word in search_text:
                        score += 1

                for word in program_detail.split('-'):
                    if word and word in search_text:
                        score += 1

            # Additional scoring for partial matches within the program_id
            words_in_search = search_text.split()
            for word in words_in_search:
                if len(word) > 2 and word in program_lower:  # Avoid very short words
                    score += 1

            if score > 0:
                matches.append((program_id, score))

        # Sort by score and return program IDs
        matches.sort(key=lambda x: x[1], reverse=True)
        return [match[0] for match in matches]

    def classify_query_mode(self, question: str) -> str:
        """Simple query mode classification with greeting detection."""
        question_lower = question.lower().strip()

        # Check for greetings/casual conversation first
        greeting_indicators = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'whats up', "what's up", 'greetings', 'howdy'
        ]

        if any(greeting in question_lower for greeting in greeting_indicators):
            return "greeting"

        # Check for very short/vague queries
        if len(question_lower) <= 10 and question_lower in ['help', 'info', 'start', 'begin']:
            return "greeting"

        # Check for exploratory queries
        exploratory_indicators = [
            'tell me about your', 'what programs', 'all programs', 'list of programs',
            'what do you offer', 'show me all', 'available programs'
        ]

        if any(indicator in question_lower for indicator in exploratory_indicators):
            return "exploratory"

        return "specific"

    def build_simple_scope(self, question: str, history: str = "", available_program_ids: List[str] = None) -> Dict[str, Any]:
        """Build scope by matching against actual program_variant_ids."""
        if not available_program_ids:
            available_program_ids = []

        mode = self.classify_query_mode(question)

        if mode == "greeting":
            return {
                "mode": "greeting",
                "confidence": 0.95,
                "parsing_method": "greeting_detection"
            }

        if mode == "exploratory":
            return {
                "mode": "exploratory",
                "confidence": 0.9,
                "parsing_method": "simple_classification"
            }

        # For specific queries, find matching programs
        matching_programs = self.find_matching_programs(question, history, available_program_ids)

        if matching_programs:
            # Extract university and program info from the best match
            best_match = matching_programs[0]
            parts = best_match.split('_')

            scope = {
                "mode": "specific",
                "confidence": 0.85,
                "parsing_method": "program_id_matching",
                "matched_programs": matching_programs[:3]  # Top 3 matches
            }

            if len(parts) >= 2:
                scope["universities"] = [{"slug": parts[0]}]
                # Extract program type (remove dashes for matching)
                program_type = parts[1].replace('-', '_')
                scope["programs"] = [{"slug": program_type}]

            return scope

        return {
            "mode": "greeting",
            "confidence": 0.3,
            "parsing_method": "fallback"
        }
