"""
Simple query processor that works directly with program_variant_ids.
No need for complex specialization logic - each program is already unique.
"""

from typing import Dict, List, Any, Optional
import re


class SimpleQueryProcessor:
    """Simple processor that matches user queries directly to program_variant_ids."""

    def __init__(self):
        pass

    def find_matching_programs(self, question: str, history: str, available_program_ids: List[str]) -> List[str]:
        """
        Find program_variant_ids that match the user's question and conversation history.
        """
        search_text = (question + " " + history).lower()
        matches = []

        # ENHANCED CONTEXT PRESERVATION: Extract program context from conversation history
        history_context = self._extract_program_context_from_history(history)

        for program_id in available_program_ids:
            score = 0
            program_lower = program_id.lower()

            # Direct matching against program_variant_id components
            parts = program_id.split('_')
            if len(parts) >= 3:
                university = parts[0]
                program_type = parts[1]
                program_detail = parts[2]

                # CONTEXT BOOST: If this program matches previously discussed programs, give massive boost
                if history_context.get('universities') and university in history_context['universities']:
                    score += 10  # Huge boost for university context match

                if history_context.get('program_types') and program_type in history_context['program_types']:
                    score += 8   # Big boost for program type context match

                if history_context.get('program_details') and program_detail in history_context['program_details']:
                    score += 6   # Boost for program detail context match

                # Score based on exact matches of actual components in current question
                if university in question.lower():
                    score += 3
                if program_type.replace('-', ' ') in question.lower() or program_type in question.lower():
                    score += 3
                if any(part in question.lower() for part in program_detail.split('-')):
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

    def _extract_program_context_from_history(self, history: str) -> Dict[str, List[str]]:
        """Extract program context from conversation history to maintain continuity."""
        context = {
            'universities': [],
            'program_types': [],
            'program_details': [],
            'mentioned_program_ids': []
        }

        if not history:
            return context

        history_lower = history.lower()

        # Extract university mentions from history
        university_keywords = {
            'hull': ['hull', 'university of hull'],
            'napier': ['napier', 'edinburgh napier', 'enu'],
            'northampton': ['northampton', 'university of northampton'],
            'leicester': ['leicester', 'university of leicester'],
            'dundee': ['dundee', 'university of dundee'],
            'nottingham': ['nottingham', 'university of nottingham'],
            'derby': ['derby', 'university of derby'],
            'birmingham': ['birmingham', 'birmingham city'],
            'kings': ['kings', "king's", 'kcl']
        }

        for uni_slug, keywords in university_keywords.items():
            if any(keyword in history_lower for keyword in keywords):
                context['universities'].append(uni_slug)

        # Extract program type mentions from history
        program_keywords = {
            'mba': ['mba', 'master of business administration', 'business administration'],
            'msc': ['msc', 'master of science', 'masters'],
            'ma': ['ma', 'master of arts'],
            'global-hybrid-mba': ['global hybrid mba', 'hybrid mba'],
            'executive-mba': ['executive mba', 'emba'],
            'finance-mba': ['finance mba', 'mba finance'],
            'banking-mba': ['banking mba', 'mba banking']
        }

        for prog_slug, keywords in program_keywords.items():
            if any(keyword in history_lower for keyword in keywords):
                context['program_types'].append(prog_slug)

        # Look for actual program_variant_id patterns in history
        # This catches cases where the assistant mentioned specific program IDs
        program_id_pattern = r'([a-z]+_[a-z-]+_[a-z-]+(?:_[a-z-]+)*)'
        found_ids = re.findall(program_id_pattern, history_lower)

        for program_id in found_ids:
            context['mentioned_program_ids'].append(program_id)
            # Also extract components from found IDs
            parts = program_id.split('_')
            if len(parts) >= 3:
                if parts[0] not in context['universities']:
                    context['universities'].append(parts[0])
                if parts[1] not in context['program_types']:
                    context['program_types'].append(parts[1])
                if parts[2] not in context['program_details']:
                    context['program_details'].append(parts[2])

        return context

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
