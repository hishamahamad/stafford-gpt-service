"""
Query processing service that combines NLP intent classification with structured SQL queries.
Integrates with the existing Stafford GPT architecture while adding intelligent query parsing.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..config.settings import settings


class QueryIntent(Enum):
    """Types of queries users might ask about Stafford Global programs."""
    FIND_PROGRAM = "find_program"        # Find programs matching criteria
    COMPARE_PROGRAMS = "compare"         # Compare specific programs
    PROGRAM_DETAILS = "details"          # Get specific program information
    COST_INQUIRY = "cost"               # Questions about fees and payment
    DURATION_INQUIRY = "duration"        # Questions about program length
    ELIGIBILITY_CHECK = "eligibility"    # Check if user qualifies
    INTAKE_INQUIRY = "intake"           # Questions about start dates
    MODULES_INQUIRY = "modules"         # Questions about curriculum
    OVERVIEW_INQUIRY = "overview"       # General program overview


@dataclass
class ExtractedEntities:
    """Entities extracted from user query."""
    universities: List[str] = None       # hull, enu, birmingham-city
    programs: List[str] = None           # mba, msc-data-science, etc.
    program_names: List[str] = None      # executive-mba, general-mba, finance-mba, etc.
    budget_max: float = None
    budget_min: float = None
    duration_preference: int = None      # in months
    currency: str = "GBP"
    fields_of_interest: List[str] = None # fees, modules, duration, intakes


class QueryParser:
    """Parse natural language queries into structured components for Stafford Global programs."""

    def __init__(self):
        # University keywords mapping
        self.university_keywords = {
            'hull': ['hull', 'university of hull'],
            'enu': ['enu', 'edinburgh napier', 'napier', 'edinburgh'],
            'northampton': ['northampton', 'university of northampton'],
            'leicester': ['leicester', 'university of leicester', 'leics'],
            'dundee': ['dundee', 'university of dundee'],
            'nottingham': ['nottingham', 'university of nottingham', 'notts'],
            'derby': ['derby', 'university of derby'],
            'kings': ['kings', 'king\'s college london', 'kcl']
        }

        # Program keywords mapping
        self.program_keywords = {
            'mba': ['mba', 'master of business administration', 'business administration'],
            'msc-data-science': ['data science', 'data analytics', 'analytics', 'big data'],
            'msc-cybersecurity': ['cybersecurity', 'cyber security', 'information security'],
            'msc-finance': ['finance', 'financial', 'fintech'],
            'msc-marketing': ['marketing', 'digital marketing'],
            'msc-project-management': ['project management', 'pmp']
        }

        # Specialization keywords
        self.specialization_keywords = {
            'finance': ['finance', 'financial', 'banking', 'investment'],
            'marketing': ['marketing', 'digital marketing', 'brand'],
            'hr': ['hr', 'human resources', 'people management'],
            'healthcare': ['healthcare', 'medical', 'health management'],
            'it': ['it', 'information technology', 'digital transformation'],
            'supply_chain': ['supply chain', 'logistics', 'operations'],
            'project_management': ['project management', 'agile', 'scrum'],
            'data_analytics': ['analytics', 'data analysis', 'business intelligence'],
            'leadership': ['leadership', 'management', 'executive'],
            'executive': ['executive', 'executive mba', 'emba']
        }

        # Field keywords
        self.field_keywords = {
            'fees': ['cost', 'fee', 'price', 'pay', 'afford', 'tuition'],
            'modules': ['modules', 'curriculum', 'subjects', 'courses', 'syllabus'],
            'duration': ['duration', 'length', 'how long', 'months', 'years'],
            'intakes': ['intake', 'start', 'when', 'next intake'],
            'overview': ['about', 'overview', 'details', 'information']
        }

    def parse_query(self, query: str) -> Tuple[QueryIntent, ExtractedEntities]:
        """Parse user query into intent and entities."""
        query_lower = query.lower()
        entities = ExtractedEntities()

        # Detect intent
        intent = self._detect_intent(query_lower)

        # Extract entities
        entities.universities = self._extract_universities(query_lower)
        entities.programs = self._extract_programs(query_lower)
        entities.budget_max, entities.budget_min = self._extract_budget(query_lower)
        entities.duration_preference = self._extract_duration(query_lower)
        entities.fields_of_interest = self._extract_fields(query_lower)

        return intent, entities

    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect the primary intent of the query."""
        if any(word in query for word in ['compare', 'vs', 'versus', 'difference']):
            return QueryIntent.COMPARE_PROGRAMS
        elif any(word in query for word in ['cost', 'fee', 'price', 'pay', 'afford']):
            return QueryIntent.COST_INQUIRY
        elif any(word in query for word in ['how long', 'duration', 'months', 'years']):
            return QueryIntent.DURATION_INQUIRY
        elif any(word in query for word in ['modules', 'curriculum', 'subjects', 'courses']):
            return QueryIntent.MODULES_INQUIRY
        elif any(word in query for word in ['intake', 'start', 'when', 'next intake']):
            return QueryIntent.INTAKE_INQUIRY
        elif any(word in query for word in ['eligible', 'qualify', 'requirements']):
            return QueryIntent.ELIGIBILITY_CHECK
        elif any(word in query for word in ['tell me about', 'details', 'information about', 'overview']):
            return QueryIntent.PROGRAM_DETAILS
        else:
            return QueryIntent.FIND_PROGRAM

    def _extract_universities(self, query: str) -> List[str]:
        """Extract university slugs from query."""
        found_unis = []
        for slug, keywords in self.university_keywords.items():
            if any(keyword in query for keyword in keywords):
                found_unis.append(slug)
        return found_unis if found_unis else None

    def _extract_programs(self, query: str) -> List[str]:
        """Extract program slugs from query."""
        found_programs = []
        for slug, keywords in self.program_keywords.items():
            if any(keyword in query for keyword in keywords):
                found_programs.append(slug)
        return found_programs if found_programs else None


    def _extract_budget(self, query: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract budget constraints from query."""
        patterns = [
            r'under\s+[£$]?(\d+[,\d]*)',
            r'below\s+[£$]?(\d+[,\d]*)',
            r'less\s+than\s+[£$]?(\d+[,\d]*)',
            r'between\s+[£$]?(\d+[,\d]*)\s+and\s+[£$]?(\d+[,\d]*)',
            r'(\d+[,\d]*)\s*-\s*(\d+[,\d]*)',
            r'max\s+[£$]?(\d+[,\d]*)',
            r'budget\s+(?:of\s+)?[£$]?(\d+[,\d]*)'
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                if 'between' in pattern or '-' in pattern:
                    min_val = float(match.group(1).replace(',', ''))
                    max_val = float(match.group(2).replace(',', ''))
                    return max_val, min_val
                else:
                    max_val = float(match.group(1).replace(',', ''))
                    return max_val, None

        return None, None

    def _extract_duration(self, query: str) -> Optional[int]:
        """Extract duration preference from query."""
        patterns = [
            r'(\d+)\s*months?',
            r'(\d+)\s*years?',
            r'(\d+\.?\d*)\s*years?'
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                value = float(match.group(1))
                if 'year' in pattern:
                    return int(value * 12)  # Convert years to months
                return int(value)

        return None

    def _extract_fields(self, query: str) -> List[str]:
        """Extract fields of interest from query."""
        found_fields = []
        for field, keywords in self.field_keywords.items():
            if any(keyword in query for keyword in keywords):
                found_fields.append(field)
        return found_fields if found_fields else ["overview"]  # Default to overview


class ScopeBuilder:
    """Build scope objects compatible with existing Stafford GPT architecture."""

    def build_scope(self, intent: QueryIntent, entities: ExtractedEntities) -> Dict[str, Any]:
        """Convert parsed intent and entities into scope format expected by existing system."""

        # Map intent to mode
        if intent == QueryIntent.COMPARE_PROGRAMS:
            mode = "compare"
        elif intent in [QueryIntent.FIND_PROGRAM, QueryIntent.PROGRAM_DETAILS]:
            mode = "specific" if entities.universities or entities.programs else "exploratory"
        else:
            mode = "specific"

        # Build universities list
        universities = []
        if entities.universities:
            universities = [{"slug": slug} for slug in entities.universities]

        # Build programs list
        programs = []
        if entities.programs:
            for program in entities.programs:
                program_obj = {"slug": program}
                # Add program name if available
                if entities.program_names:
                    program_obj["program_name"] = entities.program_names[0]
                programs.append(program_obj)

        # Determine fields of interest
        fields = entities.fields_of_interest or ["overview"]

        return {
            "mode": mode,
            "universities": universities,
            "programs": programs,
            "fields": fields,
            "confidence": 0.8,
            "budget_constraints": {
                "max": entities.budget_max,
                "min": entities.budget_min
            } if entities.budget_max or entities.budget_min else None,
            "duration_preference": entities.duration_preference
        }


class PromptsService:
    """Prompts service that uses intelligent query parsing."""

    def __init__(self):
        try:
            self.llm = ChatOpenAI(
                model_name=settings.chat_model,
                temperature=settings.temperature,
                openai_api_key=settings.openai_api_key
            )
            self.llm_available = True
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI client: {e}")
            self.llm = None
            self.llm_available = False

        self.query_parser = QueryParser()
        self.scope_builder = ScopeBuilder()

    def classify_scope(self, question: str, history: str = "") -> Dict[str, Any]:
        """Scope classification using intelligent query parsing with better follow-up handling."""

        # First, try intelligent parsing
        intent, entities = self.query_parser.parse_query(question)
        scope = self.scope_builder.build_scope(intent, entities)

        # If we have high confidence in our parsing, return it
        if entities.universities or entities.programs or intent != QueryIntent.FIND_PROGRAM:
            scope["confidence"] = 0.9
            scope["parsing_method"] = "intelligent"
            return scope

        # For follow-up questions, pluck out specific university/program context from history
        if history and not entities.universities and not entities.programs:
            # Extract entities from conversation history to maintain context
            history_intent, history_entities = self.query_parser.parse_query(history)
            if history_entities.universities or history_entities.programs:
                # Use history context for follow-up questions
                scope = self.scope_builder.build_scope(intent, history_entities)
                scope["mode"] = "specific"  # Maintain specific mode for follow-ups
                scope["confidence"] = 0.8
                scope["parsing_method"] = "history_context"
                return scope

        # If LLM is not available, return the scope we have
        if not self.llm_available or not self.llm:
            scope["confidence"] = 0.5
            scope["parsing_method"] = "parser_only"
            return scope

        # Otherwise, fall back to LLM-based classification with enhanced context awareness
        try:
            prompt_text = f"""
            Task: classify the query as "specific", "exploratory", or "compare".
            Return JSON:
            
            {{
              "mode": "specific|exploratory|compare",
              "universities": [{{"slug": "<hull|enu|birmingham-city>"}}],
              "programs": [{{"slug": "<mba|msc-data-science|msc-cybersecurity>", "specialisation":"<optional>"}}],
              "fields": ["fees|modules|duration|intakes|overview"],
              "confidence": 0.0
            }}
            
            Rules:
            - hull = University of Hull
            - enu = Edinburgh Napier University  
            - birmingham-city = Birmingham City University
            - If user names uni/program OR if history shows they were discussing a specific uni/program, it's "specific"
            - If they ask for alternatives without prior context, it's "exploratory"
            - If they say "vs" or "compare", it's "compare"
            - For follow-up questions (like "what are the modules?", "how much?"), check history for context
            
            History: {history}
            Current Question: {question}
            """

            messages = [
                SystemMessage(content="You are a Stafford Global query classifier. Maintain context from conversation history for follow-up questions. Return only valid JSON."),
                HumanMessage(content=prompt_text)
            ]

            response = self.llm.invoke(messages)

            llm_scope = json.loads(response.content)
            llm_scope["parsing_method"] = "llm_fallback"
            return llm_scope
        except Exception as e:
            print(f"LLM classification failed: {e}")
            # Final fallback
            scope["confidence"] = 0.2
            scope["parsing_method"] = "fallback"
            return scope

    def rewrite_queries(self, scope: Dict[str, Any], question: str) -> List[str]:
        """Rewrite queries based on scope understanding."""
        # Handle None scope gracefully
        if scope is None:
            return [question]

        queries = [question]  # Always include original

        # Add university-specific queries
        if scope.get("universities"):
            for uni in scope["universities"]:
                if uni and uni.get("slug"):
                    uni_query = f"{uni['slug']} {question}"
                    queries.append(uni_query)

        # Add program-specific queries
        if scope.get("programs"):
            for program in scope["programs"]:
                if program and program.get("slug"):
                    prog_query = f"{program['slug']} {question}"
                    queries.append(prog_query)

        # Add field-specific queries based on intent
        fields = scope.get("fields", [])
        if fields:
            for field in fields[:2]:  # Limit to 2 additional field queries
                if field:
                    field_query = f"{field} {question}"
                    queries.append(field_query)

        return queries[:5]  # Limit total queries

    def ground_facts(
        self,
        scope: Dict[str, Any],
        rows: Dict[str, Dict[str, Any]],
        chunks_by_variant: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Fact grounding using only canonical data."""

        # Handle None scope gracefully
        if scope is None:
            scope = {"mode": "exploratory"}

        # Always use canonical data if available - it's reliable
        if rows:
            return self._ground_facts_from_canonical(scope, rows)

        # If we have no canonical data, return empty
        return {"mode": scope.get("mode", "exploratory"), "items": []}

    def _ground_facts_from_canonical(self, scope: Dict[str, Any], rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create grounded facts directly from canonical data when vector chunks are not available."""
        items = []

        for program_variant_id, row in rows.items():
            # Extract basic info
            basic_info = json.loads(row.get('basic_info', '{}')) if isinstance(row.get('basic_info'), str) else row.get('basic_info', {})
            duration_info = json.loads(row.get('duration', '{}')) if isinstance(row.get('duration'), str) else row.get('duration', {})
            fees_info = json.loads(row.get('fees', '{}')) if isinstance(row.get('fees'), str) else row.get('fees', {})

            # Build fields from canonical data
            fields = {}

            # Fees information
            if fees_info:
                currency_symbol = "$" if fees_info.get('currency') == 'USD' else "£"
                total_fee = fees_info.get('total_fee', fees_info.get('discounted_total_fee'))
                if total_fee:
                    fields['fees'] = {
                        'value': f"{currency_symbol}{total_fee:,.0f}",
                        'currency': fees_info.get('currency', 'USD'),
                        'notes': f"Total program fee. Installments: {currency_symbol}{fees_info.get('instalment_fee', 0):.0f} per {fees_info.get('instalment_period', 1)} month(s)" if fees_info.get('instalment_fee') else "",
                        'evidence': []
                    }

            # Duration information
            if duration_info:
                months = duration_info.get('months', duration_info.get('minimum_duration'))
                if months:
                    fields['duration'] = {
                        'value': f"{months} months",
                        'evidence': []
                    }

                # Intake information
                intake_start = duration_info.get('intake_start')
                if intake_start:
                    fields['intakes'] = {
                        'value': [intake_start],
                        'evidence': []
                    }

            # Program overview
            program_name = basic_info.get('program_name', 'Unknown Program')
            university_name = basic_info.get('university_name', 'Unknown University')
            program_type = basic_info.get('program_type', row.get('program_type', 'Unknown'))

            fields['overview'] = {
                'value': f"The {program_name} is a {program_type.lower()} program offered by {university_name} through Stafford Global.",
                'evidence': []
            }

            # Determine university and program slugs
            university_slug = row.get('university_id', 'unknown')
            program_slug = 'mba' if 'mba' in program_type.lower() else 'unknown'

            item = {
                'program_variant_id': program_variant_id,
                'university_slug': university_slug,
                'program_slug': program_slug,
                'meets_budget': True,  # Will be filtered later if needed
                'meets_duration': True,  # Will be filtered later if needed
                'fields': fields
            }

            items.append(item)

        result = {
            "mode": scope.get("mode", "specific"),
            "items": items
        }

        # Apply filters
        if scope.get('budget_constraints') or scope.get('duration_preference'):
            result = self._apply_filters(result, scope)

        return result

    def _ground_facts_from_canonical_with_evidence(
        self,
        scope: Dict[str, Any],
        rows: Dict[str, Dict[str, Any]],
        chunks_by_variant: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Create grounded facts from canonical data enhanced with vector evidence."""
        items = []

        for program_variant_id, row in rows.items():
            # Start with canonical data (reliable source of truth)
            basic_info = json.loads(row.get('basic_info', '{}')) if isinstance(row.get('basic_info'), str) else row.get('basic_info', {})
            duration_info = json.loads(row.get('duration', '{}')) if isinstance(row.get('duration'), str) else row.get('duration', {})
            fees_info = json.loads(row.get('fees', '{}')) if isinstance(row.get('fees'), str) else row.get('fees', {})

            # Get vector chunks for this program variant
            program_chunks = chunks_by_variant.get(program_variant_id, [])

            # Build fields from canonical data
            fields = {}

            # Fees information (canonical + evidence)
            if fees_info:
                currency_symbol = "$" if fees_info.get('currency') == 'USD' else "£"
                total_fee = fees_info.get('total_fee', fees_info.get('discounted_total_fee'))
                if total_fee:
                    # Find relevant evidence chunks about fees/cost
                    fee_evidence = []
                    for chunk in program_chunks[:3]:  # Limit to 3 most relevant chunks
                        content = chunk.get('content', '').lower()
                        if any(word in content for word in ['fee', 'cost', 'price', 'tuition', 'payment']):
                            fee_evidence.append({
                                'quote': chunk.get('content', '')[:200] + '...' if len(chunk.get('content', '')) > 200 else chunk.get('content', ''),
                                'source': chunk.get('source', 'website'),
                                'similarity': chunk.get('similarity', 0.8)
                            })

                    fields['fees'] = {
                        'value': f"{currency_symbol}{total_fee:,.0f}",
                        'currency': fees_info.get('currency', 'USD'),
                        'notes': f"Total program fee. Installments: {currency_symbol}{fees_info.get('instalment_fee', 0):.0f} per {fees_info.get('instalment_period', 1)} month(s)" if fees_info.get('instalment_fee') else "",
                        'evidence': fee_evidence
                    }

            # Duration information (canonical + evidence)
            if duration_info:
                months = duration_info.get('months', duration_info.get('minimum_duration'))
                if months:
                    # Find relevant evidence chunks about duration
                    duration_evidence = []
                    for chunk in program_chunks[:3]:
                        content = chunk.get('content', '').lower()
                        if any(word in content for word in ['month', 'duration', 'length', 'time', 'year']):
                            duration_evidence.append({
                                'quote': chunk.get('content', '')[:200] + '...' if len(chunk.get('content', '')) > 200 else chunk.get('content', ''),
                                'source': chunk.get('source', 'website'),
                                'similarity': chunk.get('similarity', 0.8)
                            })

                    fields['duration'] = {
                        'value': f"{months} months",
                        'evidence': duration_evidence
                    }

                # Intake information (canonical + evidence)
                intake_start = duration_info.get('intake_start')
                if intake_start:
                    # Find relevant evidence chunks about intakes
                    intake_evidence = []
                    for chunk in program_chunks[:3]:
                        content = chunk.get('content', '').lower()
                        if any(word in content for word in ['intake', 'start', 'begin', 'commence', 'enrollment']):
                            intake_evidence.append({
                                'quote': chunk.get('content', '')[:200] + '...' if len(chunk.get('content', '')) > 200 else chunk.get('content', ''),
                                'source': chunk.get('source', 'website'),
                                'similarity': chunk.get('similarity', 0.8)
                            })

                    fields['intakes'] = {
                        'value': [intake_start],
                        'evidence': intake_evidence
                    }

            # Program overview (canonical + enhanced with vector evidence)
            program_name = basic_info.get('program_name', 'Unknown Program')
            university_name = basic_info.get('university_name', 'Unknown University')
            program_type = basic_info.get('program_type', row.get('program_type', 'Unknown'))

            # Enhance overview with vector evidence
            overview_base = f"The {program_name} is a {program_type.lower()} program offered by {university_name} through Stafford Global."

            # Add additional context from vector chunks
            additional_context = []
            for chunk in program_chunks[:2]:  # Use top 2 chunks for overview enhancement
                content = chunk.get('content', '')
                if len(content) > 50 and 'cookie' not in content.lower():  # Filter out boilerplate
                    additional_context.append(content[:150] + '...' if len(content) > 150 else content)

            overview_evidence = []
            if additional_context:
                overview_evidence = [{
                    'quote': context,
                    'source': 'website',
                    'similarity': 0.8
                } for context in additional_context]

            fields['overview'] = {
                'value': overview_base,
                'evidence': overview_evidence
            }

            # Determine university and program slugs
            university_slug = row.get('university_id', 'unknown')
            program_slug = 'mba' if 'mba' in program_type.lower() else 'unknown'

            item = {
                'program_variant_id': program_variant_id,
                'university_slug': university_slug,
                'program_slug': program_slug,
                'meets_budget': True,  # Will be filtered later if needed
                'meets_duration': True,  # Will be filtered later if needed
                'fields': fields
            }

            items.append(item)

        result = {
            "mode": scope.get("mode", "specific"),
            "items": items
        }

        # Apply filters
        if scope.get('budget_constraints') or scope.get('duration_preference'):
            result = self._apply_filters(result, scope)

        return result

    def _apply_filters(self, grounded_json: Dict[str, Any], scope: Dict[str, Any]) -> Dict[str, Any]:
        """Apply budget and duration filters to grounded results."""
        budget = scope.get('budget_constraints', {})
        duration_pref = scope.get('duration_preference')

        filtered_items = []

        for item in grounded_json.get('items', []):
            meets_criteria = True

            # Check budget constraints
            if budget and item.get('fields', {}).get('fees', {}).get('value'):
                try:
                    fee_str = item['fields']['fees']['value']
                    # Extract numeric value (handle £9,750 format)
                    fee_val = float(re.sub(r'[£$,]', '', fee_str))

                    if budget.get('max') and fee_val > budget['max']:
                        meets_criteria = False
                        item['meets_budget'] = False
                    elif budget.get('min') and fee_val < budget['min']:
                        meets_criteria = False
                        item['meets_budget'] = False
                    else:
                        item['meets_budget'] = True
                except:
                    item['meets_budget'] = None

            # Check duration preference (±3 months tolerance)
            if duration_pref and item.get('fields', {}).get('duration', {}).get('value'):
                try:
                    duration_str = item['fields']['duration']['value']
                    duration_val = int(re.search(r'(\d+)', duration_str).group(1))

                    if abs(duration_val - duration_pref) <= 3:
                        item['meets_duration'] = True
                    else:
                        item['meets_duration'] = False
                        if scope.get('mode') == 'specific':
                            meets_criteria = False
                except:
                    item['meets_duration'] = None

            if meets_criteria or scope.get('mode') == 'exploratory':
                filtered_items.append(item)

        grounded_json['items'] = filtered_items
        return grounded_json

    def render_answer(
        self,
        question: str,
        grounded_json: Dict[str, Any],
        history: str = ""
    ) -> str:
        """Enhanced answer rendering with highly detailed responses."""

        # If LLM is not available, provide a basic response
        if not self.llm_available or not self.llm:
            items = grounded_json.get('items', [])
            if not items:
                return "I apologize, but I couldn't find any program information to answer your question. This might be because no programs matched your criteria or the system is currently operating in limited mode without AI assistance."

            # Basic response without LLM
            response_parts = []
            for item in items:
                program_id = item.get('program_variant_id', 'Unknown Program')
                response_parts.append(f"Found: {program_id}")

            return f"I found {len(items)} program(s): " + ", ".join(response_parts) + ". For detailed information, please ensure the AI service is properly configured."

        # Check if we have budget/duration filtered results
        has_budget_filter = any(item.get('meets_budget') is not None for item in grounded_json.get('items', []))
        has_duration_filter = any(item.get('meets_duration') is not None for item in grounded_json.get('items', []))

        prompt_text = f"""
        Task: Provide a HIGHLY DETAILED, comprehensive answer from the grounded_json data.
        
        DETAILED RESPONSE REQUIREMENTS:
        - Write 3-5+ paragraphs for specific programs, not just bullet points
        - Include ALL available information (fees, duration, intakes, modules, requirements, etc.)
        - For fees: Include payment plans, installment details, currency, any discounts
        - For duration: Include flexibility, part-time/full-time options, study patterns
        - For intakes: Include multiple intake dates, application deadlines, enrollment process
        - For modules/curriculum: List specific subjects, core modules, elective modules (if any), assessment methods
        - Add contextual information about the university's reputation, accreditation, career outcomes
        - Include practical details like delivery method (online/hybrid), support services, alumni network
        - Mention application process, entry requirements, and next steps
        - Use engaging, professional tone that builds confidence in the program
        
        Response Structure Guidelines:
        - SPECIFIC MODE: 
          * Start with compelling program introduction (university reputation + program overview)
          * Dedicated section for program structure and curriculum details
          * Comprehensive fees and payment information
          * Detailed intake and application process
          * Benefits, career outcomes, and student support
          * Clear call-to-action for next steps
        
        - EXPLORATORY MODE:
          * Detailed comparison of multiple programs
          * Highlight unique features of each option
          * Include pros/cons or suitability factors
        
        - COMPARE MODE:
          * Detailed side-by-side analysis
          * Feature comparison tables in prose form
          * Recommendation based on different student profiles
        
        IMPORTANT: 
        - Don't just list facts - explain WHY they matter to students
        - Use specific numbers, dates, and concrete details when available
        - Make the response feel comprehensive and authoritative
        - Include emotional appeal and practical considerations
        - If information is limited, acknowledge it but focus on what IS available
        
        Budget filtering applied: {has_budget_filter}
        Duration filtering applied: {has_duration_filter}
        
        Conversation History: {history}
        Current Question: {question}
        Grounded Data: {json.dumps(grounded_json, indent=2)}
        """

        try:
            messages = [
                SystemMessage(content="You are Stafford Global's senior education consultant providing comprehensive, detailed program guidance. Your responses should be thorough, professional, and highly informative - the kind of detailed information a prospective student would expect from a premium education consultancy."),
                HumanMessage(content=prompt_text)
            ]

            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Answer rendering failed: {e}")
            # Fallback response
            items = grounded_json.get('items', [])
            if items:
                return f"I found {len(items)} program(s) that might interest you, but I'm unable to provide detailed information at the moment due to a technical issue. Please try again later."
            else:
                return "I apologize, but I couldn't find any programs matching your criteria at the moment."

    def check_consistency(self, grounded_json: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced consistency checking."""

        # If LLM is not available, return basic validation
        if not self.llm_available or not self.llm:
            return {"ok": True, "problems": [], "fixes": {"drop_fields": [], "drop_items": []}}

        prompt_text = f"""
        Task: validate grounded_json. Return JSON:
        
        {{
          "ok": true|false,
          "problems": ["..."],
          "fixes": {{
            "drop_fields": [{{"program_variant_id":"...", "field":"fees"}}],
            "drop_items": ["<program_variant_id>"]
          }}
        }}
        
        Rules:
        - mode:"specific" ⇒ only one program_variant_id.
        - No non-empty value without evidence or explicit note it came from rows.
        - No cross-university leakage (unless mode=compare).
        - Check budget and duration filters are properly applied.
        
        Grounded JSON: {json.dumps(grounded_json)}
        """

        try:
            messages = [
                SystemMessage(content="You are a consistency checker. Return only valid JSON."),
                HumanMessage(content=prompt_text)
            ]

            response = self.llm.invoke(messages)
            return json.loads(response.content)
        except Exception as e:
            print(f"Consistency check failed: {e}")
            return {"ok": True, "problems": [], "fixes": {"drop_fields": [], "drop_items": []}}
