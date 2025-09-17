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
    specializations: List[str] = None    # banking, finance, marketing, etc.
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
            'napier': ['enu', 'edinburgh napier', 'napier', 'edinburgh', 'edinburgh napier university'],
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
            'banking': ['banking', 'bank', 'financial services'],
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
        entities.specializations = self._extract_specializations(query_lower)
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

    def _extract_specializations(self, query: str) -> List[str]:
        """Extract specialization slugs from query."""
        found_specializations = []
        for slug, keywords in self.specialization_keywords.items():
            if any(keyword in query for keyword in keywords):
                found_specializations.append(slug)
        return found_specializations if found_specializations else None


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
                model=settings.chat_model,
                temperature=settings.temperature,
                api_key=settings.openai_api_key
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

        # Check for broad exploratory queries first
        question_lower = question.lower()
        broad_query_indicators = [
            'tell me about your mba programs',
            'what mba programs',
            'all mba programs',
            'mba programs you offer',
            'tell me about your msc programs',
            'what msc programs',
            'all msc programs',
            'msc programs you offer',
            'what programs does',
            'all programs from',
            'programs available at',
            'what programs do you have',
            'show me all programs',
            'list of programs'
        ]

        # Force exploratory mode for broad queries
        if any(indicator in question_lower for indicator in broad_query_indicators):
            # Parse for university/program entities but keep exploratory mode
            intent, entities = self.query_parser.parse_query(question)
            scope = self.scope_builder.build_scope(intent, entities)
            scope["mode"] = "exploratory"  # Force exploratory mode
            scope["confidence"] = 0.95
            scope["parsing_method"] = "broad_query_detection"
            return scope

        # First, try intelligent parsing
        intent, entities = self.query_parser.parse_query(question)
        scope = self.scope_builder.build_scope(intent, entities)

        # If we have high confidence in our parsing, return it
        if entities.universities or entities.programs or intent != QueryIntent.FIND_PROGRAM:
            scope["confidence"] = 0.9
            scope["parsing_method"] = "intelligent"
            return scope

        # Enhanced context handling for follow-up questions
        if history and not entities.universities and not entities.programs:
            # Extract entities from conversation history to maintain context
            history_intent, history_entities = self.query_parser.parse_query(history)

            # Also look for university/program mentions in the full conversation history
            history_lower = history.lower()

            # Check for university mentions in history
            context_universities = []
            for slug, keywords in self.query_parser.university_keywords.items():
                if any(keyword in history_lower for keyword in keywords):
                    context_universities.append(slug)

            # Check for program mentions in history
            context_programs = []
            for slug, keywords in self.query_parser.program_keywords.items():
                if any(keyword in history_lower for keyword in keywords):
                    context_programs.append(slug)

            # Check for specialization mentions in history
            context_specializations = []
            for slug, keywords in self.query_parser.specialization_keywords.items():
                if any(keyword in history_lower for keyword in keywords):
                    context_specializations.append(slug)

            # If we found context from history, use it
            if context_universities or context_programs or context_specializations or history_entities.universities or history_entities.programs or history_entities.specializations:
                # Merge current intent with historical context
                merged_entities = ExtractedEntities()
                merged_entities.universities = entities.universities or history_entities.universities or context_universities
                merged_entities.programs = entities.programs or history_entities.programs or context_programs
                merged_entities.specializations = entities.specializations or history_entities.specializations or context_specializations
                merged_entities.fields_of_interest = entities.fields_of_interest
                merged_entities.budget_max = entities.budget_max
                merged_entities.budget_min = entities.budget_min
                merged_entities.duration_preference = entities.duration_preference

                scope = self.scope_builder.build_scope(intent, merged_entities)
                scope["mode"] = "specific"  # Maintain specific mode for follow-ups
                scope["confidence"] = 0.85
                scope["parsing_method"] = "enhanced_history_context"

                # Add specialization context to scope
                if merged_entities.specializations:
                    scope["specializations"] = merged_entities.specializations

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
            - If user names uni/program OR if history shows they were discussing a specific uni/program, it's "specific"
            - If they ask for alternatives without prior context, it's "exploratory"
            - If they say "vs" or "compare", it's "compare"
            - For follow-up questions (like "what are the modules?", "how much?"), check history for context
            - Pay special attention to university mentions in the conversation history
            
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
            curriculum_info = json.loads(row.get('curriculum', '{}')) if isinstance(row.get('curriculum'), str) else row.get('curriculum', {})
            learning_outcomes_info = json.loads(row.get('learning_outcomes', '[]')) if isinstance(row.get('learning_outcomes'), str) else row.get('learning_outcomes', [])

            # Build fields from canonical data
            fields = {}

            # Fees information - prioritize discounted fees and mention Stafford Grant
            if fees_info:
                currency_symbol = "$" if fees_info.get('currency') == 'USD' else "£"

                # Prioritize discounted fees over total fees
                discounted_fee = fees_info.get('discounted_total_fee')
                total_fee = fees_info.get('total_fee')
                discounted_instalment = fees_info.get('discounted_instalment_fee')
                regular_instalment = fees_info.get('instalment_fee')

                if discounted_fee and total_fee and discounted_fee != total_fee:
                    # Show discounted fee as primary with Stafford Grant mention
                    stafford_grant = total_fee - discounted_fee
                    display_fee = discounted_fee
                    instalment_fee = discounted_instalment or regular_instalment

                    notes = f"Discounted program fee (includes Stafford Grant of {currency_symbol}{stafford_grant:,.0f})"
                    if instalment_fee:
                        notes += f". Installments: {currency_symbol}{instalment_fee:.0f} per {fees_info.get('instalment_period', 1)} month(s)"
                else:
                    # Fall back to total fee
                    display_fee = total_fee
                    instalment_fee = regular_instalment
                    notes = "Total program fee"
                    if instalment_fee:
                        notes += f". Installments: {currency_symbol}{instalment_fee:.0f} per {fees_info.get('instalment_period', 1)} month(s)"

                if display_fee:
                    fields['fees'] = {
                        'value': f"{currency_symbol}{display_fee:,.0f}",
                        'currency': fees_info.get('currency', 'USD'),
                        'notes': notes,
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

            # Curriculum/Modules information - THE MISSING PIECE!
            if curriculum_info:
                modules_data = []

                # Add core modules
                core_modules = curriculum_info.get('core_modules', [])
                if core_modules:
                    modules_data.append("**Core Modules:**")
                    for module in core_modules:
                        module_name = module.get('module_name', 'Unknown Module')
                        module_desc = module.get('description', 'No description available')
                        modules_data.append(f"• **{module_name}**: {module_desc[:200]}{'...' if len(module_desc) > 200 else ''}")

                # Add optional/specialization modules - DYNAMIC LABELING
                optional_modules = curriculum_info.get('optional_modules', [])
                if optional_modules:
                    # Determine specialization name dynamically from scope or program data
                    specialization_name = "Specialization"  # Default fallback

                    # Try to get specialization from scope first
                    if scope and scope.get('specializations'):
                        spec_slug = scope['specializations'][0]  # Take first specialization
                        # Convert slug to readable name
                        specialization_mapping = {
                            'banking': 'Banking',
                            'finance': 'Finance',
                            'marketing': 'Marketing',
                            'hr': 'Human Resources',
                            'healthcare': 'Healthcare',
                            'it': 'Information Technology',
                            'supply_chain': 'Supply Chain',
                            'project_management': 'Project Management',
                            'data_analytics': 'Data Analytics',
                            'leadership': 'Leadership',
                            'executive': 'Executive'
                        }
                        specialization_name = specialization_mapping.get(spec_slug, specialization_name)

                    # Alternatively, try to infer from program name
                    elif basic_info.get('program_name', '').lower():
                        program_name_lower = basic_info['program_name'].lower()
                        if 'banking' in program_name_lower:
                            specialization_name = 'Banking'
                        elif 'finance' in program_name_lower:
                            specialization_name = 'Finance'
                        elif 'marketing' in program_name_lower:
                            specialization_name = 'Marketing'
                        elif 'hr' in program_name_lower or 'human resources' in program_name_lower:
                            specialization_name = 'Human Resources'
                        # Add more as needed

                    modules_data.append(f"\n**{specialization_name} Specialization Modules:**")
                    for module in optional_modules:
                        module_name = module.get('module_name', 'Unknown Module')
                        module_desc = module.get('description', 'No description available')
                        modules_data.append(f"• **{module_name}**: {module_desc[:200]}{'...' if len(module_desc) > 200 else ''}")

                # Add dissertation/project
                dissertation = curriculum_info.get('dissertation_project')
                if dissertation and dissertation.get('required'):
                    modules_data.append(f"\n**{dissertation.get('name', 'Final Project')}**: Required capstone project")

                if modules_data:
                    fields['modules'] = {
                        'value': '\n'.join(modules_data),
                        'evidence': []
                    }

            # Learning outcomes
            if learning_outcomes_info:
                outcomes_text = "Upon completion, students will be able to:\n" + '\n'.join([f"• {outcome}" for outcome in learning_outcomes_info[:8]])  # Limit to 8 outcomes
                fields['learning_outcomes'] = {
                    'value': outcomes_text,
                    'evidence': []
                }

            # Program overview
            program_name = basic_info.get('program_name', 'Unknown Program')
            university_name = basic_info.get('university_name', 'Unknown University')
            program_type = basic_info.get('program_type', row.get('program_type', 'Unknown'))
            program_overview = basic_info.get('program_overview', '')

            # Use the detailed program overview if available, otherwise use basic description
            if program_overview:
                overview_text = program_overview
            else:
                overview_text = f"The {program_name} is a {program_type.lower()} program offered by {university_name} through Stafford Global."

            fields['overview'] = {
                'value': overview_text,
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
        """Enhanced answer rendering with intelligent intent-based responses."""

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

        # Get the mode to adjust response style
        mode = grounded_json.get('mode', 'specific')

        # Use intelligent intent discovery for specific mode
        if mode == 'exploratory':
            prompt_text = f"""
            Task: Provide a high-level overview of ONLY the programs found in the data. Keep response under 250 words.
            
            EXPLORATORY MODE REQUIREMENTS:
            - ONLY mention programs that actually exist in the provided data
            - If no programs are found, clearly state that no programs are available for that type
            - Group programs by type based on what's actually available in the data
            - Mention only the universities that have programs in the provided data
            - DO NOT invent, assume, or mention programs not in the data
            - Use engaging, professional tone that encourages further exploration
            - If data is empty or insufficient, say so honestly in a natural, conversational way
            
            CRITICAL RULES:
            - NEVER mention program types that don't exist in the data
            - NEVER mention universities that don't have programs in the data
            - If no relevant programs found, suggest contacting for more information in a friendly, natural tone
            - Base response ENTIRELY on the provided data
            - Keep language conversational and helpful, not overly formal
            - DO NOT mention technical terms like "data", "system", "database", or "available in our data"
            - Speak as if you're a knowledgeable education consultant, not a technical system
            - Frame missing programs as "we don't currently offer" rather than "not available in the data"
            
            Current Question: {question}
            Available Programs Data: {json.dumps(grounded_json, indent=2)}
            """

            system_message = "You are Stafford Global's education consultant. Provide high-level program overviews for exploratory queries. Focus on breadth of offerings and encourage further specific inquiries."
        else:
            # For specific mode, use intelligent intent discovery with specialization context
            prompt_text = f"""
            Task: Understand what the user is specifically asking about and provide a focused, relevant answer.
            
            INTENT-BASED RESPONSE APPROACH:
            1. First, analyze the user's question to understand their specific intent
            2. Look at the available data and identify what information is relevant to their question
            3. Check the conversation history for specialization context (e.g., if they previously asked about Banking)
            4. Provide a direct, focused answer that addresses their specific need
            5. Only include information that's directly relevant to what they asked
            6. If they ask about something not available in the data, acknowledge this clearly
            
            SPECIALIZATION CONTEXT HANDLING:
            - If the conversation history shows the user was interested in a specific specialization (e.g., Banking, Marketing, Finance), 
              prioritize programs with that specialization in your response
            - When user asks general questions like "tell me about the Global Hybrid MBA option" after previously discussing Banking,
              focus specifically on the Global Hybrid MBA Banking option, not all Global Hybrid MBA programs
            - Maintain the specialization context from previous exchanges unless user explicitly asks for something different
            
            RESPONSE GUIDELINES:
            - Be conversational and helpful, like a knowledgeable education consultant
            - Answer the specific question being asked - don't dump all available information
            - ONLY include fees/costs if user specifically asks about pricing, costs, fees, or affordability
            - ONLY include duration/timeline if user specifically asks about time, duration, or how long
            - ONLY include specific intake dates if user specifically asks about start dates or when programs begin
            - For general program inquiries, focus on program types, specializations, and general overview
            - Keep responses under 250 words and well-structured
            - Use specific data from the grounded information when available
            - If information is missing, suggest next steps or how to get more details
            
            CRITICAL RESTRICTIONS:
            - DO NOT mention specific fees, costs, or pricing unless explicitly asked
            - DO NOT mention specific durations, months, or timeframes unless explicitly asked  
            - DO NOT mention specific intake dates unless explicitly asked
            - DO NOT include payment installment details unless explicitly asked
            - Focus on program content, types, and academic aspects for general queries
            
            IMPORTANT:
            - Don't assume what the user wants beyond what they asked
            - Don't include unrelated information just because it's available
            - Be honest about what information is and isn't available
            - Focus on being helpful rather than comprehensive
            - Maintain a professional but approachable tone
            - Pay special attention to specialization context from conversation history
            
            Conversation History: {history}
            User's Question: {question}
            Available Program Data: {json.dumps(grounded_json, indent=2)}
            
            Budget filtering applied: {has_budget_filter}
            Duration filtering applied: {has_duration_filter}
            """

            system_message = "You are Stafford Global's education consultant. Analyze user questions carefully and provide focused, intent-driven responses. Only include information that directly addresses what the user is asking about. Pay special attention to specialization context from conversation history. Be helpful, conversational, and honest about available information."

        try:
            messages = [
                SystemMessage(content=system_message),
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
