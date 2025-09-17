"""
Simplified prompts service for the Stafford Global GPT system.
Contains only the essential functionality used by the orchestrator.
"""

import json
import re
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..config.settings import settings


class PromptsService:
    """Simplified prompts service with only essential functionality."""

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
        """Create grounded facts directly from canonical data with strict data isolation."""
        items = []

        for program_variant_id, row in rows.items():
            # STRICT DATA ISOLATION: Only use data from THIS specific program variant
            # Clear any potential cross-contamination by validating data belongs to this variant

            try:
                # Extract and validate basic info for THIS program only
                basic_info = {}
                if row.get('basic_info'):
                    if isinstance(row['basic_info'], str):
                        basic_info = json.loads(row['basic_info'])
                    else:
                        basic_info = row['basic_info']

                # Validate this data belongs to the current program variant
                if not basic_info:
                    print(f"Warning: No basic_info for program_variant_id {program_variant_id}, skipping")
                    continue

                # Extract and validate duration info for THIS program only
                duration_info = {}
                if row.get('duration'):
                    if isinstance(row['duration'], str):
                        duration_info = json.loads(row['duration'])
                    else:
                        duration_info = row['duration']

                # Extract and validate fees info for THIS program only
                fees_info = {}
                if row.get('fees'):
                    if isinstance(row['fees'], str):
                        fees_info = json.loads(row['fees'])
                    else:
                        fees_info = row['fees']

                # Extract and validate curriculum info for THIS program only
                curriculum_info = {}
                if row.get('curriculum'):
                    if isinstance(row['curriculum'], str):
                        curriculum_info = json.loads(row['curriculum'])
                    else:
                        curriculum_info = row['curriculum']

                # Extract and validate learning outcomes for THIS program only
                learning_outcomes_info = []
                if row.get('learning_outcomes'):
                    if isinstance(row['learning_outcomes'], str):
                        learning_outcomes_info = json.loads(row['learning_outcomes'])
                    else:
                        learning_outcomes_info = row['learning_outcomes']

                # Extract and validate entry requirements for THIS program only
                entry_requirements_info = {}
                if row.get('entry_requirements'):
                    if isinstance(row['entry_requirements'], str):
                        entry_requirements_info = json.loads(row['entry_requirements'])
                    else:
                        entry_requirements_info = row['entry_requirements']

                    # DEBUG: Log entry requirements data availability (no debugger)
                    print(f"DEBUG: Found entry_requirements for {program_variant_id}: {bool(entry_requirements_info)}")
                    if entry_requirements_info:
                        print(f"DEBUG: Entry requirements keys: {list(entry_requirements_info.keys())}")
                else:
                    print(f"DEBUG: No entry_requirements column for {program_variant_id}")

                # Build fields from canonical data - ONLY from this specific program
                fields = {}

                # Fees information - STRICT: Only use fees from THIS program variant
                if fees_info and isinstance(fees_info, dict):
                    currency_symbol = "$" if fees_info.get('currency') == 'USD' else "£"

                    # Prioritize discounted fees over total fees
                    discounted_fee = fees_info.get('discounted_total_fee')
                    total_fee = fees_info.get('total_fee')
                    discounted_instalment = fees_info.get('discounted_instalment_fee')
                    regular_instalment = fees_info.get('instalment_fee')

                    # VALIDATION: Ensure fees are numeric and reasonable
                    display_fee = None
                    instalment_fee = None

                    if discounted_fee and total_fee and discounted_fee != total_fee:
                        # Validate discounted fee is reasonable
                        if isinstance(discounted_fee, (int, float)) and discounted_fee > 0:
                            stafford_grant = total_fee - discounted_fee
                            display_fee = discounted_fee
                            instalment_fee = discounted_instalment or regular_instalment

                            notes = f"Discounted program fee (includes Stafford Grant of {currency_symbol}{stafford_grant:,.0f})"
                            if instalment_fee and isinstance(instalment_fee, (int, float)) and instalment_fee > 0:
                                notes += f". Installments: {currency_symbol}{instalment_fee:.0f} per {fees_info.get('instalment_period', 1)} month(s)"
                    else:
                        # Fall back to total fee with validation
                        if total_fee and isinstance(total_fee, (int, float)) and total_fee > 0:
                            display_fee = total_fee
                            instalment_fee = regular_instalment
                            notes = "Total program fee"
                            if instalment_fee and isinstance(instalment_fee, (int, float)) and instalment_fee > 0:
                                notes += f". Installments: {currency_symbol}{instalment_fee:.0f} per {fees_info.get('instalment_period', 1)} month(s)"

                    # Only add fees if we have valid data
                    if display_fee and display_fee > 0:
                        fields['fees'] = {
                            'value': f"{currency_symbol}{display_fee:,.0f}",
                            'currency': fees_info.get('currency', 'USD'),
                            'notes': notes,
                            'evidence': [],
                            'source_program_id': program_variant_id  # Track data source
                        }

                # Duration information - STRICT: Only use duration from THIS program variant
                if duration_info and isinstance(duration_info, dict):
                    months = duration_info.get('months') or duration_info.get('minimum_duration')
                    if months and isinstance(months, (int, float)) and months > 0:
                        fields['duration'] = {
                            'value': f"{int(months)} months",
                            'evidence': [],
                            'source_program_id': program_variant_id  # Track data source
                        }

                    # Intake information - STRICT: Only use intake from THIS program variant
                    intake_start = duration_info.get('intake_start')
                    if intake_start and isinstance(intake_start, str):
                        fields['intakes'] = {
                            'value': [intake_start],
                            'evidence': [],
                            'source_program_id': program_variant_id  # Track data source
                        }

                # Curriculum/Modules information - STRICT: Only use curriculum from THIS program variant
                if curriculum_info and isinstance(curriculum_info, dict):
                    modules_data = []

                    # Add core modules
                    core_modules = curriculum_info.get('core_modules', [])
                    if isinstance(core_modules, list) and core_modules:
                        modules_data.append("**Core Modules:**")
                        for module in core_modules:
                            if isinstance(module, dict):
                                module_name = module.get('module_name', 'Unknown Module')
                                module_desc = module.get('description', 'No description available')
                                if module_name and module_name != 'Unknown Module':
                                    modules_data.append(f"• **{module_name}**: {module_desc[:200]}{'...' if len(module_desc) > 200 else ''}")

                    # Add optional/specialization modules
                    optional_modules = curriculum_info.get('optional_modules', [])
                    if isinstance(optional_modules, list) and optional_modules:
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

                        modules_data.append(f"\n**{specialization_name} Specialization Modules:**")
                        for module in optional_modules:
                            if isinstance(module, dict):
                                module_name = module.get('module_name', 'Unknown Module')
                                module_desc = module.get('description', 'No description available')
                                if module_name and module_name != 'Unknown Module':
                                    modules_data.append(f"• **{module_name}**: {module_desc[:200]}{'...' if len(module_desc) > 200 else ''}")

                    # Add dissertation/project
                    dissertation = curriculum_info.get('dissertation_project')
                    if dissertation and isinstance(dissertation, dict) and dissertation.get('required'):
                        project_name = dissertation.get('name', 'Final Project')
                        modules_data.append(f"\n**{project_name}**: Required capstone project")

                    # Only add modules if we have valid data
                    if modules_data:
                        fields['modules'] = {
                            'value': '\n'.join(modules_data),
                            'evidence': [],
                            'source_program_id': program_variant_id  # Track data source
                        }

                # Learning outcomes - STRICT: Only use outcomes from THIS program variant
                if learning_outcomes_info and isinstance(learning_outcomes_info, list) and learning_outcomes_info:
                    # Filter out empty or invalid outcomes
                    valid_outcomes = [outcome for outcome in learning_outcomes_info if outcome and isinstance(outcome, str)]
                    if valid_outcomes:
                        outcomes_text = "Upon completion, students will be able to:\n" + '\n'.join([f"• {outcome}" for outcome in valid_outcomes[:8]])  # Limit to 8 outcomes
                        fields['learning_outcomes'] = {
                            'value': outcomes_text,
                            'evidence': [],
                            'source_program_id': program_variant_id  # Track data source
                        }

                # Entry Requirements - STRICT: Only use requirements from THIS program variant
                if entry_requirements_info and isinstance(entry_requirements_info, dict):
                    requirements_data = []

                    # Degree requirements (mapped from 'degree_requirement')
                    degree_reqs = entry_requirements_info.get('degree_requirement', [])
                    if isinstance(degree_reqs, list) and degree_reqs:
                        requirements_data.append("**Academic Requirements:**")
                        for req in degree_reqs:
                            if isinstance(req, str) and req.strip():
                                requirements_data.append(f"• {req}")
                    elif isinstance(degree_reqs, str) and degree_reqs.strip():
                        requirements_data.append("**Academic Requirements:**")
                        requirements_data.append(f"• {degree_reqs}")

                    # English language requirements (mapped from 'english_requirement')
                    english_reqs = entry_requirements_info.get('english_requirement', {})
                    if isinstance(english_reqs, dict) and english_reqs:
                        requirements_data.append("\n**English Language Requirements:**")

                        # IELTS requirements
                        ielts_score = english_reqs.get('ielts_score')
                        if ielts_score:
                            requirements_data.append(f"• IELTS: {ielts_score} overall")

                        # TOEFL requirements
                        toefl_score = english_reqs.get('toefl_score')
                        if toefl_score:
                            requirements_data.append(f"• TOEFL: {toefl_score}")

                        # Other accepted tests
                        other_tests = english_reqs.get('other_accepted_tests')
                        if other_tests:
                            if isinstance(other_tests, list):
                                for test in other_tests:
                                    if isinstance(test, str) and test.strip():
                                        requirements_data.append(f"• {test}")
                            elif isinstance(other_tests, str):
                                requirements_data.append(f"• {other_tests}")

                        # English interview option
                        if english_reqs.get('english_interview_option'):
                            requirements_data.append("• English interview available as alternative")

                        # Note for non-native speakers
                        if english_reqs.get('required_if_non_native'):
                            requirements_data.append("• Required for non-native English speakers")

                    # Work experience requirements (mapped from 'work_requirement')
                    work_req = entry_requirements_info.get('work_requirement', {})
                    if isinstance(work_req, dict) and work_req:
                        requirements_data.append("\n**Work Experience:**")

                        years = work_req.get('work_experience_years')
                        management_required = work_req.get('management_experience_required')

                        if years:
                            exp_text = f"Minimum {years} years of professional experience"
                            if management_required:
                                exp_text += " with management responsibilities"
                            requirements_data.append(f"• {exp_text}")
                        elif management_required:
                            requirements_data.append("• Management experience required")

                    # Only add entry requirements if we have valid data
                    if requirements_data:
                        fields['entry_requirements'] = {
                            'value': '\n'.join(requirements_data),
                            'evidence': [],
                            'source_program_id': program_variant_id  # Track data source
                        }

                # Program overview - STRICT: Only use overview from THIS program variant
                program_name = basic_info.get('program_name', 'Unknown Program')
                university_name = basic_info.get('university_name', 'Unknown University')
                program_type = basic_info.get('program_type', row.get('program_type', 'Unknown'))
                program_overview = basic_info.get('program_overview', '')

                # Validate program identifiers
                if not program_name or program_name == 'Unknown Program':
                    print(f"Warning: Invalid program_name for program_variant_id {program_variant_id}")
                if not university_name or university_name == 'Unknown University':
                    print(f"Warning: Invalid university_name for program_variant_id {program_variant_id}")

                # Use the detailed program overview if available, otherwise use basic description
                if program_overview and isinstance(program_overview, str):
                    overview_text = program_overview
                else:
                    overview_text = f"The {program_name} is a {program_type.lower()} program offered by {university_name} through Stafford Global."

                fields['overview'] = {
                    'value': overview_text,
                    'evidence': [],
                    'source_program_id': program_variant_id  # Track data source
                }

                # Determine university and program slugs with validation
                university_slug = row.get('university_id', 'unknown')
                program_slug = 'mba' if 'mba' in program_type.lower() else 'unknown'

                # Create the item with strict data isolation
                item = {
                    'program_variant_id': program_variant_id,
                    'university_slug': university_slug,
                    'program_slug': program_slug,
                    'meets_budget': True,  # Will be filtered later if needed
                    'meets_duration': True,  # Will be filtered later if needed
                    'fields': fields,
                    'data_validation': {
                        'has_valid_basic_info': bool(basic_info),
                        'has_valid_fees': 'fees' in fields,
                        'has_valid_duration': 'duration' in fields,
                        'has_valid_curriculum': 'modules' in fields,
                        'has_valid_entry_requirements': 'entry_requirements' in fields,
                        'source_verified': True
                    }
                }

                items.append(item)

            except Exception as e:
                print(f"Error processing program_variant_id {program_variant_id}: {e}")
                # Skip this program variant if there's any error to prevent contamination
                continue

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
                    # Extract numeric value (handle ��9,750 format)
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
            - ONLY include entry requirements if user specifically asks about requirements, qualifications, eligibility, or admission criteria
            - For general program inquiries, focus on program types, specializations, and general overview
            - Keep responses under 250 words and well-structured
            - Use specific data from the grounded information when available
            - If information is missing, suggest next steps or how to get more details
            
            CRITICAL RESTRICTIONS:
            - DO NOT mention specific fees, costs, or pricing unless explicitly asked
            - DO NOT mention specific durations, months, or timeframes unless explicitly asked  
            - DO NOT mention specific intake dates unless explicitly asked
            - DO NOT mention entry requirements, IELTS, TOEFL, academic qualifications unless explicitly asked
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
