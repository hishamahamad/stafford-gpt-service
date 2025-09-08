"""
AI-powered program data extraction service.
Takes a URL and generates comprehensive structured program data using LLM analysis.
"""

import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..config.settings import settings
from ..config.program_mappings import (
    get_university_name,
    get_program_type_name,
    get_degree_type_name,
    get_all_degree_type_names,
    extract_program_info_from_url,
    generate_program_variant_id
)
from ..core.utils import scrape_with_playwright


class ProgramDataExtractor:
    """Extract comprehensive program data from web content using AI analysis."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=settings.chat_model,
            temperature=0.1,  # Low temperature for factual extraction
            openai_api_key=settings.openai_api_key
        )

    async def extract_program_data(self, url: str, university_id: str, degree_type: str) -> Dict[str, Any]:
        """
        Extract comprehensive program data from a URL.

        Args:
            url: The program webpage URL (e.g., https://www.staffordglobal.org/online-mba/executive-mba/)
            university_id: University identifier (hull, napier, leicester, etc.)
            degree_type: Degree type identifier (management-degree, finance-degree, etc.)

        Returns:
            Comprehensive program data following the Claude Opus schema
        """

        # Step 1: Extract program info from URL
        program_info = extract_program_info_from_url(url)
        program_type = program_info['program_type']
        program_slug = program_info['program_slug']
        program_type_name = program_info['program_type_name']

        if not program_type or not program_slug:
            raise ValueError(f"Could not extract program information from URL: {url}")

        # Step 2: Scrape the webpage content
        content = await scrape_with_playwright(url)
        if not content:
            raise ValueError(f"Could not scrape content from URL: {url}")

        # Step 3: Use AI to extract structured data
        extracted_data = await self._extract_with_ai(
            content, url, university_id, program_type, program_slug, degree_type
        )

        # Step 4: Generate program_variant_id and add required identifiers
        program_variant_id = generate_program_variant_id(university_id, program_type, program_slug)

        extracted_data['program_variant_id'] = program_variant_id
        extracted_data['university_id'] = university_id
        extracted_data['program_type'] = program_type
        extracted_data['program_slug'] = program_slug
        extracted_data['degree_type'] = degree_type
        extracted_data['url'] = url

        return extracted_data

    async def _extract_with_ai(
        self,
        content: str,
        url: str,
        university_id: str,
        program_type: str,
        program_slug: str,
        degree_type: str
    ) -> Dict[str, Any]:
        """Use AI to extract structured program data from webpage content."""

        university_name = get_university_name(university_id)
        program_type_name = get_program_type_name(program_type)
        degree_type_name = get_degree_type_name(degree_type)

        # Get all available degree names for context
        available_degrees = get_all_degree_type_names()

        prompt = f"""
You are an expert education data analyst. Extract comprehensive program information from the webpage content below and structure it according to the updated schema.

IMPORTANT EXTRACTION RULES:
1. Extract ONLY factual information explicitly stated in the content
2. If information is not available, use null
3. For fees, look for exact amounts and installment details
4. For duration, extract in months only. If only years is mentioned, convert it to months and extract
5. For modules, extract exact module names if listed
6. For intakes, extract specific dates or periods mentioned
7. Be conservative - don't invent or assume information

KNOWN CONTEXT:
- University: {university_name} (id: {university_id})
- Program Type: {program_type_name} (identifier: {program_type})
- Program Slug: {program_slug}
- Degree Type: {degree_type_name} (identifier: {degree_type})
- URL: {url}

AVAILABLE DEGREE TYPES (use these for degree_requirement matching if applicable):
{', '.join(available_degrees)}

Return ONLY valid JSON following this structure:

{{
  "basic_info": {{
    "program_name": "extract specific specialization/variant name from content or null",
    "program_type": "{program_type_name}",
    "university_name": "{university_name}",
  }},
  "duration": {{
    "months": "integer months or null",
    "delivery_mode": "online|online+workshop or null",
    "flexible": true|false|null,
    "minimum_duration": "integer or null",
    "maximum_duration": "integer or null",
    "intake_start": "when intake starts or null",
    "enrollment_deadline": "enrollment deadline or null"
  }},
  "fees": {{
    "currency": "GBP|USD|EUR or null",
    "total_fee": "number or null",
    "instalment_fee": "monthly/periodic fee or null",
    "instalment_period": "period in months or null",
    "discounted_total_fee": "discounted total or null",
    "discounted_instalment_fee": "discounted installment or null"
  }},
  "intake_info": {{
    "next_intake": "next intake if mentioned or null",
    "application_deadline": "deadline if mentioned or null",
    "rolling_admissions": true|false|null,
    "cohort_size": "number or null",
    "current_availability": "available|limited|closed or null"
  }},
  "entry_requirements": {{
    "degree_requirement": ["Use standard degree names from the available types above, or extract exact requirements if mentioned"],
    "english_requirement": {{
      "required_if_non_native": true|false|null,
      "ielts_score": "number or null",
      "toefl_score": "number or null",
      "english_interview_option": true|false|null,
      "other_accepted_tests": ["array of accepted tests or null"]
    }},
    "work_requirement": {{
      "work_experience_years": "number or null",
      "management_experience_required": true|false|null
    }}
  }},
  "accreditation": {{
    "aacsb": true|false|null,
    "amba": true|false|null,
    "equis": true|false|null,
    "other_accreditations": ["array or null"],
    "tef_rating": "Gold|Silver|Bronze or null",
    "qaa_assured": true|false|null,
    "wes_approved": true|false|null
  }},
  "curriculum": {{
    "total_credits": "number or null",
    "core_modules": [
      {{
        "module_name": "exact name",
        "credits": "number or null",
        "description": "description if available"
      }}
    ],
    "optional_modules": ["array or null"],
    "dissertation_project": {{
      "required": true|false|null,
      "name": "project name or null",
      "credits": "number or null"
    }},
    "professional_certifications": ["array or null"]
  }},
  "support_services": {{
    "personal_tutor": true|false|null,
    "academic_advisor": true|false|null,
    "career_services": true|false|null,
    "library_access": true|false|null,
    "digital_library": true|false|null,
    "student_portal": true|false|null,
    "technical_support": true|false|null,
    "networking_opportunities": true|false|null,
    "alumni_network": true|false|null
  }},
  "career_outcomes": {{
    "target_positions": ["array if mentioned or null"],
    "target_industries": ["array if mentioned or null"],
    "career_progression_support": true|false|null,
    "placement_rate": "percentage or null",
    "average_salary_increase": "text or null",
    "notable_alumni_companies": ["array if mentioned or null"]
  }},
  "geographic_focus": {{
    "target_regions": ["array of regions or null"],
    "local_relevance": true|false|null,
    "international_focus": true|false|null,
    "study_locations": ["array of locations or null"]
  }},
  "completion_rates": {{
    "overall_completion_rate": "percentage or null",
    "average_completion_time": "months or null",
    "student_satisfaction_rate": "percentage or null"
  }}
}}

WEBPAGE CONTENT:
{content[:8000]}  # Limit content to avoid token limits
"""

        messages = [
            SystemMessage(content="You are an expert education data extraction system. Return only valid JSON."),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)

        # Clean the response content to handle markdown code blocks
        response_content = response.content.strip()

        # Remove markdown code block syntax if present
        if response_content.startswith('```json'):
            response_content = response_content[7:]  # Remove '```json'
        elif response_content.startswith('```'):
            response_content = response_content[3:]   # Remove '```'

        if response_content.endswith('```'):
            response_content = response_content[:-3]  # Remove trailing '```'

        response_content = response_content.strip()

        try:
            extracted_data = json.loads(response_content)
            return extracted_data
        except json.JSONDecodeError as e:
            # Don't create fallback data - fail clearly if extraction fails
            raise ValueError(f"Failed to parse AI response as JSON. Raw response: {response.content[:500]}...") from e


def create_program_data_extractor() -> ProgramDataExtractor:
    """Factory function to create ProgramDataExtractor instance."""
    return ProgramDataExtractor()
