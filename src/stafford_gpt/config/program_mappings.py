"""
Program and university mapping configurations.
Centralized configuration for all program-related identifiers and their display names.
"""

from typing import Dict, Optional
import re

# University identifier mappings
UNIVERSITY_NAMES: Dict[str, str] = {
    'hull': 'University of Hull',
    'napier': 'Edinburgh Napier University',
    'leicester': 'University of Leicester',
    'northampton': 'University of Northampton',
    'dundee': 'University of Dundee',
    'nottingham': 'University of Nottingham',
    'derby': 'University of Derby',
    'kings': 'King\'s College London',
}

# Program type mappings (from URL paths)
PROGRAM_TYPES: Dict[str, str] = {
    'doctorate': 'Doctorate',
    'online-bachelors': 'Online Bachelors',
    'online-llm': 'Online LLM',
    'online-ma': 'Online MA',
    'online-mba': 'Online MBA',
    'online-med': 'Online MEd',
    'online-msc': 'Online MSc',
    'pgce': 'PGCE'
}

# Degree type mappings (manually specified)
DEGREE_TYPES: Dict[str, str] = {
    'computing-degree': 'Computing Degree',
    'data-analytics-degree': 'Data Analytics Degree',
    'doctorate-degree': 'Doctorate Degree',
    'education-degree': 'Education Degree',
    'engineering-degree': 'Engineering Degree',
    'finance-degree': 'Finance Degree',
    'healthcare-degree': 'Healthcare Degree',
    'hr-degree': 'HR Degree',
    'law-degree': 'Law Degree',
    'logistics-degree': 'Logistics Degree',
    'management-degree': 'Management Degree',
    'marketing-degree': 'Marketing Degree',
    'project-management-degree': 'Project Management Degree',
    'psychology-degree': 'Psychology Degree',
    'psychology-masters': 'Psychology Masters',
    'social-science-degree': 'Social Science Degree'
}


def get_university_name(identifier: str) -> str:
    return UNIVERSITY_NAMES.get(identifier, f'Unknown University ({identifier})')


def get_program_type_name(identifier: str) -> str:
    return PROGRAM_TYPES.get(identifier, f'Unknown Program Type ({identifier})')


def get_degree_type_name(identifier: str) -> str:
    return DEGREE_TYPES.get(identifier, f'Unknown Degree Type ({identifier})')


def get_all_degree_type_names() -> list[str]:
    return list(DEGREE_TYPES.values())


def extract_program_info_from_url(url: str) -> Dict[str, Optional[str]]:
    match = re.search(r'staffordglobal\.org/([^/]+)/([^/]+)/?', url)

    if not match:
        return {
            'program_type': None,
            'program_slug': None,
            'program_type_name': None
        }

    program_type = match.group(1)
    program_slug = match.group(2)
    program_type_name = get_program_type_name(program_type)

    return {
        'program_type': program_type,
        'program_slug': program_slug,
        'program_type_name': program_type_name
    }


def generate_program_variant_id(university_id: str, program_type: str, program_slug: str) -> str:
    return f"{university_id}_{program_type}_{program_slug}"


def get_all_university_ids() -> list[str]:
    return list(UNIVERSITY_NAMES.keys())


def get_all_program_type_identifiers() -> list[str]:
    return list(PROGRAM_TYPES.keys())


def get_all_degree_type_identifiers() -> list[str]:
    return list(DEGREE_TYPES.keys())
