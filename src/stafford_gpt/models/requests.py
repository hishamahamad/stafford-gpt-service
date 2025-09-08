"""Enhanced Pydantic models for comprehensive program data."""

from typing import List, Optional, Union, Dict, Any
from datetime import datetime
from pydantic import BaseModel


# Nested models for better organization
class BasicInfo(BaseModel):
    program_name: str
    program_type: Optional[str] = None
    university_name: str


class Duration(BaseModel):
    months: Optional[int] = None  # Keep only months, remove years for simplicity
    delivery_mode: Optional[str] = None  # "online" or "online+workshop"
    flexible: Optional[bool] = None
    minimum_duration: Optional[int] = None
    maximum_duration: Optional[int] = None
    intake_start: Optional[str] = None  # When intake starts
    enrollment_deadline: Optional[str] = None  # Deadline for enrollment


class PaymentPlan(BaseModel):
    plan_name: str
    installments: int
    amount_per_installment: float


class Fees(BaseModel):
    currency: Optional[str] = None
    total_fee: Optional[float] = None
    instalment_fee: Optional[float] = None
    instalment_period: Optional[int] = None  # in months
    discounted_total_fee: Optional[float] = None
    discounted_instalment_fee: Optional[float] = None


class IntakeInfo(BaseModel):
    next_intake: Optional[str] = None
    application_deadline: Optional[str] = None  # Changed from date to str
    rolling_admissions: Optional[bool] = None  # Should be determined from content
    cohort_size: Optional[int] = None
    current_availability: Optional[str] = None  # available, limited, closed - extract from content


class EnglishRequirement(BaseModel):
    required_if_non_native: Optional[bool] = None  # Should be determined from content
    ielts_score: Optional[float] = None
    toefl_score: Optional[int] = None
    english_interview_option: Optional[bool] = None  # Should be determined from content
    other_accepted_tests: Optional[List[str]] = None  # Renamed from other_tests_accepted


class WorkRequirement(BaseModel):
    work_experience_years: Optional[int] = None
    management_experience_required: Optional[bool] = None


class EntryRequirements(BaseModel):
    degree_requirement: Optional[List[str]] = None  # Array of eligible degrees
    english_requirement: Optional[EnglishRequirement] = None
    work_requirement: Optional[WorkRequirement] = None


class Ranking(BaseModel):
    ranking_body: str
    rank: str
    year: int


class Accreditation(BaseModel):
    aacsb: Optional[bool] = None  # Should be determined from content, not assumed False
    amba: Optional[bool] = None  # Should be determined from content, not assumed False
    equis: Optional[bool] = None  # Should be determined from content, not assumed False
    other_accreditations: Optional[List[str]] = None
    tef_rating: Optional[str] = None  # Gold, Silver, Bronze
    qaa_assured: Optional[bool] = None  # Should be determined from content
    wes_approved: Optional[bool] = None  # Should be determined from content
    rankings: Optional[List[Ranking]] = None


class Module(BaseModel):
    module_name: str
    credits: Optional[int] = None
    description: Optional[str] = None
    learning_outcomes: Optional[List[str]] = None


class DissertationProject(BaseModel):
    required: Optional[bool] = None  # Should be determined from content, not assumed True
    name: Optional[str] = None  # Should be extracted, not assumed to be "Business Research Project"
    credits: Optional[int] = None
    description: Optional[str] = None


class Curriculum(BaseModel):
    total_credits: Optional[int] = None
    core_modules: Optional[List[Module]] = None
    optional_modules: Optional[List[Module]] = None
    specialization_modules: Optional[List[Module]] = None
    dissertation_project: Optional[DissertationProject] = None
    professional_certifications: Optional[List[str]] = None


class Assessment(BaseModel):
    assessment_types: Optional[List[str]] = None  # essays, reports, case studies, exams
    exams_required: Optional[bool] = None  # Should be determined from content
    continuous_assessment: Optional[bool] = None  # Should be determined from content
    group_work: Optional[bool] = None  # Should be determined from content
    practical_projects: Optional[bool] = None  # Should be determined from content
    submission_method: Optional[str] = None  # online, in-person, mixed - extract from content


class SupportServices(BaseModel):
    personal_tutor: Optional[bool] = None  # Should be determined from content
    academic_advisor: Optional[bool] = None  # Should be determined from content
    career_services: Optional[bool] = None  # Should be determined from content
    library_access: Optional[bool] = None  # Should be determined from content
    digital_library: Optional[bool] = None  # Should be determined from content
    student_portal: Optional[bool] = None  # Should be determined from content
    technical_support: Optional[bool] = None  # Should be determined from content
    networking_opportunities: Optional[bool] = None  # Should be determined from content
    alumni_network: Optional[bool] = None  # Should be determined from content


class CareerOutcomes(BaseModel):
    target_positions: Optional[List[str]] = None
    target_industries: Optional[List[str]] = None
    career_progression_support: Optional[bool] = None  # Should be determined from content
    placement_rate: Optional[str] = None  # Changed from float to str
    average_salary_increase: Optional[str] = None
    notable_alumni_companies: Optional[List[str]] = None


class AcademicProgression(BaseModel):
    further_study_options: Optional[List[str]] = None  # PhD, DBA
    credit_transfer_options: Optional[bool] = None  # Should be determined from content
    articulation_agreements: Optional[List[str]] = None


class Faculty(BaseModel):
    name: str
    title: Optional[str] = None
    photo_url: Optional[str] = None
    expertise: Optional[List[str]] = None
    profile_url: Optional[str] = None


class Testimonial(BaseModel):
    student_name: str
    program_completed: Optional[str] = None
    testimonial_text: str
    graduation_year: Optional[int] = None
    current_position: Optional[str] = None


class GeographicFocus(BaseModel):
    target_regions: Optional[List[str]] = None  # Should be extracted, not assumed to be ["Global"]
    local_relevance: Optional[bool] = None  # Should be determined from content
    international_focus: Optional[bool] = None  # Should be determined from content
    study_locations: Optional[List[str]] = None


class TechnologyPlatform(BaseModel):
    lms_platform: Optional[str] = None  # Canvas, Moodle
    mobile_app: Optional[bool] = None  # Should be determined from content, not assumed False
    vr_ar_components: Optional[bool] = None  # Should be determined from content
    simulation_tools: Optional[bool] = None  # Should be determined from content
    collaboration_tools: Optional[List[str]] = None


class CompletionRates(BaseModel):
    overall_completion_rate: Optional[str] = None  # Changed from float to str
    regional_completion_rate: Optional[str] = None  # Changed from float to str
    average_completion_time: Optional[str] = None  # Changed from float to str
    student_satisfaction_rate: Optional[str] = None  # Changed from float to str


class ContactInfo(BaseModel):
    program_coordinator: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    whatsapp: Optional[str] = None
    office_location: Optional[str] = None


class Metadata(BaseModel):
    last_updated: Optional[datetime] = None
    active: Optional[bool] = None  # Should be determined, not assumed True
    featured: Optional[bool] = None  # Should be determined, not assumed False
    popularity_score: Optional[int] = None
    search_tags: Optional[List[str]] = None
    seo_keywords: Optional[List[str]] = None
    language: Optional[str] = None  # Should be extracted, not assumed to be "English"


# Enhanced main program data request
class ProgramDataRequest(BaseModel):
    """Comprehensive program data request model based on Claude Opus schema."""

    # Core identifiers (required)
    program_variant_id: str
    university_id: str
    program_type: str
    url: Optional[str] = None

    # Comprehensive nested data (all optional for gradual adoption)
    basic_info: BasicInfo
    duration: Duration
    fees: Optional[Fees] = None
    intake_info: Optional[IntakeInfo] = None
    entry_requirements: Optional[EntryRequirements] = None
    accreditation: Optional[Accreditation] = None
    curriculum: Optional[Curriculum] = None
    assessment: Optional[Assessment] = None
    support_services: Optional[SupportServices] = None
    career_outcomes: Optional[CareerOutcomes] = None
    academic_progression: Optional[AcademicProgression] = None
    faculty: Optional[List[Faculty]] = None
    testimonials: Optional[List[Testimonial]] = None
    geographic_focus: Optional[GeographicFocus] = None
    technology_platform: Optional[TechnologyPlatform] = None
    completion_rates: Optional[CompletionRates] = None
    contact_info: Optional[ContactInfo] = None
    metadata: Optional[Metadata] = None


class QueryRequest(BaseModel):
    """Request model for chat queries."""
    question: str
    namespace: str = "customer"
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for chat queries."""
    answer: str
    session_id: str
    metadata: dict


# Simplified request models for the new URL-based system
class ExtractRequest(BaseModel):
    """Request model for AI-powered program data extraction from URL."""
    url: str  # Full Stafford Global URL (e.g., https://www.staffordglobal.org/online-mba/executive-mba/)
    university_id: str  # hull, napier, leicester, etc.
    degree_type: str  # management-degree, finance-degree, etc.


class ContentRequest(BaseModel):
    """Request for adding evidence content to existing programs."""
    program_variant_id: str
    sources: List[Union[str, Dict[str, str]]]  # URLs, raw text, or structured content


class IngestionResponse(BaseModel):
    """Response model for ingestion operations."""
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
