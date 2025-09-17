"""
Main orchestration service for the Stafford Global GPT system.
Implements the complete pipeline with simplified query processing.
"""
import logging
import uuid
from typing import Dict, List, Any, Optional

from .query_processing import PromptsService
from .simple_query_processor import SimpleQueryProcessor
from .data_access import DataAccessService


class OrchestratorService:
    """
    Main orchestration service with simplified query intelligence.
    Works directly with program_variant_ids without complex specialization logic.
    """

    def __init__(self, data_access: DataAccessService):
        self.data_access = data_access
        self.prompts = PromptsService()
        self.simple_processor = SimpleQueryProcessor()

    def process_query(
        self,
        question: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simplified query processing pipeline.
        """
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())

            # Get conversation history
            history = self._format_history(self.data_access.get_session_history(session_id))
            if history is None:
                history = ""

            # Get all available program variant IDs first
            all_programs = self.data_access.get_all_programs()
            available_program_ids = [prog['program_variant_id'] for prog in all_programs]

            # Step 1: Simple scope classification and program matching
            scope = self.simple_processor.build_simple_scope(question, history, available_program_ids)

            # Step 2: Get program variant IDs based on scope
            program_variant_ids = []

            if scope.get("mode") == "specific":
                # Use directly matched programs from simple processor
                if scope.get("matched_programs"):
                    program_variant_ids = scope["matched_programs"]
                else:
                    # Fallback to data access resolution
                    program_variant_ids = self.data_access.resolve_program_variants(scope)
            elif scope.get("mode") == "exploratory":
                # For exploratory, get a diverse sample across universities and program types
                diverse_programs = self._get_diverse_program_sample(available_program_ids)
                program_variant_ids = diverse_programs
            elif scope.get("mode") == "greeting":
                # For greetings, don't load any program data
                program_variant_ids = []

            # Step 3: Load canonical data
            rows = {}
            if program_variant_ids:
                rows = self.data_access.get_canonical_data(program_variant_ids)

            # Step 4: Ground facts (simplified - no specialization complexity)
            grounded = self.prompts.ground_facts(scope, rows, {})

            # Step 5: Check consistency
            consistency_check = self.prompts.check_consistency(grounded)
            if not consistency_check.get("ok", True):
                grounded = self._apply_consistency_fixes(grounded, consistency_check)

            # Step 6: Render final answer
            answer = self.prompts.render_answer(question, grounded, history)

            # Store conversation
            self.data_access.store_message(session_id, "user", question)
            self.data_access.store_message(session_id, "assistant", answer)

            return {
                "answer": answer,
                "session_id": session_id,
                "scope": scope,
                "program_variant_ids": program_variant_ids,
                "consistency_issues": consistency_check.get("problems", []) if consistency_check else []
            }

        except Exception as e:
            # Ensure we always return a valid response structure even on error
            if not session_id:
                session_id = str(uuid.uuid4())

            # Provide user-friendly error message instead of technical details
            error_answer = "I apologize, but I'm experiencing some technical difficulties at the moment. Please try asking your question again, or contact our support team if the issue persists."

            # Still try to store the conversation if possible
            try:
                self.data_access.store_message(session_id, "user", question)
                self.data_access.store_message(session_id, "assistant", error_answer)
            except:
                pass  # If storage fails, don't fail the whole response

            return {
                "answer": error_answer,
                "session_id": session_id,
                "scope": {},
                "program_variant_ids": [],
                "consistency_issues": [],
                "error": str(e)
            }

    def ingest_program_data(self, program_data: Dict[str, Any]) -> bool:
        """
        Ingest canonical program data (source of truth).
        """
        return self.data_access.store_program_data(program_data)

    def ingest_enhanced_program_data(self, program_data: Dict[str, Any]) -> bool:
        """
        Ingest comprehensive program data using the enhanced Claude Opus schema.
        Stores nested JSONB objects for maximum flexibility and rich queries.
        """
        return self.data_access.store_enhanced_program_data(program_data)

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for prompts with better context preservation."""
        if not history:
            return ""

        formatted = []
        for msg in history[-6:]:  # Last 3 exchanges
            role = msg["role"].title()
            content = msg["content"]
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def _apply_consistency_fixes(
        self,
        grounded: Dict[str, Any],
        consistency_check: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply consistency fixes to grounded data."""
        fixes = consistency_check.get("fixes", {})

        # Remove problematic items
        items_to_drop = set(fixes.get("drop_items", []))
        if items_to_drop:
            grounded["items"] = [
                item for item in grounded.get("items", [])
                if item.get("program_variant_id") not in items_to_drop
            ]

        # Remove problematic fields
        fields_to_drop = fixes.get("drop_fields", [])
        for field_fix in fields_to_drop:
            variant_id = field_fix.get("program_variant_id")
            field_name = field_fix.get("field")

            for item in grounded.get("items", []):
                if item.get("program_variant_id") == variant_id:
                    if "fields" in item and field_name in item["fields"]:
                        del item["fields"][field_name]

        return grounded

    def _get_diverse_program_sample(self, available_program_ids: List[str]) -> List[str]:
        """
        Get a diverse sample of programs across different universities.
        Simple approach: group by university and take 1-2 programs from each.
        """
        # Group programs by university (first part of program_variant_id)
        programs_by_uni = {}
        for program_id in available_program_ids:
            university = program_id.split('_')[0] if '_' in program_id else 'unknown'
            if university not in programs_by_uni:
                programs_by_uni[university] = []
            programs_by_uni[university].append(program_id)

        # Take 1-2 programs from each university to ensure diversity
        diverse_sample = []
        for university, programs in programs_by_uni.items():
            # Take up to 2 programs per university
            diverse_sample.extend(programs[:2])

        # Limit total to reasonable number for exploratory overview
        return diverse_sample[:15]
