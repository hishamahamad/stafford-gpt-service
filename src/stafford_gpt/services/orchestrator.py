"""
Main orchestration service for the Stafford Global GPT system.
Implements the complete pipeline with enhanced query processing.
"""

import uuid
from typing import Dict, List, Any, Optional

from .query_processing import PromptsService
from .data_access import DataAccessService


class OrchestratorService:
    """
    Main orchestration service with enhanced query intelligence.
    Uses only canonical program data as the source of truth.
    """

    def __init__(self, data_access: DataAccessService):
        self.data_access = data_access
        self.prompts = PromptsService()

    def process_query(
        self,
        question: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simplified query processing pipeline using only canonical data.

        Steps:
        1. Classify scope (specific/exploratory/compare)
        2. Resolve to program_variant_ids if possible
        3. Load canonical data from rows
        4. Ground facts (canonical data only)
        5. Check consistency
        6. Render final answer
        """
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())

            # Get conversation history
            history = self._format_history(self.data_access.get_session_history(session_id))
            if history is None:
                history = ""

            # Step 1: Classify scope
            scope = self.prompts.classify_scope(question, history)
            if scope is None:
                scope = {"mode": "exploratory", "universities": [], "programs": [], "fields": ["overview"]}

            # Step 2: Resolve program variants if scope allows
            program_variant_ids = []
            if scope and scope.get("mode") in ["specific", "compare"]:
                program_variant_ids = self.data_access.resolve_program_variants(scope)
                if program_variant_ids is None:
                    program_variant_ids = []

            # Step 3: Load canonical data
            rows = {}
            if program_variant_ids:
                rows = self.data_access.get_canonical_data(program_variant_ids)
                if rows is None:
                    rows = {}

            # Step 4: Ground facts (canonical data only)
            grounded = self.prompts.ground_facts(scope, rows, {})  # Empty chunks
            if grounded is None:
                grounded = {"mode": scope.get("mode", "exploratory"), "items": []}

            # Step 5: Check consistency
            consistency_check = self.prompts.check_consistency(grounded)
            if consistency_check is None:
                consistency_check = {"ok": True, "problems": [], "fixes": {"drop_fields": [], "drop_items": []}}

            # Apply fixes if needed
            if not consistency_check.get("ok", True):
                grounded = self._apply_consistency_fixes(grounded, consistency_check)

            # Step 6: Render final answer
            answer = self.prompts.render_answer(question, grounded, history)
            if answer is None:
                answer = "I apologize, but I'm unable to provide an answer at the moment due to a technical issue."

            # Store conversation
            self.data_access.store_message(session_id, "user", question)
            self.data_access.store_message(session_id, "assistant", answer)

            return {
                "answer": answer,
                "session_id": session_id,
                "scope": scope or {},
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
