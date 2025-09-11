"""
Data access service for the Stafford Global GPT system.
Handles canonical program data only - no vector functionality.
"""

import json
from typing import Dict, List, Any
from psycopg2.extras import RealDictCursor


class DataAccessService:
    def __init__(self, connection):
        self.conn = connection

    def resolve_program_variants(self, scope: Dict[str, Any]) -> List[str]:
        cur = self.conn.cursor()

        # Build query based on scope
        where_conditions = ["pv.active = true"]
        params = []

        # Filter by universities if specified
        if scope.get("universities"):
            university_ids = [u.get("slug") for u in scope["universities"] if u.get("slug")]
            if university_ids:
                placeholders = ",".join(["%s"] * len(university_ids))
                where_conditions.append(f"u.id IN ({placeholders})")
                params.extend(university_ids)

        # Filter by programs if specified
        if scope.get("programs"):
            program_identifiers = [p.get("slug") for p in scope["programs"] if p.get("slug")]
            if program_identifiers:
                # Create conditions to match program_type - handle variations like 'mba' matching 'online-mba'
                program_conditions = []
                for identifier in program_identifiers:
                    program_conditions.append("p.program_type LIKE %s")
                    params.append(f"%{identifier}%")
                where_conditions.append(f"({' OR '.join(program_conditions)})")

        query = f"""
            SELECT pv.program_variant_id
            FROM program_variants pv
            JOIN universities u ON pv.university_id = u.id
            JOIN programs p ON pv.program_id = p.id
            WHERE {' AND '.join(where_conditions)}
            ORDER BY pv.created_at DESC
        """

        try:
            cur.execute(query, params)
            results = cur.fetchall()
            return [row[0] for row in results]
        except Exception as e:
            return []
        finally:
            cur.close()

    def get_canonical_data(self, program_variant_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not program_variant_ids:
            return {}

        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        placeholders = ",".join(["%s"] * len(program_variant_ids))

        query = f"""
            SELECT 
                pv.program_variant_id,
                u.id as university_id,
                u.name as university_name,
                p.university as program_identifier,
                p.program_type as program_type,
                p.program_name as program_name,
                pv.basic_info,
                pv.program_overview,
                pv.program_benefits,
                pv.duration,
                pv.fees,
                pv.intake_info,
                pv.curriculum,
                pv.accreditation,
                pv.career_outcomes,
                pv.entry_requirements,
                pv.geographic_focus,
                pv.completion_rates,
                pv.support_services,
                pv.academic_progression
            FROM program_variants pv
            JOIN universities u ON pv.university_id = u.id
            JOIN programs p ON pv.program_id = p.id
            WHERE pv.program_variant_id IN ({placeholders})
            AND pv.active = true
        """

        try:
            cur.execute(query, program_variant_ids)
            results = cur.fetchall()
            return {row['program_variant_id']: dict(row) for row in results}
        except Exception as e:
            return {}
        finally:
            cur.close()

    def get_all_programs(self) -> List[Dict[str, Any]]:
        """Get all active programs with basic information."""
        cur = self.conn.cursor(cursor_factory=RealDictCursor)

        query = """
            SELECT 
                pv.program_variant_id,
                u.id as university_id,
                u.name as university_name,
                p.university as program_identifier,
                p.program_type as program_type,
                p.program_name as program_name,
                pv.basic_info,
                pv.duration,
                pv.fees,
                pv.created_at,
                pv.updated_at
            FROM program_variants pv
            JOIN universities u ON pv.university_id = u.id
            JOIN programs p ON pv.program_id = p.id
            WHERE pv.active = true
            ORDER BY u.name, p.program_name
        """

        try:
            cur.execute(query)
            results = cur.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            return []
        finally:
            cur.close()

    def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        cur = self.conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (session_id, limit))

            results = cur.fetchall()
            return [dict(row) for row in reversed(results)]

        except Exception as e:
            return []
        finally:
            cur.close()

    def store_message(self, session_id: str, role: str, content: str) -> bool:
        cur = self.conn.cursor()

        try:
            # Ensure session exists
            cur.execute(
                "INSERT INTO sessions (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
                (session_id,)
            )

            # Store message
            cur.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content)
            )

            self.conn.commit()
            return True

        except Exception as e:
            self.conn.rollback()
            return False
        finally:
            cur.close()

    def store_enhanced_program_data(self, program_data: Dict[str, Any]) -> bool:
        cur = self.conn.cursor()

        try:
            # Extract core identifiers
            program_variant_id = program_data['program_variant_id']
            university_id = program_data['university_id']
            program_type = program_data['program_type']

            basic_info = program_data['basic_info']
            duration = program_data['duration']

            # Ensure university exists
            cur.execute(
                "INSERT INTO universities (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                (university_id, basic_info['university_name'])
            )

            # Ensure program exists
            cur.execute(
                "INSERT INTO programs (university, program_type, program_name) VALUES (%s, %s, %s) ON CONFLICT (program_name, university) DO NOTHING",
                (university_id, program_type, basic_info['program_name'])
            )

            # Get program ID
            cur.execute("SELECT id FROM programs WHERE university = %s AND program_name = %s",
                       (university_id, basic_info['program_name']))
            program_id = cur.fetchone()[0]

            # Insert or update comprehensive program variant
            upsert_query = """
                INSERT INTO program_variants (
                    program_variant_id, university_id, program_id, url,
                    basic_info, program_overview, program_benefits,duration, fees, intake_info,
                    entry_requirements, accreditation, curriculum, assessment,
                    support_services, career_outcomes,
                    academic_progression, faculty, testimonials, geographic_focus,
                    technology_platform, completion_rates, academic_progression, contact_info, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (program_variant_id) DO UPDATE SET
                    university_id = EXCLUDED.university_id,
                    program_id = EXCLUDED.program_id,
                    url = EXCLUDED.url,
                    basic_info = EXCLUDED.basic_info,
                    program_overview = EXCLUDED.program_overview,
                    program_benefits = EXCLUDED.program_benefits,
                    duration = EXCLUDED.duration,
                    fees = EXCLUDED.fees,
                    intake_info = EXCLUDED.intake_info,
                    entry_requirements = EXCLUDED.entry_requirements,
                    accreditation = EXCLUDED.accreditation,
                    curriculum = EXCLUDED.curriculum,
                    assessment = EXCLUDED.assessment,
                    support_services = EXCLUDED.support_services,
                    career_outcomes = EXCLUDED.career_outcomes,
                    academic_progression = EXCLUDED.academic_progression,
                    faculty = EXCLUDED.faculty,
                    testimonials = EXCLUDED.testimonials,
                    geographic_focus = EXCLUDED.geographic_focus,
                    technology_platform = EXCLUDED.technology_platform,
                    completion_rates = EXCLUDED.completion_rates,
                    academic_progression = EXCLUDED.academic_progression,
                    contact_info = EXCLUDED.contact_info,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """

            cur.execute(upsert_query, (
                program_variant_id,
                university_id,
                program_id,
                program_data.get('url'),
                json.dumps(basic_info),
                json.dumps(duration),
                json.dumps(program_data.get('program_overview')),
                json.dumps(program_data.get('program_benefits')),
                json.dumps(program_data.get('fees')),
                json.dumps(program_data.get('intake_info')),
                json.dumps(program_data.get('entry_requirements')),
                json.dumps(program_data.get('accreditation')),
                json.dumps(program_data.get('curriculum')),
                json.dumps(program_data.get('assessment')),
                json.dumps(program_data.get('support_services')),
                json.dumps(program_data.get('career_outcomes')),
                json.dumps(program_data.get('academic_progression')),
                json.dumps(program_data.get('faculty')),
                json.dumps(program_data.get('testimonials')),
                json.dumps(program_data.get('geographic_focus')),
                json.dumps(program_data.get('academic_progression')),
                json.dumps(program_data.get('technology_platform')),
                json.dumps(program_data.get('completion_rates')),
                json.dumps(program_data.get('contact_info')),
                json.dumps(program_data.get('metadata')),
            ))

            self.conn.commit()
            return True

        except Exception as e:
            self.conn.rollback()
            return False
        finally:
            cur.close()
