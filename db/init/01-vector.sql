-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Sessions table for chat history
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Messages table for chat history
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Universities table for canonical university data
CREATE TABLE IF NOT EXISTS universities (
    id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Programs table for canonical program data
CREATE TABLE IF NOT EXISTS programs (
    id SERIAL PRIMARY KEY,
    university VARCHAR(100) REFERENCES universities(id),
    program_type VARCHAR(255) NOT NULL, -- e.g., 'MBA', 'MSc', 'BSc'
    program_name VARCHAR(255) NOT NULL, -- e.g., 'Master of Business Administration', 'MSc Data Science'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(program_name, university)
);

-- Program variants table - THE SOURCE OF TRUTH for program data
CREATE TABLE IF NOT EXISTS program_variants (
    id SERIAL PRIMARY KEY,
    program_variant_id VARCHAR(100) NOT NULL UNIQUE,
    university_id VARCHAR(100) REFERENCES universities(id),
    program_id INTEGER REFERENCES programs(id),
    url TEXT,
    basic_info JSONB NOT NULL,
    program_overview TEXT NOT NULL,
    program_benefits JSONB,
    duration JSONB NOT NULL,
    fees JSONB,
    intake_info JSONB,
    entry_requirements JSONB,
    accreditation JSONB,
    curriculum JSONB,
    assessment JSONB,
    support_services JSONB,
    career_outcomes JSONB,
    unique_features JSONB,
    academic_progression JSONB,
    faculty JSONB,
    testimonials JSONB,
    geographic_focus JSONB,
    academic_progression JSONB,
    technology_platform JSONB,
    completion_rates JSONB,
    contact_info JSONB,
    metadata JSONB,

    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Documents table for vector chunks
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    source VARCHAR(500) NOT NULL,
    namespace VARCHAR(100) DEFAULT 'customer',
    program_variant_id VARCHAR(100),
    university_id VARCHAR(100),
    program_identifier VARCHAR(100),
    chunk_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_documents_namespace ON documents(namespace);
CREATE INDEX IF NOT EXISTS idx_documents_program_variant_id ON documents(program_variant_id);
CREATE INDEX IF NOT EXISTS idx_documents_university_id ON documents(university_id);
CREATE INDEX IF NOT EXISTS idx_documents_program_identifier ON documents(program_identifier);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_program_variants_university_id ON program_variants(university_id);
CREATE INDEX IF NOT EXISTS idx_program_variants_program_id ON program_variants(program_id);
CREATE INDEX IF NOT EXISTS idx_program_variants_active ON program_variants(active);

-- Function to update program variant timestamps
CREATE OR REPLACE FUNCTION update_program_variant_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at
CREATE TRIGGER trigger_update_program_variant_updated_at
    BEFORE UPDATE ON program_variants
    FOR EACH ROW
    EXECUTE FUNCTION update_program_variant_updated_at();
