"""
Round Configuration Service - Defines interview round structures.
Maps real company interview processes to practice rounds.
"""

from typing import Dict, List, Optional
from app.schemas import InterviewRound, RoundConfig, QuestionDifficulty


class RoundConfigService:
    """Service for managing interview round configurations."""
    
    @staticmethod
    def get_difficulty_for_experience(experience_years: int) -> QuestionDifficulty:
        """
        Calculate difficulty based on years of experience.
        
        Args:
            experience_years: Years of experience (0-30)
            
        Returns:
            QuestionDifficulty: EASY for juniors, MEDIUM for mid-level, HARD for seniors
        """
        if experience_years <= 2:
            return QuestionDifficulty.EASY
        elif experience_years <= 6:
            return QuestionDifficulty.MEDIUM
        else:
            return QuestionDifficulty.HARD
    
    # Round configurations matching real interview processes
    ROUND_CONFIGS: Dict[InterviewRound, RoundConfig] = {
        InterviewRound.HR_SCREENING: RoundConfig(
            round_type=InterviewRound.HR_SCREENING,
            name="HR Screening Round",
            description="Initial screening focusing on background, motivation, and culture fit",
            duration_minutes=20,
            question_count=4,
            difficulty=QuestionDifficulty.EASY,
            question_time_limit=90,
            categories=["behavioral", "motivation", "background"]
        ),
        
        InterviewRound.TECHNICAL_ROUND_1: RoundConfig(
            round_type=InterviewRound.TECHNICAL_ROUND_1,
            name="Technical Round 1 - Fundamentals",
            description="Core technical concepts, DSA basics, and domain fundamentals",
            duration_minutes=45,
            question_count=6,
            difficulty=QuestionDifficulty.MEDIUM,
            question_time_limit=120,
            categories=["technical", "coding", "fundamentals"]
        ),
        
        InterviewRound.TECHNICAL_ROUND_2: RoundConfig(
            round_type=InterviewRound.TECHNICAL_ROUND_2,
            name="Technical Round 2 - Deep Dive",
            description="Advanced technical concepts, architecture, and problem-solving",
            duration_minutes=60,
            question_count=6,
            difficulty=QuestionDifficulty.HARD,
            question_time_limit=150,
            categories=["technical", "architecture", "advanced_coding"]
        ),
        
        InterviewRound.SYSTEM_DESIGN: RoundConfig(
            round_type=InterviewRound.SYSTEM_DESIGN,
            name="System Design Round",
            description="Scalability, architecture decisions, and system tradeoffs",
            duration_minutes=60,
            question_count=2,
            difficulty=QuestionDifficulty.HARD,
            question_time_limit=180,
            categories=["system_design", "scalability", "architecture"]
        ),
        
        InterviewRound.BEHAVIORAL: RoundConfig(
            round_type=InterviewRound.BEHAVIORAL,
            name="Behavioral Round",
            description="STAR method scenarios, teamwork, conflict resolution",
            duration_minutes=40,
            question_count=5,
            difficulty=QuestionDifficulty.MEDIUM,
            question_time_limit=120,
            categories=["behavioral", "teamwork", "leadership"]
        ),
        
        InterviewRound.MANAGERIAL: RoundConfig(
            round_type=InterviewRound.MANAGERIAL,
            name="Managerial/Director Round",
            description="Strategic thinking, vision, cross-functional collaboration",
            duration_minutes=45,
            question_count=4,
            difficulty=QuestionDifficulty.HARD,
            question_time_limit=150,
            categories=["leadership", "strategy", "vision"]
        ),
        
        InterviewRound.MACHINE_LEARNING: RoundConfig(
            round_type=InterviewRound.MACHINE_LEARNING,
            name="Machine Learning Specialist Round",
            description="ML algorithms, model selection, feature engineering, deployment",
            duration_minutes=60,
            question_count=5,
            difficulty=QuestionDifficulty.HARD,
            question_time_limit=150,
            categories=["machine_learning", "algorithms", "ml_systems"]
        ),
        
        InterviewRound.DATA_ENGINEERING: RoundConfig(
            round_type=InterviewRound.DATA_ENGINEERING,
            name="Data Engineering Round",
            description="Data pipelines, ETL, big data technologies, data modeling",
            duration_minutes=60,
            question_count=5,
            difficulty=QuestionDifficulty.HARD,
            question_time_limit=150,
            categories=["data_engineering", "pipelines", "big_data"]
        ),
        
        InterviewRound.FRONTEND_SPECIALIST: RoundConfig(
            round_type=InterviewRound.FRONTEND_SPECIALIST,
            name="Frontend Specialist Round",
            description="React/Vue/Angular, performance optimization, UI/UX",
            duration_minutes=60,
            question_count=6,
            difficulty=QuestionDifficulty.HARD,
            question_time_limit=150,
            categories=["frontend", "react", "performance"]
        ),
        
        InterviewRound.BACKEND_SPECIALIST: RoundConfig(
            round_type=InterviewRound.BACKEND_SPECIALIST,
            name="Backend Specialist Round",
            description="APIs, databases, microservices, backend architecture",
            duration_minutes=60,
            question_count=6,
            difficulty=QuestionDifficulty.HARD,
            question_time_limit=150,
            categories=["backend", "apis", "databases"]
        ),
        
        InterviewRound.DEVOPS: RoundConfig(
            round_type=InterviewRound.DEVOPS,
            name="DevOps/SRE Round",
            description="CI/CD, infrastructure, monitoring, reliability",
            duration_minutes=60,
            question_count=5,
            difficulty=QuestionDifficulty.HARD,
            question_time_limit=150,
            categories=["devops", "infrastructure", "reliability"]
        ),
        
        InterviewRound.SECURITY: RoundConfig(
            round_type=InterviewRound.SECURITY,
            name="Security Specialist Round",
            description="Application security, vulnerabilities, secure coding",
            duration_minutes=60,
            question_count=5,
            difficulty=QuestionDifficulty.HARD,
            question_time_limit=150,
            categories=["security", "vulnerabilities", "secure_coding"]
        ),
        
        InterviewRound.FULL_INTERVIEW: RoundConfig(
            round_type=InterviewRound.FULL_INTERVIEW,
            name="Full Interview Day Simulation",
            description="Complete interview experience: HR → Technical 1 → Technical 2 → Behavioral (simulates real interview day)",
            duration_minutes=180,
            question_count=18,
            difficulty=QuestionDifficulty.MEDIUM,
            question_time_limit=120,
            categories=["hr", "technical", "behavioral"]  # Mixed sequence
        ),
    }
    
    # Recommended round sequences by experience level
    ROUND_SEQUENCES = {
        "junior": [
            InterviewRound.HR_SCREENING,
            InterviewRound.TECHNICAL_ROUND_1,
            InterviewRound.BEHAVIORAL
        ],
        "mid": [
            InterviewRound.HR_SCREENING,
            InterviewRound.TECHNICAL_ROUND_1,
            InterviewRound.TECHNICAL_ROUND_2,
            InterviewRound.BEHAVIORAL
        ],
        "senior": [
            InterviewRound.HR_SCREENING,
            InterviewRound.TECHNICAL_ROUND_2,
            InterviewRound.SYSTEM_DESIGN,
            InterviewRound.BEHAVIORAL,
            InterviewRound.MANAGERIAL
        ]
    }
    
    @classmethod
    def get_round_config(cls, round_type: InterviewRound) -> RoundConfig:
        """Get configuration for a specific round."""
        return cls.ROUND_CONFIGS[round_type]
    
    @classmethod
    def get_all_rounds(cls) -> List[RoundConfig]:
        """Get all available round configurations."""
        return list(cls.ROUND_CONFIGS.values())
    
    @classmethod
    def get_recommended_round(cls, experience_years: int) -> InterviewRound:
        """Get recommended starting round based on experience."""
        if experience_years < 3:
            return InterviewRound.TECHNICAL_ROUND_1
        elif experience_years < 7:
            return InterviewRound.TECHNICAL_ROUND_2
        else:
            return InterviewRound.SYSTEM_DESIGN
    
    @classmethod
    def get_recommended_sequence(cls, experience_years: int) -> List[InterviewRound]:
        """Get recommended round sequence based on experience."""
        if experience_years < 3:
            return cls.ROUND_SEQUENCES["junior"]
        elif experience_years < 7:
            return cls.ROUND_SEQUENCES["mid"]
        else:
            return cls.ROUND_SEQUENCES["senior"]
    
    @classmethod
    def get_rounds_for_domain(cls, domain: str) -> List[InterviewRound]:
        """
        Get relevant rounds for a specific domain.
        
        Returns ONLY the rounds applicable to the selected domain.
        Filters out irrelevant specialist rounds.
        
        Example:
        - "Python Backend" → Shows Backend Specialist, NOT ML/Frontend/Data Eng
        - "Machine Learning" → Shows ML Specialist, NOT Frontend/Backend
        """
        domain_lower = domain.lower()
        
        # CORE rounds - Always shown (universal for all domains)
        core_rounds = [
            InterviewRound.HR_SCREENING,           # Every role has HR screening
            InterviewRound.TECHNICAL_ROUND_1,      # Fundamentals (domain-specific)
            InterviewRound.BEHAVIORAL,             # STAR method, teamwork
        ]
        
        # ADVANCED rounds - Experience-dependent
        advanced_rounds = [
            InterviewRound.TECHNICAL_ROUND_2,      # Deep dive (domain-specific)
            InterviewRound.MANAGERIAL,             # Leadership (senior roles)
        ]
        
        # SPECIALIST rounds - Domain-specific (ONLY show if domain matches!)
        specialist_rounds = []
        
        # Machine Learning / AI / Data Science
        if any(keyword in domain_lower for keyword in ["machine learning", "ml", "ai", "data science", "deep learning", "nlp"]):
            specialist_rounds.append(InterviewRound.MACHINE_LEARNING)
            core_rounds.append(InterviewRound.SYSTEM_DESIGN)  # ML often needs system design
        
        # Data Engineering / Big Data
        elif any(keyword in domain_lower for keyword in ["data engineer", "big data", "etl", "data pipeline"]):
            specialist_rounds.append(InterviewRound.DATA_ENGINEERING)
            core_rounds.append(InterviewRound.SYSTEM_DESIGN)
        
        # Frontend Development
        elif any(keyword in domain_lower for keyword in ["frontend", "react", "vue", "angular", "ui", "javascript", "typescript"]):
            specialist_rounds.append(InterviewRound.FRONTEND_SPECIALIST)
            advanced_rounds.append(InterviewRound.SYSTEM_DESIGN)  # Frontend system design
        
        # Backend Development
        elif any(keyword in domain_lower for keyword in ["backend", "api", "microservice", "python", "java", "node", "go", "c#"]):
            specialist_rounds.append(InterviewRound.BACKEND_SPECIALIST)
            core_rounds.append(InterviewRound.SYSTEM_DESIGN)  # Backend always has system design
        
        # DevOps / SRE / Cloud
        elif any(keyword in domain_lower for keyword in ["devops", "sre", "cloud", "infrastructure", "kubernetes", "docker"]):
            specialist_rounds.append(InterviewRound.DEVOPS)
            core_rounds.append(InterviewRound.SYSTEM_DESIGN)
        
        # Security Engineering
        elif any(keyword in domain_lower for keyword in ["security", "cybersecurity", "penetration", "appsec"]):
            specialist_rounds.append(InterviewRound.SECURITY)
            advanced_rounds.append(InterviewRound.SYSTEM_DESIGN)
        
        # Generic/Unknown domain - Show general technical path
        else:
            core_rounds.append(InterviewRound.SYSTEM_DESIGN)
            advanced_rounds.append(InterviewRound.TECHNICAL_ROUND_2)
        
        # Combine: Core → Specialist → Advanced → Full Interview
        all_rounds = core_rounds + specialist_rounds + advanced_rounds
        
        # Always add Full Interview Day at the end
        all_rounds.append(InterviewRound.FULL_INTERVIEW)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_rounds = []
        for r in all_rounds:
            if r not in seen:
                seen.add(r)
                unique_rounds.append(r)
        
        return unique_rounds
    
    @classmethod
    def get_full_interview_breakdown(cls, experience_years: int) -> Dict[str, int]:
        """
        Get the breakdown for full interview simulation.
        Returns dict with round_type: question_count distribution.
        
        Simulates a real interview day:
        - Junior: HR (3) → Tech 1 (6) → Behavioral (4) → Tech 2 (5) = 18 questions
        - Mid: HR (2) → Tech 1 (5) → Tech 2 (6) → Behavioral (5) = 18 questions  
        - Senior: HR (2) → Tech 2 (5) → System Design (3) → Behavioral (4) → Managerial (4) = 18 questions
        """
        if experience_years < 3:
            # Junior full day
            return {
                InterviewRound.HR_SCREENING: 3,
                InterviewRound.TECHNICAL_ROUND_1: 6,
                InterviewRound.BEHAVIORAL: 4,
                InterviewRound.TECHNICAL_ROUND_2: 5
            }
        elif experience_years < 7:
            # Mid-level full day
            return {
                InterviewRound.HR_SCREENING: 2,
                InterviewRound.TECHNICAL_ROUND_1: 5,
                InterviewRound.TECHNICAL_ROUND_2: 6,
                InterviewRound.BEHAVIORAL: 5
            }
        else:
            # Senior full day
            return {
                InterviewRound.HR_SCREENING: 2,
                InterviewRound.TECHNICAL_ROUND_2: 5,
                InterviewRound.SYSTEM_DESIGN: 3,
                InterviewRound.BEHAVIORAL: 4,
                InterviewRound.MANAGERIAL: 4
            }
