import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px

from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Resume & Job Match Analyzer",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# Custom UI Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f9ff 0%, #eef5ff 45%, #ffffff 100%);
    color: #1f2937;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

.hero-card {
    background: linear-gradient(135deg, #243b63 0%, #47b2e4 100%);
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 18px 45px rgba(55, 81, 126, 0.25);
}

.hero-card h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
    color: white;
}

.hero-card p {
    font-size: 18px;
    line-height: 1.7;
    opacity: 0.95;
    color: white;
}

h1, h2, h3 {
    color: #243b63;
    font-weight: 800;
}

[data-testid="stMetric"] {
    background: #ffffff;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e6eef8;
    box-shadow: 0 10px 28px rgba(55, 81, 126, 0.10);
}

[data-testid="stMetricLabel"] {
    color: #243b63;
    font-weight: 700;
}

[data-testid="stMetricValue"] {
    color: #111827;
    font-weight: 800;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(135deg, #47b2e4, #243b63);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(71, 178, 228, 0.25);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #243b63, #111827);
    color: white;
    border: none;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #47b2e4;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(55, 81, 126, 0.08);
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 8px 22px rgba(55, 81, 126, 0.08);
}

[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid rgba(36, 59, 99, 0.14) !important;
    box-shadow: 0 6px 18px rgba(55, 81, 126, 0.06) !important;
}

/* Fix Streamlit alert/readability issue: some themes make info/warning/success text too light */
[data-testid="stAlert"],
[data-testid="stAlert"] *,
.stAlert,
.stAlert * {
    color: #1f2937 !important;
    opacity: 1 !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] li,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #1f2937 !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}

[data-testid="stAlert"] svg {
    color: #243b63 !important;
    fill: #243b63 !important;
}

/* Text area / input readability */
textarea,
input,
[data-baseweb="textarea"] textarea,
[data-baseweb="input"] input {
    color: #111827 !important;
    background-color: #ffffff !important;
    opacity: 1 !important;
}

label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] * {
    color: #243b63 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

.footer-box {
    margin-top: 40px;
    padding: 20px;
    text-align: center;
    background: #ffffff;
    border-radius: 18px;
    border: 1px solid #e6eef8;
    color: #243b63;
    box-shadow: 0 8px 20px rgba(55, 81, 126, 0.08);
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero-card">
    <h1>📄 AI Resume & Job Match Analyzer</h1>
    <p>
        Upload your resume, paste a job description, and get an explained ATS score,
        job match score, missing skills, missing job keywords, role recommendations,
        resume section check, improvement suggestions, and a downloadable report.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Skill Database
# -----------------------------
SKILLS = [
    # Tech / Data
    "python", "java", "javascript", "typescript", "c++", "c#", "html", "css",
    "sql", "mysql", "postgresql", "mongodb", "sqlite",
    "excel", "power bi", "tableau", "looker", "google analytics",
    "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "plotly",
    "machine learning", "deep learning", "data analysis", "data visualization",
    "statistics", "nlp", "natural language processing", "computer vision",
    "tensorflow", "keras", "pytorch", "streamlit", "flask", "django", "fastapi",
    "api", "rest api", "git", "github", "docker", "kubernetes",
    "aws", "azure", "gcp", "cloud computing",
    "data cleaning", "data mining", "data preprocessing", "eda",
    "business intelligence", "dashboard", "forecasting", "regression",
    "classification", "clustering", "time series", "random forest",
    "linear regression", "logistic regression", "xgboost",

    # Teaching / education
    "teaching", "teacher", "lesson planning", "classroom management",
    "curriculum development", "curriculum", "student assessment", "assessment",
    "instruction", "training", "learning outcomes", "student engagement",
    "english language", "spoken english", "grammar", "communication skills",
    "educational technology", "online teaching", "academic writing",
    "syllabus planning", "mentoring", "tutoring",

    # General professional
    "communication", "teamwork", "leadership", "problem solving",
    "project management", "agile", "scrum", "critical thinking",
    "research", "report writing", "presentation", "customer service",
    "sales", "marketing", "finance", "accounting", "hr", "operations"
]


ROLE_PROFILES = {
    "English Language Teacher": [
        "teaching", "english language", "spoken english", "grammar", "lesson planning",
        "classroom management", "assessment", "communication", "student engagement"
    ],
    "Classroom Instructor": [
        "teaching", "instruction", "classroom management", "lesson planning",
        "curriculum", "assessment", "communication", "presentation"
    ],
    "Teaching Assistant": [
        "teaching", "student assessment", "mentoring", "tutoring",
        "communication", "classroom management", "education"
    ],
    "Data Analyst": [
        "python", "sql", "excel", "pandas", "numpy", "data analysis",
        "data visualization", "statistics", "dashboard", "power bi", "tableau"
    ],
    "Business Intelligence Analyst": [
        "business intelligence", "power bi", "tableau", "dashboard", "sql",
        "excel", "data visualization", "data analysis", "report writing"
    ],
    "Junior Data Scientist": [
        "python", "machine learning", "statistics", "scikit-learn", "pandas",
        "numpy", "regression", "classification", "clustering", "eda"
    ],
    "Python Developer": [
        "python", "flask", "django", "fastapi", "api", "rest api",
        "git", "github", "sql", "docker"
    ],
    "Business Analyst": [
        "excel", "power bi", "data analysis", "business intelligence",
        "communication", "presentation", "report writing", "project management"
    ],
    "Marketing Analyst": [
        "marketing", "google analytics", "excel", "data analysis",
        "data visualization", "dashboard", "presentation", "statistics"
    ]
}


CUSTOM_STOPWORDS = {
    "job", "role", "candidate", "required", "requirements", "responsibilities",
    "ability", "work", "working", "team", "company", "organization", "including",
    "must", "should", "will", "using", "use", "good", "excellent", "strong",
    "knowledge", "experience", "years", "skills", "skill", "based", "related",
    "position", "apply", "application", "preferred", "qualification",
    "qualifications", "description", "duties"
}


# -----------------------------
# Helper Functions
# -----------------------------
def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"PDF reading error: {e}")
    return text


def extract_text_from_docx(file):
    text = ""
    try:
        document = Document(file)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        st.error(f"DOCX reading error: {e}")
    return text


def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_resume_text(uploaded_file):
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif file_name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    return ""


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#.\s-]", " ", text)
    return text.strip()


def calculate_similarity(resume_text, job_text):
    if not resume_text.strip() or not job_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(similarity * 100, 2)


def find_skills(text, skills_list):
    text = clean_text(text)
    found_skills = []

    for skill in skills_list:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text):
            found_skills.append(skill.title())

    return sorted(list(set(found_skills)))


def extract_job_keywords(job_text, top_n=35):
    cleaned = clean_text(job_text)

    try:
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=90
        )
        matrix = vectorizer.fit_transform([cleaned])
        keywords = vectorizer.get_feature_names_out()
        counts = matrix.toarray()[0]

        keyword_df = pd.DataFrame({
            "keyword": keywords,
            "count": counts
        }).sort_values(by="count", ascending=False)

        final_keywords = []
        for keyword in keyword_df["keyword"].tolist():
            parts = keyword.split()
            if any(part in CUSTOM_STOPWORDS for part in parts):
                continue
            if len(keyword) < 3:
                continue
            if keyword not in final_keywords:
                final_keywords.append(keyword)

        return final_keywords[:top_n]

    except Exception:
        return []


def match_keywords(resume_text, job_keywords):
    resume_clean = clean_text(resume_text)
    matched = []
    missing = []

    for keyword in job_keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        if re.search(pattern, resume_clean):
            matched.append(keyword)
        else:
            missing.append(keyword)

    return matched, missing


def check_section_presence(text):
    text_lower = text.lower()

    sections = {
        "Contact Information": bool(re.search(r"@|\+?\d[\d\s\-()]{7,}|linkedin|github|phone|email|contact", text_lower)),
        "Professional Summary": bool(re.search(r"summary|profile|objective|career objective|professional summary|about me", text_lower)),
        "Education": bool(re.search(r"education|degree|university|college|bachelor|master|phd|matric|intermediate", text_lower)),
        "Experience": bool(re.search(r"experience|employment|work history|internship|teaching experience|professional experience|job", text_lower)),
        "Skills": bool(re.search(r"skills|technical skills|core skills|tools|technologies|competencies", text_lower)),
        "Certifications": bool(re.search(r"certification|certificate|course|training|workshop", text_lower)),
        "Projects": bool(re.search(r"projects|portfolio|github|project work|academic project", text_lower))
    }

    return sections


def calculate_formatting_score(resume_text):
    word_count = len(resume_text.split())
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]

    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", resume_text))
    has_phone = bool(re.search(r"\+?\d[\d\s\-()]{7,}", resume_text))
    has_reasonable_lines = len(lines) >= 8
    has_too_many_symbols = len(re.findall(r"[^a-zA-Z0-9\s@+.#,;:/()\-]", resume_text)) > 80

    score = 0
    if has_email:
        score += 3
    if has_phone:
        score += 3
    if has_reasonable_lines:
        score += 2
    if not has_too_many_symbols:
        score += 2

    return min(score, 10)


def calculate_ats_breakdown(
    resume_text,
    resume_skills,
    job_skills,
    sections,
    matched_keywords,
    job_keywords
):
    word_count = len(resume_text.split())

    # This stricter ATS scoring gives more weight to job-specific alignment.
    # Total = 100 points.

    # 1. Contact Information: 5 points
    contact_points = 5 if sections.get("Contact Information") else 0

    # 2. Skills: 20 points
    if len(job_skills) > 0:
        matched_skill_count = len(set(resume_skills).intersection(set(job_skills)))
        skills_points = (matched_skill_count / len(job_skills)) * 20
    else:
        skills_points = 0

    # 3. Education: 7 points
    education_points = 7 if sections.get("Education") else 0

    # 4. Experience: 10 points
    experience_points = 10 if sections.get("Experience") else 0

    # 5. Job Keywords: 35 points
    if len(job_keywords) > 0:
        keyword_points = (len(matched_keywords) / len(job_keywords)) * 35
    else:
        keyword_points = 0

    # 6. Formatting: 5 points
    formatting_points = min(calculate_formatting_score(resume_text) / 10 * 5, 5)

    # 7. Resume Length: 8 points
    if 300 <= word_count <= 900:
        length_points = 8
    elif 150 <= word_count < 300 or 900 < word_count <= 1300:
        length_points = 5
    else:
        length_points = 3

    # 8. Section Headings: 10 points
    important_headings = [
        "Professional Summary", "Education", "Experience",
        "Skills", "Certifications", "Projects"
    ]
    found_headings = sum(1 for heading in important_headings if sections.get(heading))
    heading_points = (found_headings / len(important_headings)) * 10

    breakdown = {
        "Contact Information": round(contact_points, 2),
        "Skills": round(skills_points, 2),
        "Education": round(education_points, 2),
        "Experience": round(experience_points, 2),
        "Job Keywords": round(keyword_points, 2),
        "Formatting": round(formatting_points, 2),
        "Resume Length": round(length_points, 2),
        "Section Headings": round(heading_points, 2)
    }

    max_points = {
        "Contact Information": 5,
        "Skills": 20,
        "Education": 7,
        "Experience": 10,
        "Job Keywords": 35,
        "Formatting": 5,
        "Resume Length": 8,
        "Section Headings": 10
    }

    total_score = sum(breakdown.values())
    return round(min(total_score, 100), 2), breakdown, max_points

def get_match_level(ats_score):
    if ats_score >= 80:
        return "Strong Match", "Your resume is strongly aligned with this job description."
    elif ats_score >= 60:
        return "Moderate Match", "Your resume is reasonably aligned, but it can be improved with better keywords and missing skills."
    return "Weak Match", "Your resume needs stronger alignment with the job description before applying."


def recommend_job_roles(resume_skills, matched_keywords):
    resume_terms = {skill.lower() for skill in resume_skills}
    resume_terms.update({kw.lower() for kw in matched_keywords})

    recommendations = []

    for role, role_skills in ROLE_PROFILES.items():
        role_skill_set = {skill.lower() for skill in role_skills}
        matched = resume_terms.intersection(role_skill_set)
        missing = role_skill_set - resume_terms

        score = (len(matched) / len(role_skill_set)) * 100 if role_skill_set else 0

        recommendations.append({
            "Recommended Role": role,
            "Role Fit Score": round(score, 2),
            "Matched Role Skills": ", ".join(sorted([skill.title() for skill in matched])) if matched else "None",
            "Skills to Improve": ", ".join(sorted([skill.title() for skill in list(missing)[:6]])) if missing else "None"
        })

    rec_df = pd.DataFrame(recommendations)
    rec_df = rec_df.sort_values(by="Role Fit Score", ascending=False).head(5).reset_index(drop=True)

    # Ensure it never looks empty
    if rec_df["Role Fit Score"].max() == 0:
        rec_df.loc[0, "Role Fit Score"] = 5.0
        rec_df.loc[0, "Matched Role Skills"] = "General profile detected"
        rec_df.loc[0, "Skills to Improve"] = "Add clearer job-specific skills"

    return rec_df


def extract_candidate_name(resume_text):
    """Extract candidate name automatically from resume text.

    This function is conservative. It first tries explicit labels such as
    "Name: ...", then checks the top resume lines, then tries email/LinkedIn
    based inference. If it is not confident, it returns "Candidate" instead
    of using a wrong heading such as Experience, Education, or Skills.
    """
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    first_block = "\n".join(lines[:35])

    blocked_terms = {
        "resume", "cv", "curriculum", "vitae", "email", "phone", "contact",
        "address", "linkedin", "github", "objective", "summary", "profile",
        "education", "experience", "skills", "skill", "certification",
        "certifications", "projects", "project", "work", "employment",
        "career", "training", "languages", "interests", "references",
        "personal", "information", "professional", "academic", "qualification",
        "qualifications", "skillseducation", "skillsexperience", "experienceeducation",
        "portfolio", "about", "nationality", "address", "mobile", "telephone",
        "father", "date", "birth", "cnic", "passport", "teacher", "assistant",
        "curriculumvitae", "professionalprofile", "personalinformation",
        "office", "microsoft", "ms", "word", "powerpoint", "language", "languages",
        "lenguagems", "lenguage", "linguagem", "linguagens", "software", "computer",
        "basic", "intermediate", "advanced", "course", "courses"
    }

    blocked_phrases = [
        "experience", "education", "skills", "summary", "objective", "projects",
        "certification", "contact", "linkedin", "github", "email", "phone",
        "curriculum vitae", "professional profile", "personal information",
        "work experience", "career objective", "academic qualification",
        "teaching experience", "technical skills", "soft skills", "ms office",
        "microsoft office", "office skills", "language skills", "computer skills"
    ]

    def normalize_possible_name(value):
        value = re.sub(r"[^A-Za-z\s.'-]", " ", value)
        value = re.sub(r"\s+", " ", value).strip(" .,'-")
        words = value.split()

        if not (2 <= len(words) <= 4):
            return None
        if len(value) < 5 or len(value) > 45:
            return None

        joined = "".join(words).lower()
        value_lower = value.lower()

        if any(phrase.replace(" ", "") in joined for phrase in blocked_phrases):
            return None
        if any(term in value_lower for term in blocked_terms):
            return None
        if any(w.lower().strip(".'-") in blocked_terms for w in words):
            return None
        if any(w.lower() in {"and", "or", "with", "for", "from", "to", "in", "of", "the", "at", "by"} for w in words):
            return None
        if any(len(w) < 2 for w in words):
            return None

        # Avoid sentence-like lines. Names usually have only alphabetic tokens,
        # apostrophes, hyphens, or initials.
        for word in words:
            if not re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", word):
                return None

        return " ".join(w.capitalize() if w.isupper() or w.islower() else w for w in words)

    # 1) Direct patterns such as "Name: Rafiu Ali"
    direct_patterns = [
        r"(?:^|\n)\s*name\s*[:\-]\s*([A-Za-z][A-Za-z\s.'-]{3,45})",
        r"(?:^|\n)\s*candidate\s*name\s*[:\-]\s*([A-Za-z][A-Za-z\s.'-]{3,45})",
        r"(?:^|\n)\s*applicant\s*name\s*[:\-]\s*([A-Za-z][A-Za-z\s.'-]{3,45})",
    ]

    for pattern in direct_patterns:
        match = re.search(pattern, first_block, re.IGNORECASE)
        if match:
            name = normalize_possible_name(match.group(1))
            if name:
                return name.title()

    # 2) Top-line name detection. Most ATS-friendly resumes place the name near top.
    candidates = []
    for idx, line in enumerate(lines[:30]):
        raw_line = line.strip()
        lower_compact = raw_line.lower().replace(" ", "")

        if len(raw_line) < 3 or len(raw_line) > 60:
            continue
        if any(phrase.replace(" ", "") in lower_compact for phrase in blocked_phrases):
            continue
        if "@" in raw_line or re.search(r"\+?\d[\d\s\-()]{7,}", raw_line):
            continue
        if raw_line.lower().startswith(("http", "www")):
            continue
        if re.search(r"\d", raw_line):
            continue

        name = normalize_possible_name(raw_line)
        if not name:
            continue

        words = name.split()
        score = 0
        if idx <= 4:
            score += 5
        elif idx <= 10:
            score += 3
        else:
            score += 1

        # Real names commonly use Title Case or ALL CAPS in resume headers.
        original_words = re.sub(r"[^A-Za-z\s.'-]", " ", raw_line).split()
        title_like = sum(1 for w in original_words if w[:1].isupper() or w.isupper())
        if title_like >= max(1, len(original_words) - 1):
            score += 3

        if 8 <= len(name) <= 35:
            score += 1

        candidates.append((score, idx, name.title()))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        best_score, _, best_name = candidates[0]
        final_check = best_name.lower()
        unsafe_name_words = {
            "office", "language", "languages", "lenguagems", "microsoft", "word",
            "powerpoint", "skills", "education", "experience", "summary", "profile",
            "course", "courses", "software", "computer", "basic", "intermediate", "advanced"
        }
        if best_score >= 7 and not any(word in final_check for word in unsafe_name_words):
            return best_name

    # 3) Last-resort inference from email address, e.g. rafiu.ali@gmail.com.
    # This is only used when it looks like a real two-word personal name.
    email_match = re.search(r"([A-Za-z]+)[._-]([A-Za-z]+)@", first_block)
    if email_match:
        parts = [email_match.group(1), email_match.group(2)]
        if all(part.lower() not in blocked_terms and len(part) >= 2 for part in parts):
            return f"{parts[0].title()} {parts[1].title()}"

    linkedin_match = re.search(r"linkedin\.com/in/([A-Za-z]+)[-_]([A-Za-z]+)", first_block, re.IGNORECASE)
    if linkedin_match:
        parts = [linkedin_match.group(1), linkedin_match.group(2)]
        if all(part.lower() not in blocked_terms and len(part) >= 2 for part in parts):
            return f"{parts[0].title()} {parts[1].title()}"

    return "Candidate"

def extract_job_title(job_text):
    lines = [line.strip() for line in job_text.splitlines() if line.strip()]

    patterns = [
        r"job title\s*[:\-]\s*(.+)",
        r"position\s*[:\-]\s*(.+)",
        r"role\s*[:\-]\s*(.+)"
    ]

    for line in lines[:15]:
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()

    role_keywords = [
        "english teacher", "teacher", "teaching assistant", "data analyst",
        "business analyst", "data scientist", "python developer",
        "business intelligence analyst", "finance analyst", "marketing analyst"
    ]

    job_lower = job_text.lower()
    for role in role_keywords:
        if role in job_lower:
            return role.title()

    return "the advertised role"


def extract_organization_name(job_text):
    lines = [line.strip() for line in job_text.splitlines() if line.strip()]

    patterns = [
        r"company\s*[:\-]\s*(.+)",
        r"organization\s*[:\-]\s*(.+)",
        r"institute\s*[:\-]\s*(.+)",
        r"school\s*[:\-]\s*(.+)",
        r"college\s*[:\-]\s*(.+)",
        r"university\s*[:\-]\s*(.+)",
        r"at\s+([A-Z][A-Za-z&.,\s]{2,60})"
    ]

    for line in lines[:20]:
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                org = match.group(1).strip()
                org = re.split(r"\s{2,}|,|\|", org)[0].strip()
                if len(org) >= 3:
                    return org.title()

    return "your organization"


def generate_suggestions(match_score, missing_skills, missing_keywords, sections, resume_text, ats_score):
    suggestions = []

    if ats_score < 60:
        suggestions.append("Before applying, improve the resume alignment by adding more role-specific keywords and relevant skills from the job description.")
    elif ats_score < 80:
        suggestions.append("Your resume is moderately aligned. Improve it by adding missing skills, stronger achievements, and clearer project or work impact.")
    else:
        suggestions.append("Your resume is strongly aligned. Make final improvements by adding measurable results and role-specific achievements.")

    if match_score < 50:
        suggestions.append("The Job Match Score is low because the resume and job description have limited keyword overlap.")
    elif match_score < 70:
        suggestions.append("Your resume has moderate job-description alignment. Add more job-specific keywords to improve the match.")
    else:
        suggestions.append("Your resume has strong textual alignment with the job description.")

    if missing_skills:
        suggestions.append("Add or strengthen these missing skills if you genuinely have experience with them: " + ", ".join(missing_skills[:10]) + ".")

    if missing_keywords:
        suggestions.append("Include important missing job keywords naturally in your experience, skills, or summary section: " + ", ".join(missing_keywords[:12]) + ".")

    for section, present in sections.items():
        if not present:
            suggestions.append(f"Consider adding a clear '{section}' section to improve ATS readability.")

    word_count = len(resume_text.split())
    if word_count < 250:
        suggestions.append("The resume seems short. Add measurable achievements, project details, tools used, responsibilities, and impact.")
    elif word_count > 1200:
        suggestions.append("The resume seems long. Make it more concise and focused on the target role.")

    if not re.search(r"\d+%|\d+\+|\$\d+|\d+ years|\d+ months", resume_text.lower()):
        suggestions.append("Add measurable achievements such as percentages, number of students taught, accuracy improvement, revenue impact, or project outcomes.")

    return suggestions


def generate_score_explanation(match_score, ats_score, ats_breakdown, ats_max_points, matched_keywords, missing_keywords, matched_skills, missing_skills):
    explanation = []

    explanation.append(f"**Job Match Score ({match_score}%)** is calculated using TF-IDF and cosine similarity between the resume text and job description text.")
    explanation.append(f"**ATS Score ({ats_score}%)** is calculated from contact information, skills, education, experience, job keywords, formatting, resume length, and section headings.")

    category_percentages = {
        category: (ats_breakdown[category] / ats_max_points[category]) * 100 if ats_max_points[category] else 0
        for category in ats_breakdown
    }

    weakest = min(category_percentages, key=category_percentages.get)
    strongest = max(category_percentages, key=category_percentages.get)

    explanation.append(
        f"Strongest area: **{strongest}** with **{ats_breakdown[strongest]} / {ats_max_points[strongest]}** points."
    )
    explanation.append(
        f"Weakest area: **{weakest}** with **{ats_breakdown[weakest]} / {ats_max_points[weakest]}** points."
    )

    if matched_keywords:
        explanation.append("Some matched job keywords found in the resume: **" + ", ".join(matched_keywords[:8]) + "**.")
    if missing_keywords:
        explanation.append("Important missing job keywords include: **" + ", ".join(missing_keywords[:10]) + "**.")
    if missing_skills:
        explanation.append("Missing skills detected from the job description include: **" + ", ".join(missing_skills[:8]) + "**.")

    return "\n\n".join(explanation)

def generate_cover_letter(resume_text, job_text, matched_skills, ats_score):
    candidate_name = extract_candidate_name(resume_text)
    job_title = extract_job_title(job_text)
    organization_name = extract_organization_name(job_text)
    skills_line = ", ".join(matched_skills[:8]) if matched_skills else "relevant professional skills"

    if ats_score >= 75:
        alignment_sentence = f"My background aligns well with the requirements of the {job_title} position, particularly through my experience with {skills_line}."
    elif ats_score >= 55:
        alignment_sentence = f"My profile shows a reasonable foundation for the {job_title} position, and I am prepared to further strengthen the required skills for this role."
    else:
        alignment_sentence = f"I am interested in the {job_title} position and am actively developing the skills and experience required to contribute effectively in this role."

    if candidate_name == "Candidate":
        opening_line = f"I am writing to express my interest in the {job_title} opportunity at {organization_name}."
        signature = "[Your Name]"
    else:
        opening_line = f"My name is {candidate_name}, and I am writing to express my interest in the {job_title} opportunity at {organization_name}."
        signature = candidate_name

    cover_letter = f"""
Dear Hiring Manager,

{opening_line}

{alignment_sentence}

I am motivated to apply my communication, learning, problem-solving, and professional abilities in a practical work environment. I am also willing to continue improving my skills according to the requirements of the position and the goals of your organization.

Thank you for considering my application. I would welcome the opportunity to discuss how my background and potential can contribute to {organization_name}.

Sincerely,
{signature}
"""
    return cover_letter.strip()

def create_report(
    match_score,
    ats_score,
    match_level,
    ats_breakdown,
    ats_max_points,
    section_df,
    matched_skills,
    missing_skills,
    matched_keywords,
    missing_keywords,
    role_recommendations,
    suggestions,
    score_explanation
):
    report = []

    report.append("AI Resume & Job Match Analyzer Report")
    report.append("=" * 45)
    report.append("")
    report.append(f"Job Match Score: {match_score}%")
    report.append(f"ATS Score: {ats_score}%")
    report.append(f"Match Level: {match_level}")
    report.append("")
    report.append("Score Explanation")
    report.append("-" * 25)
    report.append(score_explanation.replace("**", ""))
    report.append("")
    report.append("ATS Score Breakdown")
    report.append("-" * 25)

    for component, points in ats_breakdown.items():
        report.append(f"{component}: {points} / {ats_max_points[component]}")

    report.append("")
    report.append("Resume Section Check")
    report.append("-" * 25)
    for _, row in section_df.iterrows():
        report.append(f"{row['Section']}: {row['Status']}")

    report.append("")
    report.append("Matched Skills")
    report.append("-" * 20)
    report.append(", ".join(matched_skills) if matched_skills else "No matched skills found.")

    report.append("")
    report.append("Missing Skills")
    report.append("-" * 20)
    report.append(", ".join(missing_skills) if missing_skills else "No major missing skills found.")

    report.append("")
    report.append("Matched Keywords")
    report.append("-" * 20)
    report.append(", ".join(matched_keywords) if matched_keywords else "No matched keywords found.")

    report.append("")
    report.append("Missing Keywords")
    report.append("-" * 20)
    report.append(", ".join(missing_keywords) if missing_keywords else "No missing keywords found.")

    report.append("")
    report.append("Recommended Job Roles")
    report.append("-" * 25)
    for _, row in role_recommendations.iterrows():
        report.append(f"{row['Recommended Role']}: {row['Role Fit Score']}%")

    report.append("")
    report.append("Improvement Suggestions")
    report.append("-" * 25)
    for i, suggestion in enumerate(suggestions, start=1):
        report.append(f"{i}. {suggestion}")

    report.append("")
    report.append("Generated by AI Resume & Job Match Analyzer.")

    return "\n".join(report)


# -----------------------------
# Main App
# -----------------------------
st.subheader("1. Upload Resume and Job Description")

col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx", "txt"]
    )

with col2:
    job_description = st.text_area(
        "Paste Job Description",
        height=250,
        placeholder="Paste the complete job description here..."
    )

analyze_btn = st.button("Analyze Resume")

if analyze_btn:
    if uploaded_resume is None:
        st.error("Please upload a resume file.")
        st.stop()

    if not job_description.strip():
        st.error("Please paste a job description.")
        st.stop()

    resume_text = extract_resume_text(uploaded_resume)

    if not resume_text.strip():
        st.error("Could not extract text from the resume. Try another PDF/DOCX/TXT file.")
        st.stop()

    cleaned_resume = clean_text(resume_text)
    cleaned_job = clean_text(job_description)

    match_score = calculate_similarity(cleaned_resume, cleaned_job)

    resume_skills = find_skills(cleaned_resume, SKILLS)
    job_skills = find_skills(cleaned_job, SKILLS)

    matched_skills = sorted(list(set(resume_skills).intersection(set(job_skills))))
    missing_skills = sorted(list(set(job_skills) - set(resume_skills)))

    job_keywords = extract_job_keywords(job_description, top_n=35)
    matched_keywords, missing_keywords = match_keywords(resume_text, job_keywords)

    sections = check_section_presence(resume_text)

    ats_score, ats_breakdown, ats_max_points = calculate_ats_breakdown(
        resume_text,
        resume_skills,
        job_skills,
        sections,
        matched_keywords,
        job_keywords
    )

    match_level, match_message = get_match_level(ats_score)

    role_recommendations = recommend_job_roles(resume_skills, matched_keywords)

    suggestions = generate_suggestions(
        match_score,
        missing_skills,
        missing_keywords,
        sections,
        resume_text,
        ats_score
    )

    score_explanation = generate_score_explanation(
        match_score,
        ats_score,
        ats_breakdown,
        ats_max_points,
        matched_keywords,
        missing_keywords,
        matched_skills,
        missing_skills
    )

    st.subheader("2. Resume Analysis Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Job Match Score", f"{match_score}%")
    c2.metric("ATS Score", f"{ats_score}%")
    c3.metric("Matched Skills", len(matched_skills))
    c4.metric("Missing Keywords", len(missing_keywords))

    if ats_score >= 80:
        st.success(f"### {match_level}\n{match_message}")
    elif ats_score >= 60:
        st.warning(f"### {match_level}\n{match_message}")
    else:
        st.error(f"### {match_level}\n{match_message}")

    st.subheader("3. Score Explanation")
    st.markdown(score_explanation)

    st.subheader("4. ATS Score Breakdown")

    breakdown_df = pd.DataFrame({
        "Category": list(ats_breakdown.keys()),
        "Score": list(ats_breakdown.values()),
        "Max Score": [ats_max_points[key] for key in ats_breakdown.keys()]
    })

    st.dataframe(breakdown_df, use_container_width=True)

    fig_breakdown = px.bar(
        breakdown_df,
        x="Category",
        y="Score",
        text="Score",
        title="ATS Score Breakdown by Category"
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

    st.subheader("5. Recommended Job Roles")

    st.dataframe(role_recommendations, use_container_width=True)

    fig_roles = px.bar(
        role_recommendations,
        x="Role Fit Score",
        y="Recommended Role",
        orientation="h",
        title="Top Recommended Roles Based on Resume Skills and Keywords",
        text="Role Fit Score"
    )
    st.plotly_chart(fig_roles, use_container_width=True)

    st.subheader("6. Skills and Job Keywords Match")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### ✅ Matched Skills")
        if matched_skills:
            st.success(", ".join(matched_skills))
        else:
            st.warning("No matched skills found.")

        st.markdown("### ✅ Matched Keywords")
        if matched_keywords:
            st.success(", ".join(matched_keywords[:20]))
        else:
            st.warning("No matched job keywords found.")

    with col4:
        st.markdown("### ⚠️ Missing Skills")
        if missing_skills:
            st.warning(", ".join(missing_skills))
        else:
            st.success("No major missing skills found.")

        st.markdown("### ⚠️ Missing Keywords")
        if missing_keywords:
            st.warning(", ".join(missing_keywords[:25]))
        else:
            st.success("No major missing keywords found.")

    keyword_df = pd.DataFrame({
        "Keyword Type": ["Matched Keywords", "Missing Keywords"],
        "Count": [len(matched_keywords), len(missing_keywords)]
    })

    fig_keywords = px.pie(
        keyword_df,
        names="Keyword Type",
        values="Count",
        title="Matched vs Missing Job Keywords"
    )
    st.plotly_chart(fig_keywords, use_container_width=True)

    st.subheader("7. Resume Section Check")

    section_df = pd.DataFrame({
        "Section": list(sections.keys()),
        "Status": ["Found" if v else "Not Found" for v in sections.values()]
    })

    st.dataframe(section_df, use_container_width=True)

    fig_sections = px.bar(
        section_df,
        x="Section",
        color="Status",
        title="Resume Section Availability"
    )
    st.plotly_chart(fig_sections, use_container_width=True)

    st.subheader("8. Improvement Suggestions")

    for suggestion in suggestions:
        st.info(suggestion)

    st.subheader("9. Cover Letter Generator")

    detected_candidate_name = extract_candidate_name(resume_text)
    detected_job_title = extract_job_title(job_description)
    detected_organization = extract_organization_name(job_description)

    display_candidate = detected_candidate_name if detected_candidate_name != "Candidate" else "Not confidently detected"

    cover_letter = generate_cover_letter(
        resume_text,
        job_description,
        matched_skills,
        ats_score
    )

    st.text_area(
        "Generated Cover Letter",
        cover_letter,
        height=300
    )

    st.info(
        "Note: Candidate name is extracted automatically from the resume text. Some PDF CVs store the name inside images, "
        "headers, text boxes, or two-column layouts, so the extracted text order can be different from what you see visually. "
        "When the app is not confident, it uses [Your Name] instead of guessing a wrong name. If a wrong name still appears, "
        "use a more ATS-friendly PDF/DOCX version with selectable text at the top."
    )

    st.subheader("10. Download Report")

    report = create_report(
        match_score,
        ats_score,
        match_level,
        ats_breakdown,
        ats_max_points,
        section_df,
        matched_skills,
        missing_skills,
        matched_keywords,
        missing_keywords,
        role_recommendations,
        suggestions,
        score_explanation
    )

    st.download_button(
        label="Download Resume Analysis Report",
        data=report,
        file_name="resume_job_match_report.txt",
        mime="text/plain"
    )

    st.download_button(
        label="Download Cover Letter",
        data=cover_letter,
        file_name="cover_letter.txt",
        mime="text/plain"
    )

    with st.expander("Show Extracted Resume Text"):
        st.write(resume_text)

else:
    st.info("Upload your resume and paste a job description, then click Analyze Resume.")


st.markdown("""
<div class="footer-box">
    <strong>AI Resume & Job Match Analyzer</strong><br>
    Built with Python, Streamlit, Scikit-learn, TF-IDF, Cosine Similarity, and keyword matching.
</div>
""", unsafe_allow_html=True)
