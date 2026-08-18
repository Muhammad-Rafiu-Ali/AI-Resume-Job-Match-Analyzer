# AI Resume & Job Match Analyzer

AI Resume & Job Match Analyzer is a Python and Streamlit-based web application that helps job seekers evaluate how well their resume matches a specific job description. The system analyzes resume content, calculates an ATS-style score, identifies matched and missing skills, extracts job keywords, recommends suitable job roles, and generates a professional cover letter.

This project is designed as a practical AI/NLP portfolio project for career technology, resume screening, and intelligent job application support.

---

## 🚀 Live Demo

- https://www.linkedin.com/posts/rafiu-ali_artificialintelligence-machinelearning-nlp-ugcPost-7493192382679629824-Lza2/?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEO2c-wBiWextr8AEHfaKyElOfVRbj-DAVQ

---
## Project Overview

Many job applicants submit resumes without knowing whether their CV is aligned with the job description. Recruiters and Applicant Tracking Systems often look for relevant skills, experience, keywords, formatting quality, and clear resume sections.

This project solves that problem by allowing users to:

1. Upload their resume.
2. Paste a job description.
3. Analyze resume-job fit.
4. View ATS-style score breakdown.
5. Identify missing skills and keywords.
6. Receive improvement suggestions.
7. Generate a cover letter.
8. Download an analysis report.

---

## Key Features

### Resume Upload

The application supports multiple resume formats:

- PDF
- DOCX
- TXT

It extracts resume text and uses it for further analysis.

---

### Job Match Score

The app compares the resume with the job description using NLP-based text similarity. It helps users understand how closely their resume matches the target job.

---

### ATS Score Breakdown

The system provides an explainable ATS-style score based on:

- Contact information
- Skills
- Education
- Experience
- Job keywords
- Resume formatting
- Resume length
- Section headings

This makes the score more transparent and useful for applicants.

---

### Matched and Missing Skills

The system detects skills from both the resume and the job description. It shows:

- Skills already present in the resume
- Important skills missing from the resume

This helps users improve their resume before applying.

---

### Matched and Missing Keywords

The app extracts important keywords from the job description and compares them with the resume text.

It identifies:

- Matched job keywords
- Missing job keywords

This is useful for improving resume relevance and ATS compatibility.

---

### Job Role Recommendation

Based on detected skills and matched keywords, the app recommends suitable job roles. This feature helps users understand which roles are most aligned with their current profile.

---

### Resume Section Check

The system checks whether the resume contains important sections such as:

- Contact information
- Professional summary
- Education
- Experience
- Skills
- Projects
- Certifications

---

### Improvement Suggestions

The app gives practical recommendations to improve the resume, such as:

- Add missing skills
- Include important job keywords
- Improve section headings
- Add measurable achievements
- Strengthen experience descriptions
- Improve job-specific alignment

---

### Cover Letter Generator

The application generates a professional cover letter based on the resume and job description.

A note is also included in the app to explain that some PDF resumes may have text extraction limitations due to image-based formatting, two-column layouts, or complex designs.

---

### Downloadable Report

Users can download a complete resume analysis report containing:

- Job match score
- ATS score
- Matched skills
- Missing skills
- Matched keywords
- Missing keywords
- Recommended job roles
- Resume improvement suggestions

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- PyPDF2
- python-docx
- Plotly
- Bootstrap
- HTML / CSS

---

## NLP and Analysis Logic

The project uses explainable NLP and rule-based scoring methods.

### TF-IDF Vectorization

TF-IDF is used to convert resume and job description text into numerical vectors.

### Cosine Similarity

Cosine similarity is used to calculate how closely the resume matches the job description.

### Keyword Matching

Important job-description keywords are extracted and compared with resume content.

### ATS-Style Rules

A rule-based scoring system checks resume structure, skills, education, experience, formatting, and keyword alignment.

---
## Install required libraries:
```text
pip install -r requirements.txt
```
## Run the Streamlit app:
```text
python -m streamlit run app.py
```
---
## How to Use
1. Open the Streamlit application.
2. Upload your resume in PDF, DOCX, or TXT format.
3. Paste the target job description.
4. Click the Analyze Resume button.
5. Review the ATS score, job match score, skills, keywords, and suggestions.
6. Generate and download the cover letter.
7. Download the full resume analysis report.
---
## Landing Page
This project also includes a Bootstrap-based landing website created using the Orbit template from BootstrapMade.
The landing page includes:
- Project overview
- Features
- Workflow
- Screenshots section
- Analysis logic
- Developer profile
- Contact and GitHub links
---
## Developer

Rafiu Ali

Data Science, AI & Business Intelligence Developer

- GitHub: https://github.com/Muhammad-Rafiu-Ali
- LinkedIn: https://www.linkedin.com/in/rafiu-ali/

---
## Project Structure

```text
AI-Resume-Job-Match-Analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── index.html
│
├── assets/
│   ├── css/
│   ├── js/
│   ├── img/
│   └── vendor/
│
└── sample/
    └── sample_resume.pdf
```
