from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (only first time)
nltk.download("punkt")
nltk.download("stopwords")


# Step 1: Extract text from PDF
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""

    for para in doc.paragraphs:
        text += para.text + " "

    return text

# Step 2: Preprocess text
def preprocess_text(text):
    words = word_tokenize(text.lower())

    filtered_words = []

    for word in words:
        if word.isalnum() and word not in stopwords.words("english"):
            filtered_words.append(word)

    return " ".join(filtered_words)


# Step 3: Calculate match score using TF-IDF + Cosine Similarity
def calculate_match_score(resume_text, jd_text):
    tfidf = TfidfVectorizer()

    vectors = tfidf.fit_transform([resume_text, jd_text])

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])

    return round(similarity[0][0] * 100, 2)


# Step 4: Skill extraction
skills_list = [
    "python",
    "sql",
    "flask",
    "fastapi",
    "machine learning",
    "nlp",
    "tensorflow",
    "pytorch",
    "cnn"
]


def extract_skills(text):
    found_skills = []

    for skill in skills_list:
        if skill in text.lower():
            found_skills.append(skill)

    return found_skills


# Step 5: Find missing skills
def find_missing_skills(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    missing = []

    for skill in jd_skills:
        if skill not in resume_skills:
            missing.append(skill)

    return missing


# Step 6: Main program
resume_path = "resume.docx"
jd_path = "job_description.txt"

resume_text = extract_text_from_docx(resume_path)

with open(jd_path, "r", encoding="utf-8") as file:
    job_description = file.read()

resume_clean = preprocess_text(resume_text)
jd_clean = preprocess_text(job_description)

match_score = calculate_match_score(resume_clean, jd_clean)

skills_found = extract_skills(resume_text)
missing_skills = find_missing_skills(resume_text, job_description)


# Step 7: Recommendation
if match_score >= 50:
    recommendation = "Strong Match - Shortlist Candidate"
elif match_score >= 30:
    recommendation = "Moderate Match - Consider for Interview"
else:
    recommendation = "Low Match - Needs Improvement"


# Final Output
print("\n===== Resume Screening Result =====")
print("Match Score:", match_score, "%")
print("Skills Found:", skills_found)
print("Missing Skills:", missing_skills)
print("Recommendation:", recommendation)