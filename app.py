from flask import Flask, request, render_template
import pdfplumber
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def analyze_resume(resume_text, job_description):
    prompt = f"""
You are an expert HR recruiter and resume analyst.

Resume:
{resume_text}

Job Description:
{job_description}

Analyze the resume against the job description and return EXACTLY in this format:

MATCH SCORE: (a number 0-100)

MISSING KEYWORDS:
- keyword 1
- keyword 2
- keyword 3
- keyword 4
- keyword 5

SUGGESTIONS:
- suggestion 1
- suggestion 2
- suggestion 3
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        pdf_file = request.files["resume"]
        job_desc = request.form["job_description"]
        resume_text = extract_text_from_pdf(pdf_file)
        result = analyze_resume(resume_text, job_desc)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)