from flask import Flask, request, render_template
import pdfplumber
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Shared: extract text from uploaded PDF ──────────────────────
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()


def truncate(text, max_chars=6000):
    """Truncate text to avoid hitting Groq context limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated for length]"


def validate_resume_text(text):
    """Raise if extracted text is too short to be a real resume."""
    if not text or len(text.strip()) < 100:
        raise ValueError(
            "Could not extract enough text from your PDF. "
            "Make sure it is a text-based PDF (not a scanned image). "
            "Try copy-pasting text from it to verify."
        )


# ── Original: simple keyword match analyze ──────────────────────
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


# ── New: 3-call deep JD analysis pipeline ───────────────────────
def groq_json(messages, call_name=""):
    """Call Groq and parse JSON response."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=4000,
            messages=messages,
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"AI returned invalid JSON in {call_name or 'unknown'} call: {str(e)}")


def analyze_jd_deep(resume_text, jd_text):
    """
    3-call pipeline:
    1. Deep JD analysis — extract hidden signals
    2. Resume analysis — structure, language, gaps
    3. Cross-reference — ranked actionable feedback
    """
    # Validate and truncate inputs
    validate_resume_text(resume_text)
    resume_short = truncate(resume_text, 6000)
    jd_short     = truncate(jd_text, 4000)

    # ── Call 1: JD Deep Analysis ────────────────────────────────
    jd_data = groq_json([
        {
            "role": "system",
            "content": (
                "You are an expert recruiter and talent strategist with 15 years experience. "
                "Analyze job descriptions at a deep level — not just keywords but hidden signals. "
                "Always respond with valid JSON only."
            )
        },
        {
            "role": "user",
            "content": f"""Analyze this job description deeply. Extract hidden signals beyond surface keywords.

JOB DESCRIPTION:
{jd_short}

Return JSON with this exact structure:
{{
  "company_stage": "early-stage startup | growth startup | mid-size company | enterprise | unknown",
  "company_stage_evidence": "specific phrase from JD that signals this",
  "culture_signals": ["signal 1", "signal 2", "signal 3"],
  "real_vs_stated": {{
    "must_have": ["truly non-negotiable skill or trait"],
    "nice_to_have": ["listed as required but actually flexible"],
    "unstated_but_critical": ["what they really need but didn't write"]
  }},
  "ideal_candidate_profile": "2-3 sentence description of who they actually want",
  "red_flags": ["potential red flag in the JD if any"],
  "language_tone": "ownership-driven | process-heavy | research-focused | growth-hacker | bureaucratic",
  "years_experience_real": "actual minimum years needed based on role complexity",
  "role_summary": "one plain-english sentence of what this person will actually do day to day"
}}"""
        }
    ], call_name="JD Analysis")

    # ── Call 2: Resume Analysis ──────────────────────────────────
    resume_data = groq_json([
        {
            "role": "system",
            "content": (
                "You are an expert resume reviewer and career coach. "
                "Analyze resumes for structure, language quality, and impact — not just keywords. "
                "Always respond with valid JSON only."
            )
        },
        {
            "role": "user",
            "content": f"""Analyze this resume deeply. Focus on language patterns, structure, and what is missing.

RESUME:
{resume_short}

Return JSON with this exact structure:
{{
  "language_style": "ownership-driven | passive | mixed",
  "passive_bullets_count": 0,
  "strong_bullets_count": 0,
  "has_metrics": true,
  "metrics_quality": "strong | weak | missing",
  "structure_issues": ["issue 1", "issue 2"],
  "buried_achievements": ["achievement that should be more prominent"],
  "skills_depth": "deep | shallow | mixed",
  "experience_level_signals": "junior | mid | senior | unclear",
  "resume_type": "specialist | generalist | unclear",
  "opening_effectiveness": "strong | weak | missing",
  "top_3_strengths": ["strength 1", "strength 2", "strength 3"],
  "top_3_weaknesses": ["weakness 1", "weakness 2", "weakness 3"]
}}"""
        }
    ], call_name="Resume Analysis")

    # ── Call 3: Cross-reference + Actionable Feedback ────────────
    # Pass BOTH the structured analysis AND the actual content
    # so feedback references real specifics from the resume/JD
    feedback_data = groq_json([
        {
            "role": "system",
            "content": (
                "You are a top-tier career coach who gives brutally honest, specific, actionable advice. "
                "You never give generic feedback. Every point must reference specific details from the "
                "actual resume text and job description provided. "
                "Always respond with valid JSON only."
            )
        },
        {
            "role": "user",
            "content": f"""Cross-reference this JD analysis and resume analysis WITH the actual source content.
Generate specific, ranked, actionable feedback that references real details from the resume and JD.

JD ANALYSIS:
{json.dumps(jd_data, indent=2)}

RESUME ANALYSIS:
{json.dumps(resume_data, indent=2)}

ACTUAL JD TEXT (for specific references):
{jd_short}

ACTUAL RESUME TEXT (for specific references):
{resume_short}

Return JSON with this exact structure:
{{
  "match_verdict": "strong | moderate | weak | mismatch",
  "match_summary": "2 honest sentences explaining overall fit",
  "biggest_gap": "the single most important thing missing",
  "actions": [
    {{
      "priority": "critical | high | medium | low",
      "category": "language | structure | content | positioning | skills",
      "title": "short action title",
      "problem": "what is wrong right now — reference a specific part of the resume",
      "fix": "exactly what to do, referencing specific resume content",
      "example": "before/after example using real content from the resume, otherwise null"
    }}
  ],
  "positioning_advice": "how to frame yourself for THIS specific company/role in 2-3 sentences",
  "one_thing": "if the candidate could only do ONE thing before applying, what is it"
}}

Order actions by priority (critical first). Include 4-7 actions.
Every action MUST reference a specific detail from the actual resume or JD text — never be generic."""
        }
    ], call_name="Cross-reference")

    return jd_data, resume_data, feedback_data


# ── Routes ───────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        pdf_file = request.files["resume"]
        job_desc = request.form["job_description"]
        resume_text = extract_text_from_pdf(pdf_file)
        result = analyze_resume(resume_text, job_desc)
    return render_template("index.html", result=result)


@app.route("/jd-analyze", methods=["GET", "POST"])
def jd_analyze():
    jd_signals = None
    resume_analysis = None
    feedback = None
    error = None

    if request.method == "POST":
        try:
            pdf_file = request.files.get("resume")
            jd_text  = request.form.get("job_description", "").strip()

            if not pdf_file or not jd_text:
                error = "Please upload a resume PDF and paste a job description."
            else:
                resume_text = extract_text_from_pdf(pdf_file)
                jd_signals, resume_analysis, feedback = analyze_jd_deep(resume_text, jd_text)
        except ValueError as e:
            error = str(e)
        except json.JSONDecodeError:
            error = "The AI returned an unexpected response. Please try again."
        except Exception as e:
            error = f"Something went wrong: {str(e)}"

    return render_template(
        "jd_analyzer.html",
        jd_signals=jd_signals,
        resume_analysis=resume_analysis,
        feedback=feedback,
        error=error
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))