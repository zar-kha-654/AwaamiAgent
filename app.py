
import os
import json
import re
import io
import streamlit as st
from rag.knowledge_base import ensure_knowledge_base
from rag.rag_pipeline import query_civic_rag
from groq import Groq
from pypdf import PdfReader
from PIL import Image
import pytesseract

# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-120b"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AwaamiAgent | Civic Assistance",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.markdown("""
<style>

    /* Main application */
    .main {
        background-color: #f8fafc;
    }

    /* Main content width */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .awaami-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    .awaami-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    /* Section cards */
    .section-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.4rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .section-description {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    /* Result cards */
    .result-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.3rem;
        margin: 0.8rem 0;
    }

    .result-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }

    .result-value {
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Small feature badges */
    .feature-badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        margin: 0.2rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
        padding: 1.5rem 0 0.5rem 0;
    }

    /* Improve buttons */
    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 2.7rem;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# API KEY
# ============================================================

def get_api_key():
    """
    Get the Groq API key from:
    1. Streamlit Secrets during deployment
    2. Environment variable during Colab/local development
    """

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    return api_key


api_key = get_api_key()

if not api_key:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Add it to the environment or Streamlit Secrets."
    )
    st.stop()


client = Groq(api_key=api_key)

# ============================================================
# RAG KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def load_rag_knowledge_base():
    return ensure_knowledge_base()


try:
    rag_status = load_rag_knowledge_base()

except Exception as e:
    rag_status = None
    st.warning(
        "Official civic knowledge base could not be loaded yet. "
        "AwaamiAgent will continue without RAG grounding."
    )
    st.caption(f"RAG technical error: {str(e)}")

# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes):
    """Extract selectable text from a PDF."""
    
    reader = PdfReader(io.BytesIO(file_bytes))

    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    full_text = "\n\n".join(pages).strip()

    return {
        "text": full_text,
        "page_count": len(reader.pages),
        "extraction_method": "pdf_text"
    }


def extract_image_text(file_bytes):
    """Extract text from an image using OCR."""
    
    image = Image.open(io.BytesIO(file_bytes))

    text = pytesseract.image_to_string(
        image,
        lang="eng"
    )

    return {
        "text": text.strip(),
        "page_count": 1,
        "extraction_method": "ocr"
    }


def process_uploaded_document(uploaded_file):
    """Process an uploaded PDF or image."""

    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.getvalue()

    metadata = {
        "filename": uploaded_file.name,
        "file_type": uploaded_file.type,
        "file_size_bytes": len(file_bytes)
    }

    try:

        if uploaded_file.type == "application/pdf":

            result = extract_pdf_text(file_bytes)

        elif uploaded_file.type in [
            "image/png",
            "image/jpeg"
        ]:

            result = extract_image_text(file_bytes)

        else:
            raise ValueError("Unsupported file type.")

        metadata.update({
            "page_count": result["page_count"],
            "extraction_method": result["extraction_method"]
        })

        return {
            "metadata": metadata,
            "text": result["text"]
        }

    except Exception as e:

        raise ValueError(
            f"Could not process the uploaded document: {str(e)}"
        )
# ============================================================
# AI ANALYSIS
# ============================================================

# ============================================================
# AI ANALYSIS
# ============================================================

def analyze_civic_problem(
    problem,
    language,
    document_context=None
):
    """
    Analyze a user's civic problem using RAG + Groq.
    If Groq is unavailable, return the official RAG evidence
    instead of crashing the application.
    """

    language_instruction = (
        "Respond in English."
        if language == "English"
        else "Respond in natural, simple Urdu. Keep important English terms "
             "in parentheses when useful."
    )

    # --------------------------------------------------------
    # DOCUMENT CONTEXT
    # --------------------------------------------------------

    if document_context:

        document_text = document_context.get(
            "text",
            ""
        )

        document_metadata = json.dumps(
            document_context.get(
                "metadata",
                {}
            ),
            ensure_ascii=False,
            indent=2
        )

    else:

        document_text = "No document uploaded."
        document_metadata = "{}"

    # --------------------------------------------------------
    # RAG RETRIEVAL
    # --------------------------------------------------------

    rag_context = ""
    rag_sources = []

    try:

        # If the user only uploaded a document, use its text
        # as the RAG query instead of an empty problem.
        rag_query = problem.strip()

        if not rag_query and document_text:
            rag_query = document_text[:3000]

        if rag_query:

            rag_result = query_civic_rag(
                rag_query,
                top_k=5
            )

            rag_context = rag_result.get(
                "context",
                ""
            )

            rag_sources = rag_result.get(
                "sources",
                []
            )

    except Exception as e:

        print(
            f"RAG retrieval error: {e}"
        )

        rag_context = ""
        rag_sources = []

    # --------------------------------------------------------
    # GROQ PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are AwaamiAgent, an AI civic assistance system.

Your job is to help an ordinary person understand a civic problem
and identify practical next steps.

{language_instruction}

IMPORTANT SAFETY RULES:

- Do not invent laws.
- Do not invent government departments.
- Do not invent deadlines.
- Do not invent fees.
- Do not invent procedures.
- Do not claim uncertain information is verified.
- If jurisdiction-specific information is missing, clearly say so.
- Do not present the response as legal advice.
- Give general practical guidance only.

Return ONLY valid JSON using exactly these six fields:

{{
  "issue_category": "short category",
  "explanation": "simple explanation of the problem",
  "important_information": [
    "important point 1",
    "important point 2"
  ],
  "next_steps": [
    "practical step 1",
    "practical step 2"
  ],
  "required_documents": [
    "document or information 1",
    "document or information 2"
  ],
  "complaint": "short initial complaint/application draft"
}}

Keep the response concise.

User's civic problem:

{problem if problem.strip() else "No problem description provided."}

Uploaded document metadata:

{document_metadata}

Official civic source evidence:

{rag_context if rag_context else "No matching official source evidence was found."}

IMPORTANT:

Use the official civic source evidence above whenever it is relevant.

Do not invent laws, procedures, deadlines, fees, or departments.

If the official evidence does not contain enough information,
clearly say that the information should be verified with the
relevant official authority.

Uploaded document text:

{document_text}
"""

    # --------------------------------------------------------
    # GROQ ANALYSIS
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        raw_output = response.choices[0].message.content.strip()

        result = parse_json_response(
            raw_output
        )

        # Preserve RAG sources so the UI can display them.
        result["_rag_sources"] = rag_sources

        return result

    except Exception as e:

        print(
            f"Groq API error: {e}"
        )

        # ----------------------------------------------------
        # FALLBACK WHEN GROQ IS UNAVAILABLE
        # ----------------------------------------------------

        return {
            "issue_category": "Civic Issue",

            "explanation": (
                "AwaamiAgent retrieved information from its "
                "official civic knowledge base. The AI analysis "
                "service is temporarily unavailable, so the "
                "retrieved information is shown without additional "
                "AI interpretation."
            ),

            "important_information": (
                [
                    "Relevant official civic information was retrieved."
                ]
                if rag_context
                else
                [
                    "No matching official civic source evidence "
                    "was retrieved."
                ]
            ),

            "next_steps": [
                "Review the available official source information.",
                "Verify the applicable procedure with the relevant "
                "government authority before taking action."
            ],

            "required_documents": [],

            "complaint": (
                "AI complaint generation is temporarily unavailable. "
                "Please use the official information and enter your "
                "personal details before submitting a complaint."
            ),

            "_rag_sources": rag_sources
        }


def parse_json_response(text):
    """
    Parse JSON even if the model accidentally surrounds it
    with markdown code fences.
    """

    # Remove markdown code fences if present
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE
    )

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to locate the JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("The AI returned an invalid JSON response.")

        try:
            result = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            raise ValueError("The AI returned an invalid JSON response.")

    required_fields = [
        "issue_category",
        "explanation",
        "important_information",
        "next_steps",
        "required_documents",
        "complaint"
    ]

    for field in required_fields:
        if field not in result:
            result[field] = []

    return result


# ============================================================
# SEPARATE COMPLAINT GENERATOR
# ============================================================

def generate_complaint(
    issue_category,
    explanation,
    important_information,
    language
):
    """
    Generate a polished complaint/application locally.
    This does not require the Groq API.
    """

    information = "\n".join(
        f"- {item}"
        for item in important_information
    )

    if language == "Urdu":

        return f"""درخواست / شکایت

موضوع: {issue_category}

محترم متعلقہ افسر،

میں اس مسئلے کے حوالے سے درخواست/شکایت پیش کرنا چاہتا/چاہتی ہوں۔

مسئلے کی تفصیل:
{explanation}

اہم معلومات:
{information}

براہِ کرم اس معاملے کا جائزہ لے کر ضروری کارروائی اور رہنمائی فراہم کی جائے۔

شکریہ۔

نام: [آپ کا نام]
پتہ: [آپ کا پتہ]
رابطہ نمبر: [آپ کا فون نمبر]
اکاؤنٹ/ریفرنس نمبر: [اگر قابل اطلاق ہو]
تاریخ: [تاریخ]
"""

    return f"""COMPLAINT / APPLICATION

Subject: {issue_category}

To,
The Relevant Authority

Dear Sir/Madam,

I am writing to request assistance regarding the following civic issue.

Issue:
{explanation}

Important Information:
{information}

I kindly request that this matter be reviewed and that I be provided
with the appropriate guidance or assistance.

Thank you for your consideration.

Sincerely,

Name: [Your Name]
Address: [Your Address]
Contact Number: [Your Phone Number]
Account/Reference Number: [If applicable]
Date: [Date]
"""
# ============================================================
# SESSION STATE
# ============================================================

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "complaint" not in st.session_state:
    st.session_state.complaint = None


# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <div class="awaami-title">🏛️ AwaamiAgent</div>
    <div class="awaami-subtitle">
        AI-powered civic assistance to help citizens understand problems,
        find practical next steps, and prepare complaints.
    </div>
    """,
    unsafe_allow_html=True
)
with st.sidebar:

    st.markdown("## 🏛️ AwaamiAgent")

    st.markdown(
        """
        Your civic assistance companion.

        Describe a civic problem or upload a supporting document,
        and AwaamiAgent will help you understand the situation.
        """
    )

    st.divider()

    st.markdown("### How it works")

    st.markdown(
        """
        **1. Describe**  
        Tell us what happened.

        **2. Analyze**  
        AwaamiAgent examines the information.

        **3. Understand**  
        Get a simple explanation.

        **4. Act**  
        Review practical next steps and prepare a complaint.
        """
    )

    st.divider()

    st.markdown("### Supported documents")

    st.markdown(
        """
        - PDF documents
        - PNG images
        - JPG/JPEG images
        - Scanned civic documents
        - Bills and notices
        """
    )

    st.divider()

    st.caption(
        "AwaamiAgent provides general civic assistance. "
        "Always verify jurisdiction-specific information through "
        "reliable official sources."
    )

st.info(
    "ℹ️ AwaamiAgent provides general civic assistance. "
    "Laws, procedures, deadlines, fees, and responsible departments "
    "should always be verified through reliable official sources."
)


# ============================================================
# INPUT
# ============================================================

st.markdown(
    """
    <div class="section-card">
        <div class="section-title">📝 Describe Your Civic Problem</div>
        <div class="section-description">
            Explain your problem in simple words. You can mention what happened,
            where the problem occurred, and any important details you know.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

problem = st.text_area(
    "Your problem",
    placeholder=(
        "Example: My electricity bill is much higher than usual "
        "and I do not understand why."
    ),
    height=150,
    label_visibility="collapsed"
)

st.markdown(
    """
    <div class="section-card">
        <div class="section-title">📎 Supporting Document</div>
        <div class="section-description">
            Have a bill, notice, scanned document, or other relevant file?
            Upload it here for additional context.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a supporting document",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Upload a PDF, scanned document, bill, notice, or image."
)

language = st.selectbox(
    "🌐 Response language",
    ["English", "Urdu"]
)

# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔎 Analyze My Problem",
    type="primary",
    use_container_width=True
):

    if not problem.strip() and uploaded_file is None:
        st.warning(
            "Please describe your civic problem or upload a document."
        )
    else:
        st.session_state.analysis = None
        st.session_state.complaint = None

        with st.spinner("AwaamiAgent is analyzing your problem..."):
            try:
                document_context = None

                if uploaded_file is not None:
                    with st.spinner("Reading uploaded document..."):
                        document_context = process_uploaded_document(
                            uploaded_file
                        )

                result = analyze_civic_problem(
                    problem.strip(),
                    language,
                    document_context
                )

                st.session_state.analysis = result

            except Exception as e:
                st.error(
                    "Something went wrong while analyzing the problem."
                )

                st.caption(f"Technical error: {str(e)}")


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.analysis:

    result = st.session_state.analysis

    st.divider()

    st.markdown("## 📊 Your Civic Assessment")

    # --------------------------------------------------------
    # ISSUE IDENTIFIED
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Issue Identified</div>
            <div class="result-value">
                {result.get("issue_category", "Not specified")}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    st.markdown("### 💡 What's happening?")

    st.markdown(
        f"""
        <div class="result-card">
            {result.get("explanation", "No explanation available.")}
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # IMPORTANT INFORMATION
    # --------------------------------------------------------

    st.markdown("### 🔎 Important Information")

    important_information = result.get(
        "important_information",
        []
    )

    if important_information:

        for item in important_information:

            st.markdown(
                f"""
                <div class="result-card">
                    • {item}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.write("No additional information provided.")
    # --------------------------------------------------------
    # OFFICIAL SOURCES
    # --------------------------------------------------------

    rag_sources = result.get(
        "_rag_sources",
        []
    )

    st.markdown("### 🔗 Official Sources Used")

    if rag_sources:

        for source in rag_sources:

            if isinstance(source, dict):

                source_name = source.get(
                    "source_name",
                    "Official Source"
                )

                authority = source.get(
                    "authority",
                    ""
                )

                category = source.get(
                    "category",
                    ""
                )

                source_url = source.get(
                    "source_url",
                    ""
                )

                last_verified = source.get(
                    "last_verified",
                    ""
                )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <strong>{source_name}</strong>
                        <br>
                        {authority}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if category:
                    st.caption(
                        f"Category: {category}"
                    )

                if last_verified:
                    st.caption(
                        f"Last verified: {last_verified}"
                    )

                if source_url:
                    st.markdown(
                        f"[🔗 Open official source]({source_url})"
                    )

            else:
                st.write(f"• {source}")

    else:

        st.info(
            "No official source information was available "
            "for this assessment."
        )

    # --------------------------------------------------------
    # NEXT STEPS
    # --------------------------------------------------------

    st.markdown("### 🧭 What You Can Do Next")

    next_steps = result.get(
        "next_steps",
        []
    )

    if next_steps:

        for index, step in enumerate(next_steps, 1):

            st.markdown(
                f"""
                <div class="result-card">
                    <strong>Step {index}</strong><br>
                    {step}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.write("No next steps provided.")

    # --------------------------------------------------------
    # REQUIRED DOCUMENTS
    # --------------------------------------------------------

    st.markdown("### 📄 Documents / Information You May Need")

    required_documents = result.get(
        "required_documents",
        []
    )

    if required_documents:

        for item in required_documents:

            st.markdown(
                f"""
                <div class="result-card">
                    • {item}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.write("No specific documents identified.")

    # --------------------------------------------------------
    # INITIAL COMPLAINT
    # --------------------------------------------------------

    st.markdown("### 📝 Initial Complaint / Application")

    st.text_area(
        "AI-generated draft",
        value=result.get("complaint", ""),
        height=250,
        disabled=True
    )

    st.divider()

    # --------------------------------------------------------
    # POLISHED COMPLAINT
    # --------------------------------------------------------

    st.markdown("## ✍️ Prepare Your Complaint")

    st.caption(
        "Turn the analyzed information into a more polished "
        "complaint or application."
    )

    if st.button(
        "📝 Generate Polished Complaint",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Generating complaint..."):

            try:

                complaint = generate_complaint(
                    result.get("issue_category", ""),
                    result.get("explanation", ""),
                    important_information,
                    language
                )

                st.session_state.complaint = complaint

            except Exception as e:

                st.error(
                    "Something went wrong while generating the complaint."
                )

                st.caption(
                    f"Technical error: {str(e)}"
                )


# ============================================================
# DISPLAY POLISHED COMPLAINT
# ============================================================

if st.session_state.complaint:

    st.markdown("## 📄 Polished Complaint / Application")

    st.text_area(
        "Your complaint/application",
        value=st.session_state.complaint,
        height=400
    )

    st.download_button(
        label="⬇️ Download Complaint",
        data=st.session_state.complaint,
        file_name="awaamiagent_complaint.txt",
        mime="text/plain",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>AwaamiAgent</strong> — Helping citizens understand,
        verify, and act on civic issues.<br>
        This tool provides general civic assistance and is not a government
        authority or a substitute for professional legal advice.
    </div>
    """,
    unsafe_allow_html=True
)
