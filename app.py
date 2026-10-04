
import os
import json
import re
import io
import streamlit as st

from rag.knowledge_base import ensure_knowledge_base
from rag.rag_pipeline import query_civic_rag
from openai import OpenAI
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
    page_title="AwaamiAgent",
    page_icon="🏛️",
    layout="centered"
)


# ============================================================
# PROFESSIONAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .section-label {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1rem;
    }

    .status-box {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin: 0.5rem 0 1rem 0;
        border: 1px solid #ddd;
    }

    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
    }

    div[data-testid="stFileUploader"] {
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API KEY
# ============================================================

def get_api_key():
    """Get Groq API key from Streamlit Secrets or environment."""

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
        "Add it to Streamlit Secrets or the environment."
    )
    st.stop()


client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


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

    st.caption(
        f"RAG technical error: {str(e)}"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🏛️ AwaamiAgent</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Civic Assistant")

    st.write(
        "Turn a civic problem into a clear action path:"
    )

    st.markdown(
        """
        **Problem → Understand → Verify → Act**
        """
    )

    st.divider()

    st.markdown("### 🔎 Civic Knowledge")

    st.write(
        "AwaamiAgent retrieves relevant civic information "
        "from its official-source knowledge base before analysis."
    )

    if rag_status is not None:
        st.success("Knowledge base loaded")
    else:
        st.warning("Knowledge base unavailable")

    st.divider()

    st.markdown("### 📄 Supported Input")

    st.write(
        "• Text problem\n"
        "• PDF document\n"
        "• JPG / PNG image"
    )

    st.divider()

    st.markdown("### 🌐 Response")

    st.write(
        "Choose English or Urdu from the main panel."
    )

    st.divider()

    st.caption(
        "AwaamiAgent provides general civic assistance. "
        "Always verify important information with the relevant "
        "official authority."
    )


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes):
    """Extract selectable text from a PDF."""

    reader = PdfReader(
        io.BytesIO(file_bytes)
    )

    pages = []

    for page in reader.pages:
        pages.append(
            page.extract_text() or ""
        )

    return {
        "text": "\n\n".join(pages).strip(),
        "page_count": len(reader.pages),
        "extraction_method": "pdf_text"
    }


def extract_image_text(file_bytes):
    """Extract text from an image using OCR."""

    image = Image.open(
        io.BytesIO(file_bytes)
    )

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

            result = extract_pdf_text(
                file_bytes
            )

        elif uploaded_file.type in [
            "image/png",
            "image/jpeg"
        ]:

            result = extract_image_text(
                file_bytes
            )

        else:
            raise ValueError(
                "Unsupported file type."
            )

        metadata.update(
            {
                "page_count": result["page_count"],
                "extraction_method": result[
                    "extraction_method"
                ]
            }
        )

        return {
            "metadata": metadata,
            "text": result["text"]
        }

    except Exception as e:

        raise ValueError(
            f"Could not process the uploaded document: {str(e)}"
        )


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_response(text):
    """Parse JSON even if the model returns markdown code fences."""

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE
    )

    try:

        result = json.loads(
            cleaned
        )

    except json.JSONDecodeError:

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1:

            raise ValueError(
                "The AI returned an invalid JSON response."
            )

        try:

            result = json.loads(
                cleaned[start:end + 1]
            )

        except json.JSONDecodeError:

            raise ValueError(
                "The AI returned an invalid JSON response."
            )

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
# LOCAL FALLBACK
# ============================================================

def create_analysis_fallback(
    problem,
    language,
    rag_context,
    rag_sources
):
    """
    Provide useful civic guidance when Groq is unavailable.
    """

    problem_text = problem.strip()

    if not problem_text:
        problem_text = (
            "The user uploaded a civic/government document "
            "but did not provide a separate problem description."
        )

    problem_lower = problem_text.lower()

    # --------------------------------------------------------
    # BASIC LOCAL ISSUE CLASSIFICATION
    # --------------------------------------------------------

    if any(
        word in problem_lower
        for word in [
            "electricity",
            "electric bill",
            "electricity bill",
            "wapda",
            "lesco",
            "fesco",
            "iesco",
            "k-electric"
        ]
    ):

        category_en = "Electricity / Billing Issue"
        category_ur = "بجلی / بل کا مسئلہ"

    elif any(
        word in problem_lower
        for word in [
            "gas",
            "sngpl",
            "ssgc"
        ]
    ):

        category_en = "Gas / Utility Issue"
        category_ur = "گیس / یوٹیلیٹی کا مسئلہ"

    elif any(
        word in problem_lower
        for word in [
            "water",
            "sewerage",
            "sewage",
            "drainage"
        ]
    ):

        category_en = "Water / Sewerage Issue"
        category_ur = "پانی / سیوریج کا مسئلہ"

    elif any(
        word in problem_lower
        for word in [
            "road",
            "street",
            "pothole",
            "traffic"
        ]
    ):

        category_en = "Road / Public Infrastructure Issue"
        category_ur = "سڑک / عوامی انفراسٹرکچر کا مسئلہ"

    elif any(
        word in problem_lower
        for word in [
            "document",
            "notice",
            "letter",
            "application"
        ]
    ):

        category_en = "Government Document / Application Issue"
        category_ur = "سرکاری دستاویز / درخواست کا مسئلہ"

    else:

        category_en = "Civic Service Issue"
        category_ur = "شہری مسئلہ"


    # --------------------------------------------------------
    # EXTRACT RAG INFORMATION
    # --------------------------------------------------------

    rag_points = []

    if rag_context:

        lines = [
            line.strip()
            for line in rag_context.splitlines()
            if line.strip()
        ]

        for line in lines:

            cleaned = re.sub(
                r"^\s*(?:[-•*]|\d+[.)])\s*",
                "",
                line
            ).strip()

            if len(cleaned) >= 30:
                rag_points.append(
                    cleaned
                )

            if len(rag_points) >= 5:
                break


    # --------------------------------------------------------
    # URDU FALLBACK
    # --------------------------------------------------------

    if language == "Urdu":

        explanation = (
            f"Aap ne yeh civic masla report kiya hai: "
            f"“{problem_text}”\n\n"
            f"Isay {category_ur} ke taur par identify kiya gaya hai. "
            "AwaamiAgent ne available civic knowledge base se relevant "
            "information retrieve ki hai. Neeche di gayi information ko "
            "apne case ke mutabiq check karein."
        )

        important = []

        if rag_points:

            important.extend(
                rag_points[:3]
            )

        else:

            important.extend(
                [
                    "Apne maslay se related bill, notice ya doosre records sambhal kar rakhein.",
                    "Complaint submit karne se pehle apni relevant information verify karein.",
                    "Jahan zaroori ho, relevant official authority se confirmation lein."
                ]
            )

        next_steps = [
            "Apne maslay ke relevant documents aur evidence collect karein.",
            "Retrieved official information ko apne case ke saath compare karein.",
            "Relevant official department ya authority ko complaint/report submit karein.",
            "Complaint/reference number aur submitted documents ka record rakhein.",
            "Agar official information aapke specific case ko cover nahi karti, authority se confirmation lein."
        ]

        documents = [
            "CNIC / relevant identification document",
            "Related bill, notice, application or reference number",
            "Relevant supporting documents or photographs",
            "Previous complaint/reference number, if available"
        ]

        complaint = (
            "Mohtaram Sir/Madam,\n\n"
            f"Main {category_ur} ke hawalay se "
            "apni darkhwast/complaint pesh karna chahta/chahti hoon.\n\n"
            f"Maslay ki tafseel:\n{problem_text}\n\n"
            "Barah-e-karam meri complaint ka jaiza le kar, "
            "munasib rehnumai aur zaroori action faraham kiya jaye.\n\n"
            "Shukriya.\n\n"
            "[Aap ka Naam]\n"
            "[CNIC / Reference Number]\n"
            "[Contact Information]\n"
            "[Date]"
        )

        category = category_ur


    # --------------------------------------------------------
    # ENGLISH FALLBACK
    # --------------------------------------------------------

    else:

        explanation = (
            f"You reported the following civic problem: "
            f"“{problem_text}”\n\n"
            f"AwaamiAgent identified this as a "
            f"{category_en}. Relevant information was retrieved "
            "from the civic knowledge base. Review the information "
            "below and verify case-specific details with the "
            "relevant official authority."
        )

        important = []

        if rag_points:

            important.extend(
                rag_points[:3]
            )

        else:

            important.extend(
                [
                    "Keep the bill, notice, application, or other relevant records.",
                    "Verify important case-specific information before submitting a complaint.",
                    "Contact the relevant official authority when confirmation is required."
                ]
            )

        next_steps = [
            "Collect the documents and evidence related to your problem.",
            "Compare your situation with the retrieved official civic information.",
            "Submit the complaint/report to the relevant official department or authority.",
            "Keep the complaint/reference number and copies of submitted documents.",
            "If the retrieved information does not cover your specific case, verify it with the relevant authority."
        ]

        documents = [
            "CNIC / relevant identification document",
            "Related bill, notice, application or reference number",
            "Relevant supporting documents or photographs",
            "Previous complaint/reference number, if available"
        ]

        complaint = (
            "To Whom It May Concern,\n\n"
            f"I would like to submit a complaint regarding "
            f"a {category_en}.\n\n"
            f"Details of the issue:\n{problem_text}\n\n"
            "Kindly review this matter and provide the appropriate "
            "guidance and necessary action.\n\n"
            "Sincerely,\n"
            "[Your Name]\n"
            "[CNIC / Reference Number]\n"
            "[Contact Information]\n"
            "[Date]"
        )

        category = category_en


    return {
        "issue_category": category,
        "explanation": explanation,
        "important_information": important,
        "next_steps": next_steps,
        "required_documents": documents,
        "complaint": complaint,
        "_rag_sources": rag_sources,
        "_ai_unavailable": True
    }


# ============================================================
# AI CIVIC ANALYSIS
# ============================================================

def analyze_civic_problem(
    problem,
    language,
    document_context=None
):
    """Analyze a civic problem using Civic RAG + Groq."""

    language_instruction = (
        "Respond in English."
        if language == "English"
        else
        "Respond in natural, simple Urdu. Keep important English "
        "terms in parentheses when useful."
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
    # CIVIC RAG
    # --------------------------------------------------------

    rag_context = ""
    rag_sources = []

    try:

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

        response = client.responses.create(
            model=MODEL,
            input=prompt
        )

        raw_output = response.output_text.strip()

        result = parse_json_response(
            raw_output
        )

        result["_rag_sources"] = rag_sources
        result["_ai_unavailable"] = False

        return result

    except Exception as e:

        print(
            f"Groq unavailable: {e}"
        )

        return create_analysis_fallback(
            problem,
            language,
            rag_context,
            rag_sources
        )


# ============================================================
# COMPLAINT GENERATOR
# ============================================================

def generate_complaint(
    issue_category,
    explanation,
    important_information,
    language
):
    """Generate a polished complaint without inventing facts."""

    language_instruction = (
        "Write in English."
        if language == "English"
        else
        "Write in natural, simple Urdu."
    )

    prompt = f"""
You are AwaamiAgent's complaint drafting assistant.

{language_instruction}

Create a professional complaint/application based ONLY on the
information provided below.

IMPORTANT:
- Do not invent names.
- Do not invent account numbers.
- Do not invent addresses.
- Do not invent dates.
- Do not invent laws.
- Do not invent government departments.
- Do not invent fees or deadlines.
- Use placeholders where personal information is missing.
- Do not provide legal advice.
- Keep the complaint concise and practical.

Issue:
{issue_category}

Explanation:
{explanation}

Important information:
{json.dumps(important_information, ensure_ascii=False)}

Use appropriate placeholders such as:
[Your Name]
[Your Address]
[Account/Reference Number]
[Date]

Return only the complaint/application text.
"""

    try:

        response = client.responses.create(
            model=MODEL,
            input=prompt
        )

        return response.output_text.strip()

    except Exception as e:

        print(
            f"Groq complaint generation unavailable: {e}"
        )

        if language == "Urdu":

            return (
                "Mohtaram Sir/Madam,\n\n"
                "Main apne civic maslay ke silsilay mein "
                "darkhwast pesh karna chahta/chahti hoon.\n\n"
                f"Masla: {issue_category}\n\n"
                f"Tafseel: {explanation}\n\n"
                "Barah-e-karam is matter ka jaiza le kar "
                "munasib action aur rehnumai faraham ki jaye.\n\n"
                "Shukriya.\n"
                "[Aap ka Naam]\n"
                "[Date]"
            )

        return (
            "To Whom It May Concern,\n\n"
            "I would like to submit a complaint regarding "
            "the following civic issue.\n\n"
            f"Issue: {issue_category}\n\n"
            f"Details: {explanation}\n\n"
            "Kindly review this matter and provide appropriate "
            "assistance and guidance.\n\n"
            "Sincerely,\n"
            "[Your Name]\n"
            "[Your Address]\n"
            "[Date]"
        )


# ============================================================
# SESSION STATE
# ============================================================

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "complaint" not in st.session_state:
    st.session_state.complaint = None


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏛️ AwaamiAgent</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your AI-powered civic assistance system'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
Describe a civic problem in simple words or upload a
government/civic document. AwaamiAgent uses retrieved civic
information to help you understand the issue, identify practical
next steps, and prepare a complaint or application.
"""
)

st.info(
    "AwaamiAgent provides general civic assistance. "
    "Jurisdiction-specific laws, procedures, deadlines, fees, "
    "and departments should always be verified through reliable "
    "official sources."
)


# ============================================================
# INPUT
# ============================================================

st.markdown(
    "### 📝 Describe Your Problem"
)

problem = st.text_area(
    "Civic problem",
    placeholder=(
        "Example: My electricity bill is much higher than usual "
        "and I do not understand why."
    ),
    height=160,
    label_visibility="collapsed"
)


uploaded_file = st.file_uploader(
    "📄 Upload a civic/government document (optional)",
    type=[
        "pdf",
        "png",
        "jpg",
        "jpeg"
    ],
    help=(
        "Upload a PDF, scanned document, bill, notice, "
        "or image."
    )
)


language = st.selectbox(
    "🌐 Response language",
    [
        "English",
        "Urdu"
    ]
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔎 Analyze Problem",
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

        with st.spinner(
            "AwaamiAgent is analyzing your problem..."
        ):

            try:

                document_context = None

                if uploaded_file is not None:

                    with st.spinner(
                        "Reading uploaded document..."
                    ):

                        document_context = (
                            process_uploaded_document(
                                uploaded_file
                            )
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

                st.caption(
                    f"Technical error: {str(e)}"
                )


# ============================================================
# DISPLAY ANALYSIS RESULTS
# ============================================================

if st.session_state.analysis:

    result = st.session_state.analysis

    st.divider()

    # --------------------------------------------------------
    # AI STATUS
    # --------------------------------------------------------

    if result.get("_ai_unavailable"):

        st.warning(
            "The AI analysis service is temporarily unavailable. "
            "AwaamiAgent is showing local civic guidance and "
            "retrieved official-source information instead."
        )

    else:

        st.success(
            "AI analysis completed successfully."
        )


    # --------------------------------------------------------
    # ISSUE
    # --------------------------------------------------------

    st.markdown(
        "### 📌 Issue Identified"
    )

    st.write(
        result.get(
            "issue_category",
            "Not specified"
        )
    )


    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    st.markdown(
        "### 💡 What's Happening?"
    )

    st.write(
        result.get(
            "explanation",
            "No explanation available."
        )
    )


    # --------------------------------------------------------
    # IMPORTANT INFORMATION
    # --------------------------------------------------------

    st.markdown(
        "### 🔎 Important Information"
    )

    important_information = result.get(
        "important_information",
        []
    )

    if important_information:

        for item in important_information:

            st.write(
                f"• {item}"
            )

    else:

        st.write(
            "No additional information provided."
        )


    # --------------------------------------------------------
    # NEXT STEPS
    # --------------------------------------------------------

    st.markdown(
        "### 🧭 What Should I Do?"
    )

    next_steps = result.get(
        "next_steps",
        []
    )

    if next_steps:

        for index, step in enumerate(
            next_steps,
            1
        ):

            st.write(
                f"**{index}.** {step}"
            )

    else:

        st.write(
            "No next steps provided."
        )


    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------

    st.markdown(
        "### 📄 Documents / Information You May Need"
    )

    required_documents = result.get(
        "required_documents",
        []
    )

    if required_documents:

        for item in required_documents:

            st.write(
                f"• {item}"
            )

    else:

        st.write(
            "No specific documents identified."
        )


    # --------------------------------------------------------
    # RAG SOURCES
    # --------------------------------------------------------

    rag_sources = result.get(
        "_rag_sources",
        []
    )

    st.markdown(
        "### 🔗 Official Civic Sources"
    )

    if rag_sources:

        for source in rag_sources:

            if isinstance(
                source,
                dict
            ):

                source_name = (
                    source.get("title")
                    or source.get("source")
                    or source.get("name")
                    or "Official source"
                )

                source_url = source.get(
                    "url"
                )

                if source_url:

                    st.markdown(
                        f"- [{source_name}]({source_url})"
                    )

                else:

                    st.write(
                        f"• {source_name}"
                    )

            else:

                st.write(
                    f"• {source}"
                )

    else:

        st.caption(
            "No matching official civic sources were returned."
        )


    # --------------------------------------------------------
    # INITIAL COMPLAINT
    # --------------------------------------------------------

    st.markdown(
        "### 📝 Initial Complaint / Application"
    )

    st.text_area(
        "Initial draft",
        value=result.get(
            "complaint",
            ""
        ),
        height=250,
        disabled=True
    )


    # --------------------------------------------------------
    # POLISHED COMPLAINT
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### ✍️ Generate a Polished Complaint"
    )

    if st.button(
        "Generate Complaint",
        use_container_width=True
    ):

        with st.spinner(
            "Preparing your complaint..."
        ):

            try:

                complaint = generate_complaint(
                    result.get(
                        "issue_category",
                        ""
                    ),
                    result.get(
                        "explanation",
                        ""
                    ),
                    important_information,
                    language
                )

                st.session_state.complaint = complaint

                st.rerun()

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

    st.divider()

    st.markdown(
        "### 📄 Polished Complaint / Application"
    )

    st.text_area(
        "Your draft",
        value=st.session_state.complaint,
        height=400
    )

    st.download_button(
        label="⬇️ Download Complaint as TXT",
        data=st.session_state.complaint,
        file_name="awaamiagent_complaint.txt",
        mime="text/plain",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AwaamiAgent is an AI civic assistance tool and is not a "
    "government authority or a substitute for professional "
    "legal advice."
)
