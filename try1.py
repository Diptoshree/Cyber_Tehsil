####zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
import os
import re
import json
import logging
import base64
import openai
import streamlit as st
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError

# ----------- PAGE CONFIG FIRST -----------
st.set_page_config(page_title="Objection Classifier", layout="centered")

# ----------- ADD BACKGROUND IMAGE -----------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("a5.jpg")  # Replace with your image filename

# ----------- SSL CERTIFICATE BYPASS -----------
import httpx
from openai._client import OpenAI as RawOpenAI

class UnsafeOpenAI(OpenAI):
    def __init__(self, api_key):
        super().__init__(
            api_key=api_key,
            http_client=httpx.Client(verify=False)  # Disable SSL cert verification
        )

# ----------- ENV & CLIENT SETUP -----------
load_dotenv()
#api_key = os.getenv("OPENAI_API_KEY")
api_key=st.secrets["secret_section"]["openai_api_key"]
client = UnsafeOpenAI(api_key=api_key)

# ----------- LOGGER SETUP -----------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ----------- UTILITIES -----------
def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\u200b|\u200c', '', text)
    return text

def count_words(text): return len(text.split())

def max_word_length(text):
    words = text.split()
    return max(len(word) for word in words) if words else 0

def reason(Objection_new):
    Objection = normalize_text(Objection_new)
    word_count = count_words(Objection_new)
    max_word_len = max_word_length(Objection_new)

    if re.fullmatch(r'[^a-zA-Z0-9\s]+', Objection_new): return 'All Special Characters'
    if re.fullmatch(r'[^a-zA-Z\s]+', Objection_new): return 'Number and Special Characters'
    if len(Objection) > 15 and re.search(r'[a-zA-Zअ-हऒ-௽०-९\s]', Objection_new): return 'More than 15 characters'
    if len(Objection) < 5 and re.search(r'[a-zA-Zअ-हऒ-௽०-९\s]', Objection_new): return 'Small word'
    if word_count == 1 and max_word_len > 15: return 'Long word'
    return 'No Objection'

def categorize_statement_openai(objection_text):
    messages = [
        {
            "role": "system",
            "content": f"""
You are a helpful AI assistant specializing in categorizing statements (referred to as the objection text) that may indicate objections related to: 
1. Sale of land property
2. Map modification (Naksha Tarmeem)
3. Name change (Namantaran)
4. Government land-related matters
5. Court related matter
6. Financial Dispute

The statements may be in English, Hindi, or Hinglish (Hindi written using English phonetics).

Your task is to determine whether a statement indicates an objection and classify it into one of the two categories:

Categories:
Valid Objection: If the statement contains an objection in the above 6 categories .
No Objection: If the statement does not indicate any objection in the above 6 categories .

Classification Process (Sequential Steps):
Step 1: Keyword Detection (Primary Step)
Check if the statement contains any of the following keywords (including possible spelling mistakes or incomplete words):
['आपत्ति', "आपत्ति", "आपत्ति ", 'आपत्‍ती', "apatati", "aapatti", "aapaati", "appati", "apatti", "apti",
"रोक", "बंजर", "लौलाश", "पैतृक", "अन्‍य", "पैतरक", "भुमि", "जमीन", "विवाद", "फर्जी", "नामांतरण",
"बाबद्", "विवाद", "अवरोध", "विवादित", "आवश्यक", "सरकारी", "शासकीय", "धोखेबाजी", "कब्‍जा", "जमीन",
"naksha", "सिविल", "पूर्वजों", "mere", "mera", "case", "civil", "court", "bhumi", "land", "meri", "bandhak",
"attached", "namantran", "illegal", "legal", "अवैध", "galat", "registry", "registration", "rasta",
"nisast", "निरस्त", "avedan", "appeal", "धोखा-दाड़ी", "धोखा", "fraud", "नक्‍शा", "तरमीम", "वसीयत"]

Case 1: If a keyword is found, strictly classify it as "Valid Objection" and do not analyze intent further.
Case 2 (Exception): If the statement implies "no objection" (e.g., "आपत्ति नहीं", "no objection"), classify it as "No Objection".

Stop processing after Step 1 if a keyword is detected.

Step 2: If no keyword is found:
Case 3: If the statement is not meaningful and contains no keyword, classify as "No Objection".
Case 4: If the statement is meaningful but does not contain any keyword, analyze the intent:
      If it expresses concerns about land disputes, legal issues, ownership conflicts, or government restrictions, classify as "Valid Objection".
      Otherwise, classify as "No Objection".

Return the result in the following JSON format:
{{
    "Objection Statement": "{objection_text}",
    "Explanation": "Provide a clear and concise explanation for your classification. Your explanation should justify why the statement falls into the chosen category.",
    "Classification": "Classify the statement into one of the two categories: 'Valid Objection' or 'No Objection'."
}}
**Important: Strictly do not provide any additional information or explanation outside the defined format.**
                """
        },
        {
            "role": "user",
            "content": objection_text
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        result = response.choices[0].message.content.strip()
        data = json.loads(result)
        return data.get("Objection Statement", ""), data.get("Explanation", ""), data.get("Classification", "")
    except Exception as e:
        logger.error(f"OpenAI error: {str(e)}")
        return objection_text, f"Error: {str(e)}", "Classification Failed"

def classify_objection(objection):
    objection_new = normalize_text(objection)
    no_objection_phrases = ['no objection', 'आपत्ति नहीं', 'dont have objection', 'dont have any objection', 'apatti nehi', 'आपत्ति नहि','apatti nahi','apatti nhi','आपत्ति नही']
    trigger_words = [ 'आपत्ति', "आपत्ति ", 'आपत्‍ती', "apatati", "aapatti", "aapaati", "appati", "apatti", "apti",
        "रोक", "बंजर", "लौलाश", "पैतृक", "अन्‍य", "पैतरक", "भुमि","भूमि", "जमीन", "विवाद", "फर्जी", "नामांतरण","अपराध","आराजी",
        "सहमति नहीं", "बाबद्", "अवरोध", "विवादित", "आवश्यक", "सरकारी", "शासकीय", "धोखेबाजी", "कब्‍जा","शासन",
        "आराजियों","अपील", "naksha", "सिविल", "पूर्वजों", "mere", "mera", "case", "civil", "court", "bhumi", 
        "land", "meri", "bandhak", "attached", "namantran", "illegal", "legal", "अवैध", "galat", "registry", 
        "registration", "rasta", "nisast", "निरस्त", "avedan", "appeal", "धोखा-दाड़ी", "धोखा", "fraud", 
        "नक्‍शा", "तरमीम", "वसीयत"]

    if any(phrase in objection_new for phrase in no_objection_phrases):
        return "Statement implies no objection.", "No Objection"
    elif any(word in objection_new for word in trigger_words):
        return "Contains objection keyword indicating a valid objection.", "Valid Objection"
    elif reason(objection_new) in ['All Special Characters', 'Number and Special Characters', 'Small word']: 
        return reason(objection_new), "No Objection"
    else:
        _, explanation, classification = categorize_statement_openai(objection)
        return explanation or "No valid objection found.", classification or "No Objection"

# ----------- UI NAVIGATION -----------
st.title("Cyber Tehsil Objection Classification")
option = st.radio("Choose processing mode:", ["Single Objection Processing", "Bulk Objection Processing"])

# ----------- SINGLE OBJECTION -----------
if option == "Single Objection Processing":
    objection_input = st.text_area(" Objection Text", placeholder="Enter your objection here...", height=120)
    if st.button("Classify"):
        if objection_input.strip() == "":
            st.warning(" Please enter an objection text before classification.")
        else:
            logger.info(f"Input Received: {objection_input}")
            explanation, classification = classify_objection(objection_input)
            logger.info(f"Classification Result: {classification} | Explanation: {explanation}")

            st.success("✅ Classification Complete")
            st.markdown("### 🧾 Result")
            st.markdown(f"**Objection Text:** {objection_input}")
            st.markdown(f"**Explanation:** {explanation}")
            st.markdown(f"**Classification:** `{classification}`")

# ----------- BULK OBJECTION -----------
elif option == "Bulk Objection Processing":
    uploaded_file = st.file_uploader(" Upload Excel file with 'Objection' column", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            if "Objection" not in df.columns:
                st.error("❌ Excel must contain a column named 'Objection'.")
            else:
                st.write("### Preview of Uploaded Data:")
                st.dataframe(df.head())

                if st.button("Classify All"):
                    results = []
                    for obj in df["Objection"]:
                        explanation, classification = classify_objection(obj)
                        results.append({
                            "Objection": obj,
                            "Explanation": explanation,
                            "Classification": classification
                        })

                    result_df = pd.DataFrame(results)
                    st.success("✅ All rows classified successfully!")
                    st.dataframe(result_df)

                    # Download as Excel
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        result_df.to_excel(writer, index=False, sheet_name='Results')
                    output.seek(0)

                    st.download_button(
                        label="📥 Download Results as Excel",
                        data=output,
                        file_name="objection_classification_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ----------- SIDEBAR HIGHLIGHTS -----------
with st.sidebar:
    st.markdown("""
    <style>
    /* Reduce sidebar width */
    section[data-testid="stSidebar"] {
        width: 300px !important;  /* Adjust width here (e.g., 220px, 250px) */
        min-width: 300px !important;
    }
    /* Make sidebar background transparent and remove padding */
    [data-testid="stSidebar"] {
        background-color: transparent !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: transparent !important;
        padding: 1rem !important;
    }
    .paragraphs {
        font-size: 14px;
        line-height: 1.5;
        color: black;
    }

    .paragraphs p {
        margin-bottom: 0.8rem;
    }
    </style>

    <div class="paragraphs">
        <p><strong>Background of Cyber Tehsil:</strong></p>
        <p>During any land property transaction, Cyber Tehsil asks villagers to submit objections, if any.</p>
        <p>Cyber Tehsildar evaluates the objections and objected cases mandatorily go to territorial court.</p>
        <p>This AI utility will classify objections into ‘Valid Objection’ and ‘No Objection’.</p>
        <p>The classification shall save court’s time in processing invalid objections.</p>
    </div>

    """, unsafe_allow_html=True)



