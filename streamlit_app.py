import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import streamlit as st
from groq import Groq
from io import BytesIO
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file.

    :param pdf_file: A file-like object containing the PDF.
    :return: A string containing the extracted text.
    """
    document = fitz.open(stream=pdf_file, filetype="pdf")
    text = ""

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()

    return text

def preprocess_text(text):
    """
    Preprocess the text by tokenizing and removing stopwords.

    :param text: The text to preprocess.
    :return: A list of sentences with stopwords removed.
    """
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    processed_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
        processed_sentences.append(' '.join(filtered_words))

    return processed_sentences

def extract_key_points(sentences):
    """
    Extracts key points such as education, experience, skills, certifications, and accomplishments from the sentences.

    param sentences: The list of preprocessed sentences.
    return: A dictionary containing key points.
    """
    key_points = {
        "Education": [],
        "Experience": [],
        "Skills": [],
        "Certifications": [],
        "Accomplishments": [],
        "Projects": [],
        'technical_skills': [],
        'project_management': []
    }

    education_keywords = ["university", "college", "degree", "bachelor", "master", "phd", "course", "education"]
    experience_keywords = ["experience", "worked", "job", "position", "role", "company", "responsible", "managed"]
    skills_keywords = ["skills", "proficient", "knowledge", "expertise", "tools", "technologies"]
    certifications_keywords = ["certified", "certification", "certificate", "accreditation"]
    accomplishments_keywords = ["accomplished", "achieved", "project", "successfully", "led", "improved", "award"]
    project_keywords = ["Project", "Skills"]
    technical_skills_keywords = [
        "Python", "Java", "JavaScript", "C++", "SQL", "HTML", "CSS", "Machine Learning", "Artificial Intelligence",
        "Data Analysis", "Cloud Computing", "AWS", "Azure", "Google Cloud", "DevOps", "Docker", "Kubernetes", "Git",
        "Agile", "Scrum"
    ]
    project_management_keywords = [
        "Project Planning", "Project Execution", "Risk Management", "Stakeholder Management", "Budgeting",
        "Scheduling", "Resource Allocation", "Team Leadership", "Scrum Master", "Agile Methodologies",
        "Waterfall Methodologies", "Kanban", "JIRA", "Trello", "Asana"
    ]

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(word in sentence_lower for word in education_keywords):
            key_points["Education"].append(sentence)
        elif any(word in sentence_lower for word in experience_keywords):
            key_points["Experience"].append(sentence)
        elif any(word in sentence_lower for word in skills_keywords):
            key_points["Skills"].append(sentence)
        elif any(word in sentence_lower for word in certifications_keywords):
            key_points["Certifications"].append(sentence)
        elif any(word in sentence_lower for word in accomplishments_keywords):
            key_points["Accomplishments"].append(sentence)
        elif any(word in sentence_lower for word in project_keywords):
            key_points["Projects"].append(sentence)
        elif any(word in sentence_lower for word in technical_skills_keywords):
            key_points["technical_skills"].append(sentence)
        elif any(word in sentence_lower for word in project_management_keywords):
            key_points["project_management"].append(sentence)

    return key_points

def summarize_key_points(key_points):
    """
    Summarizes the key points into a list of five points for easy review.

    :param key_points: A dictionary containing key points.
    :return: A list of summary points.
    """
    summary = []

    for category, points in key_points.items():
        if points:
            summary.append(f"{category}: {points[0]}")
        if len(summary) >= 8:
            break

    # Ensure the summary has exactly 5 elements
    if len(summary) < 8:
        summary += ["(Additional point not available)"] * (8 - len(summary))

    return summary

def ats_check(text, keywords):
    """
    Check the text against a list of keywords and calculate the ATS score.

    :param text: The text to check.
    :param keywords: A list of keywords to match.
    :return: The ATS score.
    """
    matches = []
    for keyword in keywords:
        if keyword.lower() in text.lower():
            matches.append(keyword)
    score = len(matches) / len(keywords) * 100  # Calculate score as a percentage
    return score, matches

def generate_content(prompt, context):
    """
    Generate content using the Groq API.

    :param prompt: The prompt to generate content.
    :param context: The context to provide to the model.
    :return: The generated content.
    """
    client = Groq(api_key=("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"))
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": context
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

def create_docx(text):
    """
    Create a DOCX file with the extracted text right-aligned.

    :param text: The text to include in the DOCX file.
    :return: A BytesIO object containing the DOCX file.
    """
    doc = Document()
    paragraph = doc.add_paragraph(text)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT

    doc_io = BytesIO()
    doc.save(doc_io)
    doc_io.seek(0)
    return doc_io

def main():
    st.title("Resume Summary and Chatbot Comparison")

    # Upload PDF
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(BytesIO(uploaded_file.read()))

        # Initialize summary
        summary = []

        # Buttons for different actions
        if st.button("Full Text Extraction"):
            st.write("Full Text:")
            st.write(pdf_text)

        if st.button("Summary Generation"):
            processed_sentences = preprocess_text(pdf_text)
            key_points = extract_key_points(processed_sentences)
            summary = summarize_key_points(key_points)

            st.write("Summary:")
            for i, point in enumerate(summary, 1):
                st.write(f"{i}. {point}")

        # ATS Check
        if st.button("ATS Check"):
            keywords = ['Python', 'Data Analysis', 'Machine Learning']  # Example keywords
            ats_score, matched_keywords = ats_check(pdf_text, keywords)
            st.write(f"ATS Score: {ats_score}%")
            st.write(f"Matched Keywords: {', '.join(matched_keywords)}")

        # Download DOCX
        if st.button("Download DOCX"):
            docx_file = create_docx(pdf_text)
            st.download_button(
                label="Download Extracted Text",
                data=docx_file,
                file_name="extracted_text.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

        # User input for chatbot
        st.subheader("Chatbot Comparison")
        user_input = st.text_input("Ask you:")
        submit_button = st.button("Send")

        if submit_button and user_input:
            context = f"Summary: {', '.join(summary)}\nFull Text: {pdf_text[:500]}..." if summary else f"Full Text: {pdf_text[:500]}..."
            chatbot_response = generate_content(user_input, context)

            st.write("User: ", user_input)
            st.write("Chatbot: ", chatbot_response)

if __name__ == "__main__":
    main()
