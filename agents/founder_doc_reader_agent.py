import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import docx
import io

class FounderDocReaderAgent:
    def extract_text(self, uploaded_file):
        file_type = uploaded_file.name.split(".")[-1].lower()
        if file_type == "pdf":
            return self._extract_pdf_with_fitz(uploaded_file)
        elif file_type == "docx":
            return self._extract_docx(uploaded_file)
        elif file_type == "txt":
            return uploaded_file.read().decode("utf-8", errors="ignore")
        return ""

    def _extract_pdf_with_fitz(self, file):
        try:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "\n".join(page.get_text("text") for page in doc)
            text = text.strip()
            if len(text) < 100:
                print("⚠️ Very little text extracted with fitz.")
            return text
        except Exception as e:
            print(f"❌ PyMuPDF extraction failed: {e}")
            return self._fallback_extract_pdf_with_pypdf2(file)

    def _fallback_extract_pdf_with_pypdf2(self, file):
        try:
            file.seek(0)  # reset file pointer
            reader = PdfReader(file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text.strip()
        except Exception as e:
            print(f"❌ PyPDF2 fallback also failed: {e}")
            return ""

def _extract_docx(self, file):
    try:
        doc = docx.Document(io.BytesIO(file.read()))
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()]).strip()
    except Exception as e:
        print(f"❌ DOCX extraction failed: {e}")
        return ""


