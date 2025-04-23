from PyPDF2 import PdfReader
import docx
import io

class FounderDocReaderAgent:
    def extract_text(self, uploaded_file):
        file_type = uploaded_file.name.split(".")[-1].lower()
        if file_type == "pdf":
            return self._extract_pdf(uploaded_file)
        elif file_type == "docx":
            return self._extract_docx(uploaded_file)
        elif file_type == "txt":
            return uploaded_file.read().decode("utf-8", errors="ignore")
        return ""

    def _extract_pdf(self, file):
        try:
            reader = PdfReader(file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""

    def _extract_docx(self, file):
        try:
            doc = docx.Document(io.BytesIO(file.read()))
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception:
            return ""
