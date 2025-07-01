import requests
from PyPDF2 import PdfReader, PdfWriter


class ExamGeneratorClient:
    def __init__(self, base_url: str):
        """
        Initialize the ExamGeneratorClient.

        Args:
            base_url (str): The base URL of the API (e.g., "http://localhost:8000").
        """
        self.base_url = base_url

    def convert_pdf_to_txt(self, pdf_path: str, txt_path: str) -> str:
        """
        Convert a PDF file to plain text and save it to a .txt file.

        Args:
            pdf_path (str): Path to the PDF file.
            txt_path (str): Path to save the extracted text.

        Returns:
            str: Path to the .txt file.
        """
        try:
            reader = PdfReader(pdf_path)
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        txt_file.write(text + "\n")
            print(f"Extracted text from '{pdf_path}' and saved to '{txt_path}'.")
            return txt_path
        except Exception as e:
            raise Exception(f"Failed to convert PDF to text: {e}")

    def upload_document(self, file_path: str):
        """
        Upload a document to the server.

        Args:
            file_path (str): Path to the document file to be uploaded.

        Returns:
            dict: Server response.
        """
        # If the input is a PDF, convert it to a .txt file
        if file_path.lower().endswith(".pdf"):
            txt_path = file_path.replace(".pdf", ".txt")
            file_path = self.convert_pdf_to_txt(file_path, txt_path)

        endpoint = f"{self.base_url}/documents/upload"
        with open(file_path, 'rb') as file:
            files = {"file": file}
            print(f"Uploading document: {file_path}")
            response = requests.post(endpoint, files=files)

        if response.status_code == 200:
            print(f"Document '{file_path}' uploaded successfully.")
            return response.json()
        else:
            raise Exception(
                f"Failed to upload document: {response.status_code}, {response.text}"
            )

    def search_documents(self, query: str):
        """
        Search for relevant documents based on a query.

        Args:
            query (str): The search query (can be null).

        Returns:
            list: A list of relevant documents with metadata.
        """
        endpoint = f"{self.base_url}/documents/search"
        params = {"query": query} if query else {}
        query_text = query if query else "general context (no specific topic)"
        print(f"Searching for documents with query: '{query_text}'")
        response = requests.get(endpoint, params=params)

        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data['documents'])} relevant documents.")
            return data["documents"]
        else:
            raise Exception(
                f"Failed to search documents: {response.status_code}, {response.text}"
            )

    def generate_exam(self, num_exercises: int, topic: str = None, complexity: float = 5.0):
        """
        Generate an exam based on the topic, complexity, and retrieved documents.

        Args:
            num_exercises (int): Number of exercises to include in the exam.
            topic (str, optional): The topic for the exam (can be null).
            complexity (float): The complexity level of the exam (1-10).

        Returns:
            str: The generated exam from the assistant.
        """
        # Search for documents; use a general query if topic is None
        documents = self.search_documents(query=topic)

        # Combine document content into a single context
        if documents:
            context = "\n\n".join(
                [
                    f"Document: {doc['filename']}\nContent: {doc['content'][:500]}..."
                    for doc in documents
                ]
            )
        else:
            context = "No relevant documents found. The assistant will generate questions without additional context."

        # Prepare the chat message for exam generation
        topic_text = topic if topic else "general knowledge"
        message = (
            f"Context:\n{context}\n\nUser: Please create an exam with {num_exercises} "
            f"exercises on the topic '{topic_text}' with a complexity of {complexity}/10.\nAssistant:"
        )

        # Call the chat endpoint to generate the exam
        endpoint = f"{self.base_url}/chat"
        payload = {
            "message": message,
            "model": "gpt-4",  # Specify the desired model
        }
        print(f"Sending exam generation request for topic: '{topic_text}'")
        response = requests.post(endpoint, json=payload)

        if response.status_code == 200:
            exam = response.json()["message"]
            print("Exam generated successfully.")
            return exam
        else:
            raise Exception(
                f"Failed to generate exam: {response.status_code}, {response.text}"
            )


# Usage Example
if __name__ == "__main__":
    # Set the base URL of the API
    base_url = "http://13.49.241.125:8000/api"
    client = ExamGeneratorClient(base_url)

    try:
        # Step 1: Upload a document
        document_path = "Algorithmen.pdf"
        client.upload_document(document_path)

        # Step 2: Generate an exam
        num_exercises = 5
        topic = None  # Topic can be null
        complexity = 7.5
        exam = client.generate_exam(num_exercises, topic, complexity)

        print("\nGenerated Exam:")
        print(exam)

    except Exception as e:
        print(f"An error occurred: {e}")