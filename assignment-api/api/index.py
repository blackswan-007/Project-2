import os
import tempfile
import zipfile
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM
llm = OpenAI(api_key=os.environ.get("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"), temperature=0)

@app.post("/api/")
async def process_question(
    question: str = Form(...),
    file: UploadFile = None
):
    try:
        # Process file if provided
        file_content = None
        if file:
            # Create temporary directory for file processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, file.filename)
                
                # Save uploaded file
                with open(temp_file_path, "wb") as f:
                    f.write(await file.read())
                
                # Handle zip files
                if file.filename.endswith('.zip'):
                    with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Look for CSV files in extracted content
                    for filename in os.listdir(temp_dir):
                        if filename.endswith('.csv'):
                            csv_path = os.path.join(temp_dir, filename)
                            df = pd.read_csv(csv_path)
                            file_content = df.to_string()
                else:
                    # Read file content directly
                    with open(temp_file_path, "r") as f:
                        file_content = f.read()
        
        # Create prompt template for the LLM
        prompt = PromptTemplate(
            input_variables=["question", "file_content"],
            template="""
            You are an AI assistant helping a student with their graded assignment for an IIT Madras
            Data Science course. Analyze the following question and provide a concise, accurate answer.
            
            Question: {question}
            
            File Content (if provided): {file_content}
            
            Provide only the answer that should be entered in the assignment, with no additional explanation.
            """
        )
        
        # Create and run LLMChain
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(question=question, file_content=file_content or "No file provided")
        
        # Return formatted response
        return {"answer": response.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Handler for serverless deployment
from mangum import Mangum
handler = Mangum(app)
