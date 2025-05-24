import os
from load_dotenv import load_dotenv
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Load environment variables from a .env file
load_dotenv()
# Set the Google API key from environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize the Google Generative AI language model with specified parameters
llm_google = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    temperature=0,
)

# Define a Pydantic model for structured product information
class Productinformation(BaseModel):
    product_name: str = Field(description="Name of the product")
    product_price: str = Field(description="Price of the product")
    product_description: str = Field(description="Description of the product")

# Create a JSON output parser that uses the Productinformation model for validation
output_parser = JsonOutputParser(pydantic_object=Productinformation)

# Define a chat prompt template with system and human messages
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert product information specialist. Your task is to provide accurate, current product information in JSON format only.

    IMPORTANT: Return ONLY valid JSON format. No additional text, explanations, or markdown formatting.

    {format_instructions}"""),
    ("human", "{input}")
])

# Compose the chain: prompt -> language model -> output parser
chain = prompt | llm_google | output_parser

# Invoke the chain with the input question and format instructions, get the response
response = chain.invoke({
    "input": "Tell me about the product 'iPhone 14 Pro Max'",
    "format_instructions": output_parser.get_format_instructions()
})

# Print the parsed response as pretty-formatted JSON
print(json.dumps(response, indent=4, ensure_ascii=False))
