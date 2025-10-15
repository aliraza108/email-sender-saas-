from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from agents import Agent, Runner, AsyncOpenAI, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from dataclasses import dataclass

# Disable tracing for production
set_tracing_disabled(True)
a ='AIzaSyBtQNsTGXdNsfJ'
# --- Configuration ---
API_KEY = a + "iYMWzemhi9nJFzuulydg"
MODEL = "gemini-2.5-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Setup Gemini API client
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client)
set_default_openai_api("chat_completions")

@dataclass
class output:
    subject: str
    body: str
    general_answer:str

# --- FastAPI App ---
app = FastAPI(title="Outreach Assistant API")

# Allow CORS (for frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define the Outreach Agent ---
agent = Agent(
    name="Outreach Assistant",
    instructions=(
        "You are 'Outreach Assistant' — an AI agent designed to write personalized, human-like outreach emails.\n\n"

        "### 🎯 Main Goal:\n"
        "Generate a short, conversational outreach email **subject** and **body** based on:\n"
        "- The project or service type (e.g., Shopify store, AI chatbot, automation, etc.)\n"
        "- The customer's post, description, or message found on social media, Upwork, etc.\n\n"

        "### 🧠 Your Process:\n"
        "1. Analyze the context and understand what the customer is looking for.\n"
        "2. Create a catchy, relevant, and personalized subject line.\n"
        "3. Write a short, friendly, and human-sounding email body that:\n"
        "   - Feels natural and conversational (not robotic or corporate)\n"
        "   - Matches the customer's tone and needs\n"
        "   - Clearly shows how the service or offer can help them\n"
        "   - Ends with a polite, friendly, and non-pushy call to action\n\n"

        "### 🧾 Writing Style Guidelines:\n"
        "- Keep it concise, warm, and genuine.\n"
        "- Use first-person tone ('I', 'we') to sound authentic.\n"
        "- Avoid buzzwords, jargon, or aggressive selling.\n"
        "- Always personalize based on the customer's post or message.\n"
        "- Each output should feel unique — not like a copy-paste template.\n\n"

        "### 💡 Output Format:\n"
        "Respond strictly in this JSON format:\n"
        "{\n"
        "  'subject': '<email subject>',\n"
        "  'body': '<email body>'\n"
        "}\n"
    ),
    model=MODEL,
    output_type=output
)


# --- FastAPI Chat Endpoint ---
@app.post("/generate-email")
async def generate_email(request: Request):
    """
    Input JSON example:
    {
        "project_type": "Shopify automation",
        "customer_message": "I’m looking for someone to help me automate my order tracking.",
        "email": "client@email.com"
    }
    """
    data = await request.json()
    project_type = data.get("project_type", "")
    customer_message = data.get("customer_message", "")
    email = data.get("email", "")

    prompt = (
        f"Project type: {project_type}\n"
        f"Customer message: {customer_message}\n"
        f"Target email: {email}\n\n"
        "Generate a personalized outreach subject and email body as per your instructions."
    )

    result = await Runner.run(agent, input=prompt)
    return {"email": result.final_output}


@app.get("/")
def home():
    return {"message": "Outreach Assistant API is running 🚀"}
