import os
import json
import base64
from fastapi import FastAPI, Request, Response, Query
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request as GoogleRequest
from supabase import create_client
from agents import Agent, Runner, AsyncOpenAI, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from dataclasses import dataclass
from dotenv import load_dotenv

# --- Load .env ---
load_dotenv()

# --- Disable tracing ---
set_tracing_disabled(True)

# --- Allowed CORS origins ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://email-sender-saas.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Supabase setup ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# --- Google OAuth Config ---
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GOOGLE_OAUTH_REDIRECT = os.environ.get("GOOGLE_OAUTH_REDIRECT")  # e.g. https://api.yourapp.com/auth/google/callback
FRONTEND_AFTER_CONNECT = os.environ.get(
    "FRONTEND_AFTER_CONNECT", "https://email-sender-saas.vercel.app/dashboard"
)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
]

# --- OpenAI / Gemini setup ---
API_KEY = "AIzaSyBtQNsTGXdNsfJiYMWzemhi9nJFzuulydg"
MODEL = "gemini-2.5-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client)
set_default_openai_api("chat_completions")


# --- Agent setup ---
@dataclass
class output:
    subject: str
    body: str
    general_answer: str


agent = Agent(
    name="Outreach Assistant",
    instructions=(
        "You are 'Outreach Assistant' — an AI agent designed to write personalized, human-like outreach emails.\n"
        "### 💡 Output Format:\n"
        "{\n"
        "  'subject': '<email subject>',\n"
        "  'body': '<email body>'\n"
        "}\n"
    ),
    model=MODEL,
    output_type=output,
)


# --- Helper: Google Flow ---
def make_flow(state: str = None):
    return Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uris": [GOOGLE_OAUTH_REDIRECT],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=SCOPES,
        redirect_uri=GOOGLE_OAUTH_REDIRECT,
        state=state,
    )


# --- OAuth: Start ---
@app.get("/auth/google/start")
def auth_google_start(user_id: str = Query(...)):
    """Redirects the user to Google OAuth consent page."""
    state_payload = json.dumps({"user_id": user_id})
    flow = make_flow(state=state_payload)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return RedirectResponse(auth_url)


# --- OAuth: Callback ---
@app.get("/auth/google/callback")
def auth_google_callback(code: str = Query(None), state: str = Query(None)):
    if not code:
        return JSONResponse({"error": "Missing OAuth code"}, status_code=400)

    # Extract state (user_id)
    user_id = None
    try:
        if state:
            state_json = json.loads(state)
            user_id = state_json.get("user_id")
    except Exception:
        user_id = None

    try:
        flow = make_flow(state=state)
        flow.fetch_token(code=code)
        creds = flow.credentials
    except Exception as e:
        return JSONResponse({"error": "token_fetch_failed", "detail": str(e)}, status_code=500)

    # Fetch user info
    try:
        oauth2_service = build("oauth2", "v2", credentials=creds)
        profile = oauth2_service.userinfo().get().execute()
        user_email = profile.get("email")
    except Exception:
        user_email = None

    token_data = {
        "access_token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
        "email": user_email,
    }

    if not user_id:
        return RedirectResponse(f"{FRONTEND_AFTER_CONNECT}?connected=0&error=missing_user")

    try:
        supabase.table("email_configs").upsert(
            {"user_id": user_id, "provider": "google", "data": token_data}
        ).execute()
    except Exception as e:
        return JSONResponse({"error": "supabase_upsert_failed", "detail": str(e)}, status_code=500)

    return RedirectResponse(f"{FRONTEND_AFTER_CONNECT}?connected=1")


# --- Generate Email ---
@app.post("/generate-email")
async def generate_email(request: Request):
    data = await request.json()
    project_type = data.get("project_type", "")
    customer_message = data.get("customer_message", "")
    email = data.get("email", "")

    prompt = (
        f"Project type: {project_type}\n"
        f"Customer message: {customer_message}\n"
        f"Target email: {email}\n\n"
        "Generate a personalized outreach subject and email body."
    )

    result = await Runner.run(agent, input=prompt)
    final = getattr(result, "final_output", None) or result
    out = final.__dict__ if hasattr(final, "__dict__") else final

    return {"email": out}


# --- Send Email ---
@app.post("/send-email")
async def send_email(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    to_email = data.get("to")
    subject = data.get("subject")
    body = data.get("body")

    if not user_id:
        return JSONResponse({"error": "user_id missing"}, status_code=400)

    row = supabase.table("email_configs").select("*").eq("user_id", user_id).single().execute()
    if not row.data:
        return JSONResponse({"error": "no_email_config"}, status_code=400)

    token_info = row.data.get("data", {})
    refresh_token = token_info.get("refresh_token")

    if not refresh_token:
        return JSONResponse({"error": "no_refresh_token"}, status_code=400)

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=SCOPES,
    )

    try:
        creds.refresh(GoogleRequest())
    except Exception as e:
        return JSONResponse({"error": "refresh_failed", "detail": str(e)}, status_code=500)

    try:
        service = build("gmail", "v1", credentials=creds)
        from email.mime.text import MIMEText

        message = MIMEText(body)
        message["to"] = to_email
        message["subject"] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        send_res = service.users().messages().send(userId="me", body={"raw": raw}).execute()

        # Update token in Supabase
        updated_data = {**token_info, "access_token": creds.token}
        supabase.table("email_configs").update(
            {"data": updated_data}
        ).eq("user_id", user_id).execute()

        return {"status": "sent", "result": send_res}
    except Exception as e:
        return JSONResponse({"error": "send_failed", "detail": str(e)}, status_code=500)
