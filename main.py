# main.py
import os
import json
import base64
from fastapi import FastAPI, Request, Response, Query
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from supabase import create_client
from agents import Agent, Runner, AsyncOpenAI, set_default_openai_api, set_default_openai_client, set_tracing_disabled
from dataclasses import dataclass
from dotenv import load_dotenv

origins = os.getenv("CORS_ORIGINS", "").split(",")
CORS_ORIGINS="https://email-sender-saas.vercel.app","http://localhost:3000"


load_dotenv()

set_tracing_disabled(True)

# --- OpenAI / Gemini config (keep your existing code)
a = 'AIzaSyBtQNsTGXdNsfJ'
API_KEY = a + "iYMWzemhi9nJFzuulydg"
MODEL = "gemini-2.5-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client)
set_default_openai_api("chat_completions")

@dataclass
class output:
    subject: str
    body: str
    general_answer: str

agent = Agent(
    name="Outreach Assistant",
    instructions=(
        "You are 'Outreach Assistant' — an AI agent designed to write personalized, human-like outreach emails.\n\n"
        # [keep your instruction string...]
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

# --- FastAPI app setup ---
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


# --- Supabase client (server, service role) ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# --- Google OAuth2 config ---
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GOOGLE_OAUTH_REDIRECT = os.environ.get("GOOGLE_OAUTH_REDIRECT")  # e.g., https://api.yourdomain.com/auth/google/callback
# front-end redirect target after successful connect (update this)
FRONTEND_AFTER_CONNECT = os.environ.get("FRONTEND_AFTER_CONNECT", "http://localhost:3000/dashboard")

SCOPES = ["https://www.googleapis.com/auth/gmail.send", "https://mail.google.com/"]

# --- Helper: build Flow ---
def make_flow(state: str = None):
    return Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uris": [GOOGLE_OAUTH_REDIRECT],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=SCOPES,
        redirect_uri=GOOGLE_OAUTH_REDIRECT,
        state=state
    )

# --- OAuth start endpoint ---
@app.get("/auth/google/start")
def auth_google_start(user_id: str = Query(...)):
    """
    Frontend should open /auth/google/start?user_id=<auth.uid()>
    We'll store user_id in state (url-safe json) and redirect to Google.
    """
    # Keep minimal state: include user_id
    state_payload = json.dumps({"user_id": user_id})
    flow = make_flow(state=state_payload)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )
    return RedirectResponse(auth_url)

# --- OAuth callback ---
@app.get("/auth/google/callback")
def auth_google_callback(code: str = Query(None), state: str = Query(None)):
    """
    Exchange code for tokens and store in Supabase.email_configs
    """
    if not code:
        return JSONResponse({"error": "No code provided"}, status_code=400)

    # restore state
    user_id = None
    try:
        if state:
            st = json.loads(state)
            user_id = st.get("user_id")
    except Exception:
        user_id = None

    flow = make_flow(state=state)
    flow.fetch_token(code=code)

    creds = flow.credentials  # google.oauth2.credentials.Credentials
    # get email address to store 'email' in data (optional)
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
        "email": user_email
    }

    # upsert into Supabase.email_configs
    if not user_id:
        # if we don't have a user id, cannot associate — return success but warn
        # Redirect to frontend with error
        return RedirectResponse(f"{FRONTEND_AFTER_CONNECT}?connected=0")

    # upsert
    res = supabase.table("email_configs").upsert({
        "user_id": user_id,
        "provider": "google",
        "data": token_data
    }).execute()

    # redirect back to frontend (you can add query params)
    return RedirectResponse(f"{FRONTEND_AFTER_CONNECT}?connected=1")

# --- Use Runner to generate email ---
@app.post("/generate-email")
async def generate_email(request: Request):
    data = await request.json()
    project_type = data.get("project_type", "")
    customer_message = data.get("customer_message", "") or data.get("post_text", "")
    email = data.get("email", "")

    prompt = (
        f"Project type: {project_type}\n"
        f"Customer message: {customer_message}\n"
        f"Target email: {email}\n\n"
        "Generate a personalized outreach subject and email body as per your instructions."
    )

    result = await Runner.run(agent, input=prompt)
    # Ensure result.final_output is serializable and matches the dataclass
    # result.final_output could be an object; try to convert safely:
    final = getattr(result, "final_output", None) or result
    # final can be dataclass -> convert to dict
    try:
        if hasattr(final, "__dict__"):
            out = final.__dict__
        else:
            out = final
    except Exception:
        out = {"subject": "", "body": ""}

    return {"email": out}

# --- Send email using stored refresh token (server-side) ---
@app.post("/send-email")
async def send_email(request: Request):
    """
    Input: { "user_id": "<user id>", "to": "...", "subject": "...", "body": "..." }
    The server looks up stored refresh_token in email_configs for that user, refreshes creds and sends mail.
    """
    data = await request.json()
    user_id = data.get("user_id")
    to_email = data.get("to")
    subject = data.get("subject")
    body = data.get("body")

    if not user_id:
        return JSONResponse({"error": "user_id missing"}, status_code=400)

    # lookup stored tokens
    row = supabase.table("email_configs").select("*").eq("user_id", user_id).single().execute()
    if row.error or not row.data:
        return JSONResponse({"error": "no_email_config"}, status_code=400)
    token_info = row.data.get("data", {})

    # build credentials from stored refresh token + client id/secret (so we can refresh)
    refresh_token = token_info.get("refresh_token")
    if not refresh_token:
        return JSONResponse({"error": "no_refresh_token"}, status_code=400)

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=SCOPES
    )

    try:
        # refresh access token (will use refresh_token)
        creds.refresh(RequestsSessionAdapter())
    except Exception as e:
        # fallback: attempt manual token refresh
        # (The Credentials.refresh uses google.auth.transport.requests.Request; provide wrapper)
        try:
            from google.auth.transport.requests import Request as GoogleRequest
            creds.refresh(GoogleRequest())
        except Exception as e2:
            return JSONResponse({"error": "refresh_failed", "detail": str(e2)}, status_code=500)

    # Now send via Gmail API
    try:
        service = build("gmail", "v1", credentials=creds)
        from email.mime.text import MIMEText
        raw_message = MIMEText(body)
        raw_message["to"] = to_email
        raw_message["subject"] = subject
        raw = base64.urlsafe_b64encode(raw_message.as_bytes()).decode()

        send_res = service.users().messages().send(userId="me", body={"raw": raw}).execute()

        # optionally update stored access token expiry / token
        updated_data = token_info
        updated_data["access_token"] = creds.token
        updated_data["expiry"] = creds.expiry.isoformat() if creds.expiry else None
        supabase.table("email_configs").update({"data": updated_data, "updated_at": "now()"}).eq("user_id", user_id).execute()

        return {"status": "sent", "result": send_res}
    except Exception as e:
        return JSONResponse({"error": "send_failed", "detail": str(e)}, status_code=500)

# --- helper adapter for requests transport used by google oauth refresh ---
from google.auth.transport.requests import Request as GA_Request
class RequestsSessionAdapter(GA_Request):
    def __init__(self):
        super().__init__()

