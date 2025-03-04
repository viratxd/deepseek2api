# Standard Library Imports
import base64
import ctypes
import json
import logging
import queue
import random
import re
import struct
import threading
import time

# Third-Party Imports
import transformers
from curl_cffi import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from wasmtime import Linker, Module, Store

# -------------------------- Initialize Tokenizer --------------------------
# Load the tokenizer for processing prompts and responses
chat_tokenizer_dir = "./"
tokenizer = transformers.AutoTokenizer.from_pretrained(
    chat_tokenizer_dir, trust_remote_code=True
)

# -------------------------- Logging Configuration --------------------------
# Set up logging for the application
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# -------------------------- FastAPI Application Initialization --------------------------
# Create the FastAPI app instance
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],  # Allowed HTTP methods
    allow_headers=["Content-Type", "Authorization"],  # Allowed headers
)

# -------------------------- Template Configuration --------------------------
# Set up Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")

# -------------------------- Global Constants --------------------------
# Path to the configuration file
CONFIG_PATH = "config.json"

# Path to the WASM module for PoW calculations
WASM_PATH = "sha3_wasm_bg.7b9ca65ddd.wasm"

# Keep-alive timeout for streaming responses
KEEP_ALIVE_TIMEOUT = 5

# DeepSeek API constants
DEEPSEEK_HOST = "chat.deepseek.com"
DEEPSEEK_LOGIN_URL = f"https://{DEEPSEEK_HOST}/api/v0/users/login"
DEEPSEEK_CREATE_SESSION_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat_session/create"
DEEPSEEK_CREATE_POW_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat/create_pow_challenge"
DEEPSEEK_COMPLETION_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat/completion"

# Base headers for DeepSeek API requests
BASE_HEADERS = {
    "Host": "chat.deepseek.com",
    "User-Agent": "DeepSeek/1.0.13 Android/35",
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
    "Content-Type": "application/json",
    "x-client-platform": "android",
    "x-client-version": "1.0.13",
    "x-client-locale": "zh_CN",
    "accept-charset": "UTF-8",
}
# -------------------------- Configuration Management --------------------------
def load_config():
    """
    Load the configuration from the config.json file.
    If the file is missing or invalid, return an empty dictionary and log a warning.
    """
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
            logger.info(f"[load_config] Successfully loaded configuration from {CONFIG_PATH}")
            return config
    except FileNotFoundError:
        logger.warning(f"[load_config] Configuration file not found at {CONFIG_PATH}. Using empty configuration.")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"[load_config] Invalid JSON in configuration file: {e}. Using empty configuration.")
        return {}
    except Exception as e:
        logger.error(f"[load_config] Unexpected error loading configuration: {e}. Using empty configuration.")
        return {}


def save_config(config):
    """
    Save the configuration to the config.json file.
    Logs an error if the file cannot be written.
    """
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            logger.info(f"[save_config] Successfully saved configuration to {CONFIG_PATH}")
    except Exception as e:
        logger.error(f"[save_config] Failed to save configuration: {e}")


# Load the configuration at startup
CONFIG = load_config()

# -------------------------- Global Account Queue --------------------------
# Maintain a queue of available accounts for load balancing
account_queue = []


def init_account_queue():
    """
    Initialize the account queue by loading accounts from the configuration.
    Shuffles the accounts to ensure random distribution.
    """
    global account_queue
    accounts = CONFIG.get("accounts", [])
    if not accounts:
        logger.warning("[init_account_queue] No accounts found in configuration.")
    account_queue = accounts[:]  # Create a deep copy to avoid modifying the original list
    random.shuffle(account_queue)  # Randomize the order of accounts
    logger.info(f"[init_account_queue] Initialized account queue with {len(account_queue)} accounts.")


# Initialize the account queue at startup
init_account_queue()

# -------------------------- Account Management --------------------------
def get_account_identifier(account):
    """
    Get a unique identifier for an account.
    Prefers email over mobile as the identifier.
    """
    return account.get("email", "").strip() or account.get("mobile", "").strip()


def choose_new_account(exclude_ids=None):
    """
    Choose a new account from the queue that is not in the exclude_ids list.
    Removes the selected account from the queue and returns it.
    If no accounts are available, logs a warning and returns None.
    """
    if exclude_ids is None:
        exclude_ids = set()

    for i, acc in enumerate(account_queue):
        acc_id = get_account_identifier(acc)
        if acc_id and acc_id not in exclude_ids:
            selected_account = account_queue.pop(i)
            logger.info(f"[choose_new_account] Selected account: {acc_id}")
            return selected_account

    logger.warning("[choose_new_account] No available accounts or all accounts are in use.")
    return None


def release_account(account):
    """
    Release an account back into the queue.
    Adds the account to the end of the queue for reuse.
    """
    if account:
        account_queue.append(account)
        acc_id = get_account_identifier(account)
        logger.info(f"[release_account] Released account: {acc_id}")
    else:
        logger.warning("[release_account] Attempted to release a None account.")

# -------------------------- DeepSeek API Constants --------------------------
# Base URL for DeepSeek API
DEEPSEEK_HOST = "chat.deepseek.com"
DEEPSEEK_BASE_URL = f"https://{DEEPSEEK_HOST}"

# API Endpoints
DEEPSEEK_LOGIN_URL = f"{DEEPSEEK_BASE_URL}/api/v0/users/login"
DEEPSEEK_CREATE_SESSION_URL = f"{DEEPSEEK_BASE_URL}/api/v0/chat_session/create"
DEEPSEEK_CREATE_POW_URL = f"{DEEPSEEK_BASE_URL}/api/v0/chat/create_pow_challenge"
DEEPSEEK_COMPLETION_URL = f"{DEEPSEEK_BASE_URL}/api/v0/chat/completion"

# -------------------------- DeepSeek API Headers --------------------------
# Base headers for all DeepSeek API requests
BASE_HEADERS = {
    "Host": DEEPSEEK_HOST,
    "User-Agent": "DeepSeek/1.0.13 Android/35",  # User agent to mimic Android app
    "Accept": "application/json",  # Accept JSON responses
    "Accept-Encoding": "gzip",  # Enable gzip compression
    "Content-Type": "application/json",  # Send JSON data
    "x-client-platform": "android",  # Indicate client platform
    "x-client-version": "1.0.13",  # Client version
    "x-client-locale": "zh_CN",  # Client locale
    "accept-charset": "UTF-8",  # Character encoding
}

# Additional headers for authenticated requests
def get_auth_headers(token):
    """
    Generate headers for authenticated requests by adding the authorization token.
    """
    return {**BASE_HEADERS, "Authorization": f"Bearer {token}"}


# -------------------------- WASM Module Path --------------------------
# Path to the WASM module for PoW calculations
WASM_PATH = "sha3_wasm_bg.7b9ca65ddd.wasm"

# -------------------------- Login Function --------------------------
def login_deepseek_via_account(account):
    """
    Log into DeepSeek using the provided account credentials (email or mobile).
    On success, updates the account with the new token and saves the configuration.
    Returns the new token.
    Raises HTTPException on failure.
    """
    email = account.get("email", "").strip()
    mobile = account.get("mobile", "").strip()
    password = account.get("password", "").strip()

    # Validate required fields
    if not password or (not email and not mobile):
        logger.error("[login_deepseek_via_account] Missing email/mobile or password.")
        raise HTTPException(
            status_code=400,
            detail="Account login failed: Missing email/mobile or password.",
        )

    # Prepare payload based on email or mobile
    payload = {
        "password": password,
        "device_id": "deepseek_to_api",  # Static device ID
        "os": "android",  # Mimic Android app
    }
    if email:
        payload["email"] = email
    else:
        payload["mobile"] = mobile
        payload["area_code"] = None  # No area code provided

    try:
        # Send login request to DeepSeek API
        resp = requests.post(DEEPSEEK_LOGIN_URL, headers=BASE_HEADERS, json=payload)
        resp.raise_for_status()  # Raise exception for non-200 status codes
    except requests.RequestException as e:
        logger.error(f"[login_deepseek_via_account] Login request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Account login failed: Unable to connect to DeepSeek API.",
        )

    try:
        # Parse response JSON
        data = resp.json()
    except json.JSONDecodeError as e:
        logger.error(f"[login_deepseek_via_account] Invalid JSON response: {e}")
        raise HTTPException(
            status_code=500,
            detail="Account login failed: Invalid response from DeepSeek API.",
        )

    # Validate response structure
    if not data.get("data", {}).get("biz_data", {}).get("user", {}).get("token"):
        logger.error(f"[login_deepseek_via_account] Invalid response format: {data}")
        raise HTTPException(
            status_code=500,
            detail="Account login failed: Invalid response format.",
        )

    # Extract and update token
    new_token = data["data"]["biz_data"]["user"]["token"]
    account["token"] = new_token
    save_config(CONFIG)  # Save updated configuration
    logger.info(f"[login_deepseek_via_account] Login successful for account: {get_account_identifier(account)}")
    return new_token
# -------------------------- Account Selection and Release --------------------------
def choose_new_account(exclude_ids=None):
    """
    Select a new account from the account queue that is not in the exclude_ids list.
    Removes the selected account from the queue and returns it.
    If no accounts are available, logs a warning and returns None.
    """
    if exclude_ids is None:
        exclude_ids = set()

    for i, acc in enumerate(account_queue):
        acc_id = get_account_identifier(acc)
        if acc_id and acc_id not in exclude_ids:
            selected_account = account_queue.pop(i)  # Remove the account from the queue
            logger.info(f"[choose_new_account] Selected account: {acc_id}")
            return selected_account

    logger.warning("[choose_new_account] No available accounts or all accounts are in use.")
    return None


def release_account(account):
    """
    Release an account back into the account queue.
    Adds the account to the end of the queue for reuse.
    Logs the release operation for tracking.
    """
    if account:
        account_queue.append(account)  # Add the account back to the queue
        acc_id = get_account_identifier(account)
        logger.info(f"[release_account] Released account: {acc_id}")
    else:
        logger.warning("[release_account] Attempted to release a None account.")


# -------------------------- Token and Mode Determination --------------------------
def determine_mode_and_token(request: Request):
    """
    Determine the mode of operation and extract the appropriate token for DeepSeek API requests.
    - If the Bearer token is in CONFIG["keys"], use configuration mode (select an account from CONFIG["accounts"]).
    - Otherwise, use the provided Bearer token directly.
    Updates request.state with the determined token and mode.
    Raises HTTPException on failure.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        logger.error("[determine_mode_and_token] Missing or invalid Authorization header.")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing or invalid Bearer token.",
        )

    caller_key = auth_header.replace("Bearer ", "", 1).strip()
    config_keys = CONFIG.get("keys", [])

    if caller_key in config_keys:
        # Configuration mode: Use an account from CONFIG["accounts"]
        request.state.use_config_token = True
        request.state.tried_accounts = set()  # Track accounts that have been tried

        selected_account = choose_new_account()
        if not selected_account:
            logger.error("[determine_mode_and_token] No available accounts.")
            raise HTTPException(
                status_code=429,
                detail="No accounts configured or all accounts are busy.",
            )

        # Ensure the selected account has a valid token
        if not selected_account.get("token", "").strip():
            try:
                login_deepseek_via_account(selected_account)
            except HTTPException as e:
                logger.error(f"[determine_mode_and_token] Account login failed: {e.detail}")
                raise HTTPException(status_code=500, detail="Account login failed.")

        request.state.deepseek_token = selected_account["token"]
        request.state.account = selected_account
        logger.info(f"[determine_mode_and_token] Using configuration mode with account: {get_account_identifier(selected_account)}")

    else:
        # User-provided token mode: Use the provided Bearer token directly
        request.state.use_config_token = False
        request.state.deepseek_token = caller_key
        logger.info("[determine_mode_and_token] Using user-provided token mode.")


def get_auth_headers(request: Request):
    """
    Generate headers for authenticated DeepSeek API requests.
    Includes the authorization token from request.state.
    """
    return {**BASE_HEADERS, "Authorization": f"Bearer {request.state.deepseek_token}"}

# -------------------------- Session Creation --------------------------
def create_session(request: Request, max_attempts=3):
    """
    Create a new chat session with the DeepSeek API.
    Retries up to max_attempts times in case of failure.
    In configuration mode, switches accounts if the current account fails.
    Returns the session ID on success, or None on failure.
    """
    attempts = 0
    while attempts < max_attempts:
        headers = get_auth_headers(request)
        try:
            resp = requests.post(
                DEEPSEEK_CREATE_SESSION_URL,
                headers=headers,
                json={"agent": "chat"},  # Static payload for session creation
                timeout=30,  # Timeout after 30 seconds
            )
            resp.raise_for_status()  # Raise exception for non-200 status codes
        except requests.RequestException as e:
            logger.error(f"[create_session] Request failed (attempt {attempts + 1}): {e}")
            attempts += 1
            continue

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            logger.error(f"[create_session] Invalid JSON response (attempt {attempts + 1}): {e}")
            attempts += 1
            continue

        if resp.status_code == 200 and data.get("code") == 0:
            session_id = data["data"]["biz_data"]["id"]
            logger.info(f"[create_session] Session created successfully: {session_id}")
            return session_id
        else:
            logger.warning(
                f"[create_session] Session creation failed (attempt {attempts + 1}): "
                f"code={data.get('code')}, msg={data.get('msg')}"
            )
            if request.state.use_config_token:
                # Switch to a new account in configuration mode
                current_id = get_account_identifier(request.state.account)
                request.state.tried_accounts.add(current_id)
                new_account = choose_new_account(request.state.tried_accounts)
                if not new_account:
                    logger.error("[create_session] No more accounts available.")
                    break
                try:
                    login_deepseek_via_account(new_account)
                except HTTPException as e:
                    logger.error(f"[create_session] Account login failed: {e.detail}")
                    attempts += 1
                    continue
                request.state.account = new_account
                request.state.deepseek_token = new_account["token"]
            else:
                attempts += 1

    logger.error("[create_session] Max attempts reached. Session creation failed.")
    return None


# -------------------------- PoW Handling --------------------------
def get_pow_response(request: Request, max_attempts=3):
    """
    Get a Proof of Work (PoW) response from the DeepSeek API.
    Retries up to max_attempts times in case of failure.
    In configuration mode, switches accounts if the current account fails.
    Returns the base64-encoded PoW response on success, or None on failure.
    """
    attempts = 0
    while attempts < max_attempts:
        headers = get_auth_headers(request)
        try:
            resp = requests.post(
                DEEPSEEK_CREATE_POW_URL,
                headers=headers,
                json={"target_path": "/api/v0/chat/completion"},  # Static payload for PoW
                timeout=30,  # Timeout after 30 seconds
            )
            resp.raise_for_status()  # Raise exception for non-200 status codes
        except requests.RequestException as e:
            logger.error(f"[get_pow_response] Request failed (attempt {attempts + 1}): {e}")
            attempts += 1
            continue

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            logger.error(f"[get_pow_response] Invalid JSON response (attempt {attempts + 1}): {e}")
            attempts += 1
            continue

        if resp.status_code == 200 and data.get("code") == 0:
            challenge = data["data"]["biz_data"]["challenge"]
            try:
                answer = compute_pow_answer(
                    algorithm=challenge["algorithm"],
                    challenge_str=challenge["challenge"],
                    salt=challenge["salt"],
                    difficulty=challenge.get("difficulty", 144000),
                    expire_at=challenge.get("expire_at", 1680000000),
                    signature=challenge["signature"],
                    target_path=challenge["target_path"],
                    wasm_path=WASM_PATH,
                )
            except Exception as e:
                logger.error(f"[get_pow_response] PoW computation failed: {e}")
                attempts += 1
                continue

            if answer is None:
                logger.warning(f"[get_pow_response] PoW computation returned None (attempt {attempts + 1}).")
                attempts += 1
                continue

            pow_dict = {
                "algorithm": challenge["algorithm"],
                "challenge": challenge["challenge"],
                "salt": challenge["salt"],
                "answer": answer,
                "signature": challenge["signature"],
                "target_path": challenge["target_path"],
            }
            pow_str = json.dumps(pow_dict, separators=(",", ":"), ensure_ascii=False)
            encoded = base64.b64encode(pow_str.encode("utf-8")).decode("utf-8").rstrip()
            logger.info("[get_pow_response] PoW response generated successfully.")
            return encoded
        else:
            logger.warning(
                f"[get_pow_response] PoW request failed (attempt {attempts + 1}): "
                f"code={data.get('code')}, msg={data.get('msg')}"
            )
            if request.state.use_config_token:
                # Switch to a new account in configuration mode
                current_id = get_account_identifier(request.state.account)
                request.state.tried_accounts.add(current_id)
                new_account = choose_new_account(request.state.tried_accounts)
                if not new_account:
                    logger.error("[get_pow_response] No more accounts available.")
                    break
                try:
                    login_deepseek_via_account(new_account)
                except HTTPException as e:
                    logger.error(f"[get_pow_response] Account login failed: {e.detail}")
                    attempts += 1
                    continue
                request.state.account = new_account
                request.state.deepseek_token = new_account["token"]
            else:
                attempts += 1

    logger.error("[get_pow_response] Max attempts reached. PoW request failed.")
    return None

# -------------------------- WASM Module for PoW Calculation --------------------------
def compute_pow_answer(
    algorithm: str,
    challenge_str: str,
    salt: str,
    difficulty: int,
    expire_at: int,
    signature: str,
    target_path: str,
    wasm_path: str,
) -> int:
    """
    Compute the Proof of Work (PoW) answer using a WASM module.
    Returns the computed answer as an integer, or None if computation fails.
    Raises ValueError for unsupported algorithms or RuntimeError for WASM-related errors.
    """
    if algorithm != "DeepSeekHashV1":
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Prepare the prefix for the challenge
    prefix = f"{salt}_{expire_at}_"

    # Load the WASM module
    store = Store()
    linker = Linker(store.engine)
    try:
        with open(wasm_path, "rb") as f:
            wasm_bytes = f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to load WASM file: {wasm_path}, error: {e}")

    # Instantiate the WASM module
    module = Module(store.engine, wasm_bytes)
    instance = linker.instantiate(store, module)
    exports = instance.exports(store)

    # Retrieve exported functions and memory
    try:
        memory = exports["memory"]
        add_to_stack = exports["__wbindgen_add_to_stack_pointer"]
        alloc = exports["__wbindgen_export_0"]
        wasm_solve = exports["wasm_solve"]
    except KeyError as e:
        raise RuntimeError(f"Missing WASM export: {e}")

    # Helper functions for memory operations
    def write_memory(offset: int, data: bytes):
        """Write data to WASM memory at the specified offset."""
        size = len(data)
        base_addr = ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value
        ctypes.memmove(base_addr + offset, data, size)

    def read_memory(offset: int, size: int) -> bytes:
        """Read data from WASM memory at the specified offset."""
        base_addr = ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value
        return ctypes.string_at(base_addr + offset, size)

    def encode_string(text: str):
        """Encode a string into WASM memory and return its pointer and length."""
        data = text.encode("utf-8")
        length = len(data)
        ptr_val = alloc(store, length, 1)
        ptr = int(ptr_val.value) if hasattr(ptr_val, "value") else int(ptr_val)
        write_memory(ptr, data)
        return ptr, length

    try:
        # Allocate stack space
        retptr = add_to_stack(store, -16)

        # Encode challenge and prefix into WASM memory
        ptr_challenge, len_challenge = encode_string(challenge_str)
        ptr_prefix, len_prefix = encode_string(prefix)

        # Call the WASM solve function
        wasm_solve(
            store,
            retptr,
            ptr_challenge,
            len_challenge,
            ptr_prefix,
            len_prefix,
            float(difficulty),  # Difficulty is passed as a float
        )

        # Read the status and result from WASM memory
        status_bytes = read_memory(retptr, 4)
        if len(status_bytes) != 4:
            raise RuntimeError("Failed to read status bytes from WASM memory.")
        status = struct.unpack("<i", status_bytes)[0]

        value_bytes = read_memory(retptr + 8, 8)
        if len(value_bytes) != 8:
            raise RuntimeError("Failed to read value bytes from WASM memory.")
        value = struct.unpack("<d", value_bytes)[0]

        # Restore stack pointer
        add_to_stack(store, 16)

        # Return the computed answer if status is non-zero
        if status == 0:
            logger.warning("[compute_pow_answer] WASM computation returned status 0.")
            return None
        return int(value)

    except Exception as e:
        logger.error(f"[compute_pow_answer] WASM computation failed: {e}")
        raise RuntimeError(f"WASM computation failed: {e}")

# -------------------------- Message Preprocessing --------------------------
def messages_prepare(messages: list) -> str:
    """
    Process a list of messages into a single prompt string.
    - Merges consecutive messages from the same role.
    - Adds role-specific prefixes (e.g., "### Assistant:" for assistant messages).
    - Removes markdown image syntax.
    Returns the processed prompt string.
    """
    if not messages:
        return ""

    # Merge consecutive messages from the same role
    merged_messages = []
    current_role = None
    current_text = ""

    for message in messages:
        role = message.get("role", "").strip().lower()
        content = message.get("content", "")

        # Handle content as a list (e.g., multimodal inputs)
        if isinstance(content, list):
            text_parts = [
                item.get("text", "") for item in content if item.get("type") == "text"
            ]
            text = "\n".join(text_parts)
        else:
            text = str(content).strip()

        if role == current_role:
            # Append to the current message
            current_text += "\n\n" + text
        else:
            # Save the current message and start a new one
            if current_text:
                merged_messages.append({"role": current_role, "text": current_text})
            current_role = role
            current_text = text

    # Add the last message
    if current_text:
        merged_messages.append({"role": current_role, "text": current_text})

    # Add role-specific prefixes and construct the final prompt
    final_prompt = []
    for idx, block in enumerate(merged_messages):
        role = block["role"]
        text = block["text"]

        if role == "assistant":
            # Assistant messages start with "### Assistant:"
            final_prompt.append(f"### Assistant: {text}")
        elif role in ("user", "system"):
            # User/system messages start with "### User:" (except the first one)
            if idx > 0:
                final_prompt.append(f"### User: {text}")
            else:
                final_prompt.append(text)
        else:
            # Unknown roles are added as-is
            final_prompt.append(text)

    # Join all parts into a single string
    final_prompt = "\n".join(final_prompt)

    # Remove markdown image syntax (e.g., ![alt](url))
    final_prompt = re.sub(r"!\[(.*?)\]\((.*?)\)", r"[\1](\2)", final_prompt)

    return final_prompt
# -------------------------- Chat Completions Endpoint --------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Handle chat completion requests.
    Supports both streaming and non-streaming responses.
    Manages sessions, PoW challenges, and account switching (in configuration mode).
    """
    try:
        # Determine the mode and token for the request
        try:
            determine_mode_and_token(request)
        except HTTPException as e:
            logger.error(f"[chat_completions] Token determination failed: {e.detail}")
            return JSONResponse(status_code=e.status_code, content={"error": e.detail})

        # Parse the request body
        try:
            req_data = await request.json()
        except Exception as e:
            logger.error(f"[chat_completions] Invalid request body: {e}")
            return JSONResponse(status_code=400, content={"error": "Invalid request body."})

        # Validate required fields
        model = req_data.get("model")
        messages = req_data.get("messages", [])
        if not model or not messages:
            logger.error("[chat_completions] Missing 'model' or 'messages' in request.")
            return JSONResponse(
                status_code=400,
                content={"error": "Request must include 'model' and 'messages'."},
            )

        # Determine if thinking or search is enabled based on the model
        model_lower = model.lower()
        if model_lower in ["deepseek-v3", "deepseek-chat"]:
            thinking_enabled = False
            search_enabled = False
        elif model_lower in ["deepseek-r1", "deepseek-reasoner"]:
            thinking_enabled = True
            search_enabled = False
        elif model_lower in ["deepseek-v3-search", "deepseek-chat-search"]:
            thinking_enabled = False
            search_enabled = True
        elif model_lower in ["deepseek-r1-search", "deepseek-reasoner-search"]:
            thinking_enabled = True
            search_enabled = True
        else:
            logger.error(f"[chat_completions] Unsupported model: {model}")
            return JSONResponse(
                status_code=503, content={"error": f"Model '{model}' is not available."}
            )

        # Prepare the final prompt
        final_prompt = messages_prepare(messages)

        # Create a session and get the PoW response
        session_id = create_session(request)
        if not session_id:
            logger.error("[chat_completions] Session creation failed.")
            return JSONResponse(status_code=401, content={"error": "Invalid token."})

        pow_response = get_pow_response(request)
        if not pow_response:
            logger.error("[chat_completions] PoW response generation failed.")
            return JSONResponse(
                status_code=401, content={"error": "Failed to get PoW response."}
            )

        # Prepare headers and payload for the completion request
        headers = {**get_auth_headers(request), "x-ds-pow-response": pow_response}
        payload = {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "prompt": final_prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
        }

        # Call the DeepSeek completion endpoint
        deepseek_resp = call_completion_endpoint(payload, headers, max_attempts=3)
        if not deepseek_resp:
            logger.error("[chat_completions] Completion request failed.")
            return JSONResponse(status_code=500, content={"error": "Completion failed."})

        # Handle streaming or non-streaming response
        if req_data.get("stream", False):
            return StreamingResponse(
                sse_stream(deepseek_resp, request, model, session_id),
                media_type="text/event-stream",
            )
        else:
            return await handle_non_streaming_response(deepseek_resp, request, model, session_id)

    except HTTPException as e:
        logger.error(f"[chat_completions] HTTPException: {e.detail}")
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        logger.error(f"[chat_completions] Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
    finally:
        # Release the account back to the queue in configuration mode
        if getattr(request.state, "use_config_token", False) and hasattr(
            request.state, "account"
        ):
            release_account(request.state.account)

# -------------------------- Streaming Response Handling --------------------------
def sse_stream(deepseek_resp, request, model, session_id):
    """
    Handle streaming responses from the DeepSeek API.
    Processes the response in chunks and streams it back to the client in SSE format.
    """
    try:
        final_text = ""
        final_thinking = ""
        first_chunk_sent = False
        result_queue = queue.Queue()
        last_send_time = time.time()
        citation_map = {}  # Store citation URLs for search-enabled responses

        def process_data():
            """Process the raw response data and put chunks into the result queue."""
            try:
                for raw_line in deepseek_resp.iter_lines():
                    try:
                        line = raw_line.decode("utf-8")
                    except Exception as e:
                        logger.warning(f"[sse_stream] Decoding failed: {e}")
                        result_queue.put({
                            "choices": [{
                                "index": 0,
                                "delta": {"content": "服务器繁忙，请稍候再试", "type": "text"}
                            }]
                        })
                        result_queue.put(None)
                        break

                    if not line:
                        continue

                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            result_queue.put(None)
                            break

                        try:
                            chunk = json.loads(data_str)
                            # Handle search index data
                            if (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("type")
                                == "search_index"
                            ):
                                search_indexes = chunk["choices"][0]["delta"].get("search_indexes", [])
                                for idx in search_indexes:
                                    citation_map[str(idx.get("cite_index"))] = idx.get("url", "")
                                continue

                            result_queue.put(chunk)
                        except Exception as e:
                            logger.warning(f"[sse_stream] Failed to parse chunk: {e}")
                            result_queue.put({
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": "服务器繁忙，请稍候再试", "type": "text"}
                                }]
                            })
                            result_queue.put(None)
                            break
            except Exception as e:
                logger.error(f"[sse_stream] Error processing data: {e}")
                result_queue.put({
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "服务器繁忙，请稍候再试", "type": "text"}
                    }]
                })
                result_queue.put(None)
            finally:
                deepseek_resp.close()

        # Start a thread to process the response data
        process_thread = threading.Thread(target=process_data)
        process_thread.start()

        while True:
            current_time = time.time()
            if current_time - last_send_time >= KEEP_ALIVE_TIMEOUT:
                # Send a keep-alive comment to prevent connection timeout
                yield ": keep-alive\n\n"
                last_send_time = current_time
                continue

            try:
                chunk = result_queue.get(timeout=0.1)
                if chunk is None:
                    # Send final usage statistics
                    prompt_tokens = len(tokenizer.encode(final_prompt))
                    completion_tokens = len(tokenizer.encode(final_text))
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                    finish_chunk = {
                        "id": session_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                        "usage": usage,
                    }
                    yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    break

                new_choices = []
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    ctype = delta.get("type")
                    ctext = delta.get("content", "")
                    if ctype == "thinking":
                        final_thinking += ctext
                    elif ctype == "text":
                        final_text += ctext

                    delta_obj = {}
                    if not first_chunk_sent:
                        delta_obj["role"] = "assistant"
                        first_chunk_sent = True
                    if ctype == "thinking":
                        delta_obj["reasoning_content"] = ctext
                    elif ctype == "text":
                        delta_obj["content"] = ctext

                    if delta_obj:
                        new_choices.append({
                            "delta": delta_obj,
                            "index": choice.get("index", 0),
                        })

                if new_choices:
                    out_chunk = {
                        "id": session_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": new_choices,
                    }
                    yield f"data: {json.dumps(out_chunk, ensure_ascii=False)}\n\n"
                    last_send_time = current_time

            except queue.Empty:
                continue

    except Exception as e:
        logger.error(f"[sse_stream] Unexpected error: {e}")
    finally:
        if getattr(request.state, "use_config_token", False) and hasattr(
            request.state, "account"
        ):
            release_account(request.state.account)

# -------------------------- Non-Streaming Response Handling --------------------------
async def handle_non_streaming_response(deepseek_resp, request, model, session_id):
    """
    Handle non-streaming responses from the DeepSeek API.
    Collects all response data, constructs a final response object, and returns it.
    """
    try:
        think_list = []  # Store reasoning content
        text_list = []  # Store assistant responses
        citation_map = {}  # Store citation URLs for search-enabled responses

        def collect_data():
            """Collect data from the DeepSeek API response."""
            try:
                for raw_line in deepseek_resp.iter_lines():
                    try:
                        line = raw_line.decode("utf-8")
                    except Exception as e:
                        logger.warning(f"[handle_non_streaming_response] Decoding failed: {e}")
                        text_list.append("服务器繁忙，请稍候再试")
                        break

                    if not line:
                        continue

                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            # Handle search index data
                            if (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("type")
                                == "search_index"
                            ):
                                search_indexes = chunk["choices"][0]["delta"].get("search_indexes", [])
                                for idx in search_indexes:
                                    citation_map[str(idx.get("cite_index"))] = idx.get("url", "")
                                continue

                            for choice in chunk.get("choices", []):
                                delta = choice.get("delta", {})
                                ctype = delta.get("type")
                                ctext = delta.get("content", "")
                                if ctype == "thinking":
                                    think_list.append(ctext)
                                elif ctype == "text":
                                    text_list.append(ctext)
                        except Exception as e:
                            logger.warning(f"[handle_non_streaming_response] Failed to parse chunk: {e}")
                            text_list.append("服务器繁忙，请稍候再试")
                            break
            except Exception as e:
                logger.error(f"[handle_non_streaming_response] Error processing data: {e}")
                text_list.append("服务器繁忙，请稍候再试")
            finally:
                deepseek_resp.close()

        # Collect data from the response
        collect_data()

        # Construct the final response
        final_reasoning = "".join(think_list)
        final_content = "".join(text_list)
        prompt_tokens = len(tokenizer.encode(final_prompt))
        completion_tokens = len(tokenizer.encode(final_content))

        response = {
            "id": session_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_content,
                        "reasoning_content": final_reasoning,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        logger.error(f"[handle_non_streaming_response] Unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error"},
        )
    finally:
        if getattr(request.state, "use_config_token", False) and hasattr(
            request.state, "account"
        ):
            release_account(request.state.account)

# -------------------------- Index Route --------------------------
@app.get("/")
def index(request: Request):
    """
    Serve the welcome page.
    Uses a Jinja2 template to render the HTML.
    """
    return templates.TemplateResponse("welcome.html", {"request": request})



# -------------------------- FastAPI Application Initialization --------------------------
if __name__ == "__main__":
    """
    Entry point for running the FastAPI application.
    Uses Uvicorn to serve the application.
    """
    import uvicorn

    # Configure logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "use_colors": True,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "handlers": ["default"],
            "level": "INFO",
        },
    }

    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",  # Bind to all available interfaces
        port=5001,  # Use port 5001
        log_config=logging_config,  # Apply custom logging configuration
        reload=False,  # Disable auto-reload in production
        workers=4,  # Use 4 worker processes for concurrency
    )
