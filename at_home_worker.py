import os
import time
import subprocess
import json
from datetime import datetime

# --- Configuration ---
OWNER = "seowookchoi"
REPO = "autoresearch_IRT"
ISSUE_NUMBER = 1
POLLING_INTERVAL = 30 

REPO_PATH = "/Users/awesomedasom/Desktop/autoresearch_irt"
LOG_DIR = os.path.join(REPO_PATH, "jobs/logs")
LAST_COMMENT_ID_FILE = os.path.join(REPO_PATH, ".last_run_comment_id")

def get_latest_comment():
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/{ISSUE_NUMBER}/comments"
    
    # We proved curl works on your Mac, so let's bypass Python's network stack entirely!
    cmd = [
        "curl", "-s", 
        "-H", "Accept: application/vnd.github.v3+json",
        "-H", "User-Agent: Mobile-Watcher-Script",
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"⚠️ Curl Error: {result.stderr}")
            return None
            
        data = json.loads(result.stdout)
        
        # GitHub returns a list if successful, but a dictionary if there's an error message
        if isinstance(data, list) and len(data) > 0:
            return data[-1] 
        elif isinstance(data, dict) and 'message' in data:
            print(f"⚠️ GitHub API returned a message: {data['message']}")
            
    except subprocess.TimeoutExpired:
        print("⚠️ Curl command timed out.")
    except Exception as e:
        print(f"⚠️ Unexpected JSON parsing error: {e}")
        
    return None

def get_last_run_id():
    if os.path.exists(LAST_COMMENT_ID_FILE):
        with open(LAST_COMMENT_ID_FILE, "r") as f:
            return f.read().strip()
    return None

def save_last_run_id(comment_id):
    with open(LAST_COMMENT_ID_FILE, "w") as f:
        f.write(str(comment_id))

def sync_and_run():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 👀 Checking GitHub for commands...")
    latest_comment = get_latest_comment()
    
    if not latest_comment:
        return

    comment_id = str(latest_comment['id'])
    comment_body = latest_comment['body'].strip()
    last_run_id = get_last_run_id()

    if comment_id != last_run_id and comment_body.startswith("run:"):
        command = comment_body.replace("run:", "").strip()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📲 Received mobile command: {command}")
        
        os.chdir(REPO_PATH)
        subprocess.run(["git", "checkout", "dev"], capture_output=True)
        subprocess.run(["git", "pull", "origin", "dev"], capture_output=True)
        
        print("⏳ Running task on local PC...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        os.makedirs(LOG_DIR, exist_ok=True)
        log_filename = f"issue_comment_{comment_id}.log"
        log_path = os.path.join(LOG_DIR, log_filename)
        
        with open(log_path, "w") as f:
            f.write(f"Command: {command}\n")
            f.write("-" * 30 + "\n")
            f.write(result.stdout if result.stdout else "Success (Quiet Mode)\n")
            if result.stderr:
                f.write("\n[ERRORS]\n" + result.stderr)
        
        subprocess.run(["git", "add", "."], capture_output=True)
        subprocess.run(["git", "commit", "-m", f"Log for mobile command {comment_id}"], capture_output=True)
        subprocess.run(["git", "push", "origin", "dev"], capture_output=True)
        
        save_last_run_id(comment_id)
        print(f"✅ Job finished! Log file ({log_filename}) pushed to GitHub.")

if __name__ == "__main__":
    print(f"🚀 Mobile Watcher started (Listening to Issue #{ISSUE_NUMBER}...)")
    print("Press Ctrl+C to stop.\n")
    while True:
        sync_and_run()
        time.sleep(POLLING_INTERVAL)