#!/usr/bin/env python3
import os
import json
import sys
import urllib.request
import urllib.parse

def load_env_file(env_path):
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

load_env_file(os.path.expanduser("~/.env"))
FIGMA_TOKEN = os.getenv("FIGMA_ACCESS_TOKEN")

def get_figma_file(file_key):
    if not FIGMA_TOKEN:
        print("Error: FIGMA_ACCESS_TOKEN not set in ~/.env", file=sys.stderr)
        sys.exit(1)
        
    url = f"https://api.figma.com/v1/files/{file_key}"
    req = urllib.request.Request(url, headers={"X-Figma-Token": FIGMA_TOKEN})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data
    except Exception as e:
        print(f"Error fetching Figma file {file_key}: {e}", file=sys.stderr)
        return None

def get_figma_images(file_key, ids, format="png", scale=2):
    if not FIGMA_TOKEN:
        print("Error: FIGMA_ACCESS_TOKEN not set in ~/.env", file=sys.stderr)
        sys.exit(1)
        
    ids_str = ",".join(ids) if isinstance(ids, list) else ids
    url = f"https://api.figma.com/v1/images/{file_key}?ids={urllib.parse.quote(ids_str)}&format={format}&scale={scale}"
    req = urllib.request.Request(url, headers={"X-Figma-Token": FIGMA_TOKEN})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data
    except Exception as e:
        print(f"Error fetching Figma images {file_key}: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/figma_client.py <figma_file_key_or_url>")
        sys.exit(1)
        
    arg = sys.argv[1]
    if "figma.com" in arg:
        parts = arg.split("/")
        if "file" in parts:
            idx = parts.index("file")
            file_key = parts[idx + 1]
        elif "design" in parts:
            idx = parts.index("design")
            file_key = parts[idx + 1]
        else:
            file_key = arg
    else:
        file_key = arg
        
    print(f"Fetching Figma File metadata for key: {file_key}...")
    file_data = get_figma_file(file_key)
    if file_data:
        print(f"Document Name: {file_data.get('name')}")
        print(f"Last Modified: {file_data.get('lastModified')}")
        print(f"Version: {file_data.get('version')}")
        components = file_data.get("components", {})
        print(f"Components Found: {len(components)}")
        styles = file_data.get("styles", {})
        print(f"Styles Found: {len(styles)}")
