import os
import time
import base64
import hmac
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from src.database.database import get_db

SECRET_KEY = os.environ.get("SECRET_KEY", "super-secret-enterprise-jwt-key-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

def hash_password(password: str) -> str:
    """Hash password using PBKDF2 with SHA256 and salt."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return base64.b64encode(salt + key).decode('ascii')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against stored PBKDF2 hash."""
    try:
        data = base64.b64decode(hashed_password.encode('ascii'))
        salt = data[:16]
        stored_key = data[16:]
        new_key = hashlib.pbkdf2_hmac('sha256', plain_password.encode('utf-8'), salt, 100000)
        return hmac.compare_digest(stored_key, new_key)
    except Exception:
        return False

def base64url_encode(input_bytes: bytes) -> str:
    return base64.urlsafe_b64encode(input_bytes).rstrip(b'=').decode('ascii')

def base64url_decode(input_str: str) -> bytes:
    padding = '=' * (4 - (len(input_str) % 4))
    return base64.urlsafe_b64encode((input_str + padding).encode('ascii'))

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generate lightweight, robust JWT token signed with HMAC-SHA256."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": int(expire.timestamp())})
    
    header = {"alg": "HS256", "typ": "JWT"}
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    payload_json = json.dumps(to_encode, separators=(',', ':')).encode('utf-8')
    
    encoded_header = base64.urlsafe_b64encode(header_json).rstrip(b'=').decode('ascii')
    encoded_payload = base64.urlsafe_b64encode(payload_json).rstrip(b'=').decode('ascii')
    
    signing_input = f"{encoded_header}.{encoded_payload}"
    signature = hmac.new(SECRET_KEY.encode('utf-8'), signing_input.encode('utf-8'), hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b'=').decode('ascii')
    
    return f"{signing_input}.{encoded_signature}"

def decode_access_token(token: str) -> Optional[dict]:
    """Decode and verify HMAC-SHA256 JWT token."""
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        
        signing_input = f"{parts[0]}.{parts[1]}"
        expected_sig = hmac.new(SECRET_KEY.encode('utf-8'), signing_input.encode('utf-8'), hashlib.sha256).digest()
        encoded_expected_sig = base64.urlsafe_b64encode(expected_sig).rstrip(b'=').decode('ascii')
        
        if not hmac.compare_digest(parts[2], encoded_expected_sig):
            return None
        
        # Decode payload
        payload_b64 = parts[1] + '=' * (4 - (len(parts[1]) % 4))
        payload_bytes = base64.urlsafe_b64decode(payload_b64.encode('ascii'))
        payload = json.loads(payload_bytes.decode('utf-8'))
        
        if "exp" in payload and payload["exp"] < int(time.time()):
            return None
            
        return payload
    except Exception:
        return None

def get_current_user_optional(token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Retrieve user if token is provided, returns None if not authenticated."""
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        return None
    
    from src.database.models import User
    user = db.query(User).filter(User.username == payload["sub"]).first()
    return user

def get_current_user(token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Enforce valid authentication dependency."""
    user = get_current_user_optional(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authorization credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
