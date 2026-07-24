from src.auth.security import hash_password, verify_password, create_access_token, decode_access_token

def test_password_hashing():
    raw_pass = "EnterpriseSecurePass123!"
    hashed = hash_password(raw_pass)
    assert hashed != raw_pass
    assert verify_password(raw_pass, hashed) is True
    assert verify_password("WrongPass", hashed) is False

def test_jwt_token_creation_and_decoding():
    data = {"sub": "admin_user", "role": "admin"}
    token = create_access_token(data)
    assert isinstance(token, str)
    
    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "admin_user"
    assert decoded["role"] == "admin"

def test_invalid_jwt_token():
    assert decode_access_token("invalid.token.str") is None
