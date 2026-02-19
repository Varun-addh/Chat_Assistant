# 🚀 Quick Start - Authentication System

## Installation (Already Done)

Dependencies are installed:
- ✅ PyJWT (JWT tokens)
- ✅ passlib (password hashing)
- ✅ python-jose (JWT cryptography)
- ✅ sqlalchemy (database ORM)
- ✅ alembic (database migrations)

## Starting the Server

```bash
# Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# Start the server
uvicorn app.main:app --reload --port 7860
```

## Test the System

```bash
# In a new terminal
python test_auth_system.py
```

## Quick Test with Swagger UI

1. Open http://localhost:7860/docs
2. Look for the new "Authentication" section
3. Try these endpoints:
   - POST `/auth/register` - Create account
   - POST `/auth/login` - Get JWT token
   - GET `/auth/me` - Get user info (click "Authorize" and paste token)
   - GET `/auth/quota` - Check your limits

## Example: Register & Login

### Register
```bash
curl -X POST http://localhost:7860/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "password": "YourSecurePassword123!",
    "full_name": "Your Name"
  }'
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "user_id": "abc-123-def",
  "tier": "free"
}
```

### Use the Token

```bash
# Save the token
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."

# Make authenticated requests
curl http://localhost:7860/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

## Integration with Frontend

Update your frontend to:

1. **Register/Login** → Get JWT token
2. **Store token** in localStorage or cookie
3. **Add to all requests**:
   ```javascript
   headers: {
     'Authorization': `Bearer ${token}`,
     'Content-Type': 'application/json'
   }
   ```

4. **Handle 401 errors** → Redirect to login

5. **Show user info**:
   ```javascript
   const response = await fetch('http://localhost:7860/auth/me', {
     headers: { 'Authorization': `Bearer ${token}` }
   });
   const user = await response.json();
   console.log(user.email, user.tier);
   ```

## Rate Limits

Check response headers:
```
X-RateLimit-Limit: 50
X-RateLimit-Remaining: 47
X-RateLimit-Reset: 1704643200
```

When exceeded, you'll get:
```json
{
  "detail": {
    "error": "Rate limit exceeded",
    "current_usage": 50,
    "limit": 50,
    "tier": "free",
    "message": "Upgrade to PRO for higher limits."
  }
}
```

## User Tiers

| Tier | Daily Limits | Monthly Price |
|------|--------------|---------------|
| FREE | 50 API calls, 10 questions | $0 |
| BASIC | 500 API calls, 100 questions | $19 |
| PRO | 5000 API calls, 1000 questions | $49 |
| ENTERPRISE | Unlimited | Custom |

## What's Protected

All routes are now rate-limited except:
- `/health`
- `/docs`
- `/openapi.json`
- `/auth/register`
- `/auth/login`

## Troubleshooting

### "Rate limit exceeded"
- You've hit your daily quota
- Either wait 24 hours or upgrade tier

### "401 Unauthorized"
- Token is invalid or expired
- Login again to get a new token

### "Database locked"
- Another server instance is running
- Stop all Python processes: `Get-Process python | Stop-Process -Force`

### "Module not found: jwt"
- Run: `pip install PyJWT passlib[bcrypt] python-jose[cryptography] sqlalchemy`

## Database Location

SQLite database created at:
```
data/stratax.db
```

To inspect it:
```bash
# Install SQLite browser
# Or use Python:
python
>>> from app.database import SessionLocal
>>> from app.models import User
>>> db = SessionLocal()
>>> users = db.query(User).all()
>>> for u in users: print(u.email, u.tier)
```

## Next Steps

1. ✅ Test the authentication system
2. 📊 Monitor usage in the database
3. 💰 Add Stripe integration (Phase 2)
4. 📧 Add email verification (Phase 2)
5. 🔑 Add password reset (Phase 2)

**Your app is now secure and ready to charge users!** 🎉
