# 🔐 Authentication & Rate Limiting - NEW!

**Stratax AI now has a production-ready authentication and rate limiting system!**

## What's New

### ✅ User Authentication
- **JWT-based authentication** with secure password hashing (bcrypt)
- User registration and login endpoints
- Protected routes requiring authentication
- User profile management

### ✅ Rate Limiting
- **Tier-based quotas** to protect API costs
- In-memory rate limiter (Redis-ready for production)
- Rate limit headers in responses
- Automatic cleanup of old records

### ✅ User Tiers & Pricing
- **FREE**: 50 API calls/day, 10 copilot questions, 1 mock interview, 1 practice session
- **BASIC ($19/month)**: 500 calls/day, 100 questions, 10 interviews, 5 practice sessions
- **PRO ($49/month)**: 5000 calls/day, 1000 questions, 100 interviews, 50 practice sessions
- **ENTERPRISE**: Unlimited everything

### ✅ Database Integration
- SQLAlchemy ORM with SQLite (PostgreSQL-ready)
- User accounts, usage tracking, session records
- Automatic database initialization

## API Endpoints

### Authentication

#### Register New User
```bash
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123!",
  "full_name": "John Doe",
  "username": "johndoe"
}

Response:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "user_id": "uuid-here",
  "tier": "free"
}
```

#### Login
```bash
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123!"
}

Response:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "user_id": "uuid-here",
  "tier": "free"
}
```

#### Get User Info
```bash
GET /auth/me
Authorization: Bearer <token>

Response:
{
  "id": "uuid-here",
  "email": "user@example.com",
  "username": "johndoe",
  "full_name": "John Doe",
  "tier": "free",
  "is_verified": false,
  "created_at": "2026-01-07T10:30:00Z"
}
```

#### Get Quota Info
```bash
GET /auth/quota
Authorization: Bearer <token>

Response:
{
  "tier": "free",
  "limits": {
    "daily_api_calls": 50,
    "daily_copilot_questions": 10,
    "daily_mock_interviews": 1,
    "daily_practice_sessions": 1,
    "max_tokens_per_request": 2000,
    "features": ["copilot", "basic_search"]
  },
  "message": "You are on the FREE tier"
}
```

#### Update Profile
```bash
PUT /auth/me
Authorization: Bearer <token>
Content-Type: application/json

{
  "full_name": "Jane Smith",
  "username": "janesmith",
  "user_groq_api_key": "optional-your-own-key",
  "user_gemini_api_key": "optional-your-own-key"
}
```

#### Change Password
```bash
POST /auth/change-password
Authorization: Bearer <token>
Content-Type: application/json

{
  "current_password": "OldPassword123!",
  "new_password": "NewPassword456!"
}
```

## Using Protected Routes

All API routes now support authentication. Add the JWT token to your requests:

```python
import httpx

token = "eyJ0eXAiOiJKV1QiLCJhbGc..."
headers = {"Authorization": f"Bearer {token}"}

# Make authenticated requests
response = httpx.post(
    "http://localhost:7860/api/sessions/create",
    headers=headers,
    json={...}
)
```

### Guest Access
Routes work without authentication but with stricter rate limits (10 requests/day).

## Rate Limit Headers

All responses include rate limit information:

```
X-RateLimit-Limit: 50
X-RateLimit-Remaining: 47
X-RateLimit-Reset: 1704643200
```

### Rate Limit Exceeded Response
```json
{
  "detail": {
    "error": "Rate limit exceeded",
    "current_usage": 50,
    "limit": 50,
    "tier": "free",
    "message": "You've made 50/50 requests today. Upgrade to PRO for higher limits."
  }
}
```

## Testing

Run the test suite:
```bash
# Start the server
uvicorn app.main:app --reload --port 7860

# In another terminal, run tests
python test_auth_system.py
```

## Environment Variables

Add to your `.env` file:

```env
# JWT Secret (CHANGE THIS IN PRODUCTION!)
JWT_SECRET_KEY=your-super-secret-key-change-me-in-production

# Database (optional, defaults to SQLite)
DATABASE_URL=sqlite:///./data/stratax.db
# DATABASE_URL=postgresql://user:pass@localhost/stratax  # For PostgreSQL
```

## Upgrading to PostgreSQL (Production)

1. Install PostgreSQL driver:
```bash
pip install psycopg2-binary
```

2. Update `DATABASE_URL` in `.env`:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/stratax
```

3. Database will auto-migrate on startup

## Upgrading to Redis (Production)

For distributed systems, use Redis for rate limiting:

1. Install Redis client:
```bash
pip install redis
```

2. Update `app/middleware/rate_limit.py` to use Redis instead of in-memory storage

## Security Checklist

- ✅ Passwords hashed with bcrypt
- ✅ JWT tokens with expiration (7 days)
- ✅ Protected routes require authentication
- ✅ Rate limiting prevents abuse
- ⚠️ **CHANGE `JWT_SECRET_KEY` in production!**
- ⚠️ Use HTTPS in production
- ⚠️ Enable CORS restrictions for production domains

## Next Steps

1. ✅ **Authentication** - DONE!
2. ✅ **Rate Limiting** - DONE!
3. 🔜 **Usage Tracking** - Track API usage for billing
4. 🔜 **Stripe Integration** - Accept payments
5. 🔜 **Email Verification** - Send verification emails
6. 🔜 **Password Reset** - Forgot password flow

---

**Your app is now protected and ready for production! 🚀**
