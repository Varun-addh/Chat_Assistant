# STRATAX AI — Complete Upgrade Plan
### From Solo Prototype → B2B University Platform

**Prepared:** March 2026
**Version:** 1.0
**Classification:** Internal Strategy Document

---

## TABLE OF CONTENTS

1. [Current State Audit](#1-current-state-audit)
2. [What's Missing for B2B](#2-whats-missing-for-b2b)
3. [Phase 0 — Foundation Fixes (Week 1–2)](#3-phase-0--foundation-fixes)
4. [Phase 1 — Multi-Tenancy Core (Week 3–6)](#4-phase-1--multi-tenancy-core)
5. [Phase 2 — Admin Dashboard (Week 5–8)](#5-phase-2--admin-dashboard)
6. [Phase 3 — Billing System (Week 7–10)](#6-phase-3--billing-system)
7. [Phase 4 — Production Deployment (Week 9–12)](#7-phase-4--production-deployment)
8. [Phase 5 — Compliance & Enterprise Readiness (Week 11–16)](#8-phase-5--compliance--enterprise-readiness)
9. [Phase 6 — Sales-Ready Package (Week 13–16)](#9-phase-6--sales-ready-package)
10. [Revised Revenue Projections](#10-revised-revenue-projections)
11. [Budget Estimate](#11-budget-estimate)
12. [Priority Decision: What to Build First](#12-priority-decision-what-to-build-first)

---

## 1. CURRENT STATE AUDIT

### What Is Already Built (Genuinely Impressive for a Solo Developer)

| Component | Status | Quality |
|---|---|---|
| FastAPI backend with 90+ endpoints | BUILT | Production-grade |
| JWT auth + Google OAuth + Email verification | BUILT | Production-grade |
| AI Copilot Chat with Gemini + Groq | BUILT | Solid |
| Mock Interview engine (Text + Voice) | BUILT | Good |
| Practice Mode with speech analytics | BUILT | Good |
| Deterministic scoring (4 dimensions) | BUILT | Well-engineered |
| Resume parsing + LLM-based probing | BUILT | Solid |
| Interview Intelligence (semantic search via Qdrant) | BUILT | Solid |
| System design diagram generation (Mermaid) | BUILT | Good |
| Code execution via Judge0/Piston | BUILT | Functional |
| Tiered rate limiting + quota system | BUILT | Production-grade |
| Docker + docker-compose (5-service stack) | BUILT | Good |
| Alembic database migrations | BUILT | Solid |
| 73 test files (unit + integration) | BUILT | Respectable |
| Sentry error tracking | BUILT | Production-grade |
| React 18 + TypeScript + Vite frontend | SEPARATE REPO | Functional |

---

### Critical Technical Debt Found

| Issue | File Location | Severity |
|---|---|---|
| Mock interview sessions stored in JSON file on disk | `mock_interview_service.py` | CRITICAL |
| WebSocket STT endpoint echoes "(audio)" — is a stub | `ws/stt/{session_id}` | CRITICAL |
| CI lint + security checks are `continue-on-error: true` | `.github/workflows/ci.yml` | HIGH |
| Stripe billing is a dead column — zero implementation | `models.py:37` | HIGH |
| No request body size limits (DoS vector) | `app/main.py` | HIGH |
| No LICENSE file (README claims MIT) | Root directory | HIGH |
| Email sending disabled by default | `config.py` | HIGH |
| No `requirements.lock` — builds not reproducible | Root directory | MEDIUM |
| Multi-tenancy: completely absent | Entire codebase | BLOCKING |
| Admin dashboard: completely absent | Entire codebase | BLOCKING |

---

## 2. WHAT'S MISSING FOR B2B

The business plan targets universities at $15/student/year. Here is every feature a university placement officer would demand — and current status:

| University Requirement | Current Status |
|---|---|
| "Show me my 3000 students' progress dashboard" | DOES NOT EXIST |
| "Can we group students by batch/year?" | No cohort system |
| "How do we onboard 3000 students at once?" | No bulk invite |
| "What's our department-level completion rate?" | No org analytics |
| "We use college SSO — does it integrate?" | Google OAuth only |
| "Is our student data FERPA compliant?" | No compliance docs |
| "What's your SLA and uptime guarantee?" | HuggingFace Spaces |
| "Can we brand it with our college logo?" | No white-labeling |
| "How do I pay? Can we get an invoice?" | No billing system |
| "We need a data processing agreement" | No legal docs |

**Bottom line:** You cannot onboard a single paying university today.

---

## 3. PHASE 0 — FOUNDATION FIXES (Week 1–2)

**Goal:** Make the existing product production-stable. No new features — fix what's broken.
**Cost:** $0. Pure code work.

---

### Task 0.1 — Migrate Mock Interview Sessions to Database

**Problem:** `MockInterviewService` stores all active sessions in `data/sessions/mock_interview_sessions.json`. Breaks with more than 1 Uvicorn worker or any horizontal scaling.

**Fix — Add to `app/models.py`:**

```python
class MockInterviewSession(Base):
    __tablename__ = "mock_interview_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    status = Column(String, default="active")  # active, paused, completed
    questions = Column(JSON, default=list)
    answers = Column(JSON, default=list)
    evaluations = Column(JSON, default=list)
    config = Column(JSON, default=dict)       # role, difficulty, interview_type
    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True),
                        onupdate=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)
    user = relationship("User")
```

Replace all `self.active_sessions[session_id]` read/writes with SQLAlchemy queries.

**Files to modify:**
- `app/models.py`
- `app/services/interview/mock_interview_service.py`
- Create new Alembic migration

---

### Task 0.2 — Fix CI/CD Security Gates

**Problem:** Both lint and security CI jobs have `continue-on-error: true` → security vulnerabilities are silently ignored.

**Fix in `.github/workflows/ci.yml`:**

```yaml
- name: Run security scan (bandit)
  run: bandit -r app/ -ll
  # DELETE the line: continue-on-error: true

- name: Run ruff lint
  run: ruff check app/
  # DELETE the line: continue-on-error: true
```

---

### Task 0.3 — Add Request Body Size Limits

**Problem:** A student can POST a 500MB+ file to any endpoint (DoS vector).

**Fix in `app/main.py`:**

```python
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    max_size = 10 * 1024 * 1024  # 10MB global limit
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_size:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request body too large. Max 10MB."}
        )
    return await call_next(request)
```

---

### Task 0.4 — Add LICENSE File

Create `LICENSE` in the root directory with the MIT License text (since README already claims MIT).

---

### Task 0.5 — Fix or Remove WebSocket STT Stub

**Problem:** `/ws/stt/{session_id}` echoes "(audio)" and does nothing real. Advertised feature that fails silently.

**Option A — Implement it:** Wire to the existing `faster-whisper` STT service already built in `practice_mode.py`.

**Option B — Disable it:** Remove the endpoint and frontend references until properly implemented.

---

### Task 0.6 — Pin All Dependencies

```bash
pip freeze > requirements.lock
```

Add to CI:
```yaml
- name: Verify locked dependencies
  run: pip install -r requirements.lock --dry-run
```

---

### Task 0.7 — Enable Email in Production

Update `.env.example` with SMTP defaults and add a production startup warning if email is disabled.

---

## 4. PHASE 1 — MULTI-TENANCY CORE (Week 3–6)

**Goal:** A university admin can create an organization, invite students, and see grouped data.
**This is the single most important phase. Everything else depends on it.**

---

### New Database Models (add to `app/models.py`)

```python
class OrgType(str, Enum):
    UNIVERSITY = "university"
    BOOTCAMP = "bootcamp"
    EDTECH = "edtech"
    RECRUITMENT = "recruitment"

class OrgTier(str, Enum):
    BASIC = "basic"        # $8/student/year
    STANDARD = "standard"  # $15/student/year
    PREMIUM = "premium"    # $25/student/year

class OrgRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    INSTRUCTOR = "instructor"
    STUDENT = "student"


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False, index=True)
    type = Column(String, default=OrgType.UNIVERSITY)
    logo_url = Column(String, nullable=True)
    website = Column(String, nullable=True)

    # Subscription
    tier = Column(String, default=OrgTier.STANDARD)
    max_students = Column(Integer, default=100)
    stripe_customer_id = Column(String, nullable=True)
    stripe_subscription_id = Column(String, nullable=True)
    subscription_start = Column(DateTime(timezone=True), nullable=True)
    subscription_end = Column(DateTime(timezone=True), nullable=True)

    # Config
    feature_flags = Column(JSON, default=dict)
    data_retention_days = Column(Integer, nullable=True)
    primary_contact_email = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True),
                        onupdate=lambda: datetime.now(timezone.utc))

    members = relationship("OrganizationMember", back_populates="organization")
    cohorts = relationship("Cohort", back_populates="organization")


class OrganizationMember(Base):
    __tablename__ = "organization_members"
    __table_args__ = (UniqueConstraint("organization_id", "user_id"),)

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"),
                              nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    role = Column(String, default=OrgRole.STUDENT)
    status = Column(String, default="active")  # active, invited, suspended
    invited_by = Column(String, ForeignKey("users.id"), nullable=True)
    invite_token_hash = Column(String, nullable=True)
    joined_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc))

    organization = relationship("Organization", back_populates="members")
    user = relationship("User", foreign_keys=[user_id])


class Cohort(Base):
    __tablename__ = "cohorts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"),
                              nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc))

    organization = relationship("Organization", back_populates="cohorts")
    members = relationship("CohortMember", back_populates="cohort")


class CohortMember(Base):
    __tablename__ = "cohort_members"
    __table_args__ = (UniqueConstraint("cohort_id", "user_id"),)

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    cohort_id = Column(String, ForeignKey("cohorts.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    enrolled_at = Column(DateTime(timezone=True),
                         default=lambda: datetime.now(timezone.utc))

    cohort = relationship("Cohort", back_populates="members")
    user = relationship("User")
```

### Modify Existing Models

```python
# User model — ADD:
organization_id = Column(String, ForeignKey("organizations.id"),
                         nullable=True, index=True)

# PracticeAttemptRecord — ADD:
organization_id = Column(String, ForeignKey("organizations.id"),
                         nullable=True, index=True)
cohort_id = Column(String, ForeignKey("cohorts.id"),
                   nullable=True, index=True)

# SessionRecord — ADD:
organization_id = Column(String, ForeignKey("organizations.id"),
                         nullable=True, index=True)
```

---

### New API Routes (`app/routers/org_routes.py`)

```
# Organization CRUD
POST   /api/org                                       Create organization
GET    /api/org/{slug}                                Get org details
PUT    /api/org/{slug}                                Update org
DELETE /api/org/{slug}                                Delete org (owner only)

# Member Management
POST   /api/org/{slug}/invite                         Invite (single or bulk CSV)
GET    /api/org/{slug}/members                        List members (paginated)
PUT    /api/org/{slug}/members/{user_id}              Update member role
DELETE /api/org/{slug}/members/{user_id}              Remove member
GET    /api/org/{slug}/invite/{token}                 Accept invite

# Cohort Management
POST   /api/org/{slug}/cohorts                        Create cohort
GET    /api/org/{slug}/cohorts                        List cohorts
GET    /api/org/{slug}/cohorts/{cohort_id}            Get cohort
PUT    /api/org/{slug}/cohorts/{cohort_id}            Update cohort
DELETE /api/org/{slug}/cohorts/{cohort_id}            Delete cohort
POST   /api/org/{slug}/cohorts/{cohort_id}/enroll     Enroll students (bulk)
DELETE /api/org/{slug}/cohorts/{cohort_id}/students/{user_id}

# Analytics
GET    /api/org/{slug}/analytics/overview             Org-wide dashboard stats
GET    /api/org/{slug}/analytics/students             Per-student performance table
GET    /api/org/{slug}/analytics/cohort/{cohort_id}   Cohort-level analytics
GET    /api/org/{slug}/analytics/export               Export as CSV
```

---

### Data Isolation — CRITICAL RULE

Every query touching student data must be org-scoped:

```python
# WRONG — returns data across all organizations:
db.query(PracticeAttemptRecord).filter_by(user_id=user_id)

# CORRECT — tenant fence enforced:
db.query(PracticeAttemptRecord).filter(
    PracticeAttemptRecord.organization_id == current_user.organization_id,
    PracticeAttemptRecord.user_id == user_id
)
```

---

### Analytics Response Shapes

**`GET /api/org/{slug}/analytics/overview`**
```json
{
  "total_students": 3200,
  "active_students_7d": 847,
  "active_students_30d": 1923,
  "total_sessions_completed": 12440,
  "average_score": 71.3,
  "score_trend": "+4.2 vs last month",
  "completion_rate": "68%",
  "top_weak_areas": ["System Design", "Dynamic Programming", "Behavioral STAR"],
  "cohort_summary": [
    {"name": "CS Batch 2026", "students": 450, "avg_score": 73.1, "sessions": 2100},
    {"name": "CS Batch 2025", "students": 380, "avg_score": 78.4, "sessions": 3200}
  ]
}
```

**`GET /api/org/{slug}/analytics/students`**
```json
{
  "students": [
    {
      "user_id": "uuid",
      "name": "Student Name",
      "cohort": "CS Batch 2026",
      "sessions_completed": 14,
      "average_score": 76.2,
      "score_trend": "improving",
      "last_active": "2026-03-12",
      "weak_dimension": "delivery",
      "dimensions": {
        "correctness": 81,
        "delivery": 62,
        "clarity": 74,
        "structure": 79
      }
    }
  ],
  "pagination": {"page": 1, "total": 3200, "per_page": 50}
}
```

---

## 5. PHASE 2 — ADMIN DASHBOARD (Week 5–8)

**Goal:** University placement officer has a usable web interface to manage their institution.
**Frontend work (separate repo). Can overlap with Phase 1 backend.**

---

### Pages Required

#### Page 1 — Overview Dashboard (`/admin`)

**4 KPI Cards (top row):**
- Total Students enrolled
- Active This Month (with percentage)
- Avg Practice Score (with trend arrow)
- Sessions Completed This Month

**Charts (row 2):**
- Left: Line chart — average score over time (6 months)
- Right: Radar chart — dimension averages (correctness, delivery, clarity, structure)

**Cohort table (row 3):**

| Cohort Name | Students | Avg Score | Sessions | Completion Rate | Actions |
|---|---|---|---|---|---|
| CS Batch 2026 | 450 | 73.1 | 2100 | 67% | View |

---

#### Page 2 — Student Roster (`/admin/students`)

**Filterable, sortable table:**

| Column | Description |
|---|---|
| Name | Full name |
| Email | Email address |
| Cohort | Assigned cohort |
| Sessions | Total completed |
| Avg Score | Overall score 0-100 |
| Trend | Up/Down/Stable arrow |
| Last Active | Days since last session |
| Actions | View, Move, Suspend |

**Filters:** By cohort, score range, date range, activity status

**Export button:** Download as CSV

---

#### Page 3 — Student Detail (`/admin/students/:userId`)

- Profile card: name, email, cohort, member since, last active
- Score progression line chart (all sessions)
- Dimension radar chart (current averages)
- Practice frequency heatmap (GitHub-style, last 6 months)
- Session history table

---

#### Page 4 — Cohort Management (`/admin/cohorts`)

- Create cohort form
- List all cohorts with quick stats
- Bulk import students via CSV upload
- Assign students to cohorts

---

#### Page 5 — Settings (`/admin/settings`)

- Organization profile (name, logo, website, contact)
- Feature toggles per org
- Member role management
- Data retention settings
- Danger zone (delete organization)

---

### Frontend Tech (Use Existing Stack)

| Library | Use | Already In Stack? |
|---|---|---|
| React 18 + TypeScript | Framework | YES |
| Tailwind CSS | Styling | YES |
| Chart.js / react-chartjs-2 | Analytics charts | YES |
| Zustand | Admin state | YES |
| TanStack Table v8 | Student roster table | ADD — install `@tanstack/react-table` |
| react-dropzone | CSV bulk import | ADD — install `react-dropzone` |

---

## 6. PHASE 3 — BILLING SYSTEM (Week 7–10)

**Goal:** Can generate real invoices and process payments.

---

### Stripe Integration (Global Markets)

**Step 1: Create Stripe Products**

In the Stripe dashboard, create 3 products with per-seat yearly pricing:
- Basic: $8.00/student/year
- Standard: $15.00/student/year
- Premium: $25.00/student/year

**Step 2: Install**
```bash
pip install stripe
```

**Step 3: New Router (`app/routers/billing_routes.py`)**

```python
# Checkout — creates a Stripe Checkout Session
@router.post("/billing/checkout")
async def create_checkout(org_slug: str, tier: str, student_count: int,
                          current_user: User, db: Session):
    org = get_org_or_403(org_slug, current_user, db, required_role="owner")
    price_id = STRIPE_PRICE_IDS[tier]

    session = stripe.checkout.Session.create(
        customer=org.stripe_customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": student_count}],
        success_url=f"{settings.frontend_url}/admin/billing?success=true",
        cancel_url=f"{settings.frontend_url}/admin/billing?cancelled=true",
        metadata={"org_id": org.id}
    )
    return {"checkout_url": session.url}


# Webhook — handles Stripe events
@router.post("/billing/webhook")
async def stripe_webhook(request: Request, db: Session):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig, settings.stripe_webhook_secret
        )
    except Exception:
        raise HTTPException(400, "Invalid webhook signature")

    handlers = {
        "checkout.session.completed": activate_subscription,
        "invoice.payment_failed": handle_payment_failure,
        "customer.subscription.deleted": deactivate_subscription,
    }

    handler = handlers.get(event["type"])
    if handler:
        handler(event["data"]["object"], db)

    return {"received": True}
```

**All billing routes:**
```
POST   /api/billing/checkout       Create Stripe Checkout Session
POST   /api/billing/webhook        Stripe event handler (MUST be public, no auth)
GET    /api/billing/portal         Get Stripe Customer Portal URL
GET    /api/billing/usage          Current period active student count
GET    /api/billing/invoices       Invoice history from Stripe
```

---

### Razorpay Integration (India)

For Indian universities — Razorpay is more common than Stripe and supports UPI, netbanking, NEFT.

```bash
pip install razorpay
```

```python
@router.post("/billing/razorpay/order")
async def create_razorpay_order(org_slug: str, tier: str, student_count: int,
                                 current_user: User):
    client = razorpay.Client(auth=(settings.razorpay_key_id,
                                    settings.razorpay_key_secret))
    price_inr = INR_PRICES[tier] * student_count  # in rupees

    order = client.order.create({
        "amount": price_inr * 100,  # convert to paise
        "currency": "INR",
        "notes": {"org_slug": org_slug, "tier": tier, "students": student_count}
    })
    return {"order_id": order["id"], "amount": price_inr, "currency": "INR"}
```

---

### Pricing Reference

| Tier | USD (Global) | INR (India) |
|---|---|---|
| Basic | $8/student/year | Rs 600/student/year |
| Standard | $15/student/year | Rs 1,200/student/year |
| Premium | $25/student/year | Rs 2,000/student/year |

---

### Quota Enforcement After Billing

```python
# app/middleware/quota.py — NEW FILE

async def check_org_quota(org: Organization, db: Session) -> bool:
    # Check subscription expiry
    if org.subscription_end and org.subscription_end < datetime.now(timezone.utc):
        return False  # Subscription expired

    # Check student count limit
    active_count = db.query(OrganizationMember).filter_by(
        organization_id=org.id, status="active"
    ).count()

    if active_count > org.max_students:
        return False  # Over seat limit

    return True
```

---

## 7. PHASE 4 — PRODUCTION DEPLOYMENT (Week 9–12)

**Goal:** Move off HuggingFace Spaces. Reliable hosting with actual uptime guarantees.

---

### Recommended Platform: Railway.app (MVP Phase)

**Why Railway:**
- Docker-based deploy (Dockerfile already ready)
- Managed PostgreSQL + Redis included
- Auto-deploy from GitHub push
- Built-in staging environments
- $50-150/month for MVP with 10 universities
- No Kubernetes complexity

**Setup:**

```bash
npm install -g @railway/cli
railway login
railway init
railway add postgresql
railway add redis
railway variables set JWT_SECRET_KEY="$(openssl rand -hex 32)"
railway up
```

---

### Add Full CD Pipeline to CI/CD

```yaml
# .github/workflows/ci.yml — ADD:

deploy-staging:
  name: Deploy to Staging
  needs: [test, lint, security]
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/main'
  steps:
    - uses: actions/checkout@v4
    - name: Deploy to Railway Staging
      uses: bervproject/railway-deploy@latest
      with:
        railway-token: ${{ secrets.RAILWAY_TOKEN }}
        service: stratax-staging
    - name: Smoke test staging
      run: |
        sleep 30
        curl -f https://staging.stratax.ai/health || exit 1

deploy-production:
  name: Deploy to Production
  needs: [deploy-staging]
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/release'
  environment: production  # Requires manual approval
  steps:
    - name: Deploy to Railway Production
      uses: bervproject/railway-deploy@latest
      with:
        railway-token: ${{ secrets.RAILWAY_TOKEN }}
        service: stratax-production
```

---

### Add Prometheus Metrics

```bash
pip install prometheus-fastapi-instrumentator
```

```python
# app/main.py — ADD:
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

---

### Production Infrastructure Architecture

```
Internet
    |
    v
Cloudflare (CDN + DDoS protection, free tier)
    |
    v
Load Balancer (Railway built-in)
    |
    +------------------+------------------+
    |                  |                  |
    v                  v                  v
App Replica 1    App Replica 2    App Replica 3
(FastAPI, 2w)   (FastAPI, 2w)   (FastAPI, 2w)
    |
    +------------------+------------------+-------------------+
    |                  |                  |                   |
    v                  v                  v                   v
PostgreSQL          Redis            Qdrant Cloud        S3/R2 Storage
(Managed)          (Managed)        (Free tier up        (Resumes,
                                    to 1M vectors)        media files)
```

---

### Monthly Cost Estimate (10 Universities, 30K Students)

| Service | Cost/Month |
|---|---|
| Railway (3 app replicas, 2 vCPU each) | $50-80 |
| Managed PostgreSQL | $20 |
| Managed Redis | $10 |
| Qdrant Cloud | $0 (free tier) |
| Cloudflare R2 storage | $5-15 |
| LLM APIs (Gemini + Groq) | $300-800 |
| Cloudflare CDN | $0 (free plan) |
| **Total** | **$385-925/month** |

---

## 8. PHASE 5 — COMPLIANCE & ENTERPRISE READINESS (Week 11–16)

This is what university legal teams will scrutinize.

---

### Priority 1: Legal Documents (Week 11–12)

Create and publish these at your domain:

| Document | URL | Notes |
|---|---|---|
| Privacy Policy | `/privacy` | Cover student data, LLM providers, retention |
| Terms of Service | `/terms` | Usage rights, liabilities |
| Data Processing Agreement | Template PDF | For signing with each university |
| Acceptable Use Policy | `/aup` | What students may/may not do |

**Critical disclosure in Privacy Policy:**
"Student responses are processed by Google Gemini API and Groq API. These providers do not train on API data by default. Student email and name are not transmitted to third-party LLM providers."

---

### Priority 2: FERPA Awareness (Week 12)

FERPA protects student education records in the US.

**Code changes:**

```python
# Add to student data endpoints:
"""
FERPA NOTICE: This endpoint returns student education records.
Access is restricted to authorized school officials with legitimate
educational interest per FERPA 34 CFR § 99.31(a)(1).
"""

# Ensure org-scoped access only (Phase 1 already covers this)
```

**Frontend change — add consent checkbox to registration when joining via org invite:**

```typescript
<label className="flex items-start gap-2">
  <input type="checkbox" required />
  <span>
    I consent to my interview practice session data being shared with{' '}
    <strong>{organization.name}</strong> placement officials
    for academic placement purposes.
  </span>
</label>
```

---

### Priority 3: Data Rights Controls (Week 13)

**Admin controls:**
- Set auto-delete after N days (GDPR retention minimization)
- Export all student data as CSV (GDPR portability)
- Delete individual student's data (GDPR erasure / CCPA)

**Scheduled task for auto-deletion:**

```python
# app/tasks/cleanup.py
async def cleanup_expired_data():
    """Run nightly via APScheduler."""
    orgs = db.query(Organization).filter(
        Organization.data_retention_days.isnot(None)
    ).all()

    for org in orgs:
        cutoff = datetime.now(timezone.utc) - timedelta(days=org.data_retention_days)
        db.query(PracticeAttemptRecord).filter(
            PracticeAttemptRecord.organization_id == org.id,
            PracticeAttemptRecord.created_at < cutoff
        ).delete()
        db.commit()
```

---

### Priority 4: SSO Integration (Week 14–15)

Universities use SAML/Shibboleth — Google OAuth alone won't satisfy enterprise procurement.

**Recommended approach — WorkOS (fastest to implement):**

```bash
pip install workos
```

```python
@router.get("/auth/sso/initiate")
async def initiate_sso(organization_id: str):
    """Redirect to university's IdP via WorkOS."""
    url = workos_client.sso.get_authorization_url(
        organization=organization_id,
        redirect_uri=f"{settings.backend_base_url}/auth/sso/callback"
    )
    return RedirectResponse(url)

@router.get("/auth/sso/callback")
async def sso_callback(code: str, db: Session):
    """Handle IdP callback, create or link user account."""
    profile = workos_client.sso.get_profile_and_token(code)
    # Find or create user, set organization_id, issue JWT
```

**Cost:** WorkOS AuthKit free up to 1M MAU.

---

### Priority 5: SOC 2 (Month 6+, Not Immediate)

Do NOT spend $30K on SOC 2 before you have 3 paying customers. Start when:
- You have 3+ paying universities
- You have $50K+ in revenue or funding
- Use **Vanta** ($900/month) or **Drata** ($1,200/month) for automated evidence collection

---

### Compliance Checklist Before First University Pilot

```
[ ] Privacy Policy published at stratax.ai/privacy
[ ] Terms of Service published at stratax.ai/terms
[ ] DPA template ready to share with universities
[ ] Student consent checkbox added to org invite flow
[ ] Admin can configure data retention period
[ ] Admin can export all student data (CSV)
[ ] Admin can delete individual student data
[ ] No student PII in plain-text logs (JSONL auditor already strips PII - confirmed)
[ ] LLM provider data processing terms reviewed and documented
[ ] CORS configured to specific domain(s) in production
```

---

## 9. PHASE 6 — SALES-READY PACKAGE (Week 13–16)

The best product in the world sells nothing without these assets.

---

### Required Assets Before Cold-Emailing University #1

| Asset | Priority | Notes |
|---|---|---|
| Live demo server at `demo.stratax.ai` | MUST HAVE | Dedicated instance, not HuggingFace. With seed data (fake university, 3 cohorts, 50 students with realistic scores). |
| 3-minute product video (Loom) | MUST HAVE | Admin creates org → invites students → student does mock interview → admin sees analytics. |
| One-pager PDF | MUST HAVE | Problem / Solution / Features / Pricing / Contact. One page. Placement officers won't read more. |
| Free 30-day pilot offer | MUST HAVE | "No credit card. 50 students. See results." |
| Case study (post-pilot) | Before paid contracts | Real numbers from your pilot university. Even 50 students' data is enough. |
| ROI calculator (simple spreadsheet) | Helpful | "Your 2000 students × Rs 1200 = Rs 24L revenue. Placement rate improvement." |

---

### Demo Seed Data Script

Before any demo, the demo server must have realistic data:

```python
# scripts/seed_demo_data.py — creates:
# - 1 organization: "Demo University"
# - 3 cohorts: "CS Batch 2026", "CS Batch 2025", "AI/ML Specialization"
# - 50 students each with:
#   - 5-20 completed practice sessions
#   - Realistic scores (avg 55-85, distributed naturally)
#   - Some students improving, some declining (realistic trajectories)
#   - Mixed dimension strengths (some strong in correctness, weak in delivery)
# - 2 faculty instructor accounts

# An empty demo is a deal-killer.
```

---

### Cold Email Template for Indian Universities

```
Subject: Free AI mock interview platform pilot for [College Name] students

Hi [Name],

I'm [Your Name], building Stratax AI — an AI-powered interview simulation
platform built specifically for campus placement preparation.

We're offering a completely free 30-day pilot for [College Name]:
  - 50 students get full access (no cost, no commitment)
  - Real-time AI feedback on technical + behavioral answers
  - Speech analytics (filler words, pace, confidence scoring)
  - Dashboard so you can track placement readiness across batches
  - Works on any device, no installation needed

We've already built: resume-probing interviews, voice-based practice mode,
scoring across 4 dimensions, and cohort-level analytics for TPOs.

Would you be open to a 15-minute demo this week?

[Your Name]
[Phone]
[demo.stratax.ai]
```

---

### Target Outreach List — India First

| Institution Type | Target Person | Channel |
|---|---|---|
| IITs / NITs | Training & Placement Officer (TPO) | LinkedIn + Email |
| Tier-2 Engineering Colleges | Dean of Placements | LinkedIn DM |
| Coding bootcamps (Masai, Newton, Scaler) | Head of Curriculum / Co-founder | LinkedIn DM |
| PGDM/MBA colleges | Placement Coordinator | LinkedIn + Email |
| EdTech platforms | Business Development Lead | LinkedIn |

---

## 10. REVISED REVENUE PROJECTIONS

### Year 1 — Honest Scenario

| Milestone | When | Revenue |
|---|---|---|
| Phase 0-2 complete, product ready | Month 1-2 | Rs 0 |
| Free pilot #1 launched | Month 3-4 | Rs 0 |
| Free pilots 2-3 (build case studies) | Month 4-5 | Rs 0 |
| First paying university (discounted: Rs 800/student) | Month 6-8 | Rs 2-6L |
| 2-3 more paying universities | Month 9-12 | Rs 5-15L |
| **Year 1 total** | | **Rs 7-21L ($8K-25K)** |

---

### Year 2 — Realistic Growth

| Quarter | Universities | Students | Avg Price | Revenue |
|---|---|---|---|---|
| Q1 Y2 | 5 | 5,000 | Rs 1000 | Rs 50L |
| Q2 Y2 | 8 | 12,000 | Rs 1100 | Rs 132L |
| Q3 Y2 | 12 | 20,000 | Rs 1200 | Rs 240L |
| Q4 Y2 | 15 | 28,000 | Rs 1200 | Rs 336L |
| **Year 2 total** | | | | **Rs 758L (~Rs 7.5 Crore)** |

---

### The $450K / Rs 3.7 Crore Target

This IS achievable — but the timeline is Year 2, not Year 1. And only with:
- Multi-tenancy built (Phase 1)
- Admin dashboard that impresses placement officers (Phase 2)
- At least 3 pilot case studies (Month 5-6)
- 1 dedicated salesperson hired by Month 8
- Consistent outreach to 50-100 universities

---

## 11. BUDGET ESTIMATE

### Pre-Revenue Phase (Months 1–6)

| Item | Monthly | 6-Month Total |
|---|---|---|
| Railway hosting (staging + prod) | Rs 8,000 | Rs 48,000 |
| LLM APIs (Gemini, Groq — dev + demo) | Rs 4,000-8,000 | Rs 24,000-48,000 |
| Domain + Email (Cloudflare + Zoho) | Rs 400 | Rs 2,400 |
| Legal (Privacy Policy + DPA review) | One-time | Rs 25,000-50,000 |
| WorkOS for SSO (free tier) | Rs 0 | Rs 0 |
| Marketing materials (Canva Pro) | Rs 1,000 | Rs 6,000 |
| **Total 6-month burn** | | **Rs 1.05L-1.54L** |

---

### After First Revenue (Months 7–12)

| Item | Monthly Cost |
|---|---|
| Hosting (scaling with users) | Rs 15,000-40,000 |
| LLM APIs (3K active students) | Rs 20,000-50,000 |
| Salesperson (hire after Month 8) | Rs 60,000-80,000 |
| Misc tools + operations | Rs 10,000 |
| **Total monthly burn** | **Rs 1.05L-1.80L** |

**Total Year 1 investment: Approx Rs 8-12 Lakh (mostly hosting + LLMs + salesperson)**

---

## 12. PRIORITY DECISION: WHAT TO BUILD FIRST

If you can only focus on one thing at a time, follow this exact order:

```
WEEK 1-2 .... Phase 0 — Fix the existing bugs (0 new features)
WEEK 3-6 .... Phase 1 — Multi-tenancy: Organization + Cohort + Analytics APIs
WEEK 5-8 .... Phase 2 — Admin dashboard (can overlap with Phase 1 backend)
WEEK 7-10 ... Phase 3 — Billing: Stripe (global) + Razorpay (India)
WEEK 9-12 ... Phase 4 — Move to Railway, set up staging + CI/CD
WEEK 11-14 .. Phase 5 — Privacy Policy, DPA, FERPA, data retention controls
WEEK 13-16 .. Phase 6 — Demo server with seed data + outreach materials

MONTH 5-6 ... Launch first FREE university pilot (50-200 students)
MONTH 7-8 ... Collect real usage data, build case study
MONTH 8-9 ... Send first invoice to a paying university
MONTH 9-12 .. Close 3-5 paying contracts
YEAR 2 ....... Scale to 15 universities, hire salesperson, target Rs 7 Crore
```

---

### The Single Most Important Rule

> **Build multi-tenancy first. Everything else is secondary.**
>
> Without it:
> - Billing has no customer to bill
> - Analytics has no audience to show
> - Sales demos display an empty system
> - No university will evaluate your product seriously
>
> With it: Every other phase becomes possible.

---

## APPENDIX A — Key Files to Modify by Phase

| Phase | File | What Changes |
|---|---|---|
| Phase 0 | `app/services/interview/mock_interview_service.py` | Replace JSON file with DB queries |
| Phase 0 | `.github/workflows/ci.yml` | Remove `continue-on-error: true` |
| Phase 0 | `app/main.py` | Add body size limit middleware |
| Phase 0 | Root | Add `LICENSE` file |
| Phase 1 | `app/models.py` | Add Organization, OrganizationMember, Cohort, CohortMember |
| Phase 1 | `app/models.py` | Add `organization_id` to User, PracticeAttemptRecord, SessionRecord |
| Phase 1 | `app/routers/` | Add `org_routes.py` |
| Phase 1 | `alembic/` | New migration for multi-tenancy tables |
| Phase 1 | `app/main.py` | Register org_routes router |
| Phase 2 | Frontend repo | Add `/admin/*` route group with 5 pages |
| Phase 2 | Frontend repo | Install `@tanstack/react-table`, `react-dropzone` |
| Phase 3 | `app/routers/` | Add `billing_routes.py` |
| Phase 3 | `app/config.py` | Add Stripe + Razorpay keys to Settings |
| Phase 3 | `app/models.py` | Add billing columns to Organization |
| Phase 4 | `.github/workflows/ci.yml` | Add deploy-staging + deploy-production jobs |
| Phase 4 | Root | Add `railway.json` config |
| Phase 5 | Root | Add `PRIVACY.md`, `TERMS.md`, `DPA_TEMPLATE.md` |
| Phase 5 | `app/routers/auth_routes.py` | Add consent checkbox to org invite flow |
| Phase 5 | `app/models.py` | Add `data_retention_days` to Organization |
| Phase 5 | `app/tasks/` | Add `cleanup.py` for data retention enforcement |

---

## APPENDIX B — Architecture After Full Upgrade

```
+=========================================================+
|                  STRATAX AI v2.0                        |
+==========================+==============================+
|    STUDENT PORTAL        |      ADMIN PORTAL           |
|                          |                             |
|  /chat                   |  /admin/overview            |
|  /mock-interview         |  /admin/students            |
|  /practice               |  /admin/cohorts             |
|  /interview-intelligence |  /admin/analytics           |
|  /architecture           |  /admin/settings            |
|                          |  /admin/billing             |
+==========================+==============================+
                           |
              FastAPI Backend (v2 — Multi-tenant)
                           |
       +-------------------+-------------------+
       |                   |                   |
  Auth Service        Org Service         AI Services
  (JWT + SSO +      (Multi-tenant:      (LLM + STT +
  WorkOS SAML)      org, cohort,         TTS, Scoring)
                    analytics)
       |                   |                   |
       +-------------------+-------------------+
                           |
       +-------------------+-------------------+
       |                   |                   |
  PostgreSQL            Redis             Qdrant Cloud
  (All structured      (Rate limits       (Interview
   data)               + caching)         Intelligence)
                           |
                    S3/R2 Storage
                (Resumes + Media files)
```

---

*Stratax AI Upgrade Plan v1.0 — March 2026*
*This document is based on a full codebase audit of the InterviewAst repository.*
*All file paths, model names, and endpoint names are verified against actual source code.*
