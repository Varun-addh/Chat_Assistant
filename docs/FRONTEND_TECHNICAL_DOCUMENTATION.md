# Frontend Technical Documentation (InterviewAstfe / Stratax AI)

This document describes the frontend (Vite + React + TypeScript) codebase: how it boots, routes, authenticates, talks to the backend, and implements core features (Interview Assistant, Code Runner, Mock Interview, Architecture Generator), plus PWA/service worker and icon pipeline.

---

## 1) Tech Stack

- **Runtime/UI**: React 18, TypeScript, Vite
- **Styling**: TailwindCSS + shadcn/ui (Radix primitives)
- **Animation**: framer-motion
- **State/data**:
  - Local React state for most UI and flows
  - TanStack Query provider is wired in, but much of the fetching is direct `fetch()` wrappers.
- **PWA**: Web manifest + custom service worker (`public/sw.js`)
- **Hosting**: Firebase Hosting (serves `dist/`, SPA rewrite to `index.html`)

---

## 2) Repo Layout (Frontend-Relevant)

- `index.html`: Vite entry HTML (includes meta theme color / icons)
- `public/`
  - `manifest.webmanifest`: PWA manifest
  - `sw.js`: service worker (cache + offline)
  - `icons/`: generated icons + source
- `src/`
  - `main.tsx`: app bootstrap + providers + SW registration
  - `App.tsx`: router + global modals/providers
  - `pages/`: top-level route pages
  - `components/`: feature components (assistant, runner, auth, etc.)
  - `context/`: React contexts (auth)
  - `lib/`: API clients, backend wrappers, utilities, feature-specific API modules
  - `hooks/`: custom hooks (toast, speech recognition, theme, etc.)
- `scripts/`: build-time tooling (icon generation)

---

## 3) Application Bootstrap

### Entry point

- The application starts in `src/main.tsx`.
- Providers/wrappers:
  - `AuthProvider` provides user + token state and auth actions.
  - `ErrorBoundary` wraps the UI to prevent hard crashes.
  - `App` renders routing and global UI.
  - `EvaluationOverlayHost` mounts overlay UI for evaluation features.

### Service worker registration

`src/main.tsx` registers `public/sw.js` on `window.load` **only in production**. In development, it proactively unregisters SWs to avoid stale cache issues.

---

## 4) Routing

Defined in `src/App.tsx` using `react-router-dom`.

### Routes

- `/` → Landing page (`src/pages/Index.tsx`)
- `/login` → Auth page (`src/pages/Auth.tsx`) wrapped with `ProtectedRoute requireAuth={false}`
- `/auth/google/callback` → OAuth callback page (`src/pages/GoogleCallback.tsx`)
- `/app` → Main assistant UI (`src/components/InterviewAssistant.tsx`)
- `/run` → Code Runner page (`src/pages/Runner.tsx`) guarded by `ProtectedRoute`
- `/architecture` → Architecture generator page (`src/pages/Architecture.tsx`) guarded by `ProtectedRoute`
- `/progress` → Progress page (`src/pages/Progress.tsx`) guarded by `ProtectedRoute`
- `*` → Not Found (`src/pages/NotFound.tsx`)

### Route guards

`src/components/ProtectedRoute.tsx`:
- Shows a loading spinner while auth state is being resolved.
- If `requireAuth=true` and there’s no user → redirects to `/login`.
- If `requireAuth=false` and user exists → redirects to `/app`.

---

## 5) Authentication

### Auth state

`src/context/AuthContext.tsx`:
- Stores:
  - `user` (id, email, full_name, tier)
  - `token` (JWT from `localStorage`)
  - `loading` (bootstrapping state)
- Key behaviors:
  - On mount/token change: if `token` exists it calls `/auth/me`.
  - Login/register store JWT and basic user identifiers in `localStorage`.
  - Logout clears `token`, `user`, and storage keys.

### OAuth (Google)

- `loginWithGoogle()` opens a popup to `${API_BASE_URL}/auth/google`.
- Popup returns to `/auth/google/callback` which posts a `window.postMessage` back to the opener.
- On success: token/user fields are stored and `user` is set.

### API base URL alignment

AuthContext uses:
- `VITE_AUTH_API_URL` if set, else falls back to `STRATAX_API_BASE_URL`.

This matters because JWT must be recognized by the same backend you use for `/api/*` endpoints.

### Global logout listener

`src/lib/authHelpers.ts` provides `setupAuthListener()` which listens for `auth:logout` events and redirects to `/login`. It is invoked in `src/main.tsx`.

---

## 6) API Layer

There are **two main styles** of backend calling:

1) **Unified fetch wrapper** (preferred / newer): `src/lib/strataxClient.ts`
2) **Auth-specific helper**: `src/lib/authApi.ts` + `src/lib/authHelpers.ts`

### 6.1 `strataxClient.ts` (unified client)

Core responsibilities:
- Computes `STRATAX_API_BASE_URL` from `VITE_API_BASE_URL` (fallback: hosted default).
- Builds consistent headers:
  - `Authorization: Bearer <jwt>` (if token exists)
  - Stable guest identity headers:
    - `X-Stratax-Guest-Id`
    - `X-Client-Id`
    - `X-User-ID` (aligned with guest id)
  - Optional BYOK headers **only when authenticated**:
    - `X-API-Key`
    - `X-Gemini-Key`
- Emits window events for demo gating / rate limiting (`demo:limit-reached`, `demo:unavailable`).
- Captures effective session id from response headers (`X-Stratax-Session-Id`) and persists it.

### 6.2 `api.ts` (core product endpoints)

`src/lib/api.ts` wraps product endpoints using `strataxFetch()` and `buildStrataxHeaders()`.

Important calls used throughout the app:
- `apiCreateSession()` → `POST /api/session`
- `apiSubmitQuestion()` → `POST /api/question` (non-stream)
- `apiSubmitQuestionStream()` → `POST /api/question` with streaming body
- `apiGetHistory()` → `GET /api/session/{sessionId}/chat` (gracefully handles 404 as empty)
- `apiGetSessions()` → `GET /api/sessions` (backend may return `{ items: [...] }`)
- `apiDeleteSession()` → `DELETE /api/session/{sessionId}` (404 treated as already deleted)
- `apiUpdateSessionTitle()` → `PUT /api/session/{sessionId}/title`
- `apiExecuteCode()` → `POST /api/code/execute` (backend-only code execution + optional tracing)

#### Session id handling
- `apiSubmitQuestion()` attempts to read session id from headers and attach it to the returned JSON if missing.
- Some UI flows also “adopt” a session id returned by the backend in response payload.

### 6.3 `authApi.ts` (authenticated helper)

`src/lib/authApi.ts` provides a generic `apiCall()` that:
- Adds JWT if present
- Adds guest identity headers
- Handles:
  - `401`: clears auth *only when it looks like JWT/session expiry*
  - `429`: dispatches `ratelimit:exceeded`

This is mainly useful for endpoints under `/auth/*` and auth-gated flows.

---

## 7) Core Feature Modules

### 7.1 Interview Assistant (`/app`)

Primary component: `src/components/InterviewAssistant.tsx`

Responsibilities:
- Owns the main multi-mode UI:
  - Answer (default)
  - Mirror (Feedback)
  - Intelligence
  - Mock Interview
  - Practice
- Manages chat sessions:
  - Creates/loads a session id
  - Loads history for a session
  - Saves some UI state to `localStorage` (tab, mode, sidebar open, etc.)
- Calls `apiSubmitQuestion()` or `apiSubmitQuestionStream()` depending on streaming.

#### Answer vs Mirror mode
- **Answer**: submit just the question.
- **Mirror**: requires `user_answer` (draft answer) as input.
  - If missing, the UI opens a dialog to collect it before submitting.

#### History and persistence
- Session id is stored under a localStorage key (`ia_session_id`).
- The component has resilience logic for “session not found” by clearing the stored id and creating a new session.

### 7.2 Code Runner (`/run`)

Key files:
- `src/components/CodeRunner.tsx`
- `src/components/MonacoEditor.tsx`

Capabilities:
- Runs code by calling the backend endpoint `POST /api/code/execute`.
- No browser-side execution (no Pyodide, no direct Judge0/RapidAPI calls).
- Stores code, stdin, outputs, timer config, etc. in `localStorage`.
- Output panels show stdout/stderr and any backend-provided timing/memory.

#### Backend execution contract (frontend-facing)

Request (frontend → backend):
- `language`: runner language id (e.g. `python`)
- `code`: source
- `stdin`: optional input
- `trace`: optional boolean (frontend enables this for Python Visualize)
- `trace_max_events`: optional number (limits trace volume)
- `explain_trace`: optional boolean (when `true`, backend adds per-line explanations)
- `explain_max_lines`: optional number (default backend-side; frontend currently uses `200`)

Response (backend → frontend):
- `stdout`, `stderr`, `time_seconds`, `memory_kb`
- `trace_events[]` (Python-only today):
  - `step`, `line`, `event`
  - `locals` (backend-filtered/sanitized; frontend also hides system locals by default)
  - `explanation` (optional short explanation for that executed line)
- `line_explanations` (optional): map of `{ "<lineNumber>": "explanation" }` for quick lookup

#### Visualize (debugger-style)

The Visualize tab in `src/components/CodeRunner.tsx` provides a “debugger grade” experience:
- Timeline of steps with click-to-jump
- Scrubber (range slider) + prev/next step controls
- Current line preview (line number + source line)
- Locals panel rendered as a table with:
  - Search
  - Changed-only vs All toggle
  - Hide/Show system locals toggle

#### Trace explanations

When backend returns explanations:
- Timeline rows show a short explanation (when present)
- State panel shows an **Explanation** block
- The UI prefers `trace_events[].explanation` and falls back to `line_explanations[line]`

#### Security note

All code execution is backend-mediated:
- Never ship sandbox keys (Judge0/RapidAPI/etc.) to the browser.
- Avoid `VITE_*` secrets entirely—Vite embeds them into the client bundle.

### 7.3 Mock Interview

Key files:
- `src/components/MockInterviewMode.tsx`
- `src/lib/mockInterviewApi.ts`

Capabilities:
- Runs a session-based mock interview with:
  - Setup phase
  - Interview questions
  - Feedback
  - Summary
  - History view
- Supports voice input via `useSpeechRecognition`.
- Persists mock session state into `localStorage` so the user can resume.

### 7.4 Architecture Generator (`/architecture`)

Key files:
- `src/pages/Architecture.tsx`
- `src/components/ArchitectureGenerator.tsx`
- `src/lib/architectureApi.ts`

High-level:
- Presents an architecture generation UI.
- Submits to backend architecture generation APIs (see `architectureApi.ts`).

---

## 8) PWA (Manifest + Service Worker)

### Manifest

`public/manifest.webmanifest`:
- `start_url: /`, `display: standalone`
- Sets `background_color` and `theme_color` (also used for Android splash background matching).
- Declares icons:
  - Standard: `stratax-ai-192.png`, `stratax-ai-512.png`
  - Maskable: `stratax-ai-maskable-192.png`, `stratax-ai-maskable-512.png`
- Uses query-string cache busting (e.g. `?v=15`).

### Service worker

`public/sw.js`:
- Cache name is versioned (e.g. `stratax-ai-v15`).
- Pre-caches a minimal list of shell assets (index, manifest, icons).
- **Resilient install**: caches assets individually; a single failed fetch does not block SW installation.
- Fetch strategy:
  - Same-origin GET only
  - SPA navigation fallback to `/index.html`
  - Cache-first for static assets

Operational note: when you change icons or caching logic, bump BOTH the icon query param and `CACHE_NAME` to minimize stale-client issues.

---

## 9) Icon Generation Pipeline

`script`: `scripts/generate-icons.mjs` (Sharp)

Purpose:
- Takes `public/icons/source.png` and generates:
  - PWA icons (192/512)
  - Maskable icons (192/512)
  - `apple-touch-icon.png`

Key behaviors:
- `--background auto` samples edge colors from the source image to choose a background that avoids visible borders.
- Outputs are flattened and alpha is removed (`flatten()` + `removeAlpha()`), since some Android launchers can render alpha edges as dark/black artifacts.
- Supports a small `--bleed-crop` to trim subtle source borders before resizing.

Common commands:
- `npm run icons:generate`
- `npm run icons:generate -- --background auto --maskable-scale 0.78 --bleed-crop 0.03`

---

## 10) Environment Variables

Frontend env vars (Vite):

- `VITE_API_BASE_URL`
  - Base URL for product APIs via `STRATAX_API_BASE_URL`.
- `VITE_AUTH_API_URL`
  - Base URL for auth endpoints; should typically match `VITE_API_BASE_URL`.

---

## 11) Build, Run, Deploy

### Local dev
- `npm install`
- `npm run dev`

### Build
- `npm run build` (outputs `dist/`)

### Firebase Hosting
Firebase config serves `dist` and rewrites all routes to `index.html` for SPA behavior.

---

## 12) Troubleshooting Runbook

### PWA icon changes not showing
- Android/browser aggressively caches icons and manifest.
- Recommended:
  - Bump icon query param in `manifest.webmanifest`.
  - Bump `CACHE_NAME` in `public/sw.js`.
  - (Device) Clear site data OR uninstall PWA + reinstall.

### Android splash “border/edges”
- The splash background comes from `theme_color` / `background_color` and OS-level rules.
- Best mitigation: set `manifest.webmanifest` colors to match the icon edge/background color.

### “Installing…” but app never completes
- If SW install fails due to a missing precache asset, installation can stall.
- This repo’s SW caches assets individually to reduce that risk.
- If it still happens:
  - Verify `public/sw.js` `ASSETS` URLs exist post-build.
  - Bump cache name and reinstall.

### Auth loops / 401 handling
- The code tries to avoid logging users out for non-JWT-related 401s (e.g., invalid BYOK key).
- Ensure `VITE_AUTH_API_URL` and `VITE_API_BASE_URL` point to the same backend if JWT validation is failing.

---

## 13) Notes on Code Conventions

- Local storage keys are used heavily for UX persistence (sessions, tabs, runner settings).
- The codebase uses a mix of:
  - direct `fetch()` (auth context)
  - `api.ts` wrappers around `strataxFetch`
  - `authApi.ts` wrappers

If you want a single consistent path going forward, prefer routing all new calls through `strataxFetch` + `buildStrataxHeaders`.
