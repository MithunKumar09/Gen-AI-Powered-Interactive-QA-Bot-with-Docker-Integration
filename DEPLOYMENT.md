# Deployment Guide

Getting this live on Render, from a repository whose credentials are currently
public.

Read [Step 0](#step-0-rotate-your-credentials-do-this-first) before anything
else. It is not optional and nothing later works without it.

---

## Step 0: Rotate your credentials (do this first)

**Your Cohere and Pinecone API keys are exposed in two separate public places.**
They must be treated as permanently compromised.

### Exposure 1: this repository's git history

Public GitHub repositories are continuously scraped for credentials. Assume both
keys have already been collected.

The history also contains **three previous attempts to remove these files**
(`ef94ed4`, `77498bb`, `fedb0d3`). Each deleted the file from the working tree;
every blob stayed reachable. Deleting a secret in a later commit does not remove
it — which is exactly why revocation, not history editing, is the actual fix.

### Exposure 2: a public Docker Hub image — verified

`docker.io/mithun1634/rag-backend:latest` is **public**, has been **pulled 124
times**, and its `/app/.env` layer contains both keys in plaintext:

```console
$ docker run --rm --entrypoint sh mithun1634/rag-backend:latest -c 'cat /app/.env'
COHERE_API_KEY=byLT…
PINECONE_API_KEY=acb4…
```

This happened because the build context is `./Backend` and there was no
`Backend/.dockerignore`, so `COPY . .` baked the file into a layer. (Both
`.dockerignore` files now exist, and the current images are verified clean.)

**This exposure is worse than the git one in a specific way:** you cannot rewrite
an image's layers. Deleting the tag is the only remediation, and 124 pulls means
the image may already have been downloaded and retained by others.

`mithun1634/rag-frontend` is also public (68 pulls) but carried no secrets — the
old `Frontend/.env` held only `BACKEND_URL`.

### 0.0 Delete the exposed images

Do this alongside key revocation:

1. <https://hub.docker.com/r/mithun1634/rag-backend> → **Settings** → **Delete
   repository**.
2. Do not re-push until you have confirmed a fresh build has no `.env`:
   ```bash
   docker run --rm --entrypoint sh <your-image> -c 'find / -name ".env" 2>/dev/null'
   # must print nothing
   ```

Deleting the repository does not un-leak the keys either. Step 0.1 is still the
remedy.

### 0.1 Revoke and reissue

| Service | Where | What to do |
|---|---|---|
| Cohere | <https://dashboard.cohere.com/api-keys> | Delete the existing key, create a new one |
| Pinecone | <https://app.pinecone.io/> → API Keys | Delete the existing key, create a new one |

### 0.2 Verify the old keys are dead

Do not skip this. A key you believe is revoked is not necessarily revoked.

Perform the verification locally using the old credentials stored outside the
repository. Never paste old or replacement credentials into this document,
Git history, issue trackers, or CI logs.

Expected invalid-key results:

- Cohere: the key-check response contains `"valid": false`
- Pinecone: the API returns HTTP `401` or `403`

Do not proceed until both old credentials are confirmed invalid.

### 0.3 Generate the two local secrets

```bash
python -c "import secrets; print('BACKEND_API_KEY=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('SESSION_SCOPE_KEY=' + secrets.token_urlsafe(32))"
```

`SESSION_SCOPE_KEY` is the HMAC key that turns a browser session id into a storage
scope, so the raw id never reaches Pinecone or the logs. **Rotating it later makes
every stored document unreachable** — not leaked, since nobody can derive the
scope, just unreachable. Retention cleanup removes them on the normal cutoff.

### 0.4 Create your local `.env`

```bash
cp .env.example .env
# then fill in the four secrets you just obtained
```

`.env` is gitignored (`.env` with no slash matches at any depth, so
`Backend/.env` and `Frontend/.env` are covered too). It is also excluded from both
Docker build contexts by `Backend/.dockerignore` and `Frontend/.dockerignore` — the
files that stop `COPY . .` baking your keys into a published image layer.

### 0.5 Purge the history (optional, after revoking)

Optional because revocation is what actually protects you. Worth doing so the
public repository does not visibly contain credentials.

```bash
# 1. Back up first, and verify the backup opens.
git clone --mirror . ../qa-bot-backup.git
git -C ../qa-bot-backup.git log --oneline -1   # should show 5a32138

# 2. Install the tool.
pip install git-filter-repo

# 3. Drop the .env blobs from every commit.
git filter-repo --invert-paths \
  --path .env \
  --path "Backend (RAG Model)/.env" \
  --path "Frontend (Interactive QA Bot)/.env"

# 4. Scrub the key that was hard-coded in manage_index.py. Use --replace-text,
#    not --path: that file existed at two historical paths and should be
#    transformed, not deleted from history.
printf 'OLD_PINECONE_KEY==>REDACTED\nOLD_COHERE_KEY==>REDACTED\n' > /tmp/secrets.txt
git filter-repo --force --replace-text /tmp/secrets.txt
rm /tmp/secrets.txt

# 5. Confirm nothing remains.
git log --all -S 'OLD_PINECONE_KEY' --oneline   # expect no output
gitleaks detect --no-banner                      # if installed

# 6. Re-add the remote (filter-repo removes it) and push.
git remote add origin https://github.com/MithunKumar09/Gen-Ai-RAG-Enhanced-Interactive-QA-Bot.git
git fetch origin
git push --force-with-lease=main:5a32138 origin main
```

Notes that matter:

- **Only `refs/heads/main` is pushed.** `refs/remotes/origin/main` is local
  tracking metadata, not a pushable ref. This repository has one branch and no
  tags, so one push covers everything reachable — verify with
  `git for-each-ref` before you start if that has changed.
- The explicit `--force-with-lease=main:5a32138` matters because `filter-repo`
  removes the remote, so a bare `--force-with-lease` has no recorded baseline to
  compare against and is either rejected or vacuous.
- Relax branch protection on GitHub before the push, restore it after.
- **GitHub keeps force-pushed commits reachable by SHA** until support purges
  them, and any existing clone or fork keeps the old blobs. This is why step 0.1
  is the real remedy.

---

## Step 1: Create the Pinecone index

The dimension changed. The old `sample-movies` index is 4096-dimensional because
the previous code used `embed-english-v2.0`, which Cohere **retired on
2026-04-04**. The replacement (`embed-v4.0` at 1024 dimensions) needs a new index —
vectors cannot be migrated between dimensions.

```bash
cd Backend
pip install -r requirements.txt
python scripts/init_index.py
python scripts/init_index.py --describe   # confirm dim=1024, metric=cosine
```

The script is idempotent and create-only. Deletion needs two flags
(`--delete --yes-i-am-sure`), because its predecessor deleted the index
unconditionally on every run.

**Optionally delete the old `sample-movies` index** in the Pinecone console. This
is hygiene, not a quota requirement — Starter allows 5 indexes. Leaving a stale
4096-dimensional index around mainly invites confusion later.

### Starter plan limits worth knowing

| Limit | Value | Why it shaped the design |
|---|---|---|
| Namespaces per index | **100** | Why there is one fixed namespace with metadata isolation, rather than a namespace per visitor — 100 visitors would exhaust the index with nothing reclaiming them |
| Metadata per record | **40 KB** | Why chunk metadata is measured in UTF-8 bytes before writing |
| IDs per delete/fetch | **1000** | Why `MAX_CHUNKS ≤ 1000`, so rollback stays a single request |
| Region | `aws` / `us-east-1` only | Hard-coded default |
| Storage | 2 GB per org | ~1.7 MB per 40-page document |

---

## Step 2: Verify locally

```bash
cd Backend && pytest -q          # 210 tests, no network
cd .. && docker compose up --build
```

Then, from the repo root:

```bash
# Health: public, makes no provider call
curl localhost:5000/health

# Readiness: asserts the index matches your configuration
curl -H "X-API-Key: $BACKEND_API_KEY" localhost:5000/ready

# Auth is enforced
curl -X POST localhost:5000/ask            # expect 401

# Models are actually reachable (this one does spend a few tokens)
docker compose exec backend python scripts/smoke_models.py
```

Open <http://localhost:8501>, enter your `DEMO_PASSWORD`, upload a text-based PDF,
and ask a question **that is only answerable from a later page**. That last part is
the meaningful test: the previous implementation embedded the entire PDF as a
single vector, so anything past the truncation point was unreachable. If a
later-page question is answered with a sensible page citation, chunking works.

Then ask something the document does not cover. It should say so rather than
inventing an answer.

---

## Step 3: Deploy to Render

### 3.1 Push the blueprint

`render.yaml` is committed and defines both services. Push it to `main`.

### 3.2 Create the blueprint

1. <https://dashboard.render.com> → **New** → **Blueprint**
2. Connect this repository. Render reads `render.yaml` and shows two services:
   `rag-backend` and `rag-frontend`.
3. It will prompt for every `sync: false` value. These are never stored in the
   repository:

| Service | Variable | Value |
|---|---|---|
| rag-backend | `COHERE_API_KEY` | your new Cohere key |
| rag-backend | `PINECONE_API_KEY` | your new Pinecone key |
| rag-backend | `BACKEND_API_KEY` | the shared secret from 0.3 |
| rag-backend | `SESSION_SCOPE_KEY` | the HMAC key from 0.3 |
| rag-frontend | `BACKEND_API_KEY` | **the same value** as the backend's |
| rag-frontend | `DEMO_PASSWORD` | your chosen passphrase |

`BACKEND_URL` is wired automatically from the backend service's host. The frontend
normalises a bare hostname to `https://` at startup.

### 3.3 Deploys are manual at first

`render.yaml` ships `autoDeployTrigger: off` deliberately. Deploy manually, verify,
then switch to `commit` once you trust it.

`checksPass` is a valid value but is not used here — there is no CI workflow, so
selecting it would wait for checks that never report.

### 3.4 Verify the live deployment

```bash
BACKEND=https://rag-backend-xxxx.onrender.com

curl $BACKEND/health                                    # 200
curl -H "X-API-Key: $BACKEND_API_KEY" $BACKEND/ready    # 200 + index detail
curl -X POST $BACKEND/ask                               # 401
```

Then repeat the Step 2 browser checks against the public frontend URL, plus:

- Open a **second browser session** with a different PDF. Neither session should
  see the other's document.
- Type several questions in a row and watch the backend logs. There must be
  **exactly one** `/upload` call until you change the file.

Only then flip `autoDeployTrigger` to `commit`.

---

## Free tier behaviour

| | Free | Starter ($7/service/month) |
|---|---|---|
| Sleeps when idle | After ~15 min | Never |
| Cold start | ~50 s | — |
| Suitable for | A link you send and warm up first | A link that is always live |

The frontend shows a "waking the API, this can take up to a minute" spinner
during a cold start rather than an error, and the request timeouts (10 s connect /
180 s read on upload) are sized for it.

For a demo you are actively showing someone: open the link a minute beforehand.

## Costs

| | Cost |
|---|---|
| Render, both services | $0 (free) or $14/month (both on Starter) |
| Pinecone Starter | $0 within 2 GB / 1 M reads / 2 M writes per month |
| Cohere | Trial keys are rate-limited but free; production keys are usage-based |

Cohere is the only component that can accrue real cost, which is why the demo has
a passphrase gate and three layers of rate limiting.

---

## Cleanup cadence

**Storage is not self-bounding.** `scripts/cleanup_orphans.py` does not run
itself — there is no scheduler in this deployment — so it stays bounded only as
far as you actually run it.

```bash
# Always dry-run first. This is the default.
docker compose exec backend python scripts/cleanup_orphans.py

# Then act.
docker compose exec backend python scripts/cleanup_orphans.py --delete --yes-i-am-sure

# Verify.
docker compose exec backend python scripts/cleanup_orphans.py
```

Recommended: a dry run before your first public deploy, then a confirmed run
after each demo period or weekly.

**Accepted limitation:** cleanup can remove a document belonging to a browser
session that has been left open longer than `RETENTION_HOURS`. Knowing which
document a Streamlit session is currently viewing would require a persistent
registry, which is out of scope. The degradation is graceful — the next question
returns 404, and the UI clears its state and asks for a re-upload.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Startup fails: `Invariant N violated` | Interdependent limits are inconsistent | The message names the variables and the arithmetic. See the invariants in `rag_core/config.py` |
| Startup fails: `COHERE_EMBED_MODEL ... not supported` | A retired model is configured | Use a currently-served model; `embed-english-v2.0` was retired 2026-04-04 |
| `/ready` 503, dimension mismatch | `COHERE_EMBED_DIMENSION` ≠ index dimension | Match the env var to the index, or create a new index. Dimensions cannot be changed in place |
| `/ready` 503, cannot reach Pinecone | Wrong key or no egress | Check `PINECONE_API_KEY`; the authenticated response names what failed |
| Every request 500s under gunicorn | App spec missing `()` | Must be `myapp:create_app()`. Without parentheses gunicorn calls the factory as the WSGI app |
| 502 on upload | Cohere unreachable or model retired | Run `scripts/smoke_models.py` |
| 422 on a valid-looking PDF | Scanned / image-only, so no extractable text | Only text-bearing PDFs work; there is no OCR |
| 413 on upload | Over `MAX_UPLOAD_MB` | Raise it, or use a smaller file |
| 429 sooner than expected | Limits are per gunicorn worker | With 2 workers the effective ceiling is up to 2× configured. See ARCHITECTURE.md |
| Logs are empty in `docker logs` | Missing `PYTHONUNBUFFERED` | Already set in both Dockerfiles; check you have not overridden it |
| Healthcheck always unhealthy | `curl` used in a `python:*-slim` image | The probes here use stdlib `urllib` for exactly this reason |
| Windows: `ImportError: DLL load failed ... filename or extension is too long` | `MAX_PATH`. This project's path is already ~113 characters, and `msgspec`'s binary sits deep in `site-packages` | Put the virtualenv at a short path (`C:\venv\qabot`), enable Win32 long paths, or just use Docker |

---

## What the platform is not doing for you

Worth being explicit, since a demo that looks production-shaped can imply more
than it provides:

- **The passphrase is a spend guard, not authentication.** It stops a stranger
  from burning your Cohere credits. It is shared, not per-user, and protects
  nothing else.
- **Rate limits are per worker and reset on redeploy.** They bound runaway demo
  spend. They are not an enforceable account-wide budget — set spend limits in the
  Cohere dashboard if you need a real ceiling.
- **Uploaded documents are not private in any strong sense.** They are isolated
  per browser session and deleted on the retention cutoff, but they do sit in your
  Pinecone index in the meantime. Do not upload anything genuinely sensitive.
