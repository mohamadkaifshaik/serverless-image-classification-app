# frontend

Static site for Firebase Hosting. No build step, no framework.

```
firebase use --add            # select your GCP/Firebase project once
firebase serve                # local preview on http://localhost:5000
firebase deploy --only hosting
```

`public/config.js` holds the API base URL. In production leave it as `/api` - `firebase.json`
rewrites `/api/**` to the Cloud Run service, so the browser talks to the same origin and no
CORS setup is needed. For local dev against a local container set it to `http://localhost:8080`.
