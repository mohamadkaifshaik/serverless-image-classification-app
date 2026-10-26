// API base URL.
//  - "/api"                   -> proxied to Cloud Run by the firebase.json rewrite (production)
//  - "http://localhost:8080"  -> local docker container / flask dev server
//  - "https://image-classifier-xxxx-uc.a.run.app" -> call Cloud Run directly (CORS is enabled on the API)
window.APP_CONFIG = {
  apiBase: location.hostname === "localhost" || location.hostname === "127.0.0.1"
    ? "http://localhost:8080"
    : "/api",
  maxFileMb: 10,
};
