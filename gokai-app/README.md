# GOK:AI – multiprovider (OpenAI + Gemini)

## Instalacja
```bash
npm install
```

Utwórz plik `.env` w katalogu głównym:
```
PORT=3000
OPENAI_KEYS=sk-proj-AAA,sk-proj-BBB
GEMINI_KEYS=google-AAA,google-BBB
MODEL_OPENAI=gpt-4o-mini
MODEL_GEMINI=gemini-1.5-flash
ALLOWED_ORIGINS=http://localhost:3000,https://www.mtaquestwebsidex.com
```

## Start
```bash
npm run dev
# http://localhost:3000
```

## Frontend
Jedna strona SPA: `public/index.html` (orb/terminal), `style.css`, `app.js`.
Algorytm świadomości: `public/engine/consciousness.js`.