import express from "express";
import cors from "cors";
import fetch from "node-fetch";
import dotenv from "dotenv";
dotenv.config();

/* =============== KONFIG =============== */
const PORT = process.env.PORT || 3000;
const MODEL_OPENAI = process.env.MODEL_OPENAI || "gpt-4o-mini";
const MODEL_GEMINI = process.env.MODEL_GEMINI || "gemini-1.5-flash";

const OPENAI_KEYS = (process.env.OPENAI_KEYS || "")
  .split(",").map(s => s.trim()).filter(Boolean);
const GEMINI_KEYS = (process.env.GEMINI_KEYS || "")
  .split(",").map(s => s.trim()).filter(Boolean);

if (!OPENAI_KEYS.length && !GEMINI_KEYS.length) {
  console.warn("⚠️ Brak kluczy w .env (OPENAI_KEYS i/lub GEMINI_KEYS).");
}

/* Round-robin wskaźniki w pamięci */
let rrOpenAI = 0;
let rrGemini = 0;
const pickOpenAIKey = () => OPENAI_KEYS[(rrOpenAI++) % OPENAI_KEYS.length];
const pickGeminiKey = () => GEMINI_KEYS[(rrGemini++) % GEMINI_KEYS.length];

/* =============== APP =============== */
const app = express();

/* CORS – zawęź origins produkcyjnie */
const allowed = (process.env.ALLOWED_ORIGINS || "")
  .split(",").map(s => s.trim()).filter(Boolean);
app.use(cors({
  origin: (origin, cb) => {
    if (!origin || !allowed.length || allowed.includes(origin)) return cb(null, true);
    return cb(new Error("Origin not allowed by CORS"));
  }
}));
app.use(express.json({ limit: "1mb" }));
app.use(express.static("public"));

/* ====== Helper: timeout fetch ====== */
async function fetchWithTimeout(resource, options = {}) {
  const { timeout = 20000, ...rest } = options;
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), timeout);
  try {
    const res = await fetch(resource, { ...rest, signal: ctrl.signal });
    clearTimeout(id);
    return res;
  } catch (e) {
    clearTimeout(id);
    throw e;
  }
}

/* ====== Provider: OpenAI (Chat Completions) ====== */
async function callOpenAI({ prompt, model }) {
  if (!OPENAI_KEYS.length) throw new Error("No OpenAI keys configured");
  const key = pickOpenAIKey();

  const r = await fetchWithTimeout("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${key}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: model || MODEL_OPENAI,
      messages: [
        { role: "system", content: "You are GOK:AI, a mystical yet precise assistant. Answer in Polish by default." },
        { role: "user", content: prompt }
      ]
    }),
    timeout: 20000
  });

  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`OpenAI API error ${r.status}: ${t}`);
  }
  const data = await r.json();
  const text = data?.choices?.[0]?.message?.content;
  if (!text) throw new Error("Invalid OpenAI response");
  return { text, provider: "openai", model: model || MODEL_OPENAI, keyUsed: key.slice(0, 12) + "…" };
}

/* ====== Provider: Gemini (Generative Language) ====== */
async function callGemini({ prompt, model }) {
  if (!GEMINI_KEYS.length) throw new Error("No Gemini keys configured");
  const key = pickGeminiKey();
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model || MODEL_GEMINI}:generateContent?key=${key}`;

  const r = await fetchWithTimeout(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ contents: [{ parts: [{ text: prompt }]}] }),
    timeout: 20000
  });

  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`Gemini API error ${r.status}: ${t}`);
  }
  const data = await r.json();
  const text =
    data?.candidates?.[0]?.content?.parts?.[0]?.text ??
    data?.candidates?.[0]?.output ??
    data?.text ?? null;

  if (!text) throw new Error("Invalid Gemini response");
  return { text, provider: "gemini", model: model || MODEL_GEMINI, keyUsed: key.slice(0, 8) + "…" };
}

/* ====== Routing wielodostawcy + fallback ====== */
async function callLLM({ provider, prompt, model }) {
  const want = (provider || "").toLowerCase();

  if (want === "openai") return callOpenAI({ prompt, model });
  if (want === "gemini") return callGemini({ prompt, model });

  // Auto-fallback
  try { return await callOpenAI({ prompt, model }); }
  catch (e1) {
    console.warn("OpenAI failed, trying Gemini…", e1.message);
    return await callGemini({ prompt, model });
  }
}

/* ====== Endpoint główny ====== */
app.post("/api/ask", async (req, res) => {
  try {
    const { prompt, provider, model } = req.body || {};
    if (!prompt || typeof prompt !== "string") {
      return res.status(400).json({ ok: false, error: "Invalid 'prompt'." });
    }
    const result = await callLLM({ provider, prompt, model });
    return res.json({ ok: true, ...result });
  } catch (err) {
    console.error(err);
    return res.status(502).json({ ok: false, error: err.message || "Upstream error" });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 GOK:AI multiprovider proxy on http://localhost:${PORT}`);
});