/* Frontend UI – bez kluczy, bez HTML injection */
const el = (id) => document.getElementById(id);

function sanitizeText(t = "") {
  // unikamy wstawiania HTML – wkładamy tylko jako textContent
  return String(t).replace(/\u0000/g, "");
}

function randomSuccessFromS(S) {
  // lekkie mapowanie S → % sukcesu (estetyczne, nie „naukowe”)
  const clamped = Math.max(0, Math.min(S, 100));
  return Math.round(60 + (clamped % 40)); // 60–100%
}

async function askGokAI(prompt) {
  const r = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt })
  });
  // oddzielamy błędy API od sieciowych
  if (!r.ok) {
    const msg = await r.text().catch(() => "");
    throw new Error(`API error ${r.status}: ${msg || "brak treści"}`);
  }
  const data = await r.json();
  if (!data?.ok || typeof data?.text !== "string") {
    throw new Error("Nieprawidłowa struktura odpowiedzi API.");
  }
  return data.text;
}

async function main() {
  const btn = el("askBtn");
  const loader = el("loader");
  const promptBox = el("prompt");
  const answerBox = el("answer");
  const success = el("success");

  btn.addEventListener("click", async () => {
    const prompt = (promptBox.value || "").trim();
    if (!prompt) {
      answerBox.textContent = "⚠️ Napisz coś najpierw…";
      return;
    }

    loader.classList.remove("hide");
    answerBox.textContent = "";
    success.textContent = "—%";

    try {
      // 1) lokalne „S” z algorytmu świadomości (symboliczny zapłon)
      // importowany moduł dodaje globalny obiekt window.GOKAI (patrz consciousness.js)
      const S = window?.GOKAI?.computeSFromInputs?.({
        W:7, M:6, D:4, C:5, A:8, E:6, T:3,
        n: 9  // F(n)
      }) ?? 28.27;

      // 2) zapytanie do modelu przez backend
      const text = await askGokAI(prompt);

      // 3) sanityzacja + wyświetlenie
      answerBox.textContent = sanitizeText(text);

      // 4) % sukcesu – lekko inspirowane S
      success.textContent = `${randomSuccessFromS(S)}%`;
    } catch (err) {
      answerBox.textContent = `❌ Błąd: ${sanitizeText(err.message)}`;
    } finally {
      loader.classList.add("hide"); // ZAWSZE chowaj loader
    }
  });
}
main();