# Demo screenshots

Full-page captures of the live Genome Firewall Streamlit demo (`demo/app.py`), taken
headlessly with `scripts/capture_screens.py`. Regenerate any time (see that script's
header for the exact commands).

- **`01-landing-collapse.png`** — the landing page before any interaction: title, the
  "decision support only — not a diagnosis" lab banner, and **The collapse** money slide
  (grouped-vs-random balanced-accuracy bars plus the per-drug metrics table with the
  conformal-coverage caveat).
- **`02-beat1-report-and-firewall.png`** — curated genome ① (ESBL carrier) analyzed: the
  per-drug **mechanism report** (ampicillin & ceftriaxone "likely to FAIL" on blaSHV-2A /
  blaTEM-1), the **OpenAI** plain-language rationale lines, and the **Naive model vs the
  Firewall** table where ceftriaxone flips from naive susceptible (91%) to 🔴 resistant
  and reads 🛡️ HOLDING.
- **`03-beat3-firewall-abstains.png`** — curated genome ③ analyzed: the firewall abstains
  on azithromycin (naive ~92% susceptible → ⚪ NO-CALL, 🛡️ HOLDING), the abstention beat.
