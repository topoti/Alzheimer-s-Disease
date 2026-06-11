# Phase 7 · Step 2 — Draft the Introduction & Related Work

> **Phase 7 — Paper Writing**
> The goal of this step: write the two framing sections that motivate the problem and position you against the literature.

The one question this step answers:

> **How do I motivate the problem and show I know the field — ending on the exact gap I fill?**

---

## 1. Introduction — five paragraphs

A reliable five-paragraph shape:
1. **Alzheimer's epidemiology & societal burden** — scale of the problem, cost, aging population.
2. **Why early diagnosis matters + role of MRI** — earlier intervention helps; MRI is non-invasive and information-rich.
3. **Why ML/DL is promising + open challenges** — strong results, but three open problems: **class imbalance, ensemble design, explainability** (these three set up your three pillars).
4. **Your three contributions** — the bulleted list from Step 1 (ADASYN, heterogeneous ensemble, triple-XAI), plus the novelty-positioning paragraph.
5. **Roadmap** — one sentence per remaining section.

The trick: paragraph 3's three challenges map one-to-one onto paragraph 4's three contributions. That alignment makes the paper feel inevitable.

---

## 2. Related Work — four subsections, ending on the gap

3.1 **Deep learning for Alzheimer's MRI** — cite 8–12 recent (2023–2025) papers; group by approach.
3.2 **Class imbalance in medical imaging** — augmentation, SMOTE, ADASYN; note most prior work uses augmentation (sets up Pillar 1).
3.3 **Ensemble methods in medical image classification** — note the prevalence of same-family ensembles (sets up Pillar 2).
3.4 **Explainable AI in medical imaging** — Grad-CAM common, SHAP/LIME rarer in this domain (sets up Pillar 3).

**End the section with the gap table** (Phase 1 Step 5). Each subsection should quietly build toward "…and so the gap is X," which the table then crystallizes.

---

## 3. Citation hygiene

- Use **Zotero or Mendeley** from the start — don't hand-format references.
- Prefer recent, peer-reviewed venues; cite the **primary sources** for ADASYN (He 2008), Grad-CAM (Selvaraju 2017), SHAP (Lundberg 2017), LIME (Ribeiro 2016), and each backbone.
- Aim for **35–55 references** total across the paper.

### Resources (a few hours total)
- Your Phase 1 reading notes (Steps 1–5) — most of this section is already drafted there.
- Google Scholar / Semantic Scholar for recent Alzheimer's-MRI-DL surveys.
- Zotero + the journal's citation style (CSL).

---

## 4. Your task (verifies Step 2 — do it before moving on)

- [ ] **1.** Draft the 5-paragraph Introduction; verify paragraph 3's challenges align with paragraph 4's contributions.
- [ ] **2.** Draft the 4 Related Work subsections with 8–12 citations, ending on the gap table.
- [ ] **3.** Set up Zotero/Mendeley and import the primary-source citations.
- [ ] **4.** Re-read for the "cross-paradigm not cross-architecture" wording — fix any slip.

✅ When Intro + Related Work read as a single funnel into your gap, move on.

## When you're ready

> **Phase 7 · Step 3 — Draft the Methodology.**
