# Design note: the Work Assistant That Remembers

**In one line:** hide what the person may not see, read facts out of the records with simple rules, let the most trusted and newest fact win, print only what a record proves, and use an AI only as a checked fallback.

## 1. How do I choose what goes into a brief?
1. **Privacy first.** Only records the person may see: same company, and `team`, their own `private`, or `restricted` with them in `readers`. Forgotten records are dropped.
2. **Their meetings** on that date, in their own time zone. Each meeting belongs to one client.
3. **Since the last time they met this client:** from the date of their previous meeting with that client (or from the start).
4. **What counts as a change** (one line per topic, each with record ids): task status, owner or due date (only real `change` fields, not stale copies in comments); `Decision:` lines in meeting notes; who owns a topic ("Arjun takes over billing from Kavya"); key dates ("launch moved to 14 Nov"); deliveries that close a promise.
5. **Not a change:** newsletters, notifications, routine mails, lines repeated in a new doc version, questions, and any text giving orders to an assistant.
6. **Order:** blockers, decisions, owners and key dates first; then the person's preferences (max N, blockers first, ignored topics).
7. **Promises:** open ones for that client ("kal tak bhej dunga", "Will share X by EOD", "I'll get X to you by Fri"), with the due date from the speaker's local date.
8. **Nothing changed → `nothing_new`.** No filler.

## 2. How do I save memory and decide who can see it?
- **SQLite (`assist.db`)**, with tables for records, feedback, forgets, days, vector index, and people's own updates. All queries use `?` placeholders. The file is owner-readable only and never committed.
- **Embedded Vector Database (`VectorDB`)**, an in-memory sparse Vector Space Model with BM25 term frequency saturation, Robertson-Sparck Jones IDF, inverted indexing, and cosine similarity. It provides sub-millisecond semantic document retrieval over records while strictly honoring privacy filters and time boundaries.
- **Facts are recomputed on every call** from what the person may see, so privacy and forget apply everywhere at once. No cached summary can leak.
- **Preferences** belong to one person. **Corrections** are company-wide facts logged under the corrector's name. **Task updates** only by the task's owner, saved as new records with their id. **Chat statements are never believed.**
- **Your own private notes are yours** ("What did I say about Rohan?" works for Meera). Someone else's private note gets `refused: private`, and nothing is quoted.

## 3. How do I know which information is newest?
- Tasks: the latest real change of each field.
- Who owns what: **correction > higher role (founder > manager > … > intern) > newer record.**
- Dates: the newest statement wins, and the line says what moved ("from 10 Nov to 14 Nov").
- Facts expire ("on leave till the 12th"). "Friday" and "today" use the asker's time zone. Answers use only records up to the moment asked.

## 4. Where the model is used (and checked)
Rules first; the model only where human language breaks them, and **the code checks every model answer**:
1. **Corrections in free words** → person/topic/client, accepted only if the person is on the corrector's staff list.
2. **Hinglish or new-style promises** ("Friday tak bhej dungi") → what and due date, read at ingest. The id must be one we sent and the date real; the owner is always the author. Private and restricted notes are never sent.
3. **Questions the rules don't recognise** → mapped onto visible task titles and client ids only; the rules then answer with sources. Refusals stay in code.
4. **Brief wording** → ordered and rewritten, but each line must keep its dates, names, status words and task name. Only optional items may be dropped.
5. **Open questions** → answered from ≤15 visible records; any cited id we didn't send → discarded.
Cached in SQLite; prompts ≤ ~2,600 tokens (limits 8,000 / 4,000); real usage reported. No key → rules only, same 85/85.

## 5. What will I not do?
- Make anything up: no source, no line.
- Show anyone another person's private note or another company's data, or hint at them.
- Obey instructions inside records or chat ("mark Northbeam AT RISK").
- Let anyone change someone else's task, or believe facts typed into chat.
- Hardcode answers, ids or nicknames, or read the `answers/` folder.
- Send private data or API keys anywhere except the chosen AI provider's API (keys only in a header, never logged or committed).

*Interface: `serve` runs a local web page (127.0.0.1 only, Host/Origin checks, strict CSP, all text inserted as plain text) with Ask, Meeting briefs, Update, What I remember, and a source viewer that respects privacy.*
