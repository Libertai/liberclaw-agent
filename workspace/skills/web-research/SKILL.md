---
name: web-research
description: Research topics online — search the web with web_search, then read pages and PDFs with web_fetch and read_pdf.
---
# Web Research

Combine the research tools instead of guessing URLs by hand.

## Workflow
1. **Search first.** Use `web_search` (LibertAI Search) to discover relevant
   pages — it returns titles, URLs, and snippets. Only skip this when you already
   know the exact URL.
2. **Read the promising results** with `web_fetch` (returns page text with HTML
   stripped). For PDFs, use `read_pdf` — `web_fetch` saves a PDF as a binary
   download rather than extracting its text.
3. **Check what you already know** with `search_history` before researching from
   scratch — you may have covered the topic in a past conversation.
4. **Analyze and extract** the relevant information.
5. **Save durable findings** to `MEMORY.md` (and useful URLs for next time).

## Tips
- The snippets from `web_search` are often enough — only `web_fetch` a result when
  you need the full content.
- Wikipedia and official docs fetch cleanly; fetch specific sub-pages if a page is
  truncated.
- `web_fetch` saves binary downloads to the `downloads/` directory — inspect them
  with `read_file`, `read_pdf`, or `bash`.
- If `web_search` is unavailable, fall back to `web_fetch` on specific sites.
