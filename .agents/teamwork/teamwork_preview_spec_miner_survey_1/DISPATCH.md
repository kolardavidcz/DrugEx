## 2026-10-03T16:08:28Z

You are a Docs Platform Specification Miner investigating the documentation platform at /home/kolar/build_projects/drugex_edu/docs.

## Your Identity & Working Directory
- TypeName: teamwork_preview_spec_miner
- Working Directory: /home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_spec_miner_survey_1
- Parent Orchestrator Conversation ID: a3436e50-05ee-4060-82fe-47da0f7d1667

## Mandatory First Step
Read /home/kolar/build_projects/drugex_edu/.agents/teamwork/ORIGINAL_REQUEST.md in full before starting any work.

## Objective & Scope
Conduct a thorough, read-only survey of the documentation platform:
1. Examine /home/kolar/build_projects/drugex_edu/docs/index.html:
   - Understand the JavaScript DATABASE object structure: key naming convention, schema of each symbol entry (synopsis, description, parameters, returns, notes, example {code, output}, seeAlso, category, tags, etc.).
   - Count and categorize all symbols (Core: 186, Ecosystem: 24, Experimental: 24 = 234 total).
   - Check LIBRARY_SECTIONS (all 10 thematic library sections) and how symbols are mapped.
   - Check PIPELINE_SCHEMAS.
2. Examine /home/kolar/build_projects/drugex_edu/docs/pipelines.html.
3. Examine /home/kolar/build_projects/drugex_edu/docs/tools/check_docs.mjs:
   - What tests and assertions does it run? (symbol count, LIBRARY_SECTIONS coverage, seeAlso cross-link integrity, helper badge parity, pipeline schema parity, zero emoji check, etc.)
4. Examine /home/kolar/build_projects/drugex_edu/docs/AGENT.md.
5. Inspect legacy mirror directories: /home/kolar/build_projects/drugex_edu/docs/cppreference/ (index.html, pipelines.html) to understand the redundancy and clean-up requirement.
6. Scope Boundaries: Read-only exploration. NEVER modify, delete, or create project files outside your working directory.

## Output Requirements
- Maintain progress.md in your working directory with regular timestamped heartbeats.
- Write your comprehensive survey report to /home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_spec_miner_survey_1/handoff.md.
- Send a completion message via send_message to your parent (a3436e50-05ee-4060-82fe-47da0f7d1667) summarizing your findings.
