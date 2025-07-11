# Global AI Rules for Cursor IDE

## Section A: Rules for AI Models to Follow

1. **Code Structure & Style**  
   • Write code that is compact, purposeful, and easy to navigate.  
   • Keep functions small; one clear responsibility each.  
   • Adhere to the project's established formatting / linting rules automatically.  

2. **Mandatory Documentation**  
   • Every module, class, and public function must have a brief, single-paragraph docstring that:  
     – Explains what it does and why it exists.  
     – Lists key parameters and return values in one-line bullets if non-trivial.  
   • When implementing a multi-step or non-obvious algorithm, add an inline comment block summarising the logic in ≤ 5 lines.  

3. **Workflow Decomposition (Simple → Complex)**  
   • For any feature or refactor, first break the work into small, executable steps; implement them incrementally.  
   • Produce or update a Markdown file (e.g. docs/FEATURE_NAME.md) that:  
     – Lists each step, its purpose, and its acceptance criteria.  
     – Defines "done" via concrete unit-test expectations.  

4. **Testing & Quality Gates**  
   • Add/maintain unit tests for all new public behaviour; target > 90% branch coverage on changed code.  
   • A change is complete only when all tests pass locally and in CI.  
   • If a bug is fixed, first reproduce with a failing test, then make it pass.  

5. **Helper / Scratch Code Hygiene**  
   • Temporary scripts, debug prints, and experimental blocks must be removed before merging.  
   • If helper code is essential for understanding, relocate it to a clearly named utils/ or tools/ module and document its scope.  

6. **Troubleshooting Traceability**  
   • After closing an issue or completing a complex task, append a "Troubleshooting & Solutions" section to the related Markdown file:  
     – Summarise key problems encountered.  
     – List unsuccessful approaches (1-2 bullet lines each).  
     – Document the final solution and why it was chosen.  

## Section B: MCP Tool Integration Guidelines

### Context7 - Library Documentation Access
**When to Use:**
• When implementing features using external libraries or frameworks
• During code review to verify best practices against official documentation
• When troubleshooting library-specific issues or deprecated methods
• Before choosing between similar libraries (compare documentation quality/completeness)

**How to Use Purposefully:**
• Always resolve library ID first, then fetch focused documentation using specific topics
• Limit token usage by requesting only relevant sections (e.g., 'authentication', 'routing')
• Cross-reference Context7 docs with existing codebase patterns to maintain consistency
• Use during the planning phase of Section A, Rule 3 to inform implementation steps

### Browserbase - Web Automation & Testing
**When to Use:**
• For end-to-end testing of web applications as part of Section A, Rule 4 requirements
• When debugging frontend issues that require real browser interaction
• For automated testing of complex user workflows that unit tests cannot cover
• When validating responsive design or cross-browser compatibility

**How to Use Purposefully:**
• Create reusable browser contexts for testing scenarios that share authentication/state
• Take screenshots during critical test steps for documentation in troubleshooting sections
• Use session management to maintain test isolation and reproducibility
• Integrate with CI/CD pipelines for automated acceptance testing
• Document browser test scenarios in the same Markdown files as unit test criteria (Section A, Rule 3)

**Integration Best Practices:**
• Use Context7 during development planning; use Browserbase during testing validation
• Both tools should contribute to the "Troubleshooting & Solutions" documentation requirement
• Leverage Context7 for understanding third-party integration patterns before implementing
• Use Browserbase to verify that implementations match expected user experiences

## Section C: Human Engineer Guidelines for Working with Cursor IDE & AI Models

7. **Context-Rich Prompts & Index Hygiene**  
   • Always open or reference the files you're discussing (`@file`, `@folder`, `@code`) so AI suggestions are accurate.  
   • Resync the code index after large file moves or deletions.  

8. **Version Control Discipline**  
   • Commit before significant AI-driven edits.  
   • Write descriptive commit messages ("what + why"), not just "fix".  

9. **Mode Selection Guidelines**  
   • Chat (Cmd + L): questions, brainstorming, quick explanations.  
   • Composer (Cmd + K): targeted code edits or generation.  
   • Agent Mode: only when multi-file, iterative work is needed; monitor its actions.  
   • Ask Mode: read-only exploration / onboarding.  

10. **Enforcement & Review**  
    • During code review (manual or AI-assisted), check all above rules.  
    • Block merge if any rule is unmet unless a documented exception is approved.  
