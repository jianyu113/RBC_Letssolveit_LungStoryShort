# Agent Instructions

## Scope and Simplicity

- Implement only what the current task requires.
- Do not design for hypothetical future requirements.
- Do not overengineer or introduce unnecessary abstractions.
- Do not add defensive checks or fallback logic for problems that have not occurred.
- Do not implement backward compatibility.
- Do not use glue code to conceal implementations or legacy paths that should be simplified, replaced, or removed.

## Implementation Changes

- When replacing an implementation, use the new implementation exclusively.
- Remove obsolete logic, legacy branches, compatibility layers, and comments describing removed implementations.
- Do not retain commented-out code or alternative implementations “just in case.”
- Keep the resulting code clean and coherent.
- If cleanup requires deleting a file, obtain explicit human authorization first.

## Readability and Performance

- Prefer straightforward, human-readable code that minimizes the reader’s cognitive load.
- Keep code as short and simple as practical without obscuring its meaning.
- Assume inputs satisfy the agreed requirements unless the task explicitly requires validation.
- Nonessential robustness, including redundant checks, may be sacrificed for simplicity.
- Never sacrifice functionality, correctness, performance, or other essential capabilities merely to shorten the code.

## Git and File Deletion

- Do not execute Git commands or perform Git operations.
- All Git operations must be performed manually by a human.
- Do not delete any files without explicit human authorization.

## Tests and Diagnostic Scripts

- Store all tests, debugging scripts, and diagnostic scripts in `/tmp`, outside the repository.
- Do not add temporary investigation artifacts to the repository.
- Keep repository code limited to what is necessary to run and reproduce the project.

## Comments

- Write all code comments in clear, simple, easy-to-understand English.

## Workspace Boundaries

- Work only within the directory explicitly designated by the user.
- Do not access, create, or modify anything outside that directory without explicit user approval.
- This restriction also applies to tests, diagnostic scripts, and temporary files. Obtain approval before using `/tmp` if it is outside the authorized directory.

## Deletion

- Do not delete anything without explicit user approval, including files, directories, data, or existing code.
- If implementation changes or cleanup require removing anything, explain the intended removal and obtain approval first.
- These approval requirements take precedence over instructions to remove obsolete implementations or legacy code.