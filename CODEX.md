my current development workflow is mainly with claude. 
your job is not to modify any of the code in the repository, but to generate md file for task or spec update, or explaining to me part of the code.

- only generate md file in C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\specs or C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\tasks when I explicitly tell you to do so.
- before generating md file based on our conversation, ask clarification questions so that I provide minimum freedom to claude.

Task/spec file management rules:

- New task files must follow the existing standard naming convention: `tasks/taskNN.md` for active tasks and `tasks/taskNN_done.md` only after the task is complete.
- Do not add descriptive suffixes to task filenames unless explicitly requested.
- Versioned spec files must follow the existing standard naming convention: `specs/spec_vNNN_short_name.md`.
- Outdated spec files should be kept for now, including files explicitly marked outdated. They should only be removed when the overall project is over or when I explicitly ask for removal.
- Repository cleanup tasks may remove obsolete generated artifacts, caches, histories, and inactive run outputs, but should preserve whole checkpoint folders when requested so previous models can be reused.
