# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Lint/Test Commands
- Build: `npm run build`
- Dev: `npm run dev`
- Lint: `npm run lint`
- Type check: `npm run typecheck`
- Test: `npm test`
- Run a single test: `npm test -- -t "test name"`

## Code Style Guidelines
- Framework: React with TypeScript
- Formatting: Use consistent indentation (2 spaces)
- Imports: Group imports by type (React, libraries, components, styles)
- Types: Use TypeScript types for all props, state, and function signatures
- Components: Use functional components with React hooks
- State Management: Use useState/useEffect for component state
- Naming: camelCase for variables/functions, PascalCase for components/classes
- Comments: Add descriptive comments for complex logic, especially for neural network operations
- CSS: Use className with tailwindcss for styling
- Error Handling: Implement graceful fallbacks and error handling for math operations