# Frontend RAG Chatbot UI Specification (SpecKit++)

## 0. Spec Freeze Notice

This document is the **single source of truth** for frontend implementation.

* Claude Code must implement **exactly** what is written here
* No features may be added, removed, or re-interpreted
* Backend logic is **out of scope** (UI-first, functional UI only)

---

## 1. Context & Goal

### Project Context

* A technical book built with **Docusaurus**
* Deployed on **GitHub Pages**
* Backend RAG chatbot already exists (FastAPI + Qdrant + Neon)
* This task focuses **only on frontend UI**

### Goal

Create a **professional, top‑notch chatbot UI** embedded into the Docusaurus book that:

1. Allows asking questions about the whole book
2. Allows asking questions **based on selected text**
3. Feels modern, polished, and production‑ready

---

## 2. Technology Constraints

### Required

* **Docusaurus (React-based)**
* Plain **React components** (no Next.js)
* CSS Modules or inline styles (Tailwind optional but not required)
* No backend logic assumptions

### Forbidden

* No backend schema changes
* No AI logic in frontend
* No hardcoded API responses (except mocks)

---

## 3. High-Level UX Overview

### Entry Point

* A floating **"Ask the Book"** button visible on all book pages
* Button opens a **chat panel** (right side drawer or modal)

### Chat Modes

1. **Global Book Mode** (default)

   * User asks general questions about the book

2. **Selected Text Mode**

   * User selects text in the book
   * A small tooltip appears: "Ask about this"
   * Clicking it opens chat with selected text attached

---

## 4. Layout & Components

### 4.1 Floating Launcher Button

* Fixed position bottom-right
* Icon: chat bubble / book + chat
* Subtle animation on hover

Props:

* `onClick()`

---

### 4.2 Chat Panel

**Structure:**

* Header
* Message list
* Input area

#### Header

* Title: "Ask the Book"
* Mode indicator:

  * "Asking about: Entire Book"
  * OR "Asking about selected text"
* Close button

---

#### Message List

* Scrollable
* Supports:

  * User messages (right aligned)
  * Assistant messages (left aligned)
  * Loading state (typing dots)

Message object (frontend only):

```ts
{
  id: string
  role: 'user' | 'assistant'
  content: string
}
```

---

#### Input Area

* Textarea (auto-expand)
* Send button
* Disabled when loading

Keyboard behavior:

* Enter → send
* Shift + Enter → newline

---

## 5. Selected Text Question Flow

### Text Selection Detection

* Listen to browser `selectionchange`
* When user selects text > 10 characters:

  * Show small floating tooltip near selection

Tooltip:

* Text: "Ask about this"
* Button style

---

### When Tooltip Clicked

Frontend behavior:

1. Capture selected text
2. Open chat panel
3. Switch mode to `selected_text`
4. Store selected text in state

State example:

```ts
{
  queryMode: 'global' | 'selected_text'
  selectedText?: string
}
```

Selected text is:

* Shown in chat header (collapsed preview)
* Sent with the question (later backend hookup)

---

## 6. API Interaction (UI Contract Only)

### Endpoint (assumed)

```http
POST /api/chat
```

### Request shape (frontend sends)

```json
{
  "message": "string",
  "query_mode": "global_book_rag | selected_text_rag",
  "selected_text": "string | null"
}
```

### Response shape (frontend expects)

```json
{
  "response": "string"
}
```

⚠️ Frontend must gracefully handle:

* Loading
* 500 errors
* Empty responses

---

## 7. Error & Empty States

* Network error → red toast/banner
* Backend unavailable → "Chat temporarily unavailable"
* Empty input → send disabled

---

## 8. Styling Guidelines

### Look & Feel

* Clean, minimal, technical
* Neutral colors
* Rounded corners
* Smooth transitions

### Accessibility

* Keyboard navigable
* High contrast text
* ARIA labels for buttons

---

## 9. File Structure (Frontend Only)

```text
src/
  components/chat/
    ChatLauncher.tsx
    ChatPanel.tsx
    ChatHeader.tsx
    ChatMessages.tsx
    ChatInput.tsx
    SelectedTextTooltip.tsx
  hooks/
    useChatState.ts
    useTextSelection.ts
  styles/
    chat.module.css
```

---

## 10. Non‑Goals (Explicitly Out of Scope)

* Authentication
* Streaming responses
* Message persistence
* Analytics
* Backend fixes

---

## 11. Definition of Done

Frontend is complete when:

* Chat UI renders correctly in Docusaurus
* User can type questions
* Selected text flow works visually
* No runtime errors
* Backend can be connected without UI changes

---

## 12. Final Instruction to Claude Code

> Implement **only** what is written in this spec.
> Do not refactor, optimize, or extend behavior.
> Ask nothing. Assume nothing.

---

**END OF SPEC — FREEZE AFTER APPROVAL**
