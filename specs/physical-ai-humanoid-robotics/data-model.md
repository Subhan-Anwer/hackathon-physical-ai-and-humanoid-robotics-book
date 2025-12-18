# Data Model: Physical AI & Humanoid Robotics Textbook

## Content Structure

### Module Entity
- **id**: string (unique identifier, e.g., "module-1")
- **title**: string (e.g., "Introduction to Physical AI")
- **description**: string (brief overview of the module)
- **duration**: number (estimated weeks to complete)
- **prerequisites**: array of strings (prerequisite modules or knowledge)
- **learning_objectives**: array of strings (specific learning outcomes)
- **chapters**: array of Chapter entities
- **labs**: array of Lab entities
- **rag_chunks**: array of RAGChunk entities

### Chapter Entity
- **id**: string (unique identifier, e.g., "module-1-chapter-1")
- **title**: string (e.g., "What is Physical AI?")
- **module_id**: string (reference to parent module)
- **position**: number (order within module)
- **learning_objectives**: array of strings (specific to this chapter)
- **content**: string (markdown content)
- **figures**: array of Figure entities
- **code_snippets**: array of CodeSnippet entities
- **exercises**: array of Exercise entities
- **rag_chunks**: array of RAGChunk entities

### Lab Entity
- **id**: string (unique identifier, e.g., "lab-1.1")
- **title**: string (e.g., "Physical AI simulation setup")
- **module_id**: string (reference to parent module)
- **chapter_id**: string (optional reference to related chapter)
- **duration**: string (estimated completion time)
- **objectives**: array of strings (what students will learn)
- **prerequisites**: array of strings (software/hardware requirements)
- **instructions**: string (step-by-step guide)
- **expected_outcomes**: array of strings (what students should achieve)
- **troubleshooting**: string (common issues and solutions)

### Figure Entity
- **id**: string (unique identifier)
- **title**: string (caption/description)
- **path**: string (relative path to image file)
- **alt_text**: string (accessibility description)
- **chapter_id**: string (reference to parent chapter)
- **position**: number (order within chapter)

### CodeSnippet Entity
- **id**: string (unique identifier)
- **language**: string (programming language)
- **code**: string (the actual code)
- **description**: string (what the code does)
- **chapter_id**: string (reference to parent chapter)
- **position**: number (order within chapter)
- **is_executable**: boolean (whether this code can be run)

### Exercise Entity
- **id**: string (unique identifier)
- **type**: string (e.g., "multiple-choice", "coding", "discussion")
- **question**: string (the exercise content)
- **chapter_id**: string (reference to parent chapter)
- **difficulty**: string ("easy", "medium", "hard")
- **solution**: string (optional, for instructor use)
- **position**: number (order within chapter)

### RAGChunk Entity
- **id**: string (unique identifier)
- **content**: string (text content for RAG system)
- **source_id**: string (reference to source entity - chapter, lab, etc.)
- **source_type**: string ("chapter", "lab", "figure", "code_snippet", etc.)
- **tags**: array of strings (for categorization and search)
- **embedding_vector**: array of numbers (for semantic search, computed during processing)

### UserProgress Entity
- **id**: string (unique identifier)
- **user_id**: string (identifier for the learner)
- **module_id**: string (reference to module)
- **chapter_id**: string (reference to chapter)
- **completed**: boolean (whether this content is completed)
- **progress_percentage**: number (0-100)
- **last_accessed**: datetime (timestamp of last interaction)
- **time_spent**: number (in minutes)

## Relationships

### Module contains:
- One-to-many with Chapter
- One-to-many with Lab

### Chapter contains:
- One-to-many with Figure
- One-to-many with CodeSnippet
- One-to-many with Exercise
- One-to-many with RAGChunk

### Lab contains:
- One-to-many with RAGChunk

### RAGChunk references:
- One-to-one with source content (Chapter, Lab, Figure, etc.)

## Validation Rules

### Module
- title is required and must be 1-100 characters
- duration must be between 1-4 weeks
- learning_objectives array must contain 1-10 items
- id must be unique across all modules

### Chapter
- title is required and must be 1-100 characters
- position must be unique within parent module
- content must be in valid markdown format
- id must be unique across all chapters

### Lab
- title is required and must be 1-100 characters
- duration must follow format "X hours" or "X-Y hours"
- objectives array must contain 1-5 items

### CodeSnippet
- language must be a supported programming language
- code content must be syntactically valid for the language
- position must be unique within parent chapter

### Exercise
- type must be one of allowed values
- difficulty must be one of "easy", "medium", "hard"
- position must be unique within parent chapter

### RAGChunk
- content length must be between 100-2000 characters for optimal performance
- tags array must contain 1-5 items
- source references must exist