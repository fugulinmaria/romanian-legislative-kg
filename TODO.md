# TODO - Romanian Legislative Knowledge Graph

## ✅ Completed
- [x] Create modular project structure (src/ package)
- [x] Implement legislative text generator
- [x] Add LLM-based triple extraction
- [x] Build knowledge graph with NetworkX
- [x] Add ontological reasoning (6 axiom types)
- [x] Implement vector search with Chroma
- [x] Add pre-commit hooks with ruff
- [x] Create comprehensive documentation

## 🚧 In Progress
- [ ] Improve triple extraction quality (currently low yield with gemma2:9b/27b)
  - Enhance prompt engineering
  - Better parsing logic
  - Test different models (llama3.1:8b, etc.)

## 📋 Backlog

### High Priority
- [ ] Test pipeline with larger batch of laws (20-50 laws)
- [ ] Validate ontology reasoning on realistic data
- [ ] Add error handling for LLM failures
- [ ] Create sample integration with real Romanian legislative data

### Medium Priority
- [ ] Implement RAG (Retrieval Augmented Generation) for Q&A
  - "What laws regulate digital privacy?"
  - "Show me laws modified by Law X"
- [ ] Add clustering/topic modeling for laws
- [ ] Export knowledge graph to standard formats (RDF, JSON-LD)
- [ ] Create interactive visualization (Plotly/Cytoscape)

### Low Priority
- [ ] Add unit tests for core modules
- [ ] Create CI/CD pipeline
- [ ] Add command-line interface (CLI)
- [ ] Performance optimization for large datasets
- [ ] Docker containerization

## 🐛 Known Issues
- Triple extraction yields only 5-10 triples per law (should be 20-30+)
- LLM prompt needs refinement for Romanian legislative text
- Vector search demo uses only 3 predefined queries

## 💡 Ideas
- Integration with official Romanian legislative database
- Timeline visualization of law modifications
- Conflict detection between laws
- Automated legal compliance checking
