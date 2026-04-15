package main

import (
	"embed"
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

//go:embed templates
var templatesFS embed.FS

var tmpl *template.Template

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

type Handler struct {
	store     *Store
	tools     *ToolRegistry
	ollamaURL string
	maxIter   int
	mu        sync.RWMutex
	model     string // currently active model name
}

// Index serves the full-page shell.
func (h *Handler) Index(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	sessions := h.store.ListSessions()
	render(w, "layout.html", map[string]any{
		"Sessions": sessions,
	})
}

// NewSession creates a session and returns the chat window + OOB sidebar update.
func (h *Handler) NewSession(w http.ResponseWriter, r *http.Request) {
	sess := h.store.CreateSession()
	sessions := h.store.ListSessions()
	render(w, "new_session.html", map[string]any{
		"Session":  sess,
		"Sessions": sessions,
		"Messages": []*Message{},
	})
}

// GetSession returns the chat window fragment for an existing session.
func (h *Handler) GetSession(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	sess := h.store.GetSession(id)
	if sess == nil {
		http.NotFound(w, r)
		return
	}
	msgs := h.store.GetMessages(id)
	render(w, "chat_window.html", map[string]any{
		"Session":  sess,
		"Messages": msgs,
	})
}

// SendMessage runs the agent and appends the exchange to the chat.
func (h *Handler) SendMessage(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	sessionID := r.FormValue("session_id")
	content := strings.TrimSpace(r.FormValue("message"))

	if sessionID == "" || content == "" {
		http.Error(w, "session_id and message required", http.StatusBadRequest)
		return
	}
	sess := h.store.GetSession(sessionID)
	if sess == nil {
		http.NotFound(w, r)
		return
	}

	// Persist user message.
	h.store.AddMessage(sessionID, Message{
		Role:    "user",
		Content: content,
		Time:    time.Now(),
	})

	// Derive title from first user message.
	if sess.Title == "New Chat" {
		title := content
		if len(title) > 48 {
			title = title[:48] + "…"
		}
		h.store.UpdateTitle(sessionID, title)
	}

	// Run agent with the currently selected model.
	h.mu.RLock()
	model := h.model
	h.mu.RUnlock()

	var llm LLM
	if model == "mock" {
		llm = &MockLLM{}
	} else {
		llm = &OllamaLLM{BaseURL: h.ollamaURL, Model: model}
	}
	agent := &Agent{LLM: llm, Tools: h.tools, MaxIter: h.maxIter}

	history := h.store.GetMessages(sessionID)
	result := agent.Run(r.Context(), history)

	// Persist assistant response.
	assistantMsg := h.store.AddMessage(sessionID, Message{
		Role:    "assistant",
		Content: result.Content,
		Time:    time.Now(),
		Trace:   result.Trace,
	})

	// The user bubble is already in the DOM (inserted optimistically by JS).
	// We only return the assistant response fragment, which replaces the
	// thinking indicator that JS injected before the request was sent.
	render(w, "message_pair.html", map[string]any{
		"AssistantMsg": assistantMsg,
		"IsError":      result.IsError,
	})
}

// DeleteSession removes a session.
func (h *Handler) DeleteSession(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	h.store.DeleteSession(id)
	sessions := h.store.ListSessions()
	render(w, "delete_session.html", map[string]any{
		"Sessions": sessions,
	})
}

// GetModels returns the model selector fragment, fetching available models
// from Ollama filtered to allowed families.
func (h *Handler) GetModels(w http.ResponseWriter, r *http.Request) {
	models, _ := ListModels(r.Context(), h.ollamaURL)

	h.mu.RLock()
	current := h.model
	h.mu.RUnlock()

	render(w, "model_selector.html", map[string]any{
		"Models":  models,
		"Current": current,
		"Allowed": allowedPrefixes,
	})
}

// SetModel switches the active model after validating it is in the allowed list.
func (h *Handler) SetModel(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	name := strings.TrimSpace(r.FormValue("model"))

	// Validate: must be an allowed prefix or "mock".
	if name != "mock" && !isAllowed(name) {
		http.Error(w, "model not allowed", http.StatusForbidden)
		return
	}

	// Also verify the model actually exists locally (skip check for mock).
	if name != "mock" {
		available, _ := ListModels(r.Context(), h.ollamaURL)
		found := false
		for _, m := range available {
			if m == name {
				found = true
				break
			}
		}
		if !found {
			http.Error(w, "model not found in Ollama", http.StatusBadRequest)
			return
		}
	}

	h.mu.Lock()
	h.model = name
	h.mu.Unlock()

	// Return refreshed selector so the active highlight updates.
	models, _ := ListModels(r.Context(), h.ollamaURL)
	render(w, "model_selector.html", map[string]any{
		"Models":  models,
		"Current": name,
		"Allowed": allowedPrefixes,
	})
}

// ---------------------------------------------------------------------------
// Template rendering
// ---------------------------------------------------------------------------

func render(w http.ResponseWriter, name string, data any) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := tmpl.ExecuteTemplate(w, name, data); err != nil {
		log.Printf("template %s: %v", name, err)
		http.Error(w, "render error", http.StatusInternalServerError)
	}
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	model := flag.String("model", "gemma4:26b", "Ollama model name, or 'mock' for offline testing")
	ollamaURL := flag.String("ollama", "http://localhost:11434", "Ollama server base URL")
	addr := flag.String("addr", ":3000", "Listen address")
	maxIter := flag.Int("max-iter", 5, "Maximum tool iterations per request")
	shell := flag.Bool("shell", false, "Enable shell_exec tool (use with caution)")
	flag.Parse()

	var err error
	tmpl, err = template.New("").
		Funcs(template.FuncMap{
			"timefmt": func(t time.Time) string { return t.Format("15:04:05") },
			"truncate": func(s string, n int) string {
				if len(s) <= n {
					return s
				}
				return s[:n] + "…"
			},
		}).
		ParseFS(templatesFS, "templates/*.html")
	if err != nil {
		log.Fatalf("parse templates: %v", err)
	}

	if *model == "mock" {
		log.Println("Using mock LLM (no Ollama required)")
	}

	h := &Handler{
		store:     NewStore(),
		tools:     NewToolRegistry(*shell),
		ollamaURL: *ollamaURL,
		maxIter:   *maxIter,
		model:     *model,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", h.Index)
	mux.HandleFunc("POST /session/new", h.NewSession)
	mux.HandleFunc("GET /session/{id}", h.GetSession)
	mux.HandleFunc("POST /chat/send", h.SendMessage)
	mux.HandleFunc("DELETE /session/{id}", h.DeleteSession)
	mux.HandleFunc("GET /models", h.GetModels)
	mux.HandleFunc("POST /model/set", h.SetModel)

	fmt.Printf("\n  Graupel agent harness\n")
	fmt.Printf("  Model   : %s @ %s\n", *model, *ollamaURL)
	fmt.Printf("  Shell   : %v\n", *shell)
	fmt.Printf("  URL     : http://localhost%s\n\n", *addr)

	log.Fatal(http.ListenAndServe(*addr, mux))
}
