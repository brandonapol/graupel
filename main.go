package main

import (
	"context"
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

// newContext is a thin wrapper so the call site reads clearly.
func newContext() (context.Context, context.CancelFunc) {
	return context.WithCancel(context.Background())
}

//go:embed templates
var templatesFS embed.FS

var tmpl *template.Template

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

type Handler struct {
	store     *Store
	hub       *StreamHub
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

// SendMessage starts a streaming agent run. It persists the user message,
// creates an SSE channel, launches the agent goroutine, and returns a
// streaming placeholder div that auto-connects via EventSource.
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

	// Persist user message immediately.
	h.store.AddMessage(sessionID, Message{Role: "user", Content: content, Time: time.Now()})

	// Derive title from first message.
	if sess.Title == "New Chat" {
		title := content
		if len(title) > 48 {
			title = title[:48] + "…"
		}
		h.store.UpdateTitle(sessionID, title)
	}

	// Snapshot model now so the goroutine uses the model that was active
	// when the user hit Send, not whatever is selected later.
	h.mu.RLock()
	model := h.model
	h.mu.RUnlock()

	// Create stream. The context is cancelled if the SSE client disconnects,
	// which will abort the in-flight Ollama request.
	streamID := newID()
	streamCtx, cancelStream := newContext()
	events := h.hub.Create(streamID, cancelStream)

	history := h.store.GetMessages(sessionID)

	go func() {
		var llm LLM
		if model == "mock" {
			llm = &MockLLM{}
		} else {
			llm = &OllamaLLM{BaseURL: h.ollamaURL, Model: model}
		}
		agent := &Agent{LLM: llm, Tools: h.tools, MaxIter: h.maxIter}

		result := agent.RunStream(streamCtx, history, events)

		// Persist the assistant message (even on error, to keep history intact).
		h.store.AddMessage(sessionID, Message{
			Role:    "assistant",
			Content: result.Content,
			Time:    time.Now(),
			Trace:   result.Trace,
		})

		// Give the browser a moment to open the SSE connection before we
		// send the terminal event and close the channel. This matters for
		// fast responses (mock, small prompts) where the goroutine can
		// finish before the EventSource GET even arrives at the server.
		time.Sleep(80 * time.Millisecond)

		if streamCtx.Err() == nil {
			if result.IsError {
				events <- StreamEvent{Type: "error_msg", Data: result.Content}
			} else {
				events <- StreamEvent{Type: "done", Data: ""}
			}
		}
		h.hub.Close(streamID)
	}()

	render(w, "message_pair.html", map[string]any{"StreamID": streamID})
}

// StreamHandler serves the SSE endpoint for a single in-flight agent run.
func (h *Handler) StreamHandler(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, hasFlusher := w.(http.Flusher)

	entry, ok := h.hub.Get(id)
	if !ok {
		// Stream already completed before this SSE connection arrived
		// (very fast model or mock). Send a bare done so the UI settles.
		fmt.Fprintf(w, "event: done\ndata: \n\n")
		if hasFlusher {
			flusher.Flush()
		}
		return
	}

	// Cancel the agent context if the browser disconnects.
	defer entry.cancel()

	if !hasFlusher {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case event, open := <-entry.ch:
			if !open {
				return
			}
			// SSE multi-line data: replace newlines so each line gets its own data: prefix.
			data := strings.ReplaceAll(event.Data, "\n", "\ndata: ")
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event.Type, data)
			flusher.Flush()
			if event.Type == "done" || event.Type == "error_msg" {
				return
			}
		}
	}
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
		hub:       NewStreamHub(),
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
	mux.HandleFunc("GET /stream/{id}", h.StreamHandler)
	mux.HandleFunc("DELETE /session/{id}", h.DeleteSession)
	mux.HandleFunc("GET /models", h.GetModels)
	mux.HandleFunc("POST /model/set", h.SetModel)

	fmt.Printf("\n  Graupel agent harness\n")
	fmt.Printf("  Model   : %s @ %s\n", *model, *ollamaURL)
	fmt.Printf("  Shell   : %v\n", *shell)
	fmt.Printf("  URL     : http://localhost%s\n\n", *addr)

	log.Fatal(http.ListenAndServe(*addr, mux))
}
