package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"sync"
	"time"
)

func newID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// TraceItem records one tool call within an agent turn.
type TraceItem struct {
	ToolName   string
	ToolInput  string
	ToolOutput string
}

// Message is a single chat turn.
type Message struct {
	ID        string
	SessionID string
	Role      string // "user" | "assistant" | "system"
	Content   string
	Time      time.Time
	Trace     []TraceItem
}

// Session is a named conversation.
type Session struct {
	ID        string
	Title     string
	CreatedAt time.Time
	UpdatedAt time.Time
}

// Store holds all sessions and messages in memory.
type Store struct {
	mu       sync.RWMutex
	sessions map[string]*Session
	messages map[string][]*Message
	order    []string // session IDs newest-first
}

func NewStore() *Store {
	return &Store{
		sessions: make(map[string]*Session),
		messages: make(map[string][]*Message),
	}
}

func (s *Store) CreateSession() *Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	sess := &Session{
		ID:        newID(),
		Title:     "New Chat",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	s.sessions[sess.ID] = sess
	s.messages[sess.ID] = nil
	s.order = append([]string{sess.ID}, s.order...)
	return sess
}

func (s *Store) GetSession(id string) *Session {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.sessions[id]
}

func (s *Store) ListSessions() []*Session {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*Session, 0, len(s.order))
	for _, id := range s.order {
		if sess, ok := s.sessions[id]; ok {
			out = append(out, sess)
		}
	}
	return out
}

func (s *Store) UpdateTitle(id, title string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if sess, ok := s.sessions[id]; ok {
		sess.Title = title
		sess.UpdatedAt = time.Now()
	}
}

func (s *Store) DeleteSession(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.sessions, id)
	delete(s.messages, id)
	newOrder := make([]string, 0, len(s.order))
	for _, oid := range s.order {
		if oid != id {
			newOrder = append(newOrder, oid)
		}
	}
	s.order = newOrder
}

func (s *Store) AddMessage(sessionID string, msg Message) *Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	msg.ID = newID()
	msg.SessionID = sessionID
	m := &msg
	s.messages[sessionID] = append(s.messages[sessionID], m)
	if sess, ok := s.sessions[sessionID]; ok {
		sess.UpdatedAt = time.Now()
	}
	return m
}

func (s *Store) GetMessages(sessionID string) []*Message {
	s.mu.RLock()
	defer s.mu.RUnlock()
	src := s.messages[sessionID]
	out := make([]*Message, len(src))
	copy(out, src)
	return out
}

// ---------------------------------------------------------------------------
// StreamHub — in-flight SSE channel registry
// ---------------------------------------------------------------------------

// StreamEvent is one SSE message pushed to the browser.
type StreamEvent struct {
	// Type is the SSE event name:
	//   token      — LLM token chunk to append
	//   tool_clear — discard buffered tokens (a tool call was detected)
	//   status     — status line update (e.g. "Thinking…", "Using tool: X…")
	//   done       — stream finished normally
	//   error_msg  — stream finished with an error
	Type string
	Data string
}

type streamEntry struct {
	ch     chan StreamEvent
	cancel context.CancelFunc
}

// StreamHub manages one channel per in-flight chat request.
type StreamHub struct {
	mu      sync.Mutex
	entries map[string]streamEntry
}

func NewStreamHub() *StreamHub {
	return &StreamHub{entries: make(map[string]streamEntry)}
}

// Create registers a new buffered channel for id and returns it.
// cancel is called by the SSE handler when the client disconnects.
func (h *StreamHub) Create(id string, cancel context.CancelFunc) chan StreamEvent {
	ch := make(chan StreamEvent, 128)
	h.mu.Lock()
	h.entries[id] = streamEntry{ch: ch, cancel: cancel}
	h.mu.Unlock()
	return ch
}

// Get retrieves the entry for id. The returned value is a copy; call
// entry.cancel() directly to cancel the associated context.
func (h *StreamHub) Get(id string) (streamEntry, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	e, ok := h.entries[id]
	return e, ok
}

// Close closes the channel and removes the entry. Safe to call if id is gone.
func (h *StreamHub) Close(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if e, ok := h.entries[id]; ok {
		close(e.ch)
		delete(h.entries, id)
	}
}
