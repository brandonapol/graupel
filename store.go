package main

import (
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
