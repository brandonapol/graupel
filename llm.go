package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// LLM is the abstraction over any language model backend.
type LLM interface {
	Complete(ctx context.Context, messages []Message) (string, error)
}

// ---------------------------------------------------------------------------
// Ollama client
// ---------------------------------------------------------------------------

// OllamaLLM calls a local Ollama server (default http://localhost:11434).
type OllamaLLM struct {
	BaseURL string
	Model   string
}

type ollamaMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaChatReq struct {
	Model    string      `json:"model"`
	Messages []ollamaMsg `json:"messages"`
	Stream   bool        `json:"stream"`
}

type ollamaChatResp struct {
	Message ollamaMsg `json:"message"`
	Done    bool      `json:"done"`
	Error   string    `json:"error,omitempty"`
}

func (o *OllamaLLM) Complete(ctx context.Context, messages []Message) (string, error) {
	omsgs := make([]ollamaMsg, 0, len(messages))
	for _, m := range messages {
		omsgs = append(omsgs, ollamaMsg{Role: m.Role, Content: m.Content})
	}

	body, err := json.Marshal(ollamaChatReq{
		Model:    o.Model,
		Messages: omsgs,
		Stream:   false,
	})
	if err != nil {
		return "", fmt.Errorf("marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		o.BaseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama HTTP %d: %s", resp.StatusCode, raw)
	}

	var cr ollamaChatResp
	if err := json.Unmarshal(raw, &cr); err != nil {
		return "", fmt.Errorf("unmarshal: %w", err)
	}
	if cr.Error != "" {
		return "", fmt.Errorf("ollama error: %s", cr.Error)
	}
	return cr.Message.Content, nil
}

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

// allowedPrefixes are the only model families users may select.
var allowedPrefixes = []string{"gemma3", "gemma4", "chatgpt-oss"}

// ollamaTagsResp is the shape of GET /api/tags.
type ollamaTagsResp struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

// ListModels queries Ollama for locally-pulled models and returns only those
// whose names begin with an allowed prefix. Returns an empty slice (not an
// error) when Ollama is unreachable so callers can show the install prompt.
func ListModels(ctx context.Context, baseURL string) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/api/tags", nil)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		// Ollama not running — not an error we surface as an error.
		return nil, nil //nolint:nilerr
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var tags ollamaTagsResp
	if err := json.Unmarshal(raw, &tags); err != nil {
		return nil, err
	}

	var out []string
	for _, m := range tags.Models {
		if isAllowed(m.Name) {
			out = append(out, m.Name)
		}
	}
	return out, nil
}

func isAllowed(name string) bool {
	lower := strings.ToLower(name)
	for _, p := range allowedPrefixes {
		if strings.HasPrefix(lower, p) {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Mock (useful for local testing without Ollama running)
// ---------------------------------------------------------------------------

// MockLLM returns a canned response. Activate via -model=mock flag.
type MockLLM struct{}

func (m *MockLLM) Complete(_ context.Context, messages []Message) (string, error) {
	last := ""
	for _, msg := range messages {
		if msg.Role == "user" {
			last = msg.Content
		}
	}
	return fmt.Sprintf("(mock) You said: %q — Ollama is not running or model is set to 'mock'.", last), nil
}
