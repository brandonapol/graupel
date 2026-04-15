package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Tool is the interface every tool must satisfy.
type Tool interface {
	Name() string
	Description() string
	Run(input string) (string, error)
}

// ToolRegistry holds the active set of tools.
type ToolRegistry struct {
	tools map[string]Tool
}

func NewToolRegistry(shellExec bool) *ToolRegistry {
	r := &ToolRegistry{tools: make(map[string]Tool)}
	r.add(&ReadFileTool{})
	r.add(&WriteFileTool{})
	r.add(&ListDirTool{})
	r.add(&HTTPFetchTool{})
	if shellExec {
		r.add(&ShellExecTool{})
	}
	return r
}

func (r *ToolRegistry) add(t Tool)                   { r.tools[t.Name()] = t }
func (r *ToolRegistry) Get(name string) (Tool, bool) { t, ok := r.tools[name]; return t, ok }

func (r *ToolRegistry) Descriptions() string {
	var sb strings.Builder
	for _, t := range r.tools {
		fmt.Fprintf(&sb, "  - %s: %s\n", t.Name(), t.Description())
	}
	return sb.String()
}

// ---------------------------------------------------------------------------
// Path safety
// ---------------------------------------------------------------------------

// safePath resolves p to an absolute path and verifies it is inside the cwd.
func safePath(p string) (string, error) {
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	abs, err := filepath.Abs(strings.TrimSpace(p))
	if err != nil {
		return "", err
	}
	// filepath.Abs cleans the path; ensure it stays inside wd.
	rel, err := filepath.Rel(wd, abs)
	if err != nil || strings.HasPrefix(rel, "..") {
		return "", fmt.Errorf("path %q escapes working directory", p)
	}
	return abs, nil
}

// ---------------------------------------------------------------------------
// read_file
// ---------------------------------------------------------------------------

type ReadFileTool struct{}

func (t *ReadFileTool) Name() string        { return "read_file" }
func (t *ReadFileTool) Description() string { return "Read a text file. Input: relative file path." }

func (t *ReadFileTool) Run(input string) (string, error) {
	path, err := safePath(input)
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// ---------------------------------------------------------------------------
// write_file
// ---------------------------------------------------------------------------

type WriteFileTool struct{}

func (t *WriteFileTool) Name() string { return "write_file" }
func (t *WriteFileTool) Description() string {
	return `Write content to a file (creates parent dirs). Input: JSON {"path":"rel/path","content":"text"}`
}

func (t *WriteFileTool) Run(input string) (string, error) {
	var args struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		return "", fmt.Errorf("input must be JSON {path, content}: %w", err)
	}
	path, err := safePath(args.Path)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(path, []byte(args.Content), 0o644); err != nil {
		return "", err
	}
	return fmt.Sprintf("wrote %d bytes → %s", len(args.Content), args.Path), nil
}

// ---------------------------------------------------------------------------
// list_dir
// ---------------------------------------------------------------------------

type ListDirTool struct{}

func (t *ListDirTool) Name() string        { return "list_dir" }
func (t *ListDirTool) Description() string { return "List directory contents. Input: relative directory path (or . for cwd)." }

func (t *ListDirTool) Run(input string) (string, error) {
	if strings.TrimSpace(input) == "" {
		input = "."
	}
	path, err := safePath(input)
	if err != nil {
		return "", err
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	for _, e := range entries {
		if e.IsDir() {
			fmt.Fprintf(&sb, "[dir]  %s/\n", e.Name())
		} else {
			info, _ := e.Info()
			size := int64(0)
			if info != nil {
				size = info.Size()
			}
			fmt.Fprintf(&sb, "[file] %s  (%d B)\n", e.Name(), size)
		}
	}
	if sb.Len() == 0 {
		return "(empty directory)", nil
	}
	return sb.String(), nil
}

// ---------------------------------------------------------------------------
// http_fetch
// ---------------------------------------------------------------------------

type HTTPFetchTool struct{}

func (t *HTTPFetchTool) Name() string        { return "http_fetch" }
func (t *HTTPFetchTool) Description() string { return "Fetch a URL and return the response body. Input: URL string." }

func (t *HTTPFetchTool) Run(input string) (string, error) {
	url := strings.TrimSpace(input)
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return "", fmt.Errorf("fetch %s: %w", url, err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 64*1024)) // 64 KB cap
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("HTTP %d\n%s", resp.StatusCode, body), nil
}

// ---------------------------------------------------------------------------
// shell_exec (disabled by default)
// ---------------------------------------------------------------------------

type ShellExecTool struct{}

func (t *ShellExecTool) Name() string        { return "shell_exec" }
func (t *ShellExecTool) Description() string { return "Run a shell command. Input: command string. (Use with caution.)" }

func (t *ShellExecTool) Run(input string) (string, error) {
	// ShellExecTool is only registered when -shell flag is passed; the
	// implementation is intentionally left as a stub so that accidental
	// registration without review does nothing harmful.
	return "", fmt.Errorf("shell_exec: not implemented in this build")
}
