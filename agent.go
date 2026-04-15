package main

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"
)

const agentSystemPrompt = `You are a helpful coding assistant with filesystem access.

To call a tool, output a JSON object on its own line — nothing before or after on that line:
{"tool":"TOOL_NAME","input":"INPUT_VALUE"}

For write_file, the input must itself be a JSON string with escaped quotes:
{"tool":"write_file","input":"{\"path\":\"foo.go\",\"content\":\"package main\\n\"}"}

Rules:
1. Use at most ONE tool per response.
2. After the tool result is returned, continue reasoning.
3. When you have a final answer that needs no more tools, write it normally — no JSON.
4. Never invent file contents; read first if unsure.

Available tools:
%s`

// toolCallRE matches a bare JSON object containing "tool" and "input" keys.
// It is intentionally simple: only one tool call per response is supported.
var toolCallRE = regexp.MustCompile(`(?m)^\s*(\{"tool"\s*:.*?\})\s*$`)

type toolCall struct {
	Tool  string `json:"tool"`
	Input string `json:"input"`
}

// AgentResult is what the agent returns to the handler.
type AgentResult struct {
	Content string
	Trace   []TraceItem
}

// Agent runs the LLM + tool loop.
type Agent struct {
	LLM     LLM
	Tools   *ToolRegistry
	MaxIter int // maximum tool iterations per request
}

func (a *Agent) Run(ctx context.Context, history []*Message) AgentResult {
	maxIter := a.MaxIter
	if maxIter <= 0 {
		maxIter = 5
	}

	// Build message slice: system prompt + conversation history.
	sysPrompt := fmt.Sprintf(agentSystemPrompt, a.Tools.Descriptions())
	msgs := []Message{{Role: "system", Content: sysPrompt}}
	for _, m := range history {
		if m.Role != "system" {
			msgs = append(msgs, Message{Role: m.Role, Content: m.Content})
		}
	}

	var trace []TraceItem

	for i := 0; i < maxIter; i++ {
		raw, err := a.LLM.Complete(ctx, msgs)
		if err != nil {
			return AgentResult{
				Content: fmt.Sprintf("⚠ LLM error: %v", err),
				Trace:   trace,
			}
		}

		tc := parseToolCall(raw)
		if tc == nil {
			// No tool call — this is the final answer.
			return AgentResult{Content: strings.TrimSpace(raw), Trace: trace}
		}

		// Execute tool with a hard timeout.
		toolOut := a.runTool(ctx, tc)

		trace = append(trace, TraceItem{
			ToolName:   tc.Tool,
			ToolInput:  tc.Input,
			ToolOutput: toolOut,
		})

		// Feed results back into the conversation.
		msgs = append(msgs, Message{Role: "assistant", Content: raw})
		msgs = append(msgs, Message{
			Role:    "user",
			Content: fmt.Sprintf("[tool result: %s]\n%s", tc.Tool, toolOut),
		})
	}

	return AgentResult{
		Content: fmt.Sprintf("Reached the %d-iteration tool limit without a final answer.", maxIter),
		Trace:   trace,
	}
}

func (a *Agent) runTool(ctx context.Context, tc *toolCall) string {
	tool, ok := a.Tools.Get(tc.Tool)
	if !ok {
		return fmt.Sprintf("unknown tool %q — available: %s", tc.Tool, a.Tools.Descriptions())
	}
	tctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	// Tools don't accept a context directly; use a channel to respect timeout.
	type result struct {
		out string
		err error
	}
	ch := make(chan result, 1)
	go func() {
		out, err := tool.Run(tc.Input)
		ch <- result{out, err}
	}()

	select {
	case <-tctx.Done():
		return fmt.Sprintf("tool %s timed out after 30s", tc.Tool)
	case r := <-ch:
		if r.err != nil {
			return fmt.Sprintf("error: %v", r.err)
		}
		return r.out
	}
}

func parseToolCall(response string) *toolCall {
	match := toolCallRE.FindStringSubmatch(response)
	if len(match) < 2 {
		return nil
	}
	var tc toolCall
	if err := json.Unmarshal([]byte(match[1]), &tc); err != nil {
		return nil
	}
	if tc.Tool == "" {
		return nil
	}
	return &tc
}
