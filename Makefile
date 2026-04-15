.PHONY: run mock build test vet clean help

MODEL  ?= gemma4:26b
OLLAMA ?= http://localhost:11434
ADDR   ?= :3000
ITER   ?= 5

## run: start server with Ollama (MODEL, OLLAMA, ADDR, ITER are overridable)
run:
	go run . -model=$(MODEL) -ollama=$(OLLAMA) -addr=$(ADDR) -max-iter=$(ITER)

## mock: start server with mock LLM (no Ollama required)
mock:
	go run . -model=mock -addr=$(ADDR)

## shell: start server with shell_exec enabled (use with caution)
shell:
	go run . -model=$(MODEL) -ollama=$(OLLAMA) -addr=$(ADDR) -shell

## build: compile binary to ./graupel
build:
	go build -o graupel .

## test: run all tests
test:
	go test ./...

## vet: run go vet
vet:
	go vet ./...

## clean: remove compiled binary
clean:
	rm -f graupel

## help: list available targets
help:
	@grep -E '^## ' Makefile | sed 's/## /  /'
