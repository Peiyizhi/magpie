package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

const azureOtherResource = "The requested item was created under a different Azure OpenAI resource. Use the same resource that created the item to access it."

// Azure rejects the old seal on every attempt, so removing a cache key or
// retrying the same history cannot make this fake succeed.
func TestAzureResourceReasoningRecovery(t *testing.T) {
	for _, mode := range []string{"http", "stream"} {
		t.Run(mode, func(t *testing.T) {
			fresh(t)
			type request struct {
				Input    []map[string]any `json:"input"`
				CacheKey string           `json:"prompt_cache_key"`
			}
			var mu sync.Mutex
			var sent []request
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var q request
				if err := json.NewDecoder(r.Body).Decode(&q); err != nil {
					t.Errorf("decode request: %v", err)
					w.WriteHeader(500)
					return
				}
				mu.Lock()
				sent = append(sent, q)
				mu.Unlock()
				foreign := false
				for _, it := range q.Input {
					foreign = foreign || it["encrypted_content"] == "sealed-by-old-resource"
				}
				message := "OpenAI Responses bad request: " + azureOtherResource + " [trace_id=test-trace]"
				if foreign && mode == "http" {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(400)
					json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"type": "invalid_request_error", "message": message}})
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				if foreign {
					io.WriteString(w, sse(
						`data: {"type":"response.created","response":{"id":"resp_old","model":"gpt-6-astra"}}`,
						`data: {"type":"response.failed","response":{"id":"resp_old","status":"failed","error":{"code":"invalid_request_error","message":"`+message+`"}}}`))
					return
				}
				io.WriteString(w, sse(
					`data: {"type":"response.created","response":{"id":"resp_new","model":"gpt-6-astra"}}`,
					`data: {"type":"response.output_text.delta","delta":"recovered"}`,
					`data: {"type":"response.completed","response":{"id":"resp_new","status":"completed","usage":{"input_tokens":7,"output_tokens":1}}}`))
			}))
			t.Cleanup(up.Close)
			if err := provider.Save(provider.Provider{ID: "agentrouter", Name: "agentrouter", Key: "test-key", Models: []string{"gpt-6-astra"}, Responses: up.URL + "/v1"}); err != nil {
				t.Fatal(err)
			}
			history := []map[string]any{
				{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_text", "text": "Read the example file."}}},
				{"type": "function_call", "call_id": "call_read", "name": "read_file", "arguments": `{"path":"example.txt"}`},
				{"type": "function_call_output", "call_id": "call_read", "output": "Keep this tool result intact."},
				{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "I have read the file."}}},
			}
			seal := func(resource string) map[string]any {
				return map[string]any{"type": "reasoning", "id": "rs_" + resource, "summary": []any{}, "encrypted_content": "sealed-by-" + resource + "-resource"}
			}
			old := seal("old")
			cacheKey := "thread-azure-resource-" + mode
			s := New()
			ask := func(items []map[string]any) {
				t.Helper()
				body := mustJSON(map[string]any{"model": "agentrouter/gpt-6-astra", "stream": true, "store": false, "prompt_cache_key": cacheKey, "input": items})
				rec := httptest.NewRecorder()
				r := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(string(body)))
				r.Header.Set("session_id", cacheKey)
				s.Handler().ServeHTTP(rec, r)
				if rec.Code != 200 || !strings.Contains(rec.Body.String(), "recovered") || strings.Contains(rec.Body.String(), "response.failed") {
					t.Fatalf("conversation did not recover: %d %s", rec.Code, rec.Body)
				}
			}
			ask(append(append([]map[string]any{}, history...), old))
			mu.Lock()
			first := sent
			sent = nil
			mu.Unlock()
			if len(first) != 2 || !reflect.DeepEqual(first[1].Input, history) || first[1].CacheKey != cacheKey {
				t.Fatalf("recovery did not preserve history and cache key: %+v", first)
			}

			// The rejected seal stays out next turn; the new resource's seal
			// still reaches it, along with all the visible history.
			current := seal("current")
			ask(append(append([]map[string]any{}, history...), old, current))
			mu.Lock()
			next := sent
			mu.Unlock()
			want := append(append([]map[string]any{}, history...), current)
			if len(next) != 1 || !reflect.DeepEqual(next[0].Input, want) || next[0].CacheKey != cacheKey {
				t.Fatalf("next turn did not keep only the readable seal: %+v", next)
			}
		})
	}
}

func TestAzureResourceReasoningWords(t *testing.T) {
	if !foreignReasoning.MatchString(azureOtherResource) {
		t.Fatal("Azure's cross-resource error is not recognized")
	}
	for _, message := range []string{
		"Azure OpenAI resource not found",
		"Access to this Azure OpenAI resource is forbidden",
		"Insufficient quota available for instant inference.",
		"The requested model was created under a different deployment.",
	} {
		if foreignReasoning.MatchString(message) {
			t.Errorf("unrelated error would discard history: %s", message)
		}
	}
}
