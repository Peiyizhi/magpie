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

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
)

func nativeCacheRelay(t *testing.T, proto provider.Protocol, refuse func([]byte) (int, string)) (func(string, string) (int, string), func() [][]byte) {
	t.Helper()
	fresh(t)
	var mu sync.Mutex
	var sent [][]byte
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
			w.WriteHeader(500)
			return
		}
		mu.Lock()
		sent = append(sent, body)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if refuse != nil {
			if code, reply := refuse(body); code != 0 {
				w.WriteHeader(code)
				io.WriteString(w, reply)
				return
			}
		}
		if proto == provider.Responses {
			io.WriteString(w, responsesReply)
		} else {
			io.WriteString(w, `{"id":"c","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}]}`)
		}
	}))
	t.Cleanup(up.Close)
	p := provider.Provider{ID: "fake", Name: "Relay", Key: "test-key", Models: []string{"m1"}}
	endpoint := "/v1/responses"
	if proto == provider.Responses {
		p.Responses = up.URL + "/v1"
	} else {
		p.Chat = up.URL + "/v1"
		endpoint = "/v1/chat/completions"
	}
	if err := provider.Save(p); err != nil {
		t.Fatal(err)
	}
	s := New()
	ask := func(body, session string) (int, string) {
		r := httptest.NewRequest("POST", endpoint, strings.NewReader(body))
		r.Header.Set("session_id", session)
		r.Header.Set("User-Agent", "codex_cli_rs/0.126.0")
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, r)
		return rec.Code, rec.Body.String()
	}
	return ask, func() [][]byte {
		mu.Lock()
		defer mu.Unlock()
		return append([][]byte(nil), sent...)
	}
}

func nativeCacheBody(proto provider.Protocol, more bool) string {
	field := "input"
	if proto == provider.Chat {
		field = "messages"
	}
	turns := `{"role":"user","content":"Hello"}`
	if more {
		turns += `,{"role":"assistant","content":"OK"},{"role":"user","content":"Continue"}`
	}
	return `{"model":"fake/m1","` + field + `":[` + turns + `],"client_metadata":{"source":"test-client"}}`
}

// The relay rejects every keyless request, including a retry. Both native
// APIs keep the original conversation while gaining the translated path's key.
func TestNativePromptCacheKey(t *testing.T) {
	for _, proto := range []provider.Protocol{provider.Responses, provider.Chat} {
		t.Run(string(proto), func(t *testing.T) {
			ask, sent := nativeCacheRelay(t, proto, relayNeedsCacheKey)
			var keys []string
			for _, turn := range []struct {
				session string
				more    bool
			}{{"session-one", false}, {"session-one", true}, {"session-two", false}, {"", false}, {"", true}} {
				body := nativeCacheBody(proto, turn.more)
				if code, reply := ask(body, turn.session); code != 200 {
					t.Fatalf("keyless native request was refused: %d %s", code, reply)
				}
				requests := sent()
				got := requests[len(requests)-1]
				keys = append(keys, gjson.GetBytes(got, cacheKeyField).String())
				var wantFields, gotFields map[string]any
				json.Unmarshal([]byte(body), &wantFields)
				json.Unmarshal(got, &gotFields)
				delete(wantFields, "model")
				delete(gotFields, "model")
				delete(gotFields, cacheKeyField)
				if !reflect.DeepEqual(gotFields, wantFields) {
					t.Fatalf("native request fields changed: got %s, sent %s", got, body)
				}
			}
			if len(sent()) != len(keys) || keys[0] == "" || keys[0] != keys[1] || keys[0] == keys[2] || keys[3] == "" || keys[3] != keys[4] {
				t.Fatalf("keys did not follow conversations: %q (%d requests)", keys, len(sent()))
			}
		})
	}
}

func TestNativePromptCacheKeyChoices(t *testing.T) {
	for _, proto := range []provider.Protocol{provider.Responses, provider.Chat} {
		for _, c := range []struct {
			name, field, extra, want string
			preset, asSent           bool
		}{
			{name: "client", field: `"client-key"`, want: `"client-key"`},
			{name: "empty", field: `""`, want: `""`},
			{name: "null", field: `null`, want: `null`},
			{name: "extra", extra: `{"prompt_cache_key":"${sessionId}"}`, want: `"session-one"`},
			{name: "forced extra", field: `"client-key"`, extra: `{"!prompt_cache_key":"chosen-key"}`, want: `"chosen-key"`},
			{name: "preset", preset: true},
			{name: "as sent", asSent: true},
		} {
			if c.asSent && proto != provider.Responses {
				continue
			}
			t.Run(string(proto)+"/"+c.name, func(t *testing.T) {
				ask, sent := nativeCacheRelay(t, proto, nil)
				p, err := provider.Find("fake")
				if err != nil {
					t.Fatal(err)
				}
				if c.extra != "" {
					if err := json.Unmarshal([]byte(c.extra), &p.ExtraBody); err != nil {
						t.Fatal(err)
					}
				}
				if c.preset {
					p.Preset = "deepseek"
				}
				p.CodexAsSent = c.asSent
				if err := provider.Save(*p); err != nil {
					t.Fatal(err)
				}
				body := nativeCacheBody(proto, false)
				if c.field != "" {
					body = strings.TrimSuffix(body, "}") + `,"prompt_cache_key":` + c.field + `}`
				}
				if code, reply := ask(body, "session-one"); code != 200 {
					t.Fatalf("request: %d %s", code, reply)
				}
				requests := sent()
				if len(requests) != 1 || gjson.GetBytes(requests[0], cacheKeyField).Raw != c.want {
					t.Fatalf("cache key choice changed: %q, want %s", requests, c.want)
				}
			})
		}
	}
}

func TestNativePromptCacheKeyRefused(t *testing.T) {
	for _, proto := range []provider.Protocol{provider.Responses, provider.Chat} {
		t.Run(string(proto), func(t *testing.T) {
			ask, sent := nativeCacheRelay(t, proto, func(body []byte) (int, string) {
				if gjson.GetBytes(body, cacheKeyField).Exists() {
					return 400, `{"error":{"message":"Unrecognized request argument supplied: prompt_cache_key"}}`
				}
				return 0, ""
			})
			for range 2 {
				if code, reply := ask(nativeCacheBody(proto, false), "session-one"); code != 200 {
					t.Fatalf("unsupported key was not recovered: %d %s", code, reply)
				}
			}
			var keys []bool
			for _, body := range sent() {
				keys = append(keys, gjson.GetBytes(body, cacheKeyField).Exists())
			}
			if !reflect.DeepEqual(keys, []bool{true, false, false}) {
				t.Fatalf("refused key was not left out next turn: %v", keys)
			}
		})
	}
}

func TestNativePromptCacheKeyOtherErrors(t *testing.T) {
	for _, proto := range []provider.Protocol{provider.Responses, provider.Chat} {
		for _, message := range []string{"Invalid value for prompt_cache_key", "Invalid input: expected a user message"} {
			t.Run(string(proto)+"/"+message, func(t *testing.T) {
				ask, sent := nativeCacheRelay(t, proto, func([]byte) (int, string) {
					return 400, `{"error":{"message":"` + message + `"}}`
				})
				for range 2 {
					if code, reply := ask(nativeCacheBody(proto, false), "session-one"); code != 400 || !strings.Contains(reply, message) {
						t.Fatalf("original refusal changed: %d %s", code, reply)
					}
				}
				requests := sent()
				if len(requests) != 2 {
					t.Fatalf("refusal retried without a reason: %d requests", len(requests))
				}
				for _, body := range requests {
					if gjson.GetBytes(body, cacheKeyField).String() == "" {
						t.Fatalf("key treated as unsupported: %s", body)
					}
				}
			})
		}
	}
}
