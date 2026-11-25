"""The AION-Gym server: WebSocket event stream + HTTP action API.

OWNER LAW (verbatim, 2026-11-25): "Python remains authoritative. Godot subscribes
to events and requests permitted actions. It never writes directly into plastic
state or bypasses the AION runtime's authority controls."

Surface (localhost-bound by default — the loopback bind IS the boundary):

  WebSocket  ws://<host>:<ws_port>
    on connect   -> one state.snapshot event, then every produced event pushed
                    live (the emitter is polled by the server's pump task).
    client->srv  -> {"type": "action.request", "action": ..., "args": {...},
                     "request_id": "...", "token": "..."}   (token required when
                     the server is configured with one)
    srv->client  -> schema events (validate()d before send, always) and
                    {"type": "action.result", ...} / {"type": "action.error", ...}

  HTTP       http://<host>:<http_port>
    GET  /health   -> {"ok": true, "service": "aion-gym", "schema_version": 1, ...}
    GET  /snapshot -> one serialized state.snapshot event
    GET  /events   -> the last queued events (JSON array, default cap 50)
    POST /action   -> body {"action", "args", "request_id"}; header
                      "X-AION-Token" checked when a token is configured.
                      Returns the structured action result (denials included).

AUTH LAW: reads (health / snapshot / events / the WS stream) are open on the
localhost bind; ACTION requests require the configured token (HTTP header or WS
message field). No token configured = localhost trust. The server never writes
plastic state and never executes anything itself: it forwards action requests to
GymActions, which routes to existing runtime entry points or denies.

DEPENDENCY (pre-existing in this env, NOT installed by this lane):
    websockets >= 11   (present: 15.0.1). If it is missing the WebSocket side
    raises GymBridgeError with this note at start(); the HTTP side still runs.
    No new packages were added by this module.
"""

import asyncio
import functools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from aion.gym_bridge.actions import GymActions
from aion.gym_bridge.emitter import GymBridgeError, GymEmitter
from aion.gym_bridge.schema import SCHEMA_VERSION, deserialize, serialize

try:                                   # deferred-failure: HTTP-only usage still works
    import websockets                  # noqa: F401  (see DEPENDENCY note above)
    _WEBSOCKETS_AVAILABLE = True
except ImportError:                    # pragma: no cover - env-dependent
    _WEBSOCKETS_AVAILABLE = False

WS_EVENTS_PUMP_POLL_S = 0.05           # emitter poll cadence for the WS pump
DEFAULT_EVENTS_CAP = 50


class GymServer:
    """WebSocket + HTTP front for one GymEmitter + GymActions pair."""

    def __init__(self, emitter: GymEmitter, actions: GymActions,
                 host: str = "127.0.0.1",
                 ws_port: int = 0, http_port: int = 0,
                 token: Optional[str] = None,
                 events_cap: int = DEFAULT_EVENTS_CAP):
        if emitter is None or actions is None:
            raise GymBridgeError("GymServer needs both an emitter and actions")
        self.emitter = emitter
        self.actions = actions
        self.host = host
        self.ws_port = int(ws_port)
        self.http_port = int(http_port)
        self.token = token
        self.events_cap = int(events_cap)
        self._ws_server = None
        self._http_server: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._clients: List[Any] = []           # live websockets
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._poller_task: Optional[asyncio.Task] = None
        self._last_poll_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # the action envelope (shared by the WS and HTTP paths, unit-testable)
    # ------------------------------------------------------------------ #

    def handle_action_request(self, payload: Any, authenticated: bool,
                              client: str = "wire") -> Dict:
        """One wire action request -> one action.result message. Explicit about
        malformed requests (never silent, never a crash)."""
        if not isinstance(payload, dict):
            return {"type": "action.error",
                    "detail": f"request must be a JSON object, got {type(payload).__name__}"}
        if payload.get("type") != "action.request":
            return {"type": "action.error",
                    "detail": (f"client messages must have type 'action.request', "
                               f"got {payload.get('type')!r}")}
        before = self.emitter.state_fingerprint()
        result = self.actions.request(
            action=str(payload.get("action")),
            args=payload.get("args") or {},
            client=client,
            authenticated=bool(authenticated),
            request_id=payload.get("request_id"),
        )
        out = {"type": "action.result", **result}
        if not result.get("ok", False):
            # denial audit: the observed runtime state must be byte-identical
            # across the denial — surfaced to the client, never just logged
            out["state_unchanged"] = self.emitter.state_fingerprint() == before
        return out

    def _check_token(self, presented: Any) -> bool:
        """Token gate for ACTION requests (reads stay open on localhost)."""
        if self.token is None:
            return True
        return isinstance(presented, str) and presented == self.token

    # ------------------------------------------------------------------ #
    # WebSocket side
    # ------------------------------------------------------------------ #

    async def start_async(self) -> Dict[str, int]:
        """Bind both listeners. Returns the actual (ephemeral-capable) ports."""
        if not _WEBSOCKETS_AVAILABLE:
            raise GymBridgeError(
                "the WebSocket side needs the 'websockets' package "
                "(dependency note: websockets>=11; present in this env at 15.0.1; "
                "NOT installed by this lane)")
        import websockets
        self._loop = asyncio.get_running_loop()
        self._ws_server = await websockets.serve(
            self._ws_handler, self.host, self.ws_port)
        self.ws_port = int(self._ws_server.sockets[0].getsockname()[1])

        # a subscription that is thread-safe (the emitter may be polled from
        # this loop or from the HTTP thread)
        loop = self._loop

        def on_event(event: Dict) -> None:
            for ws in list(self._clients):
                loop.call_soon_threadsafe(self._ws_queue_put, ws, event)

        self.emitter.subscribe(on_event)
        self._poller_task = loop.create_task(self._poll_loop())
        self._start_http()
        return {"ws_port": self.ws_port, "http_port": self.http_port}

    async def _poll_loop(self) -> None:
        """The server is the single poll driver: deltas become events, events
        reach every subscriber. A failed readout is RECORDED (surfaced in
        /health), never silently swallowed."""
        while True:
            await asyncio.sleep(WS_EVENTS_PUMP_POLL_S)
            try:
                self.emitter.poll()
                self._last_poll_error = None
            except Exception as exc:            # noqa: BLE001 - recorded, surfaced
                self._last_poll_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _ws_queue_put(ws: Any, event: Dict) -> None:
        q = getattr(ws, "_gym_queue", None)
        if q is not None:
            q.put_nowait(event)

    async def _ws_handler(self, ws: Any) -> None:
        from websockets.exceptions import ConnectionClosed
        self._clients.append(ws)
        ws._gym_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        try:
            await ws.send(serialize(self.emitter.snapshot()))
            pump = asyncio.create_task(self._ws_pump(ws))
            try:
                async for raw in ws:
                    # the runtime call behind an action may block (a real
                    # pathway step) — keep the loop responsive, offload it
                    reply = await loop.run_in_executor(
                        None, functools.partial(self._on_ws_message, raw,
                                                client=f"ws:{id(ws)}"))
                    if reply is not None:
                        await ws.send(json.dumps(reply, sort_keys=True))
            finally:
                pump.cancel()
        except ConnectionClosed:
            pass
        finally:
            if ws in self._clients:
                self._clients.remove(ws)

    async def _ws_pump(self, ws: Any) -> None:
        """Forward queued events (from the emitter subscription) to the client."""
        from websockets.exceptions import ConnectionClosed
        q: "asyncio.Queue[Dict]" = ws._gym_queue
        while True:
            event = await q.get()
            try:
                await ws.send(serialize(event))
            except ConnectionClosed:
                return

    def _on_ws_message(self, raw: Any, client: str) -> Optional[Dict]:
        """Parse + authorize + route one WS message (runs on the loop; the
        blocking runtime call is offloaded by the caller-side executor)."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            return {"type": "action.error", "detail": f"not valid JSON: {exc}"}
        if self.token is not None and payload.get("token") != self.token:
            # unauthenticated action attempt: structured denial, still logged
            payload = dict(payload)
            payload["token"] = None
            return self.handle_action_request(payload, authenticated=False,
                                              client=client)
        return self.handle_action_request(payload, authenticated=True, client=client)

    # ------------------------------------------------------------------ #
    # HTTP side
    # ------------------------------------------------------------------ #

    def _make_http_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:      # quiet by default
                pass

            def _json(self, code: int, body: Any) -> None:
                data = json.dumps(body, sort_keys=True).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path.startswith("/health"):
                    self._json(200, {
                        "ok": True,
                        "service": "aion-gym",
                        "schema_version": SCHEMA_VERSION,
                        "paused": bool(server.emitter.paused),
                        "ws_clients": len(server._clients),
                        "actions_logged": len(server.actions.log),
                        "poll_error": server._last_poll_error,
                    })
                elif self.path.startswith("/snapshot") \
                        or self.path.startswith("/api/snapshot"):
                    try:
                        self._json(200, server.emitter.snapshot())
                    except Exception as exc:
                        self._json(500, {"ok": False,
                                         "detail": f"{type(exc).__name__}: {exc}"})
                elif self.path.startswith("/events"):
                    q = list(server.emitter._queue)[-server.events_cap:]
                    self._json(200, {"events": q})
                else:
                    self._json(404, {"ok": False, "detail": "unknown route"})

            def do_POST(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length", 0) or 0)
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except ValueError as exc:
                    self._json(400, {"ok": False, "detail": f"bad JSON body: {exc}"})
                    return
                if not isinstance(body, dict):
                    body = {}
                path = self.path
                if path.startswith("/api/actions/"):
                    # Godot-client route: POST /api/actions/<action>. The body IS
                    # the args dict (the client sends exactly what the action
                    # needs — approve_request's nested {"op", "args", "arm"}
                    # travels inside it untouched).
                    action = path[len("/api/actions/"):].split("?")[0]
                    payload = {"action": action, "args": body}
                elif path.startswith("/action"):
                    payload = body
                else:
                    self._json(404, {"ok": False, "detail": "unknown route"})
                    return
                self._handle_action(payload)

            def _handle_action(self, payload: Dict) -> None:
                if not server._check_token(self.headers.get("X-AION-Token")):
                    self._json(403, {"ok": False, "reason": "unauthenticated",
                                     "detail": "action requests require the token "
                                               "(X-AION-Token header)"})
                    return
                payload = dict(payload)
                payload.setdefault("type", "action.request")
                self._json(200, server.handle_action_request(
                    payload, authenticated=True, client="http"))

        return Handler

    def _start_http(self) -> None:
        self._http_server = ThreadingHTTPServer((self.host, self.http_port),
                                                self._make_http_handler())
        self.http_port = int(self._http_server.server_address[1])
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            kwargs={"poll_interval": 0.05}, daemon=True, name="aion-gym-http")
        self._http_thread.start()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    async def stop_async(self) -> None:
        if self._poller_task is not None:
            self._poller_task.cancel()
            try:
                await self._poller_task
            except asyncio.CancelledError:
                pass
            self._poller_task = None
        if self._ws_server is not None:
            self._ws_server.close()
            await self._ws_server.wait_closed()
            self._ws_server = None
        self._stop_http()

    def _stop_http(self) -> None:
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=2.0)
            self._http_thread = None

    def serve_forever(self) -> None:
        """Blocking convenience: run both listeners until KeyboardInterrupt."""
        async def _run() -> None:
            ports = await self.start_async()
            print(f"aion-gym: ws :{ports['ws_port']}  http :{ports['http_port']} "
                  f"(token {'on' if self.token else 'off'})")
            try:
                await asyncio.Future()          # run until cancelled
            finally:
                await self.stop_async()
        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            pass
