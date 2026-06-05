C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria>cd aria

Starting ARIA Backend (FastAPI) using standard Python 3.12...
==================================================
Using virtual environment: C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Scripts\python.exe
INFO:     Will watch for changes in these directories: ['C:\\Users\\Cipher\\Desktop\\Python\\2.Projects\\4.Aria\\Aria']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [4272] using WatchFiles
INFO:     Started server process [13184]
INFO:     Waiting for application startup.
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
2026-06-05 11:10:44,544 [INFO] aria: LangSmith tracing enabled: aria-dev
2026-06-05 11:10:44,544 [INFO] aria.main: ARIA started successfully — graph compiled, DB ready, background tasks running.
2026-06-05 11:10:44,544 [INFO] aria.background: Memory decay loop started — interval=24h (86400s)
2026-06-05 11:10:44,544 [INFO] aria.background: Memory decay cycle firing.
2026-06-05 11:10:44,544 [INFO] aria.background: Observability maintenance loop started — interval=24h (86400s)
2026-06-05 11:10:44,544 [INFO] aria.background: Observability maintenance cycle firing.
INFO:     Application startup complete.
2026-06-05 11:10:44,579 [INFO] aria.structured: {"timestamp": "2026-06-05T05:40:44.560213Z", "session_id": null, "user_id": null, "agent": "Observability", "action": "maintenance", "success": true, "latency_ms": 0, "model_used": null, "input_tokens": 0, "output_tokens": 0, "tool_used": null, "error": null, "events_purged": "0"}
2026-06-05 11:10:44,579 [INFO] aria.background: Observability maintenance cycle complete: events_purged=0
2026-06-05 11:10:44,588 [INFO] aria.background: Memory decay cycle complete: decayed=0 pruned=0
2026-06-05 11:11:19,347 [INFO] aria.http: {"request_id": "req_a7fb8006", "method": "GET", "path": "/model-status", "status_code": 200, "latency_ms": 0}
INFO:     127.0.0.1:52362 - "GET /model-status HTTP/1.1" 200 OK
2026-06-05 11:11:21,430 [INFO] aria.http: {"request_id": "req_15a1b6ef", "method": "GET", "path": "/documents", "status_code": 307, "latency_ms": 16}
INFO:     127.0.0.1:52364 - "GET /documents?user_id=default_user HTTP/1.1" 307 Temporary Redirect
2026-06-05 11:11:21,449 [INFO] aria.http: {"request_id": "req_09c63409", "method": "GET", "path": "/documents/", "status_code": 200, "latency_ms": 16}
INFO:     127.0.0.1:52364 - "GET /documents/?user_id=default_user HTTP/1.1" 200 OK   
2026-06-05 11:11:23,513 [INFO] aria.http: {"request_id": "req_7c8d051a", "method": "GET", "path": "/memory", "status_code": 200, "latency_ms": 0}
INFO:     127.0.0.1:52366 - "GET /memory?user_id=default_user&min_importance=0.0 HTTP/1.1" 200 OK
2026-06-05 11:11:25,656 [INFO] aria.http: {"request_id": "req_fceeee0f", "method": "GET", "path": "/analytics", "status_code": 200, "latency_ms": 30}
INFO:     127.0.0.1:52368 - "GET /analytics?days=7 HTTP/1.1" 200 OK
2026-06-05 11:11:33,423 [INFO] aria.http: {"request_id": "req_bb7c659e", "method": "GET", "path": "/model-status", "status_code": 200, "latency_ms": 0}
INFO:     127.0.0.1:52370 - "GET /model-status HTTP/1.1" 200 OK
2026-06-05 11:11:35,481 [INFO] aria.http: {"request_id": "req_5232b456", "method": "GET", "path": "/documents", "status_code": 307, "latency_ms": 0}
INFO:     127.0.0.1:52372 - "GET /documents?user_id=default_user HTTP/1.1" 307 Temporary Redirect
2026-06-05 11:11:35,498 [INFO] aria.http: {"request_id": "req_76cd933e", "method": "GET", "path": "/documents/", "status_code": 200, "latency_ms": 16}
INFO:     127.0.0.1:52372 - "GET /documents/?user_id=default_user HTTP/1.1" 200 OK   
2026-06-05 11:11:40,481 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/embeddings "HTTP/1.1 200 OK"
2026-06-05 11:11:43,547 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/embeddings "HTTP/1.1 200 OK"
2026-06-05 11:11:52,505 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 11:11:53,966 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 11:11:54,002 [INFO] aria.structured: {"timestamp": "2026-06-05T05:41:53.986027Z", "session_id": "sess_c95857d81ead", "user_id": "default_user", "agent": "MemoryReader", "action": "retrieve_episodic", "success": true, "latency_ms": 0, "model_used": null, "input_tokens": 0, "output_tokens": 0, "tool_used": null, "error": null, "memories_retrieved": "0"}
2026-06-05 11:12:00,066 [INFO] httpx: HTTP Request: POST https://api.cohere.com/v1/chat "HTTP/1.1 400 Bad Request"
2026-06-05 11:12:00,244 [WARNING] models.router: Model command-a-plus-05-2026 attempt 1/3 failed: headers: {'access-control-expose-headers': 'X-Debug-Trace-ID', 'cache-control': 'no-cache, no-store, no-transform, must-revalidate, private, max-age=0', 'content-encoding': 'gzip', 'content-type': 'application/json', 'expires': 'Thu, 01 Jan 1970 00:00:00 GMT', 'pragma': 'no-cache', 'vary': 'Origin,Accept-Encoding', 'x-accel-expires': '0', 'x-debug-trace-id': 'b0aae57a1bc79ddfcba999e6897f9d06', 'x-endpoint-monthly-call-limit': '1000', 'x-trial-endpoint-call-limit': '20', 'x-trial-endpoint-call-remaining': '19', 'date': 'Fri, 05 Jun 2026 05:41:53 GMT', 'x-envoy-upstream-service-time': '18', 'server': 'envoy', 'via': '1.1 google', 'alt-svc': 'h3=":443"; ma=2592000', 'transfer-encoding': 'chunked'}, status_code: 400, body: {'id': '11864227-5c7f-4c4a-8c9b-6465ab44a26e', 'message': "invalid request: this model is not supported with '/v1/chat', please use '/v2/chat' instead. More details in the migration guide: https://docs.cohere.com/docs/migrating-v1-to-v2"}
2026-06-05 11:12:01,715 [INFO] httpx: HTTP Request: POST https://api.cohere.com/v1/chat "HTTP/1.1 400 Bad Request"
2026-06-05 11:12:01,719 [WARNING] models.router: Model command-a-reasoning-08-2025 attempt 1/3 failed: headers: {'access-control-expose-headers': 'X-Debug-Trace-ID', 'cache-control': 'no-cache, no-store, no-transform, must-revalidate, private, max-age=0', 'content-encoding': 'gzip', 'content-type': 'application/json', 'expires': 'Thu, 01 
Jan 1970 00:00:00 GMT', 'pragma': 'no-cache', 'vary': 'Origin,Accept-Encoding', 'x-accel-expires': '0', 'x-debug-trace-id': '5e90f4c6ae3ef5f8dece150c163276a5', 'x-endpoint-monthly-call-limit': '1000', 'x-trial-endpoint-call-limit': '20', 'x-trial-endpoint-call-remaining': '18', 'date': 'Fri, 05 Jun 2026 05:41:55 GMT', 'x-envoy-upstream-service-time': '20', 'server': 'envoy', 'via': '1.1 google', 'alt-svc': 'h3=":443"; ma=2592000,h3-29=":443"; ma=2592000', 'transfer-encoding': 'chunked'}, status_code: 400, body: {'id': 'b1494fca-ca9e-4b8d-bbce-85f010374def', 'message': "invalid request: this model is not supported with '/v1/chat', please use '/v2/chat' instead. More details in the migration guide: https://docs.cohere.com/docs/migrating-v1-to-v2"}
2026-06-05 11:12:03,289 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 11:12:04,807 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 11:12:04,866 [INFO] aria.structured: {"timestamp": "2026-06-05T05:42:04.865022Z", "session_id": "sess_c95857d81ead", "user_id": "default_user", "agent": "MemoryWriter", "action": "save_turn", "success": true, "latency_ms": 30, "model_used": "llama-3.3-70b-versatile", "input_tokens": 0, "output_tokens": 0, "tool_used": null, "error": null}
2026-06-05 11:12:05,620 [INFO] aria.structured: {"timestamp": "2026-06-05T05:42:05.620229Z", "session_id": "sess_c95857d81ead", "user_id": "default_user", "agent": "ChatRouter", "action": "chat_request", "success": true, "latency_ms": 28078, "model_used": "llama-3.3-70b-versatile", "input_tokens": 0, "output_tokens": 0, "tool_used": null, "error": null}
INFO:     127.0.0.1:52374 - "POST /chat HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application
  + Exception Group Traceback (most recent call last):
  |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\_utils.py", line 76, in collapse_excgroups
  |     yield
  |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\base.py", line 186, in __call__
  |     async with anyio.create_task_group() as task_group:
  |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\anyio\_backends\_asyncio.py", line 799, in __aexit__
  |     raise BaseExceptionGroup(
  | ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\uvicorn\protocols\http\httptools_impl.py", line 401, in run_asgi
    |     result = await app(  # type: ignore[func-returns-value]
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\uvicorn\middleware\proxy_headers.py", line 60, in __call__
    |     return await self.app(scope, receive, send)
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\fastapi\applications.py", line 1054, in __call__
    |     await super().__call__(scope, receive, send)
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\applications.py", line 113, in __call__
    |     await self.middleware_stack(scope, receive, send)
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\errors.py", line 187, in __call__
    |     raise exc
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\errors.py", line 165, in __call__
    |     await self.app(scope, receive, _send)
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\cors.py", line 85, in __call__
    |     await self.app(scope, receive, send)
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\base.py", line 185, in __call__
    |     with collapse_excgroups():
    |   File "C:\Users\Cipher\AppData\Local\Programs\Python\Python312\Lib\contextlib.py", line 155, in __exit__
    |     self.gen.throw(value)
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\_utils.py", line 82, in collapse_excgroups
    |     raise exc
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\base.py", line 187, in __call__
    |     response = await self.dispatch_func(request, call_next)
    |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\api\middleware\request_logger.py", line 34, in dispatch
    |     response: Response = await call_next(request)
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\base.py", line 163, in call_next
    |     raise app_exc
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\base.py", line 149, in coro
    |     await self.app(scope, receive_or_disconnect, send_no_error)
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\exceptions.py", line 62, in __call__
    |     await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)   
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    |     raise exc
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    |     await app(scope, receive, sender)
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\routing.py", line 715, in __call__
    |     await self.middleware_stack(scope, receive, send)
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\routing.py", line 735, in app
    |     await route.handle(scope, receive, send)
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\routing.py", line 288, in handle
    |     await self.app(scope, receive, send)
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\routing.py", line 76, in app
    |     await wrap_app_handling_exceptions(app, request)(scope, receive, send)     
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    |     raise exc
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    |     await app(scope, receive, sender)
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\routing.py", line 73, in app
    |     response = await f(request)
    |                ^^^^^^^^^^^^^^^^
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\fastapi\routing.py", line 301, in app
    |     raw_response = await run_endpoint_function(
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\fastapi\routing.py", line 212, in run_endpoint_function
    |     return await dependant.call(**values)
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\api\routes\chat.py", line 125, in chat
    |     primary_chain = list(cfg.primary_chain)
    |                          ^^^^^^^^^^^^^^^^^
    |   File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\pydantic\main.py", line 856, in __getattr__
    |     raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')
    | AttributeError: 'ModelTierConfig' object has no attribute 'primary_chain'      
    +------------------------------------

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\uvicorn\protocols\http\httptools_impl.py", line 401, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\uvicorn\middleware\proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\fastapi\applications.py", line 1054, in __call__
    await super().__call__(scope, receive, send)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\applications.py", line 113, in __call__
    await self.middleware_stack(scope, receive, send)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\errors.py", line 187, in __call__
    raise exc
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\errors.py", line 165, in __call__
    await self.app(scope, receive, _send)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\cors.py", line 85, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\base.py", line 185, in __call__
    with collapse_excgroups():
  File "C:\Users\Cipher\AppData\Local\Programs\Python\Python312\Lib\contextlib.py", line 155, in __exit__
    self.gen.throw(value)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\_utils.py", line 82, in collapse_excgroups
    raise exc
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\base.py", line 187, in __call__
    response = await self.dispatch_func(request, call_next)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\api\middleware\request_logger.py", line 34, in dispatch
    response: Response = await call_next(request)
                         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\base.py", line 163, in call_next
    raise app_exc
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\base.py", line 149, in coro
    await self.app(scope, receive_or_disconnect, send_no_error)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\middleware\exceptions.py", line 62, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\routing.py", line 715, in __call__
    await self.middleware_stack(scope, receive, send)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\routing.py", line 735, in app
    await route.handle(scope, receive, send)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\routing.py", line 288, in handle
    await self.app(scope, receive, send)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\routing.py", line 76, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\starlette\routing.py", line 73, in app
    response = await f(request)
               ^^^^^^^^^^^^^^^^
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\fastapi\routing.py", line 301, in app
    raw_response = await run_endpoint_function(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\fastapi\routing.py", line 212, in run_endpoint_function
    return await dependant.call(**values)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\api\routes\chat.py", line 125, in chat
    primary_chain = list(cfg.primary_chain)
                         ^^^^^^^^^^^^^^^^^
  File "C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\pydantic\main.py", line 856, in __getattr__
    raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')AttributeError: 'ModelTierConfig' object has no attribute 'primary_chain'  
2026-06-05 11:12:05,989 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/om/openai/v1/chat/completions "HTTP/1.1 429 Too Many Requests"
2026-06-05 11:12:05,991 [INFO] groq._base_client: Retrying request to /openai/v1/chatai/v1/chat/completions in 15.000000 seconds
2026-06-05 11:12:07,692 [INFO] aria.http: {"request_id": "req_6bd00501", "method": "Gethod": "GET", "path": "/memory", "status_code": 200, "latency_ms": 0}     
INFO:     127.0.0.1:52406 - "GET /memory?user_id=default_user&min_importance=0.0 HTTPe=0.0 HTTP/1.1" 200 OK
2026-06-05 11:12:09,761 [INFO] aria.http: {"request_id": "req_29f0814c", "method": "Gethod": "GET", "path": "/analytics", "status_code": 200, "latency_ms": 31} 
INFO:     127.0.0.1:52409 - "GET /analytics?days=7 HTTP/1.1" 200 OK        
2026-06-05 11:12:21,345 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/om/openai/v1/chat/completions "HTTP/1.1 200 OK"
WARNING:  WatchFiles detected changes in 'api\routes\chat.py'. Reloading...INFO:     Shutting down
INFO:     Waiting for application shutdown.
2026-06-05 11:14:10,799 [INFO] aria.main: ARIA shutting down — cancelling background tasks.
2026-06-05 11:14:10,799 [INFO] aria.background: Memory decay loop cancelled during sleep — shutting down.
2026-06-05 11:14:10,799 [INFO] aria.background: Observability maintenance loop cancelled during sleep — shutting down.
2026-06-05 11:14:10,799 [INFO] aria.main: ARIA shut down gracefully.       
INFO:     Application shutdown complete.
INFO:     Finished server process [13184]
INFO:     Started server process [16680]
INFO:     Waiting for application startup.
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
2026-06-05 11:14:24,743 [INFO] aria: LangSmith tracing enabled: aria-dev
2026-06-05 11:14:24,743 [INFO] aria.main: ARIA started successfully — graph compiled, DB ready, background tasks running.
2026-06-05 11:14:24,743 [INFO] aria.background: Memory decay loop started — interval=24h (86400s)
2026-06-05 11:14:24,743 [INFO] aria.background: Memory decay cycle firing. 
2026-06-05 11:14:24,758 [INFO] aria.background: Observability maintenance l loop started — interval=24h (86400s)
2026-06-05 11:14:24,758 [INFO] aria.background: Observability maintenance c cycle firing.
INFO:     Application startup complete.
2026-06-05 11:14:24,767 [INFO] aria.background: Memory decay cycle completete: decayed=0 pruned=0
2026-06-05 11:14:24,786 [INFO] aria.structured: {"timestamp": "2026-06-05T0T05:44:24.776386Z", "session_id": null, "user_id": null, "agent": "Observilability", "action": "maintenance", "success": true, "latency_ms": 0, "modseel_used": null, "input_tokens": 0, "output_tokens": 0, "tool_used": null,r" "error": null, "events_purged": "0"}
2026-06-05 11:14:24,787 [INFO] aria.background: Observability maintenance c cycle complete: events_purged=0
INFO:     Shutting down
INFO:     Waiting for application shutdown.
2026-06-05 11:15:45,150 [INFO] aria.main: ARIA shutting down — cancelling background tasks.
2026-06-05 11:15:45,150 [INFO] aria.background: Memory decay loop cancelled during sleep — shutting down.
2026-06-05 11:15:45,150 [INFO] aria.background: Observability maintenance loop cancelled during sleep — shutting down.
2026-06-05 11:15:45,150 [INFO] aria.main: ARIA shut down gracefully.     
INFO:     Application shutdown complete.
INFO:     Finished server process [16680]
INFO:     Stopping reloader process [4272]

Stopping backend server.

(venv) C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria>python run_backend.py
==================================================
Starting ARIA Backend (FastAPI) using standard Python 3.12...
==================================================
Using virtual environment: C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Scripts\python.exe
INFO:     Will watch for changes in these directories: ['C:\\Users\\Cipher\\Desktop\\Python\\2.Projects\\4.Aria\\Aria']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [19092] using WatchFiles
C:\Users\Cipher\Desktop\Python\2.Projects\4.Aria\Aria\venv\Lib\site-packages\langgraph\checkpoint\base\__init__.py:24: LangChainPendingDeprecationWarning: The default value of `allowed_objects` will change in a future version. Pass an explicit value (e.g., allowed_objects='messages' or allowed_objects='core') to suppress this warning.
  from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer     
INFO:     Started server process [8360]
INFO:     Waiting for application startup.
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
2026-06-05 12:27:36,541 [INFO] aria: LangSmith tracing enabled: aria-dev
2026-06-05 12:27:36,541 [INFO] aria.main: ARIA started successfully — graph compiled, DB ready, background tasks running.
2026-06-05 12:27:36,541 [INFO] aria.background: Memory decay loop started — interval=24h (86400s)
2026-06-05 12:27:36,541 [INFO] aria.background: Memory decay cycle firing.
2026-06-05 12:27:36,541 [INFO] aria.background: Observability maintenance loop started — interval=24h (86400s)
2026-06-05 12:27:36,541 [INFO] aria.background: Observability maintenance cycle firing.
INFO:     Application startup complete.
2026-06-05 12:27:36,579 [INFO] aria.structured: {"timestamp": "2026-06-05T06:57:36.569094Z", "session_id": null, "user_id": null, "agent": "Observability", "action": "maintenance", "success": true, "latency_ms": 0, "model_used": null, "input_tokens": 0, "output_tokens": 0, "tool_used": null, "error": null, "events_purged": "0"}
2026-06-05 12:27:36,579 [INFO] aria.background: Observability maintenance cycle complete: events_purged=0
2026-06-05 12:27:36,729 [INFO] aria.background: Memory decay cycle complete: decayed=12 pruned=0
2026-06-05 12:27:49,260 [INFO] aria.http: {"request_id": "req_d3f152bd", 
"method": "GET", "path": "/model-status", "status_code": 200, "latency_ms": 0}
INFO:     127.0.0.1:53309 - "GET /model-status HTTP/1.1" 200 OK
2026-06-05 12:27:51,325 [INFO] aria.http: {"request_id": "req_deb38e99", 
"method": "GET", "path": "/documents", "status_code": 307, "latency_ms": 
0}
INFO:     127.0.0.1:53311 - "GET /documents?user_id=default_user HTTP/1.1" 307 Temporary Redirect
2026-06-05 12:27:51,335 [INFO] aria.http: {"request_id": "req_d3d0eae3", 
"method": "GET", "path": "/documents/", "status_code": 200, "latency_ms": 14}
INFO:     127.0.0.1:53311 - "GET /documents/?user_id=default_user HTTP/1.1" 200 OK
2026-06-05 12:27:53,402 [INFO] aria.http: {"request_id": "req_c2a43fc1", 
"method": "GET", "path": "/memory", "status_code": 200, "latency_ms": 15}INFO:     127.0.0.1:53313 - "GET /memory?user_id=default_user&min_importance=0.0 HTTP/1.1" 200 OK
2026-06-05 12:27:55,482 [INFO] aria.http: {"request_id": "req_38a79462", 
"method": "GET", "path": "/analytics", "status_code": 200, "latency_ms": 
30}
INFO:     127.0.0.1:53316 - "GET /analytics?days=7 HTTP/1.1" 200 OK
2026-06-05 12:27:57,067 [INFO] aria.http: {"request_id": "req_390ed75d", 
"method": "GET", "path": "/model-status", "status_code": 200, "latency_ms": 0}
INFO:     127.0.0.1:53317 - "GET /model-status HTTP/1.1" 200 OK
2026-06-05 12:27:59,145 [INFO] aria.http: {"request_id": "req_bc93a1c6", 
"method": "GET", "path": "/documents", "status_code": 307, "latency_ms": 
0}
INFO:     127.0.0.1:53319 - "GET /documents?user_id=default_user HTTP/1.1" 307 Temporary Redirect
2026-06-05 12:27:59,155 [INFO] aria.http: {"request_id": "req_df4b08a7", 
"method": "GET", "path": "/documents/", "status_code": 200, "latency_ms": 15}
INFO:     127.0.0.1:53319 - "GET /documents/?user_id=default_user HTTP/1.1" 200 OK
2026-06-05 12:28:02,278 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/embeddings "HTTP/1.1 200 OK"
2026-06-05 12:28:05,178 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/embeddings "HTTP/1.1 200 OK"
2026-06-05 12:28:12,628 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 12:28:14,461 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 12:28:14,483 [INFO] aria.structured: {"timestamp": "2026-06-05T06:58:14.473838Z", "session_id": "sess_e50d6ac42657", "user_id": "default_user", "agent": "MemoryReader", "action": "retrieve_episodic", "success": true, "latency_ms": 0, "model_used": null, "input_tokens": 0, "output_tokens": 0, "tool_used": null, "error": null, "memories_retrieved": "0"} 
2026-06-05 12:28:17,655 [INFO] httpx: HTTP Request: POST https://api.cohere.com/v2/chat "HTTP/1.1 200 OK"
2026-06-05 12:28:17,681 [WARNING] models.router: Model command-a-plus-05-2026 attempt 1/3 failed: 'ThinkingAssistantMessageResponseContentItem' object has no attribute 'text'
2026-06-05 12:28:19,579 [INFO] httpx: HTTP Request: POST https://api.cohere.com/v2/chat "HTTP/1.1 200 OK"
2026-06-05 12:28:19,579 [WARNING] models.router: Model command-a-reasoning-08-2025 attempt 1/3 failed: 'ThinkingAssistantMessageResponseContentItem' object has no attribute 'text'
2026-06-05 12:28:21,029 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 12:28:22,295 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 12:28:22,356 [INFO] aria.structured: {"timestamp": "2026-06-05T06:58:22.356790Z", "session_id": "sess_e50d6ac42657", "user_id": "default_user", "agent": "MemoryWriter", "action": "save_turn", "success": true, "latency_ms": 46, "model_used": "llama-3.3-70b-versatile", "input_tokens": 0, "output_tokens": 0, "tool_used": null, "error": null}
2026-06-05 12:28:23,029 [INFO] aria.structured: {"timestamp": "2026-06-05T06:58:23.029058Z", "session_id": "sess_e50d6ac42657", "user_id": "default_user", "agent": "ChatRouter", "action": "chat_request", "success": true, "latency_ms": 21796, "model_used": "llama-3.3-70b-versatile", "input_tokens": 0, "output_tokens": 0, "tool_used": null, "error": null}
2026-06-05 12:28:23,044 [INFO] aria.http: {"request_id": "req_b3378249", 
"method": "POST", "path": "/chat", "status_code": 200, "latency_ms": 21828}
INFO:     127.0.0.1:53321 - "POST /chat HTTP/1.1" 200 OK
2026-06-05 12:28:23,413 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 12:28:25,193 [INFO] aria.http: {"request_id": "req_2a9bccce", 
"method": "GET", "path": "/model-status", "status_code": 200, "latency_ms": 0}
INFO:     127.0.0.1:53336 - "GET /model-status HTTP/1.1" 200 OK
2026-06-05 12:28:27,224 [INFO] aria.http: {"request_id": "req_f84641f4", 
"method": "GET", "path": "/documents", "status_code": 307, "latency_ms": 
0}
INFO:     127.0.0.1:53339 - "GET /documents?user_id=default_user HTTP/1.1" 307 Temporary Redirect
2026-06-05 12:28:27,236 [INFO] aria.http: {"request_id": "req_78a33663", 
"method": "GET", "path": "/documents/", "status_code": 200, "latency_ms": 0}
INFO:     127.0.0.1:53339 - "GET /documents/?user_id=default_user HTTP/1.1" 200 OK
2026-06-05 12:28:29,340 [INFO] aria.http: {"request_id": "req_dcd172ad", 
"method": "GET", "path": "/memory", "status_code": 200, "latency_ms": 14}INFO:     127.0.0.1:53341 - "GET /memory?user_id=default_user&min_importance=0.0 HTTP/1.1" 200 OK
2026-06-05 12:28:31,495 [INFO] aria.http: {"request_id": "req_66e67687", 
"method": "GET", "path": "/analytics", "status_code": 200, "latency_ms": 
46}
INFO:     127.0.0.1:53343 - "GET /analytics?days=7 HTTP/1.1" 200 OK
2026-06-05 12:29:04,429 [INFO] aria.http: {"request_id": "req_28795ac3", 
"method": "GET", "path": "/model-status", "status_code": 200, "latency_ms": 16}
INFO:     127.0.0.1:53345 - "GET /model-status HTTP/1.1" 200 OK
2026-06-05 12:29:06,498 [INFO] aria.http: {"request_id": "req_28327fa9", 
"method": "GET", "path": "/documents", "status_code": 307, "latency_ms": 
0}
INFO:     127.0.0.1:53347 - "GET /documents?user_id=default_user HTTP/1.1" 307 Temporary Redirect
2026-06-05 12:29:06,505 [INFO] aria.http: {"request_id": "req_63a95378", 
"method": "GET", "path": "/documents/", "status_code": 200, "latency_ms": 0}
INFO:     127.0.0.1:53347 - "GET /documents/?user_id=default_user HTTP/1.1" 200 OK
2026-06-05 12:29:08,629 [INFO] aria.http: {"request_id": "req_95585069", 
"method": "GET", "path": "/memory", "status_code": 200, "latency_ms": 0} 
INFO:     127.0.0.1:53349 - "GET /memory?user_id=default_user&min_importance=0.0 HTTP/1.1" 200 OK
2026-06-05 12:29:10,710 [INFO] aria.http: {"request_id": "req_ea5fe18b", 
"method": "GET", "path": "/analytics", "status_code": 200, "latency_ms": 
46}
INFO:     127.0.0.1:53351 - "GET /analytics?days=7 HTTP/1.1" 200 OK
2026-06-05 12:29:13,996 [INFO] aria.http: {"request_id": "req_b343960a", 
"method": "GET", "path": "/model-status", "status_code": 200, "latency_ms": 0}
INFO:     127.0.0.1:53353 - "GET /model-status HTTP/1.1" 200 OK
2026-06-05 12:29:16,045 [INFO] aria.http: {"request_id": "req_ed2245d8", 
"method": "GET", "path": "/documents", "status_code": 307, "latency_ms": 
0}
INFO:     127.0.0.1:53355 - "GET /documents?user_id=default_user HTTP/1.1" 307 Temporary Redirect
2026-06-05 12:29:16,064 [INFO] aria.http: {"request_id": "req_77c38905", 
"method": "GET", "path": "/documents/", "status_code": 200, "latency_ms": 16}
INFO:     127.0.0.1:53355 - "GET /documents/?user_id=default_user HTTP/1.1" 200 OK
2026-06-05 12:29:18,196 [INFO] aria.http: {"request_id": "req_fead707a", 
"method": "GET", "path": "/model-status", "status_code": 200, "latency_ms": 16}
INFO:     127.0.0.1:53357 - "GET /model-status HTTP/1.1" 200 OK
2026-06-05 12:29:20,274 [INFO] aria.http: {"request_id": "req_e564944c", 
"method": "GET", "path": "/documents", "status_code": 307, "latency_ms": 
0}
INFO:     127.0.0.1:53359 - "GET /documents?user_id=default_user HTTP/1.1" 307 Temporary Redirect
2026-06-05 12:29:20,282 [INFO] aria.http: {"request_id": "req_966f0e9a", 
"method": "GET", "path": "/documents/", "status_code": 200, "latency_ms": 0}
INFO:     127.0.0.1:53359 - "GET /documents/?user_id=default_user HTTP/1.1" 200 OK
2026-06-05 12:29:23,420 [INFO] httpx: HTTP Request: POST https://openrouter.ai/api/v1/embeddings "HTTP/1.1 200 OK"
2026-06-05 12:29:25,929 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 12:29:25,955 [INFO] aria.structured: {"timestamp": "2026-06-05T06:59:25.951287Z", "session_id": "sess_e50d6ac42657", "user_id": "default_user", "agent": "MemoryReader", "action": "retrieve_episodic", "success": true, "latency_ms": 0, "model_used": null, "input_tokens": 0, "output_tokens": 0, "tool_used": null, "error": null, "memories_retrieved": "0"} 
2026-06-05 12:29:30,158 [INFO] httpx: HTTP Request: POST https://api.cohere.com/v2/chat "HTTP/1.1 200 OK"
2026-06-05 12:29:30,181 [WARNING] models.router: Model command-a-reasoning-08-2025 attempt 1/3 failed: 'ThinkingAssistantMessageResponseContentItem' object has no attribute 'text'
2026-06-05 12:29:31,423 [INFO] httpx: HTTP Request: POST https://api.cohere.com/v2/chat "HTTP/1.1 200 OK"
2026-06-05 12:29:31,431 [WARNING] models.router: Model command-a-plus-05-2026 attempt 1/3 failed: 'ThinkingAssistantMessageResponseContentItem' object has no attribute 'text'
2026-06-05 12:29:33,552 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 12:29:35,596 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 12:29:35,658 [INFO] aria.structured: {"timestamp": "2026-06-05T06:59:35.658572Z", "session_id": "sess_e50d6ac42657", "user_id": "default_user", "agent": "MemoryWriter", "action": "save_turn", "success": true, "latency_ms": 46, "model_used": "llama-3.3-70b-versatile", "input_tokens": 0, "output_tokens": 0, "tool_used": null, "error": null}
2026-06-05 12:29:36,347 [INFO] aria.structured: {"timestamp": "2026-06-05T06:59:36.347552Z", "session_id": "sess_e50d6ac42657", "user_id": "default_user", "agent": "ChatRouter", "action": "chat_request", "success": true, "latency_ms": 14014, "model_used": "llama-3.3-70b-versatile", "input_tokens": 0, "output_tokens": 0, "tool_used": null, "error": null}
2026-06-05 12:29:36,355 [INFO] aria.http: {"request_id": "req_8cb9446b", 
"method": "POST", "path": "/chat", "status_code": 200, "latency_ms": 14030}
INFO:     127.0.0.1:53361 - "POST /chat HTTP/1.1" 200 OK
2026-06-05 12:29:38,495 [INFO] aria.http: {"request_id": "req_7121bf84", 
"method": "GET", "path": "/model-status", "status_code": 200, "latency_ms": 0}
INFO:     127.0.0.1:53371 - "GET /model-status HTTP/1.1" 200 OK
2026-06-05 12:29:38,617 [INFO] httpx: HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
2026-06-05 12:29:38,628 [WARNING] memory.episodic_memory: Memory extraction failed: 'NoneType' object has no attribute 'embed'
2026-06-05 12:29:40,577 [INFO] aria.http: {"request_id": "req_ced062b1", 
"method": "GET", "path": "/documents", "status_code": 307, "latency_ms": 
0}
INFO:     127.0.0.1:53373 - "GET /documents?user_id=default_user HTTP/1.1" 307 Temporary Redirect
2026-06-05 12:29:40,588 [INFO] aria.http: {"request_id": "req_f7f8cf37", 
"method": "GET", "path": "/documents/", "status_code": 200, "latency_ms": 14}
INFO:     127.0.0.1:53373 - "GET /documents/?user_id=default_user HTTP/1.1" 200 OK
2026-06-05 12:29:42,645 [INFO] aria.http: {"request_id": "req_b24dba32", 
"method": "GET", "path": "/memory", "status_code": 200, "latency_ms": 0} 
INFO:     127.0.0.1:53375 - "GET /memory?user_id=default_user&min_importance=0.0 HTTP/1.1" 200 OK
2026-06-05 12:29:44,729 [INFO] aria.http: {"request_id": "req_b8b5f24b", 
"method": "GET", "path": "/analytics", "status_code": 200, "latency_ms": 
30}
INFO:     127.0.0.1:53377 - "GET /analytics?days=7 HTTP/1.1" 200 OK
