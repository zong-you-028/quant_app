# Render startup recovery

The October 27 application log reports an on-demand `flet-web 1.0.3` install,
followed by `TypeError: FastAPI.__init__() got an unexpected keyword argument
'openapi_prefix'`. The broad fallback in `main.py` then masks the original
startup error with `TypeError: 'module' object is not callable`.

Install the tested Flet 0.85.2 web stack and compatible FastAPI 0.136.3,
Starlette 1.2.1 and Uvicorn 0.48.0 in the Docker build. Remove the fallback
around the entire running server so startup exceptions retain their original
traceback. Check real HTTP and WebSocket startup before an image can deploy.
The startup check uses a disposable directory, clears cloud credentials and
never submits an application login.

The separate Render certificate advisory requires a refreshed system CA store
and Python Certifi 2026.10.19 or later. Install both when rebuilding the image.
Node.js and PHP are not part of this application's Docker runtime.

Verification: the clean Python 3.12.14 deployment environment passed the real
HTTP/WebSocket startup check, `pip check` and 356 regression tests, with zero
formal-database or external-HTTP attempts. The sole remaining test deliberately
requires the historic research identity (Python 3.13.14, SQLite 3.50.4,
NumPy 2.2.6, Pandas 2.3.2). It passed separately in that exact original runtime
(1 passed in 14.91 seconds). Its initial identity rejection in the new runtime
is retained in `test_verification.log`; no guard or frozen artifact was changed.
