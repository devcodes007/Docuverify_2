# Dependencies

FastAPI has a powerful but intuitive **Dependency Injection** system. It is
designed to be very simple to use, and to make it very easy for any developer
to integrate other components with FastAPI.

A "dependency" is just a function that can take the same parameters that a
path operation function can take.

## Declaring a dependency

You declare a dependency using the `Depends` function.

```python
from fastapi import Depends, FastAPI

app = FastAPI()


def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}


@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

FastAPI will call that function with the correct parameters, just like it
does for path operation functions themselves, and will pass the value it
returns into your path operation function's parameter.

## Sub-dependencies

You can create dependencies that have sub-dependencies. Dependencies can
depend on other dependencies, defining a hierarchy of dependencies that
FastAPI will take care of resolving for you.

```python
from typing import Annotated
from fastapi import Cookie, Depends, FastAPI

app = FastAPI()


def query_extractor(q: str | None = None):
    return q


def query_or_cookie_extractor(
    q: Annotated[str, Depends(query_extractor)],
    last_query: Annotated[str | None, Cookie()] = None,
):
    if not q:
        return last_query
    return q


@app.get("/items/")
async def read_query(
    query_or_default: Annotated[str, Depends(query_or_cookie_extractor)],
):
    return {"q_or_cookie": query_or_default}
```

FastAPI resolves the entire dependency graph before calling your path
operation function, calling each dependency function once per request by
default (results are cached for the duration of the request unless you set
`use_cache=False`).

## Dependencies and exceptions

If a dependency function raises an `HTTPException`, the exception
propagates up and FastAPI stops processing the request immediately: no
further dependencies in the chain are called, and the path operation
function itself is never called. FastAPI converts the raised
`HTTPException` directly into the HTTP response sent back to the client,
using the exception's `status_code` and `detail`.

```python
from fastapi import Depends, FastAPI, HTTPException

app = FastAPI()


def verify_token(x_token: str = ""):
    if x_token != "secret-token":
        raise HTTPException(status_code=400, detail="Invalid X-Token header")


@app.get("/items/", dependencies=[Depends(verify_token)])
async def read_items():
    return [{"item": "Foo"}]
```

## Dependencies in path operation decorators

Sometimes you don't really need the return value of a dependency inside
your path operation function, but you still need it to be executed/solved.
For those cases, instead of declaring a path operation function parameter
with `Depends`, you can add a `list` of `dependencies` to the path operation
decorator.
