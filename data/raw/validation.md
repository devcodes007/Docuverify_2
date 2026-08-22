# Request Validation

FastAPI uses Python type hints and Pydantic models to validate incoming
request data automatically.

## Path and query parameters

When you declare a type for a path or query parameter, FastAPI validates
the incoming value against that type before your path operation function
is ever called.

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

If a client sends a value for `item_id` that cannot be parsed as an
integer, FastAPI automatically returns a `422 Unprocessable Entity`
response with details about the validation error, without your function
code ever running.

## Request bodies with Pydantic models

For request bodies, you declare a Pydantic `BaseModel`, and FastAPI
validates the whole JSON body against that model's fields and types.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None


@app.post("/items/")
async def create_item(item: Item):
    return item
```

## Validation and dependency injection

Request validation happens as FastAPI resolves each parameter of your path
operation function and of any dependency functions declared via `Depends`
-- both path operation parameters and dependency function parameters go
through the same type-hint-driven validation. This means dependencies can
themselves declare validated parameters (query parameters, headers,
Pydantic models), and those are validated before the dependency function
runs, in the same pass that validates the path operation's own parameters.
If a dependency's parameter fails validation, FastAPI returns the 422
error before either the dependency function or the path operation function
is called.
