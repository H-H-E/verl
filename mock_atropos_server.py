#!/usr/bin/env python3
from fastapi import FastAPI
import uvicorn, random, string

app = FastAPI()

@app.get("/batch")
def batch(size: int = 2):
    def dummy():
        s = "".join(random.choice(string.ascii_letters) for _ in range(10))
        return {"prompt": "Q:"+s, "response": "A:"+s[::-1], "reward": 1.0}
    return [dummy() for _ in range(size)]

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000, log_level="error") 