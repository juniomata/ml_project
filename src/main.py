from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def home():
    return {"hello": "world"}


@app.get("/contact")
async def contact():
    return {"contact": "0283127894129"}
