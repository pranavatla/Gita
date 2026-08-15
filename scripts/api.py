from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from answer import answer_question


app = FastAPI(
    title="Bhagavad Gita RAG API",
    description=(
        "Retrieves a relevant Bhagavad Gita passage and generates "
        "a passage-grounded practical message."
    ),
    version="1.0.0",
)

pipeline_lock = Lock()
web_index = Path(__file__).resolve().parent.parent / "web" / "index.html"


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The situation or question to analyze.",
    )


class PassageResponse(BaseModel):
    id: str
    sanskrit: str
    transliteration: str
    english: str


class AnswerResponse(BaseModel):
    message: str
    passage: PassageResponse


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(web_index)


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post(
    "/answer",
    response_model=AnswerResponse,
)
def create_answer(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        # Only one request uses the local models at a time.
        with pipeline_lock:
            answer, selected = answer_question(question)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Could not generate a grounded answer.",
        ) from error

    verse_id = answer["used_verse_ids"][0]

    candidates_by_id = {
        result["candidate"]["id"]: result["candidate"]
        for result in selected
    }

    candidate = candidates_by_id.get(verse_id)

    if candidate is None:
        raise HTTPException(
            status_code=500,
            detail="Selected passage was not available.",
        )

    metadata = candidate["metadata"]

    return AnswerResponse(
        message=answer["message"],
        passage=PassageResponse(
            id=verse_id,
            sanskrit=metadata["sanskrit"],
            transliteration=metadata["transliteration"],
            english=candidate["document"],
        ),
    )
