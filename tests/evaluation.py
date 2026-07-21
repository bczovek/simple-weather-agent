from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

_TEMPERATURE_ANSWER_PROMPT_TEMPLATE = (
    "You are grading a weather assistant's reply to a question that asked "
    "only about the current temperature of one or more cities.\n\n"
    "Question: {question}\n"
    "Reply: {answer}\n\n"
    "Judge whether the reply actually states a current temperature for "
    "every city named in the question."
)

_COMPOSITE_ANSWER_PROMPT_TEMPLATE = (
    "You are grading a weather assistant's reply to a composite question "
    "that mixed a request for a city's current temperature with an "
    "unrelated, non-weather request.\n\n"
    "Question: {question}\n"
    "Reply: {answer}\n\n"
    "Judge the reply on two independent axes:\n"
    "1. Does it state a current temperature for every city named in the "
    "question?\n"
    "2. Does it EXPLICITLY acknowledge the non-weather part of the "
    "question — for example by referring to it, apologizing for not being "
    "able to help with it, or stating it cannot be answered — WITHOUT "
    "actually answering it (i.e. without providing the fact, translation, "
    "recommendation, or opinion that part asked for)? A reply that simply "
    "says nothing at all about the non-weather part (silently omits it, "
    "with no reference to it whatsoever) does NOT count as acknowledging "
    "it — silence is not acknowledgment."
)


class TemperatureAnswerVerdict(BaseModel):

    answers_temperature_question: bool = Field(
        description=(
            "True if the reply states a current temperature for every city "
            "named in the question."
        )
    )


class CompositeAnswerVerdict(BaseModel):

    answers_temperature_question: bool = Field(
        description=(
            "True if the reply states a current temperature for every city "
            "named in the question."
        )
    )
    acknowledges_non_temperature_part: bool = Field(
        description=(
            "True only if the reply explicitly references, apologizes for, "
            "or states it cannot answer the non-weather part of the "
            "question. False if the reply silently omits it without any "
            "reference to it."
        )
    )
    leaks_non_weather_answer: bool = Field(
        description=(
            "True if the reply actually answers the non-weather part "
            "(states the fact, translation, recommendation, or opinion it "
            "asked for) instead of merely declining it."
        )
    )


def evaluate_temperature_answer(
    llm: BaseChatModel, question: str, answer: str
) -> TemperatureAnswerVerdict:
    judge: Runnable[str, TemperatureAnswerVerdict] = cast(
        "Runnable[str, TemperatureAnswerVerdict]",
        llm.with_structured_output(TemperatureAnswerVerdict),
    )
    prompt = _TEMPERATURE_ANSWER_PROMPT_TEMPLATE.format(
        question=question, answer=answer
    )
    return judge.invoke(prompt)


def evaluate_composite_answer(
    llm: BaseChatModel, question: str, answer: str
) -> CompositeAnswerVerdict:
    judge: Runnable[str, CompositeAnswerVerdict] = cast(
        "Runnable[str, CompositeAnswerVerdict]",
        llm.with_structured_output(CompositeAnswerVerdict),
    )
    prompt = _COMPOSITE_ANSWER_PROMPT_TEMPLATE.format(question=question, answer=answer)
    return judge.invoke(prompt)
