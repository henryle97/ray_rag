from dataclasses import dataclass


@dataclass
class Record:
    path: str


@dataclass
class Section:
    source: str
    text: str

    def to_dict(self):
        return {"source": self.source, "text": self.text}


@dataclass
class QueryAgentResponse:
    questions: str
    sources: list[str]
    document_ids: list[str]
    answer: str

@dataclass
class QueryAgentWithContentResponse:
    answer: str


@dataclass
class QueryAgentResponse(QueryAgentWithContentResponse):
    questions: str
    sources: list[str]
    document_ids: list[str]
