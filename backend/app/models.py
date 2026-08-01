from typing import Literal
from pydantic import BaseModel, Field

Effort = Literal["low", "medium", "high"]

class Settings(BaseModel):
    agy_reasoning_effort: Effort = "medium"

class JobCreate(BaseModel):
    claims: str = Field(min_length=1)
    agy_reasoning_effort: Effort = "medium"
    analysis_prompt: str = ""

class Evidence(BaseModel):
    document_id: str
    filename: str
    reference_number: int | None = None       # 인용발명 N
    document_number: str | None = None        # 고유 문헌 번호(US 10,987,654 A1 / 10-2020-0012345호 등)
    paragraph: str | None = None
    page: int | None = None
    excerpt: str
    original_excerpt: str | None = None       # 외국어 문헌의 원문 병기
    quality: Literal["HIGH", "MEDIUM", "LOW", "UNVERIFIED"] = "MEDIUM"

class DocumentMapping(BaseModel):
    reference_number: int
    filename: str
    document_type: str = ""
    document_id: str
    document_number: str = ""
    role: str = ""                            # 주 인용발명 / 부 인용발명

class ClaimResult(BaseModel):
    label: str = ""                           # 청구항 원문의 (A), (B) … 라벨
    claim: str
    similarity: int | None = None             # 대응 인용발명이 없으면 None
    grade: str = ""                           # 등급명(동일 / 실질적 동일 …)
    emoji: str = ""
    narrative: str = ""                       # 라벨 없는 한 문장 구성대비 서술
    difference: str | None = None             # → 차이점 (없으면 None)
    combination: bool = False                 # 결합 논리 사용 여부
    references: list[str] = []
    evidence: list[Evidence] = []
    status: str = ""
    note: str = ""

class AnalysisResult(BaseModel):
    job_id: str
    claim_mapping: list[DocumentMapping]
    claims: list[ClaimResult]
    preamble: str = ""                        # "청구항 1." 등 라벨 앞 전제부
    summary: str = ""
    summary_similarity: str = ""              # 종합 분석 요약 - 유사점 한 줄
    summary_difference: str = ""              # 종합 분석 요약 - 차이점 한 줄
    validation: list[str] = []
