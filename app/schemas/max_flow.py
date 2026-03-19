from pydantic import BaseModel


class FlowEdge(BaseModel):
    u: int
    v: int
    capacity: float


class MaxFlowInput(BaseModel):
    n_vertices: int
    edges: list[FlowEdge]
    source: int
    sink: int


class MaxFlowResult(BaseModel):
    max_flow: float
    flow_edges: list[dict]
    execution_time: float
