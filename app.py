from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import streamlit as st

from main import DEFAULT_QUESTION, evidence_agent
from query_builder import generate_pubmed_query


SAMPLE_QUESTIONS = {
    "血脂异常": "体检发现血脂偏高，生活方式干预和药物治疗分别有哪些证据？",
    "高血压": "高血压患者为什么有时要长期吃药？有哪些指南或研究依据？",
    "限钠饮食": "限钠饮食对高血压是否真的有帮助？",
    "地中海饮食": "地中海饮食对心血管风险有什么证据？",
    "糖尿病": "糖尿病患者为什么需要关注 LDL-C？",
}


QUERY_MODE_LABELS = {
    "auto": "自动模式",
    "llm": "大模型生成检索词",
    "fallback": "本地兜底检索词",
}


SOURCE_GROUPS = {
    "pubmed": ("PubMed 文献", "公共医学文献数据库"),
    "clinical_trial": ("临床试验", "ClinicalTrials.gov 注册研究"),
    "guideline": ("指南/共识", "本地整理的临床指南和共识资料"),
    "unknown": ("其他证据", "未识别来源类型"),
}


@dataclass
class RunSettings:
    question: str
    query_mode: str
    pubmed_query: str
    max_pubmed: int
    max_trials: int
    max_guidelines: int
    topk: int
    include_trials: bool
    include_guidelines: bool
    quick_demo: bool


def setup_page() -> None:
    st.set_page_config(
        page_title="临床证据助手",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.15rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu {
            visibility: hidden;
            height: 0;
        }
        .app-header {
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 1.15rem;
            padding-bottom: 0.85rem;
        }
        .app-title {
            color: #111827;
            font-size: 2.25rem;
            font-weight: 760;
            line-height: 1.15;
            margin: 0 0 0.6rem 0;
        }
        .app-subtitle {
            color: #4b5563;
            font-size: 0.96rem;
            margin: 0;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.75rem;
        }
        .chip {
            border: 1px solid #d1d5db;
            border-radius: 999px;
            color: #374151;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 650;
            line-height: 1;
            padding: 0.42rem 0.58rem;
        }
        .section-label {
            color: #374151;
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .compact-note {
            color: #6b7280;
            font-size: 0.86rem;
            margin-top: 0.35rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.3rem;
        }
        .source-line {
            color: #4b5563;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }
        .evidence-title {
            font-weight: 650;
            margin-bottom: 0.15rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">临床证据助手</div>
            <p class="app-subtitle">面向医学问题的文献、临床试验与指南证据整合。</p>
            <div class="chip-row">
                <span class="chip">PubMed</span>
                <span class="chip">ClinicalTrials.gov</span>
                <span class="chip">指南/共识</span>
                <span class="chip">RAG 回答</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> RunSettings:
    with st.sidebar:
        st.subheader("运行设置")

        sample_key = st.selectbox(
            "测试问题",
            list(SAMPLE_QUESTIONS.keys()),
            index=0,
        )

        if "question" not in st.session_state:
            st.session_state.question = DEFAULT_QUESTION

        if st.button(
            "载入测试问题",
            use_container_width=True,
        ):
            st.session_state.question = SAMPLE_QUESTIONS[
                sample_key
            ]

        quick_demo = st.toggle(
            "快速演示模式",
            value=False,
            help=(
                "减少检索数量，适合课堂展示或比赛答辩时快速跑通完整流程。"
            ),
        )

        query_mode = st.radio(
            "PubMed 检索词生成方式",
            options=list(QUERY_MODE_LABELS.keys()),
            index=0,
            horizontal=False,
            format_func=lambda value: QUERY_MODE_LABELS.get(
                value,
                value,
            ),
        )

        manual_query_enabled = st.toggle(
            "手动 PubMed query",
            value=False,
        )

        manual_query = ""

        if manual_query_enabled:
            manual_query = st.text_area(
                "英文 PubMed query",
                height=90,
                placeholder=(
                    "dyslipidemia lifestyle intervention "
                    "statin therapy cardiovascular risk"
                ),
            ).strip()

        if quick_demo:
            max_pubmed = 2
            max_trials = 1
            max_guidelines = 2
            topk = 4

            st.info(
                "快速演示：PubMed 2 篇、临床试验 1 条、指南/共识 2 条、最终证据 4 条。"
            )
        else:
            max_pubmed = st.slider(
                "PubMed 文献数",
                min_value=1,
                max_value=10,
                value=5,
            )

        include_trials = st.toggle(
            "ClinicalTrials.gov",
            value=True,
        )

        if not quick_demo:
            max_trials = st.slider(
                "临床试验数",
                min_value=0,
                max_value=10,
                value=3,
                disabled=not include_trials,
            )

        include_guidelines = st.toggle(
            "指南/共识资料",
            value=True,
        )

        if not quick_demo:
            max_guidelines = st.slider(
                "指南/共识条数",
                min_value=0,
                max_value=5,
                value=3,
                disabled=not include_guidelines,
            )

            topk = st.slider(
                "最终证据条数",
                min_value=1,
                max_value=12,
                value=5,
            )

    return RunSettings(
        question=st.session_state.question,
        query_mode=query_mode or "auto",
        pubmed_query=manual_query,
        max_pubmed=max_pubmed,
        max_trials=max_trials if include_trials else 0,
        max_guidelines=max_guidelines if include_guidelines else 0,
        topk=topk,
        include_trials=include_trials,
        include_guidelines=include_guidelines,
        quick_demo=quick_demo,
    )


def render_question_area(
    settings: RunSettings,
) -> bool:
    st.markdown(
        "<div class='section-label'>医学问题</div>",
        unsafe_allow_html=True,
    )

    st.session_state.question = st.text_area(
        "医学问题",
        value=st.session_state.get(
            "question",
            DEFAULT_QUESTION,
        ),
        height=96,
        label_visibility="collapsed",
    )

    run_clicked = st.button(
        "运行检索与回答",
        type="primary",
        use_container_width=True,
    )

    summary_parts = [
        f"PubMed {settings.max_pubmed}",
        f"临床试验 {settings.max_trials}",
        f"指南/共识 {settings.max_guidelines}",
        f"最终证据 {settings.topk}",
    ]
    st.markdown(
        (
            "<div class='compact-note'>"
            f"当前配置：{' · '.join(summary_parts)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    return run_clicked


def source_label(evidence: dict) -> str:
    metadata = evidence.get(
        "metadata",
        {},
    )
    source_type = metadata.get(
        "source_type",
        "unknown",
    )

    if source_type == "pubmed":
        return f"PubMed PMID: {metadata.get('pmid', '')}"

    if source_type == "clinical_trial":
        return f"ClinicalTrials.gov NCT ID: {metadata.get('nct_id', '')}"

    if source_type == "guideline":
        return f"Guideline ID: {metadata.get('guideline_id', '')}"

    return str(source_type)


def count_sources(
    evidence_list: list[dict],
) -> dict[str, int]:
    counts = {
        key: 0
        for key in SOURCE_GROUPS
    }

    for evidence in evidence_list:
        metadata = evidence.get("metadata", {})
        source_type = metadata.get("source_type", "unknown")

        if source_type not in counts:
            source_type = "unknown"

        counts[source_type] += 1

    return counts


def render_run_summary(
    result: dict,
) -> None:
    evidence_list = result.get("evidence", [])
    counts = count_sources(evidence_list)
    settings = result.get("settings", {})
    query_mode = settings.get("query_mode", "auto")

    col_mode, col_pubmed, col_trials, col_guidelines = st.columns(4)

    with col_mode:
        st.metric(
            "检索模式",
            QUERY_MODE_LABELS.get(query_mode, query_mode),
        )

    with col_pubmed:
        st.metric(
            "PubMed",
            counts.get("pubmed", 0),
        )

    with col_trials:
        st.metric(
            "临床试验",
            counts.get("clinical_trial", 0),
        )

    with col_guidelines:
        st.metric(
            "指南/共识",
            counts.get("guideline", 0),
        )

    if settings.get("quick_demo"):
        st.caption("当前使用快速演示模式，检索数量已自动压缩。")


def render_evidence_list(
    evidence_list: list[dict],
) -> None:
    if not evidence_list:
        st.info("当前没有可展示的检索证据。")
        return

    grouped_evidence: dict[str, list[tuple[int, dict]]] = {
        key: []
        for key in SOURCE_GROUPS
    }

    for index, evidence in enumerate(evidence_list, start=1):
        metadata = evidence.get("metadata", {})
        source_type = metadata.get("source_type", "unknown")

        if source_type not in grouped_evidence:
            source_type = "unknown"

        grouped_evidence[source_type].append((index, evidence))

    for source_type, (group_title, group_caption) in SOURCE_GROUPS.items():
        group_items = grouped_evidence.get(source_type, [])

        if not group_items:
            continue

        st.markdown(f"**{group_title}**")
        st.caption(group_caption)

        for index, evidence in group_items:
            render_evidence_item(index, evidence)


def render_evidence_item(
    index: int,
    evidence: dict,
) -> None:
    metadata = evidence.get(
        "metadata",
        {},
    )
    title = metadata.get(
        "title",
        "未命名证据",
    )
    url = metadata.get(
        "url",
        "",
    )
    document = evidence.get(
        "document",
        "",
    )

    with st.expander(
        f"[{index}] {source_label(evidence)}",
        expanded=index <= 3,
    ):
        st.markdown(
            f"<div class='evidence-title'>{title}</div>",
            unsafe_allow_html=True,
        )

        details = [
            metadata.get("journal", ""),
            metadata.get("organization", ""),
            metadata.get("status", ""),
            metadata.get("year", ""),
        ]
        details_text = " | ".join(
            str(item)
            for item in details
            if item
        )

        if details_text:
            st.markdown(
                f"<div class='source-line'>{details_text}</div>",
                unsafe_allow_html=True,
            )

        if url:
            st.markdown(f"[打开来源]({url})")

        st.write(document)


def fetch_evidence_preview(
    question: str,
    topk: int,
) -> list[dict]:
    from vector_db import search_with_details

    return search_with_details(
        query=question,
        topk=topk,
    )


def run_pipeline(
    settings: RunSettings,
) -> None:
    if not settings.question.strip():
        st.error("问题不能为空。")
        return

    log_buffer = io.StringIO()

    with st.spinner("正在生成英文检索词..."):
        pubmed_query = (
            settings.pubmed_query
            or generate_pubmed_query(
                settings.question,
                mode=settings.query_mode,
            )
        )

    with st.spinner("正在检索证据并生成回答..."):
        with contextlib.redirect_stdout(log_buffer):
            answer = evidence_agent(
                question=settings.question,
                pubmed_query=pubmed_query,
                max_results=settings.max_pubmed,
                max_trials=settings.max_trials,
                max_guidelines=settings.max_guidelines,
                topk=settings.topk,
                include_clinical_trials=settings.include_trials,
                include_guidelines=settings.include_guidelines,
                query_mode=settings.query_mode,
            )

    evidence_preview = fetch_evidence_preview(
        question=settings.question,
        topk=settings.topk,
    )

    st.session_state.last_run = {
        "answer": answer or "",
        "query": pubmed_query,
        "logs": log_buffer.getvalue(),
        "evidence": evidence_preview,
        "settings": {
            "query_mode": settings.query_mode,
            "max_pubmed": settings.max_pubmed,
            "max_trials": settings.max_trials,
            "max_guidelines": settings.max_guidelines,
            "topk": settings.topk,
            "include_trials": settings.include_trials,
            "include_guidelines": settings.include_guidelines,
            "quick_demo": settings.quick_demo,
        },
    }


def main() -> None:
    setup_page()
    settings = render_sidebar()

    st.title("临床证据助手")

    st.session_state.question = st.text_area(
        "医学问题",
        value=st.session_state.get(
            "question",
            DEFAULT_QUESTION,
        ),
        height=110,
    )

    run_clicked = st.button(
        "运行检索与回答",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        settings.question = st.session_state.question
        run_pipeline(settings)

    result = st.session_state.get(
        "last_run",
        {},
    )

    if not result:
        st.info("输入问题后运行检索与回答。")
        return

    render_run_summary(result)

    st.caption(f"PubMed 检索词：{result.get('query', '')}")

    col_answer, col_evidence = st.columns(
        [
            1.1,
            0.9,
        ],
        gap="large",
    )

    with col_answer:
        st.subheader("回答")
        st.markdown(result.get("answer", ""))

    with col_evidence:
        st.subheader("证据")
        render_evidence_list(
            result.get(
                "evidence",
                [],
            )
        )

    with st.expander("运行日志"):
        st.code(
            result.get(
                "logs",
                "",
            )
        )


if __name__ == "__main__":
    main()
