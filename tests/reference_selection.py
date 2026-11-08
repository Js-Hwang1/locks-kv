import torch


def reference_scores(page_logits, visible_pages):
    if page_logits.ndim != 4:
        raise ValueError("Expected [request, KV head, query in group, page]")
    pages = torch.arange(page_logits.shape[-1], device=page_logits.device)
    valid = pages[None, None, None, :] < visible_pages[:, None, None, None]
    logits = page_logits.double().masked_fill(~valid, float("-inf"))
    empty = visible_pages[:, None, None, None] == 0
    shares = torch.softmax(torch.where(empty, torch.zeros_like(logits), logits), dim=-1)
    return shares.masked_fill(~valid, 0).mean(dim=2)


def reference_selection(
    page_logits, visible_pages, *, total_pages, sink_pages, window_pages
):
    if sink_pages < 0 or window_pages < 1 or total_pages < sink_pages + window_pages:
        raise ValueError("Invalid total page budget or mandatory page counts")
    score = reference_scores(page_logits, visible_pages)
    selected = torch.full(score.shape, -1, device=score.device, dtype=torch.int32)
    counts = torch.zeros(score.shape[:2], device=score.device, dtype=torch.int32)
    for request in range(score.shape[0]):
        count = int(visible_pages[request])
        mandatory = sorted(
            set(range(min(sink_pages, count)))
            | set(range(max(0, count - window_pages), count))
        )
        candidates = [page for page in range(count) if page not in mandatory]
        slots = max(0, min(total_pages - len(mandatory), len(candidates)))
        for head in range(score.shape[1]):
            ranked = sorted(
                candidates, key=lambda page: (-float(score[request, head, page]), page)
            )
            ids = sorted(mandatory + ranked[:slots])
            selected[request, head, : len(ids)] = torch.tensor(
                ids, device=score.device, dtype=torch.int32
            )
            counts[request, head] = len(ids)
    return (score, selected, counts)


def stable_topb(st, n_req):
    score = st.score[:n_req]
    pages = torch.arange(st.max_pages, device=score.device)[None, None, :]
    visible = pages < st.n_pages[:n_req, None, None]
    candidates = (
        visible & (pages >= st.sink_pages) & (pages < st.n_sel_hi[:n_req, None, None])
    )
    mandatory = visible & ~candidates
    order = torch.argsort(
        score.masked_fill(~candidates, float("-inf")),
        dim=-1,
        descending=True,
        stable=True,
    )
    take = (pages < st.b_fix[:n_req, None, None]) & candidates.expand_as(score).gather(
        -1, order
    )
    keep = (
        torch.zeros_like(score, dtype=torch.bool).scatter_(-1, order, take) | mandatory
    )
    ids = torch.where(keep, pages, st.max_pages).sort(dim=-1).values
    st.page_table[:n_req].copy_(torch.where(ids < st.max_pages, ids, -1))
    st.page_cnt[:n_req].copy_(keep.sum(-1))
