from __future__ import annotations

import json
from typing import Any

from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie
from pyecharts.components import Table


def _dump_options(chart: Any) -> dict[str, Any]:
    return json.loads(chart.dump_options())


def build_model_param_pie_option(plot: dict[str, Any], topk: int = 8) -> dict[str, Any]:
    labels = list(plot.get("labels") or [])
    counts = list(plot.get("counts") or [])
    total = plot.get("total_params")

    pairs = [
        {"name": str(name), "count": int(count)}
        for name, count in zip(labels, counts)
        if isinstance(count, (int, float)) and int(count) > 0
    ]
    if not pairs and isinstance(total, (int, float)) and int(total) > 0:
        pairs = [{"name": "total_params", "count": int(total)}]
    k = max(1, int(topk))
    pairs.sort(key=lambda x: x["count"], reverse=True)
    top = pairs[:k]
    rest = sum(item["count"] for item in pairs[k:])
    if rest > 0:
        top.append({"name": "Others", "count": rest})
    chart = Pie()
    if top:
        data_pair = [(item["name"], item["count"]) for item in top]
        chart.add(
            "",
            data_pair,
            radius=["35%", "65%"],
            center=["40%", "55%"],
        )
        chart.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
    else:
        chart.add("", [("暂无数据", 1)], radius=["35%", "65%"], center=["40%", "55%"])
        chart.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}"))

    chart.set_global_opts(
        legend_opts=opts.LegendOpts(type_="scroll", orient="vertical", pos_left="68%", pos_top="12%"),
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: {c} ({d}%)"),
    )
    return _dump_options(chart)


def build_model_param_bar_option(plot: dict[str, Any], topk: int = 12) -> dict[str, Any]:
    labels = list(plot.get("labels") or [])
    counts = list(plot.get("counts") or [])
    total = plot.get("total_params")
    pairs = [
        {"name": str(name), "count": int(count)}
        for name, count in zip(labels, counts)
        if isinstance(count, (int, float)) and int(count) > 0
    ]
    if not pairs and isinstance(total, (int, float)) and int(total) > 0:
        pairs = [{"name": "total_params", "count": int(total)}]
    k = max(1, int(topk))
    pairs.sort(key=lambda x: x["count"], reverse=True)
    top = pairs[:k]
    if not top:
        top = [{"name": "暂无数据", "count": 0}]

    x = [item["name"] if len(item["name"]) <= 42 else item["name"][:39] + "..." for item in top]
    y = [item["count"] for item in top]

    chart = Bar()
    chart.add_xaxis(x)
    chart.add_yaxis("params", y, label_opts=opts.LabelOpts(is_show=True, position="right"))
    chart.reversal_axis()
    chart.set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        xaxis_opts=opts.AxisOpts(type_="value"),
        yaxis_opts=opts.AxisOpts(type_="category"),
        legend_opts=opts.LegendOpts(is_show=False),
        datazoom_opts=[opts.DataZoomOpts(type_="inside", range_start=0, range_end=100)],
    )
    return _dump_options(chart)


def _improve_pct(series: list[float]) -> list[float]:
    if not series:
        return []
    out = [0.0]
    for i in range(1, len(series)):
        prev = float(series[i - 1])
        curr = float(series[i])
        if abs(prev) < 1e-12:
            out.append(0.0)
        else:
            out.append(((prev - curr) / prev) * 100.0)
    return out


def build_loss_line_option(plot: dict[str, Any]) -> dict[str, Any]:
    xs = [int(x) for x in (plot.get("epochs") or [])]
    train = [round(float(v), 2) for v in (plot.get("train_loss") or [])]
    val = [round(float(v), 2) for v in (plot.get("val_loss") or [])]
    saved_epochs = [int(v) for v in (plot.get("saved_epochs") or [])]
    max_epoch = plot.get("max_epoch")

    chart = Line()
    if not xs or len(xs) != len(train) or len(xs) != len(val):
        chart.set_global_opts(title_opts=opts.TitleOpts(title="", subtitle="暂无 loss 数据"))
        return _dump_options(chart)

    epoch_shift = 1 if min(xs) == 0 else 0
    x_series = [str(x + epoch_shift) for x in xs]
    train_improve = [round(v, 2) for v in _improve_pct(train)]
    val_improve = [round(v, 2) for v in _improve_pct(val)]
    improve_abs_max = max(5.0, *[abs(v) for v in train_improve], *[abs(v) for v in val_improve]) * 1.1
    y_max = max(1.0, *train, *val) * 1.1

    mark_lines = [
        opts.MarkLineItem(
            name="s",
            x=str(e + epoch_shift),
            linestyle_opts=opts.LineStyleOpts(type_="dashed", color="#f59e0b", width=2, opacity=0.95),
            symbol="none",
        )
        for e in saved_epochs
    ]

    chart.add_xaxis(x_series)
    chart.add_yaxis(
        "train_loss",
        train,
        yaxis_index=0,
        is_smooth=False,
        symbol_size=6,
        label_opts=opts.LabelOpts(is_show=False),
        markline_opts=(
            opts.MarkLineOpts(
                data=mark_lines,
                is_silent=True,
                symbol="none",
                label_opts=opts.LabelOpts(is_show=True, color="#b45309", formatter="{b}"),
            )
            if mark_lines
            else None
        ),
    )
    chart.add_yaxis(
        "test_loss",
        val,
        yaxis_index=0,
        is_smooth=False,
        symbol_size=6,
        label_opts=opts.LabelOpts(is_show=False),
    )
    chart.extend_axis(
        yaxis=opts.AxisOpts(
            type_="value",
            name="improvement(%)",
            position="right",
            min_=-improve_abs_max,
            max_=improve_abs_max,
            axislabel_opts=opts.LabelOpts(formatter="function(v){return Number(v).toFixed(2) + '%';}"),
            splitline_opts=opts.SplitLineOpts(is_show=False),
        )
    )
    chart.add_yaxis(
        "train_improve_%",
        train_improve,
        yaxis_index=1,
        symbol="none",
        linestyle_opts=opts.LineStyleOpts(type_="dashed"),
        markline_opts=opts.MarkLineOpts(
            data=[
                opts.MarkLineItem(
                    name="0%",
                    y=0,
                    linestyle_opts=opts.LineStyleOpts(type_="solid", color="#000000", width=2, opacity=0.95),
                    symbol="none",
                )
            ],
            is_silent=True,
            symbol="none",
            label_opts=opts.LabelOpts(is_show=False),
        ),
        label_opts=opts.LabelOpts(is_show=False),
    )
    chart.add_yaxis(
        "test_improve_%",
        val_improve,
        yaxis_index=1,
        symbol="none",
        linestyle_opts=opts.LineStyleOpts(type_="dashed"),
        label_opts=opts.LabelOpts(is_show=False),
    )
    chart.set_global_opts(
        tooltip_opts=opts.TooltipOpts(
            trigger="axis",
            formatter="function(params){"
            "const arr=(Array.isArray(params)?params:[params]).filter(function(p){"
            "return p&&p.componentType==='series'&&p.seriesType==='line';"
            "});"
            "if(!arr.length){return '';}"
            "const axis=(arr[0].axisValueLabel||arr[0].axisValue);"
            "const h='epoch: '+axis;"
            "const seen={};"
            "const rows=[];"
            "arr.forEach(function(p){"
            "const key=String(p.seriesName||'');"
            "if(seen[key]){return;}"
            "seen[key]=1;"
            "let raw=(p.data!==undefined&&p.data!==null)?p.data:p.value;"
            "if(Array.isArray(raw)){raw=(raw.length>1?raw[1]:raw[0]);}"
            "if(raw&&typeof raw==='object'&&Array.isArray(raw.value)){"
            "raw=(raw.value.length>1?raw.value[1]:raw.value[0]);"
            "}"
            "const v=Number(raw);"
            "const text=Number.isFinite(v)?v.toFixed(2):String(raw);"
            "rows.push((p.marker||'')+key+': '+text);"
            "});"
            "return h+'<br/>'+rows.join('<br/>');"
            "}",
        ),
        xaxis_opts=opts.AxisOpts(name="epoch"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="loss",
            min_=0.0,
            max_=y_max,
            axislabel_opts=opts.LabelOpts(formatter="function(v){return Number(v).toFixed(2);}"),
        ),
        legend_opts=opts.LegendOpts(pos_top="6%"),
        datazoom_opts=[
            opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
            opts.DataZoomOpts(type_="slider", range_start=0, range_end=100),
        ],
    )
    return _dump_options(chart)


def build_prediction_line_option(chart_payload: dict[str, Any]) -> dict[str, Any]:
    pred = [round(float(v), 2) for v in (chart_payload.get("prediction") or [])]
    truth = [round(float(v), 2) for v in (chart_payload.get("truth") or [])]
    title = str(chart_payload.get("title") or "").strip()
    x_series = [str(i + 1) for i in range(max(len(pred), len(truth)))]

    chart = Line()
    if not pred or len(pred) != len(truth):
        chart.set_global_opts(title_opts=opts.TitleOpts(title=title, subtitle="暂无数据"))
        return _dump_options(chart)

    chart.add_xaxis(x_series)
    chart.add_yaxis("prediction", pred, is_smooth=False, label_opts=opts.LabelOpts(is_show=False))
    chart.add_yaxis("truth", truth, is_smooth=False, label_opts=opts.LabelOpts(is_show=False))
    global_opts: dict[str, Any] = {
        "tooltip_opts": opts.TooltipOpts(trigger="axis"),
        "xaxis_opts": opts.AxisOpts(name="step"),
        "yaxis_opts": opts.AxisOpts(type_="value"),
        "legend_opts": opts.LegendOpts(pos_top="6%"),
        "datazoom_opts": [
            opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),
            opts.DataZoomOpts(type_="slider", range_start=0, range_end=100),
        ],
    }
    if title:
        global_opts["title_opts"] = opts.TitleOpts(title=title)
    chart.set_global_opts(**global_opts)
    return _dump_options(chart)


def build_metrics_table_html(columns: list[str], rows: list[dict[str, Any]]) -> str:
    table = Table()
    if not rows:
        table.add(["状态"], [["无数据"]])
        return table.render_embed()

    headers = [str(c) for c in columns] if columns else [str(c) for c in rows[0].keys()]
    body: list[list[str]] = []
    for row in rows:
        body.append(
            [
                f"{float(row[h]):.6f}" if isinstance(row.get(h), (int, float)) else str(row.get(h, ""))
                for h in headers
            ]
        )
    table.add(headers, body)
    return table.render_embed()
