"""Dependency-free accessible HTML table and SVG figure from actual report JSON."""
import html
import json
from pathlib import Path


def render(source: Path, destination: Path):
    report = json.loads(source.read_text(encoding="utf-8"))
    escape = lambda value: html.escape(str(value), quote=True)
    rows, bars = [], []
    for index, (policy, metrics) in enumerate(report["summary"].items()):
        rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in
            (policy, metrics["runs"], f'{metrics["task_success"]:.1%}', f'{metrics["abstention"]:.1%}',
             f'{metrics["tool_calls_mean"]:.2f}', metrics["unsafe_proposed_actions"], metrics["model_calls"])) + "</tr>")
        y = 25 + 45 * index
        bars.append(f'<text x="0" y="{y + 14}">{escape(policy)}</text><rect x="165" y="{y}" width="{280 * metrics["task_success"]:.2f}" height="22" fill="#2458a6"/><text x="455" y="{y + 16}">{metrics["task_success"]:.1%}</text>')
    comparisons = "".join(f'<li>{escape(item["candidate"])} minus {escape(item["baseline"])}: '
        f'{item["success_delta"]:+.3f}; 95% interval {escape(item["ci95_percentile"])}; '
        f'{item["task_units"]} task units.</li>' for item in report["paired_comparisons"])
    limitations = "".join(f"<li>{escape(value)}</li>" for value in report["limitations"])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent reliability replay</title>
<style>body{{font:16px/1.6 system-ui;color:#192638;background:#f5f7fb;max-width:960px;margin:3rem auto;padding:0 1.2rem}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;padding:.6rem;border-bottom:1px solid #ccd4e1}}.scroll{{overflow:auto}}code{{overflow-wrap:anywhere}}svg{{max-width:100%;height:auto}}text{{font-size:14px;fill:#192638}}</style>
<main><p>SYNTHETIC · DETERMINISTIC REPLAY · NO MODEL RUN</p><h1>Tool recovery under a shared budget</h1>
<p>{escape(report["evidence_boundary"])}</p><p>Split: {escape(report["split"])}. Config SHA-256: <code>{escape(report["config_sha256"])}</code>.</p>
<figure><svg viewBox="0 0 530 220" role="img" aria-labelledby="chart-title"><title id="chart-title">Observed synthetic task success, exact values in table</title>{''.join(bars)}</svg><figcaption>Observed replay success; these are not LLM measurements.</figcaption></figure>
<div class="scroll"><table><caption>All preregistered policies</caption><thead><tr><th scope="col">Policy</th><th scope="col">Runs</th><th scope="col">Success</th><th scope="col">Abstain</th><th scope="col">Calls mean</th><th scope="col">Unsafe</th><th scope="col">Model calls</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Paired uncertainty</h2><p>Task-cluster bootstrap: average repeats within task, resample tasks 2,000 times. Conditional on this generator family.</p><ul>{comparisons}</ul>
<h2>Evidence boundaries</h2><ul>{limitations}</ul><p>Latency was not measured. Tool units do not estimate billed cost.</p></main></html>'''
    destination.write_text(document, encoding="utf-8")
    # Standalone export shares exactly the chart values above.
    destination.with_suffix(".svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 530 220" role="img"><title>Synthetic replay success</title>' + ''.join(bars) + '</svg>', encoding="utf-8")
