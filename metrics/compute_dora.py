import argparse
import csv
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any



def _parse_ts(value: str) -> datetime:
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _read_events(path: str) -> List[Dict[str, Any]]:
    events = []
    with open(path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row.get("timestamp") or not row.get("event"):
                continue
            events.append(
                {
                    "timestamp": _parse_ts(row["timestamp"]),
                    "event": row["event"].strip(),
                    "details": row.get("details", "").strip(),
                }
            )
    events.sort(key=lambda e: e["timestamp"])
    return events


def _compute_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    code_changes = [e for e in events if e["event"] == "code_change"]
    session_success = [e for e in events if e["event"] == "session_success"]
    session_failed = [e for e in events if e["event"] == "session_failed"]
    incident_start = [e for e in events if e["event"] == "incident_start"]
    incident_end = [e for e in events if e["event"] == "incident_end"]

    lead_times_hours = []
    for change in code_changes:
        next_success = next((s for s in session_success if s["timestamp"] >= change["timestamp"]), None)
        if next_success:
            delta = next_success["timestamp"] - change["timestamp"]
            lead_times_hours.append(delta.total_seconds() / 3600.0)

    success_count = len(session_success)
    failure_count = len(session_failed)
    cfr = failure_count / (success_count + failure_count) if (success_count + failure_count) else 0.0

    deploy_freq_by_day = {}
    for s in session_success:
        day = s["timestamp"].date().isoformat()
        deploy_freq_by_day[day] = deploy_freq_by_day.get(day, 0) + 1

    mttr_minutes = []
    open_incidents = []
    for e in sorted(incident_start + incident_end, key=lambda x: x["timestamp"]):
        if e["event"] == "incident_start":
            open_incidents.append(e["timestamp"])
        elif e["event"] == "incident_end" and open_incidents:
            start_ts = open_incidents.pop(0)
            delta = e["timestamp"] - start_ts
            mttr_minutes.append(delta.total_seconds() / 60.0)

    metrics = {
        "lead_time_hours": lead_times_hours,
        "lead_time_avg_hours": sum(lead_times_hours) / len(lead_times_hours) if lead_times_hours else 0.0,
        "lead_time_median_hours": (
            sorted(lead_times_hours)[len(lead_times_hours) // 2] if lead_times_hours else 0.0
        ),
        "deployment_frequency_by_day": deploy_freq_by_day,
        "deployment_frequency_avg_per_day": (
            sum(deploy_freq_by_day.values()) / len(deploy_freq_by_day) if deploy_freq_by_day else 0.0
        ),
        "change_failure_rate": cfr,
        "session_success_count": success_count,
        "session_failure_count": failure_count,
        "mttr_minutes": mttr_minutes,
        "mttr_avg_minutes": sum(mttr_minutes) / len(mttr_minutes) if mttr_minutes else 0.0,
    }
    return metrics


def _write_metrics(out_dir: str, metrics: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["lead_time_avg_hours", f"{metrics['lead_time_avg_hours']:.3f}"])
        writer.writerow(["lead_time_median_hours", f"{metrics['lead_time_median_hours']:.3f}"])
        writer.writerow(["deployment_frequency_avg_per_day", f"{metrics['deployment_frequency_avg_per_day']:.3f}"])
        writer.writerow(["change_failure_rate", f"{metrics['change_failure_rate']:.3f}"])
        writer.writerow(["mttr_avg_minutes", f"{metrics['mttr_avg_minutes']:.3f}"])
        writer.writerow(["session_success_count", metrics["session_success_count"]])
        writer.writerow(["session_failure_count", metrics["session_failure_count"]])


def _write_dashboard(out_dir: str, metrics: Dict[str, Any]) -> None:
        freq = metrics["deployment_frequency_by_day"]
        days = list(sorted(freq.keys()))
        counts = [freq[d] for d in days]
        lead_times = metrics["lead_time_hours"]
        mttr_values = metrics["mttr_minutes"]
        cfr_success = metrics["session_success_count"]
        cfr_failure = metrics["session_failure_count"]

        html = f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>DORA Metrics Dashboard</title>
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js\"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f7; color: #222; }}
        .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
        .card {{ background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.08); }}
        h1 {{ margin-top: 0; }}
        canvas {{ width: 100%; height: 280px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; }}
    </style>
</head>
<body>
    <h1>DORA Metrics Dashboard</h1>
    <div class=\"card\">
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Lead Time (avg hours)</td><td>{metrics['lead_time_avg_hours']:.3f}</td></tr>
            <tr><td>Lead Time (median hours)</td><td>{metrics['lead_time_median_hours']:.3f}</td></tr>
            <tr><td>Deployment Frequency (avg/day)</td><td>{metrics['deployment_frequency_avg_per_day']:.3f}</td></tr>
            <tr><td>Change Failure Rate</td><td>{metrics['change_failure_rate']:.3f}</td></tr>
            <tr><td>MTTR (avg minutes)</td><td>{metrics['mttr_avg_minutes']:.3f}</td></tr>
        </table>
    </div>
    <div class=\"grid\" style=\"margin-top:16px;\">
        <div class=\"card\"><canvas id=\"leadTime\"></canvas></div>
        <div class=\"card\"><canvas id=\"deployFreq\"></canvas></div>
        <div class=\"card\"><canvas id=\"cfr\"></canvas></div>
        <div class=\"card\"><canvas id=\"mttr\"></canvas></div>
    </div>
    <script>
        const leadTimes = {json.dumps(lead_times)};
        const deployDays = {json.dumps(days)};
        const deployCounts = {json.dumps(counts)};
        const mttrValues = {json.dumps(mttr_values)};
        const cfrSuccess = {cfr_success};
        const cfrFailure = {cfr_failure};

        new Chart(document.getElementById('leadTime'), {{
            type: 'bar',
            data: {{
                labels: leadTimes.map((_, i) => i + 1),
                datasets: [{{ label: 'Lead Time (hours)', data: leadTimes, backgroundColor: '#4e79a7' }}]
            }},
            options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }} }}
        }});

        new Chart(document.getElementById('deployFreq'), {{
            type: 'bar',
            data: {{
                labels: deployDays,
                datasets: [{{ label: 'Sessions per day', data: deployCounts, backgroundColor: '#59a14f' }}]
            }},
            options: {{ responsive: true }}
        }});

        new Chart(document.getElementById('cfr'), {{
            type: 'doughnut',
            data: {{
                labels: ['Success', 'Failure'],
                datasets: [{{ data: [cfrSuccess, cfrFailure], backgroundColor: ['#4e79a7', '#e15759'] }}]
            }},
            options: {{ responsive: true }}
        }});

        new Chart(document.getElementById('mttr'), {{
            type: 'bar',
            data: {{
                labels: mttrValues.map((_, i) => i + 1),
                datasets: [{{ label: 'MTTR (minutes)', data: mttrValues, backgroundColor: '#f28e2b' }}]
            }},
            options: {{ responsive: true }}
        }});
    </script>
</body>
</html>
"""
        with open(os.path.join(out_dir, "dashboard.html"), "w", encoding="utf-8") as f:
                f.write(html)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="metrics/session_events.csv")
    parser.add_argument("--out", default="metrics/out")
    args = parser.parse_args()

    events = _read_events(args.input)
    metrics = _compute_metrics(events)
    _write_metrics(args.out, metrics)
    _write_dashboard(args.out, metrics)


if __name__ == "__main__":
    main()
