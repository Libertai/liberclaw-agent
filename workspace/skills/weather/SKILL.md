---
name: weather
description: Check current weather and forecasts with wttr.in via the bash tool — no API key needed.
---
# Weather

Check current weather using wttr.in (no API key needed).

## Usage
Use the bash tool to fetch weather:
```
curl -s "wttr.in/CityName?format=3"
```

For detailed forecast:
```
curl -s "wttr.in/CityName"
```

For specific format options:
```
curl -s "wttr.in/CityName?format=%l:+%c+%t+%h+%w"
```

## Tips
- Use city names: `wttr.in/Paris`, `wttr.in/New+York`
- Airport codes work: `wttr.in/JFK`
- Add `?m` for metric, `?u` for US units
- `?format=j1` returns full JSON for programmatic use (parse with `jq`)
- wttr.in occasionally rate-limits or times out — pass `curl --max-time 10`, and
  if it fails, fall back to `web_search` for the forecast.
