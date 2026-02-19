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
- `?format=j1` returns JSON for programmatic use
