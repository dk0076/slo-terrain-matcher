# SLO terrain matcher

Finds running routes in San Luis Obispo whose gradient profiles reproduce
specific features of the Oakland Marathon course — the steep pitches at miles
3–6, the bridge climb and descent, the flat finish — so you can train on local
terrain that feels like the race.

It pulls the USGS 3DEP elevation model and the OpenStreetMap road network,
samples elevation along every road and path in the area, and slides a window
over each profile looking for stretches that match each target's length and
grade. Output is a ranked table, a map, and GPX tracks you can load onto a
watch.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/activate          # Windows; use source .venv/bin/activate elsewhere
```

```bash
pip install -r requirements.txt
```

> **Windows note.** `rasterio` ships GDAL as bundled DLLs, and Windows refuses
> to load them when the full path exceeds 260 characters. If `import rasterio`
> fails with *"DLL load failed ... the filename or extension is too long"*,
> the project is nested too deep — move it nearer the drive root, or create the
> virtual environment somewhere short such as `C:\Users\<you>\.venvs\slo-terrain`
> and use that interpreter instead.

## Usage

Run the whole pipeline:

```bash
python run.py
```

The first run downloads data and takes a couple of minutes, almost all of it
waiting on Overpass. Everything is cached under `data/`, so later runs take
about 15 seconds. Useful flags:

```bash
python run.py --check-course     # re-measure the targets against the race GPX first
```

```bash
python run.py --refresh          # ignore the cache and refetch everything
```

```bash
python run.py --skip-waterways   # skip the extra Overpass call for creek crossings
```

Check that both data sources are reachable without downloading anything:

```bash
python fetch_data.py --check
```

Run the tests:

```bash
python -m pytest
```

### Building a whole route

The segment search answers "where can I find a 5% pitch?". To get a single
route to actually run, use `route.py`, which grows a route through the road
graph preferring edges whose grade matches what the course is doing at the
same fraction of the way through:

```bash
python route.py --miles 9
```

It writes a GPX and prints how well the route matches — grade error, profile
correlation, climbing against the course's rate for that distance, and how
much of the route doubles back on itself. Useful flags: `--emphasis` (how hard
to chase the course's steep sections), `--beam` (search width), `--start
lat,lon`, and `--point-to-point` to skip closing the loop.

Because the targets are matched by *fraction* of the way through rather than
absolute distance, a shorter route meets the course's features in the same
order and proportion, at their true grades.

## Output

Everything lands in `output/`.

| File | What it is |
|---|---|
| `matches.csv` | Every matching segment, 28 columns, one row per segment. |
| `report.md` | Top 10 per feature with the caveats that apply, human-readable. |
| `routes.gpx` | Top 3 per feature as GPX tracks, for a Garmin or similar. |
| `map.html` | Folium map of the top matches, coloured by feature, with per-feature layers and stat popups. |

## Configuration

Everything tunable is in `config.yaml`; no code changes are needed to retarget
the search.

- **`area.bbox`** — the search area as `[west, south, east, north]`. Widen it
  to search more of the county; the DEM and OSM fetches follow automatically.
- **`area.reference_point`** — what "distance from town" is measured from.
- **`targets`** — the course features. Each has a length band in miles, a grade
  band in percent, an ideal grade, and flags for whether pavement is required
  or merely preferred and whether loopability should be checked.
- **`analysis`** — sample spacing, smoothing, the minimum window length, and
  how aggressively edges are chained into corridors.
- **`scoring`** — the weights behind the quality score.

To add a feature, append a target. To loosen a search that returns nothing,
widen that target's `grade_pct` or `length_mi`.

## How it works

1. **`fetch_data.py`** queries the USGS TNM Access API for 3DEP 1/3 arc-second
   tiles covering the bounding box, then reads *only the bounding box* out of
   the remote GeoTIFFs using HTTP range requests. A 1-degree tile is about
   445 MB; the clip is about 15 MB and takes roughly 12 seconds. The road
   network comes from Overpass, with retry and backoff across mirrors and an
   `osmnx` fallback. Waterways are fetched separately so creek crossings can be
   flagged. Everything is cached, keyed by bounding box.

2. **`analyze.py`** splits each OSM way at its intersections into *edges*, then
   chains edges into *corridors*: maximal lines that continue straight through
   junctions. This step matters more than it sounds. In town a single edge is
   rarely more than 100–200 m, so without chaining a two-mile climb is simply
   invisible. Chaining uses mutually-best pairing on turn angle — two edges join
   only when each is the other's straightest available continuation — which
   partitions the network cleanly and reproduces what a runner would call "the
   same road".

   Each corridor is resampled at 20 m, the DEM is sampled bilinearly at every
   point, and the elevation series is smoothed over 5 samples before any grade
   is computed. Then a sliding window finds every stretch whose length and mean
   grade fall in a target's band, overlapping windows are collapsed to the one
   closest to the ideal grade, and the survivors are scored.

3. **`report.py`** writes the four output files.

4. **`validate.py`** compares the DEM against barometric elevation from your
   own FIT files — see below.

5. **`course.py`** re-measures the configured targets against the race GPX, so
   a band that does not match the actual course shows up.

### Why grades are measured the way they are

- **Minimum 100 m window.** Grade over a shorter window is dominated by DEM
  noise, not terrain. This is a hard floor; it overrides a target that permits
  something shorter.
- **Smoothing before differencing.** A 10 m DEM differenced at its native
  resolution produces a staircase, and a staircase differenced produces fake
  15% pitches. The elevation series is smoothed over 100 m first.
- **Bilinear sampling.** Nearest-neighbour lookups against a 10 m grid give the
  same staircase problem from a different direction.
- **"Max grade" is a sustained grade.** The reported max is the steepest grade
  held over 100 m within the segment, in the direction the segment runs — not
  the steepest single sample. On a descent it is correspondingly negative.
- **Continuity is enforced.** A window that drops 40 m and climbs 20 m back has
  the same mean grade as a steady 20 m drop, but it is not the continuous climb
  or descent the spec asks for. Elevation running against a window's overall
  trend may not exceed 25% of its net change (`max_reversal_fraction`).
- **Flat means flat, not net-flat.** For a target whose band straddles zero the
  trend rule is meaningless, so total climbing per kilometre is capped instead
  (`max_undulation_m_per_km`, 20 m/km ≈ 105 ft/mile). Without it, a road of
  steep rollers that ends where it started would rank as a flat finish.

### Known limits

- **1/3 arc-second is about 10 m horizontally** — roughly 8 m at this latitude.
  That is fine for grades averaged over 50 m or more, but short pitches are
  genuinely absent from the data. A 30 m ramp between two cells cannot be seen.
- **The DEM is bare earth.** Bridges and overpasses read as the ground beneath
  them, and a creek crossing can read as a notch. Segments tagged `bridge` or
  `tunnel` in OSM, segments crossing a mapped waterway, and segments containing
  an implausible single-cell cliff are flagged in the `flags` column, scored
  down, and drawn dashed on the map. They are kept rather than dropped, because
  the flag is often a false positive on a short culvert.
- **OSM `surface` tags are incomplete.** Absent means *unknown*, never *paved*.
  Many rural roads reported as `unknown` here are in fact sealed. Paved scores
  highest, unknown sits in the middle, known-unpaved is penalised.
- **Loopability is approximate.** It asks whether a return route exists between
  the segment's endpoints that avoids the segment itself and is no more than 3×
  its length. It does not check that the return is pleasant, or legal, or open.
- **Corridors follow geometry, not intent.** A corridor can run through a gate
  or onto private land. Check anything before committing to it as a workout.

## Validating the DEM against your own runs

The grades above are only as good as the elevation model. `validate.py` takes
FIT files from runs on local roads, pulls out the GPS track and the barometric
altimeter trace, samples the DEM at the same points, and reports how far apart
they are.

```bash
python validate.py run.fit
```

```bash
python validate.py "runs/*.fit" --csv output/dem_validation.csv
```

The absolute offset is not the interesting number — a barometer drifts and the
DEM has its own datum, so a constant offset of several metres is normal and
harmless. What matters is the **grade** comparison: bias, mean absolute error
and correlation over 100 m windows. Mean absolute error at or under 0.5
percentage points with correlation above 0.9 means the reported grades are
trustworthy. Above about 1 point means treat them as indicative.

Expect the comparison to look worse on runs that include bridges, dense tree
cover or tunnels, where a bare-earth DEM and a barometer cannot agree by
construction.

## Project layout

```
config.yaml       bbox, targets, scoring weights, output paths
config.py         typed configuration loading and validation
geo.py            haversine distance, resampling, bearings
fetch_data.py     3DEP DEM and OSM acquisition, with caching
analyze.py        edges, corridors, DEM sampling, window search, scoring
report.py         CSV, Markdown, GPX and Folium map output
route.py          build whole routes that track the course's grade profile
validate.py       DEM vs. barometric elevation from FIT files
course.py         re-measure the targets against the race GPX
run.py            the whole pipeline end to end
tests/            unit tests for grade computation and window search
data/             cached downloads (git-ignored)
output/           generated reports (git-ignored)
```

## Data sources

- Elevation: [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) 1/3
  arc-second, via the TNM Access API. Public domain.
- Roads and waterways: [OpenStreetMap](https://www.openstreetmap.org/copyright)
  via Overpass. © OpenStreetMap contributors, ODbL.
