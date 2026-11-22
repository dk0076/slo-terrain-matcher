"use strict";

const $ = (id) => document.getElementById(id);

const state = { file: null, lat: null, lon: null, poll: null };

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

// Scroll-wheel zoom is off by default on both maps. With it on, scrolling the
// page while the pointer happens to be over a map zooms the map instead,
// which silently undoes the framing and strands the reader. Holding
// ctrl/cmd zooms, which is the behaviour embedded maps have trained people
// to expect.
const map = L.map("map", { scrollWheelZoom: false }).setView([39.5, -98.35], 4);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 18,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

let marker = null;
let ring = null;

function setPoint(lat, lon, zoom) {
  state.lat = lat;
  state.lon = lon;
  if (marker) marker.remove();
  marker = L.marker([lat, lon]).addTo(map);
  drawRing();
  if (zoom) map.setView([lat, lon], zoom);
  $("picked").textContent =
    `Starting near ${lat.toFixed(4)}, ${lon.toFixed(4)}. Click again to move it.`;
  $("picked").classList.add("set");
  refreshButton();
}

function drawRing() {
  if (state.lat === null) return;
  if (ring) ring.remove();
  ring = L.circle([state.lat, state.lon], {
    radius: Number($("radius").value) * 1609.344,
    color: "#2a78d6",
    weight: 1,
    fillOpacity: 0.07,
  }).addTo(map);
}

map.on("click", (event) => setPoint(event.latlng.lat, event.latlng.lng));

function enableCtrlZoom(target) {
  target.getContainer().addEventListener(
    "wheel",
    (event) => {
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        target.scrollWheelZoom.enable();
        setTimeout(() => target.scrollWheelZoom.disable(), 400);
      }
    },
    { passive: false }
  );
}
enableCtrlZoom(map);

// ---------------------------------------------------------------------------
// Place search
// ---------------------------------------------------------------------------

async function searchPlace() {
  const query = $("place").value.trim();
  if (!query) return;
  const button = $("search-btn");
  button.disabled = true;
  button.textContent = "Searching…";
  try {
    const response = await fetch(
      "https://nominatim.openstreetmap.org/search?format=json&limit=1&q=" +
        encodeURIComponent(query)
    );
    const found = await response.json();
    if (!found.length) {
      $("picked").textContent =
        "No place matched that. Try adding a state, or just click the map.";
      $("picked").classList.remove("set");
      return;
    }
    setPoint(Number(found[0].lat), Number(found[0].lon), 13);
  } catch {
    $("picked").textContent = "Could not reach the place search. Click the map instead.";
  } finally {
    button.disabled = false;
    button.textContent = "Search";
  }
}

$("search-btn").addEventListener("click", searchPlace);
$("place").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    searchPlace();
  }
});

// ---------------------------------------------------------------------------
// File input
// ---------------------------------------------------------------------------

const drop = $("drop");
const courseInput = $("course");

function acceptFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".gpx")) {
    showError("That needs to be a .gpx file. Most race sites and Garmin Connect export one.");
    return;
  }
  state.file = file;
  $("chosen").textContent = `✓ ${file.name}`;
  $("chosen").hidden = false;
  hideError();
  refreshButton();
}

courseInput.addEventListener("change", () => acceptFile(courseInput.files[0]));

["dragenter", "dragover"].forEach((name) =>
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.add("over");
  })
);
["dragleave", "drop"].forEach((name) =>
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.remove("over");
  })
);
drop.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) {
    courseInput.files = event.dataTransfer.files;
    acceptFile(file);
  }
});

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

$("radius").addEventListener("input", (event) => {
  $("radius-out").textContent = `${event.target.value} mi`;
  drawRing();
});

$("miles").addEventListener("input", (event) => {
  $("miles-out").textContent = `${event.target.value} mi`;
});

$("full-length").addEventListener("change", (event) => {
  const on = event.target.checked;
  $("miles").disabled = on;
  $("miles-out").textContent = on ? "course length" : `${$("miles").value} mi`;
});

function refreshButton() {
  const ready = Boolean(state.file) && state.lat !== null;
  $("go").disabled = !ready;
  $("go-note").textContent = ready
    ? "First run for a new area takes a minute or two."
    : "Add a course GPX and pick a spot to continue.";
}

// ---------------------------------------------------------------------------
// Submit and poll
// ---------------------------------------------------------------------------

function showError(message) {
  $("error-text").textContent = message;
  $("error").hidden = false;
}
function hideError() {
  $("error").hidden = true;
}

$("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.file || state.lat === null) return;

  hideError();
  $("result").hidden = true;
  $("status").hidden = false;
  $("bar").style.width = "5%";
  $("stage").textContent = "Starting…";
  $("go").disabled = true;

  const body = new FormData();
  body.append("course", state.file);
  body.append("lat", state.lat);
  body.append("lon", state.lon);
  body.append("radius", $("radius").value);
  if (!$("full-length").checked) body.append("miles", $("miles").value);

  let payload;
  try {
    const response = await fetch("/api/build", { method: "POST", body });
    payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "That request was refused.");
  } catch (error) {
    $("status").hidden = true;
    $("go").disabled = false;
    showError(error.message || "Could not reach the server.");
    return;
  }
  pollJob(payload.id);
});

function pollJob(id) {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    let job;
    try {
      job = await (await fetch(`/api/job/${id}`)).json();
    } catch {
      return; // transient; keep polling
    }
    $("bar").style.width = `${Math.max(5, job.progress)}%`;
    $("stage").textContent = job.stage;

    if (job.state === "done") {
      clearInterval(state.poll);
      $("status").hidden = true;
      $("go").disabled = false;
      showResult(id, job.result);
    } else if (job.state === "failed") {
      clearInterval(state.poll);
      $("status").hidden = true;
      $("go").disabled = false;
      showError(job.error || "That route could not be built.");
    }
  }, 1500);
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

let resultMap = null;
let resultLine = null;
let startDot = null;

function verdictFor(result) {
  const r = result.correlation;
  if (r >= 0.7)
    return ["good", `<b>Close match.</b> This route tracks the course's profile well.`];
  if (r >= 0.4)
    return ["", `<b>Loose match.</b> It follows the course's shape, with some stretches off.`];
  if (r >= 0.15)
    return [
      "warn",
      `<b>Weak match.</b> The terrain here doesn't offer much like the course. Try moving the start somewhere hillier, or a different distance.`,
    ];
  return [
    "bad",
    `<b>Poor match.</b> This area can't reproduce the course. Try a different start point &mdash; it matters more than any other setting.`,
  ];
}

function showResult(id, result) {
  $("result").hidden = false;
  $("result-name").textContent = `${result.length_mi} mi ${
    result.closed_loop ? "loop" : "route"
  }`;
  $("download").href = `/api/job/${id}/gpx`;

  const [tone, text] = verdictFor(result);
  const verdict = $("verdict");
  verdict.className = `verdict ${tone}`;
  verdict.innerHTML = text;

  const ratio = result.course_gain_ft
    ? (result.gain_ft / result.course_gain_ft).toFixed(2)
    : null;
  const stats = [
    ["Distance", `${result.length_mi} mi`, result.closed_loop ? "closed loop" : "point to point"],
    ["Climbing", `${result.gain_ft} ft`, ratio ? `${ratio}× the course` : ""],
    ["Steepest climb", `${result.steepest_climb}%`, `course ${result.course_steepest_climb}%`],
    ["Steepest descent", `${result.steepest_descent}%`, `course ${result.course_steepest_descent}%`],
    ["Profile match", `${result.correlation}`, "1.0 would be identical"],
    ["Grade error", `${result.grade_error} pts`, "average"],
    ["Elevation drift", `${result.max_drift_ft} ft`, "worst gap from the course"],
    ["Doubling back", `${result.repeat_pct}%`, "of the distance"],
  ];
  $("stats").innerHTML = stats
    .map(
      ([k, v, sub]) =>
        `<div class="stat"><span class="k">${k}</span><span class="v">${v}</span>` +
        (sub ? `<span class="sub">${sub}</span>` : "") +
        `</div>`
    )
    .join("");

  if (result.streets && result.streets.length) {
    $("stats").insertAdjacentHTML(
      "afterend",
      `<p class="streets">Along: ${result.streets.join(", ")}</p>`
    );
  }

  if (result.has_chart) {
    $("chart-card").hidden = false;
    $("chart").src = `/api/job/${id}/chart`;
  } else {
    $("chart-card").hidden = true;
  }

  drawResultMap(result.coords);
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawResultMap(coords) {
  if (!coords || !coords.length) return;
  if (!resultMap) {
    resultMap = L.map("result-map", { scrollWheelZoom: false });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(resultMap);
    enableCtrlZoom(resultMap);
  }
  if (resultLine) resultLine.remove();
  if (startDot) startDot.remove();
  resultLine = L.polyline(coords, { color: "#d85a30", weight: 4 }).addTo(resultMap);
  startDot = L.circleMarker(coords[0], {
    radius: 6,
    color: "#1d7a4e",
    fillColor: "#1d7a4e",
    fillOpacity: 1,
  })
    .addTo(resultMap)
    .bindTooltip("Start / finish");

  // The container was display:none until a moment ago, so Leaflet still
  // thinks it has no size. Measuring must happen before fitting, or the map
  // fits a zero-size box and lands at world zoom.
  const fit = () => {
    resultMap.invalidateSize();
    resultMap.fitBounds(resultLine.getBounds(), { padding: [20, 20] });
  };
  fit();
  setTimeout(fit, 200);
}

// Remove any previous street list before a new result renders.
new MutationObserver(() => {
  const lists = document.querySelectorAll(".streets");
  for (let i = 0; i < lists.length - 1; i += 1) lists[i].remove();
}).observe($("stats"), { childList: true });

refreshButton();
