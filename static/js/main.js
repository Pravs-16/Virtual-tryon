/* Fitting Room frontend: loads the garment rack, switches garments,
   and flips the overlay / tracking toggles via the JSON API. */

const listEl = document.getElementById("garment-list");
const overlayToggle = document.getElementById("toggle-overlay");
const landmarksToggle = document.getElementById("toggle-landmarks");

function prettyName(key) {
  return key.replaceAll("_", " ");
}

async function loadGarments() {
  const res = await fetch("/api/garments");
  const data = await res.json();
  listEl.innerHTML = "";

  data.garments.forEach((name) => {
    const btn = document.createElement("button");
    btn.className = "garment" + (name === data.current ? " active" : "");
    btn.setAttribute("role", "option");
    btn.setAttribute("aria-selected", name === data.current ? "true" : "false");

    const img = document.createElement("img");
    img.src = `/static/garments/${name}.png`;
    img.alt = "";

    const label = document.createElement("span");
    label.className = "name";
    label.textContent = prettyName(name);

    btn.append(img, label);
    btn.addEventListener("click", () => selectGarment(name));
    listEl.appendChild(btn);
  });
}

async function selectGarment(name) {
  const res = await fetch("/api/garment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) return;

  document.querySelectorAll(".garment").forEach((el) => {
    const isActive = el.querySelector(".name").textContent === prettyName(name);
    el.classList.toggle("active", isActive);
    el.setAttribute("aria-selected", isActive ? "true" : "false");
  });
}

async function pushToggles() {
  await fetch("/api/toggle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      enabled: overlayToggle.checked,
      show_landmarks: landmarksToggle.checked,
    }),
  });
}

overlayToggle.addEventListener("change", pushToggles);
landmarksToggle.addEventListener("change", pushToggles);

loadGarments();
