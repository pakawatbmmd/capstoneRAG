"use strict";

const byId = (id) => document.getElementById(id);
const askForm = byId("ask-form");
const questionInput = byId("question");
const systemSelect = byId("system");
const systemDescription = byId("system-description");
const submitButton = byId("submit");
const askStatus = byId("ask-status");
const askResult = byId("ask-result");
const savedContent = byId("saved-content");
const resultsStatus = byId("results-status");
const tabs = [byId("tab-ask"), byId("tab-results")];
let systems = [];
let savedResultsPromise;

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined && value !== null) node.textContent = String(value);
  return node;
}

function clear(node) {
  node.replaceChildren();
}

function detailMessage(body, fallback) {
  const detail = body && body.detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg).filter(Boolean).join("; ") || fallback;
  }
  return typeof detail === "string" && detail ? detail : fallback;
}

async function jsonResponse(response) {
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`Request failed (${response.status}).`);
  }
  if (!response.ok) throw new Error(detailMessage(body, `Request failed (${response.status}).`));
  return body;
}

function showError(container, message) {
  clear(container);
  const box = element("div", "error-card");
  box.setAttribute("role", "alert");
  box.append(element("h3", "", "Unable to complete request"), element("p", "", message));
  container.append(box);
}

function renderQuestion(question) {
  askResult.append(element("div", "question-bubble", question));
}

function renderSkeleton(question) {
  clear(askResult);
  renderQuestion(question);
  const card = element("div", "answer-card skeleton-card");
  card.setAttribute("aria-hidden", "true");
  for (let index = 0; index < 4; index += 1) card.append(element("span", "skeleton-bar"));
  askResult.append(card);
}

function updateSystemDescription() {
  const selected = systems.find((item) => item.name === systemSelect.value);
  systemDescription.textContent = selected
    ? `${selected.description} · τ=${selected.tau ?? "not set"}`
    : "Select an available system.";
}

async function loadSystems() {
  try {
    systems = await jsonResponse(await fetch("/api/systems"));
    for (const item of systems) {
      const option = element("option", "", item.enabled
        ? item.name
        : `${item.name} — ${item.disabled_reason}`);
      option.value = item.name;
      option.disabled = !item.enabled;
      systemSelect.append(option);
    }
    const preferred = systems.find((item) => item.default && item.enabled);
    const firstEnabled = systems.find((item) => item.enabled);
    if (preferred || firstEnabled) {
      systemSelect.value = (preferred || firstEnabled).name;
      submitButton.disabled = false;
    }
    updateSystemDescription();
  } catch (error) {
    systemDescription.textContent = "Systems could not be loaded.";
    showError(askResult, error.message);
  }
}

function passageMeta(passage) {
  const pieces = [passage.source, passage.section].filter((value) => value !== null && value !== undefined && value !== "");
  if (passage.page !== null && passage.page !== undefined && passage.page !== "") pieces.push(`p. ${passage.page}`);
  return pieces.join(" · ");
}

function passageList(passages, numbered, citedIds) {
  const list = element(numbered ? "ol" : "ul", "passage-list");
  passages.forEach((passage, index) => {
    const item = element("li");
    const heading = element("div", "passage-heading");
    heading.append(element("span", "passage-id", numbered ? `${index + 1}. ${passage.id}` : passage.id));
    const meta = passageMeta(passage);
    if (meta) heading.append(element("span", "passage-meta", meta));
    if (citedIds.has(passage.id)) heading.append(element("span", "badge", "cited"));
    item.append(heading, element("p", "passage-text", passage.text));
    list.append(item);
  });
  return list;
}

function table(headers, rows, className = "", title) {
  const wrapper = element("div", "table-scroll");
  wrapper.tabIndex = 0;
  wrapper.setAttribute("role", "region");
  wrapper.setAttribute("aria-label", title);
  const grid = element("table", className);
  const thead = element("thead");
  const headerRow = element("tr");
  headers.forEach((heading) => headerRow.append(element("th", "", heading)));
  thead.append(headerRow);
  const tbody = element("tbody");
  rows.forEach((values) => {
    const row = element("tr");
    values.forEach((value) => row.append(element("td", "", value ?? "")));
    tbody.append(row);
  });
  grid.append(thead, tbody);
  wrapper.append(grid);
  return wrapper;
}

function renderAnswer(data, question) {
  clear(askResult);
  renderQuestion(question);
  const citedIds = new Set((data.citations || []).map((passage) => passage.id));

  if (data.abstained) {
    const warning = element("section", "warning-card");
    warning.append(element("h3", "", "Not in the guidelines"));
    warning.append(element("p", "", `Blocked by: ${data.blocked_by || "unknown guard"}`));
    if (data.reason) warning.append(element("p", "", data.reason));
    askResult.append(warning);
  } else {
    const answer = element("section", "answer-card");
    answer.append(element("p", "card-kicker", "Guideline answer"), element("p", "answer-text", data.answer));
    askResult.append(answer);
    const citations = element("section", "content-card");
    citations.append(element("h3", "", "Citations"));
    citations.append(data.citations && data.citations.length
      ? passageList(data.citations, false, new Set())
      : element("p", "empty-state", "No citation passages were returned."));
    askResult.append(citations);
  }

  if (data.verdicts && data.verdicts.length) {
    const verdicts = element("section", "content-card");
    verdicts.append(element("h3", "", "Claim verdicts"));
    verdicts.append(table(["Claim", "Verdict", "Confidence"], data.verdicts.map((item) => [
      item.claim, item.verdict, typeof item.confidence === "number" ? item.confidence.toFixed(2) : "—",
    ]), "", "Claim verdicts table"));
    if (data.omitted > 0) verdicts.append(element("p", "note", `${data.omitted} claim${data.omitted === 1 ? "" : "s"} omitted.`));
    askResult.append(verdicts);
  } else if (data.omitted > 0) {
    askResult.append(element("p", "note", `${data.omitted} claim${data.omitted === 1 ? "" : "s"} omitted.`));
  }

  const retrieved = element("section", "content-card");
  retrieved.append(element("h3", "", "Top 3 retrieved passages"));
  retrieved.append(data.top3 && data.top3.length
    ? passageList(data.top3, true, citedIds)
    : element("p", "empty-state", "No passages were returned."));
  askResult.append(retrieved);
  askResult.append(element("p", "result-footer", `${data.system} · τ=${data.tau ?? "not set"} · ${data.seconds}s`));
}

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    questionInput.setCustomValidity("Enter a question.");
    questionInput.reportValidity();
    return;
  }
  if (submitButton.disabled) return;
  submitButton.disabled = true;
  askStatus.textContent = "Searching…";
  renderSkeleton(question);
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, system: systemSelect.value }),
    });
    const data = await jsonResponse(response);
    renderAnswer(data, question);
    askStatus.textContent = data.abstained
      ? `Not in the guidelines. Blocked by: ${data.blocked_by || "unknown guard"}.`
      : `Answer found with ${(data.citations || []).length} citation${(data.citations || []).length === 1 ? "" : "s"}.`;
  } catch (error) {
    showError(askResult, error.message);
    askResult.prepend(element("div", "question-bubble", question));
    askStatus.textContent = "Search failed.";
  } finally {
    submitButton.disabled = false;
  }
});

questionInput.addEventListener("input", () => questionInput.setCustomValidity(""));
document.querySelectorAll(".example-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    questionInput.value = chip.dataset.question;
    questionInput.setCustomValidity("");
    questionInput.focus();
  });
});
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    askForm.requestSubmit();
  }
});
systemSelect.addEventListener("change", updateSystemDescription);

function parseMarkdownTable(markdown) {
  const lines = markdown.split(/\r?\n/).map((line) => line.trim()).filter((line) => line.startsWith("|") && line.endsWith("|"));
  const cells = lines.map((line) => line.slice(1, -1).split("|").map((cell) => cell.trim()));
  const content = cells.filter((row) => !row.every((cell) => /^:?-{3,}:?$/.test(cell)));
  return { headers: content[0] || [], rows: content.slice(1) };
}

function makePicker(id, labelText, options) {
  const field = element("div");
  const label = element("label", "", labelText);
  label.htmlFor = id;
  const select = element("select");
  select.id = id;
  options.forEach((value) => {
    const option = element("option", "", value);
    option.value = value;
    select.append(option);
  });
  field.append(label, select);
  return { field, select };
}

function renderSavedResults(data) {
  clear(savedContent);
  const ablation = element("section", "content-card");
  const tableSplit = data.table_split || "train";
  ablation.append(element("h3", "", `${tableSplit[0].toUpperCase()}${tableSplit.slice(1)} set (τ tuned here — optimistic)`));
  const parsed = parseMarkdownTable(data.ablation_md || "");
  ablation.append(parsed.headers.length
    ? table(parsed.headers, parsed.rows, "", `${tableSplit} set tuning results table`)
    : element("p", "empty-state", "No ablation table is saved."));
  savedContent.append(ablation);

  const predictions = data.predictions || {};
  const splits = ["train", "val", "test"].filter((name) => predictions[name]);
  const section = element("section", "content-card");
  section.append(element("h3", "", "Predictions"));
  if (!splits.length) {
    section.append(element("p", "empty-state", "No predictions are saved."));
    savedContent.append(section);
    return;
  }
  const controls = element("div", "picker-row");
  const split = makePicker("prediction-split", "Dataset split", splits);
  const system = makePicker("prediction-system", "System", []);
  controls.append(split.field, system.field);
  const output = element("div");
  section.append(controls, output);
  savedContent.append(section);

  function renderPredictions() {
    clear(output);
    const rows = predictions[split.select.value][system.select.value] || [];
    output.append(rows.length
      ? table(["ID", "Question", "Answer", "Blocked by", "Top 3"], rows.map((row) => [
        row.id, row.question, row.answer, row.blocked_by, (row.top3 || []).join(", "),
      ]), "prediction-table", `${split.select.value} predictions for ${system.select.value} table`)
      : element("p", "empty-state", "No predictions are saved for this selection."));
  }

  function updateSystems() {
    const previous = system.select.value;
    clear(system.select);
    Object.keys(predictions[split.select.value] || {}).forEach((name) => {
      const option = element("option", "", name);
      option.value = name;
      system.select.append(option);
    });
    if (Array.from(system.select.options).some((option) => option.value === previous)) system.select.value = previous;
    renderPredictions();
  }

  split.select.addEventListener("change", updateSystems);
  system.select.addEventListener("change", renderPredictions);
  updateSystems();
}

function loadSavedResultsOnce() {
  if (savedResultsPromise) return;
  resultsStatus.textContent = "Loading saved results…";
  savedResultsPromise = fetch("/api/results")
    .then(jsonResponse)
    .then((data) => {
      renderSavedResults(data);
      resultsStatus.textContent = "";
    })
    .catch((error) => {
      showError(savedContent, error.message);
      resultsStatus.textContent = "Could not load saved results.";
    });
}

function activateTab(next, focus = false) {
  tabs.forEach((tab) => {
    const selected = tab === next;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    byId(tab.getAttribute("aria-controls")).hidden = !selected;
  });
  if (focus) next.focus();
  if (next.id === "tab-results") loadSavedResultsOnce();
}

document.querySelector(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  activateTab(tabs[0]);
  questionInput.focus();
});

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => activateTab(tab));
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const offset = event.key === "ArrowRight" ? 1 : -1;
    activateTab(tabs[(index + offset + tabs.length) % tabs.length], true);
  });
});

loadSystems();
