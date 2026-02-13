const form = document.getElementById("transcribe-form");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const fileInput = document.getElementById("audio-input");
const fileNameEl = document.getElementById("file-name");
const dropzone = document.getElementById("dropzone");
const progressLabelEl = document.getElementById("progress-label");
const progressPercentEl = document.getElementById("progress-percent");
const progressBarEl = document.getElementById("progress-bar");

const resultPanel = document.getElementById("result-panel");
const transcriptEl = document.getElementById("transcript");
const summaryEl = document.getElementById("summary");
const metaEl = document.getElementById("meta");
const outputDirEl = document.getElementById("output-dir");
const transcriptPathEl = document.getElementById("transcript-path");
const summaryPathEl = document.getElementById("summary-path");
const costPathEl = document.getElementById("cost-path");
const costTotalEl = document.getElementById("cost-total");
const costTranscriptionEl = document.getElementById("cost-transcription");
const costSummaryEl = document.getElementById("cost-summary");
const costPartialNoteEl = document.getElementById("cost-partial-note");

const copyTranscriptBtn = document.getElementById("copy-transcript-btn");
const copySummaryBtn = document.getElementById("copy-summary-btn");

const downloadPanel = document.getElementById("download-panel");
const downloadInputAudioEl = document.getElementById("download-input-audio");
const downloadTranscriptTxtEl = document.getElementById("download-transcript-txt");
const downloadTranscriptJsonEl = document.getElementById("download-transcript-json");
const downloadSummaryTxtEl = document.getElementById("download-summary-txt");
const downloadCostMdEl = document.getElementById("download-cost-md");
const downloadLinks = Array.from(document.querySelectorAll(".download-btn"));

const tabs = Array.from(document.querySelectorAll(".tab"));

let currentJobId = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function setWorking(working) {
  submitBtn.disabled = working;
  submitBtn.textContent = working ? "Processing..." : "Transcribe Audio";
}

function setProgress(value, label = "Progress") {
  const bounded = Math.max(0, Math.min(1, Number(value) || 0));
  const pct = Math.round(bounded * 100);
  progressLabelEl.textContent = label;
  progressPercentEl.textContent = `${pct}%`;
  progressBarEl.style.width = `${pct}%`;
}

function setSelectedFile(file) {
  if (!file) {
    fileNameEl.textContent = "No file selected";
    return;
  }
  const mb = (file.size / (1024 * 1024)).toFixed(2);
  fileNameEl.textContent = `${file.name} (${mb} MB)`;
}

function filenameFromPath(pathText, fallback = "download") {
  if (typeof pathText !== "string" || !pathText.trim()) {
    return fallback;
  }
  const normalized = pathText.trim().replaceAll("\\", "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || fallback;
}

function setDownloadLink(element, href, enabled, filename = "") {
  element.href = href || "#";
  element.dataset.filename = filename;
  if (filename) {
    element.setAttribute("download", filename);
  } else {
    element.setAttribute("download", "");
  }
  element.classList.toggle("disabled", !enabled);
}

function formatUsd(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "N/A";
  }
  return `$${value.toFixed(6)}`;
}

function resetDownloads() {
  downloadPanel.classList.add("hidden");
  setDownloadLink(downloadInputAudioEl, "", false);
  setDownloadLink(downloadTranscriptTxtEl, "", false);
  setDownloadLink(downloadTranscriptJsonEl, "", false);
  setDownloadLink(downloadSummaryTxtEl, "", false);
  setDownloadLink(downloadCostMdEl, "", false);
}

function resetCostDisplay() {
  costTotalEl.textContent = "N/A";
  costTranscriptionEl.textContent = "N/A";
  costSummaryEl.textContent = "N/A";
  costPartialNoteEl.classList.add("hidden");
}

function mimeTypeFromFilename(filename) {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".md")) {
    return "text/markdown";
  }
  if (lower.endsWith(".json")) {
    return "application/json";
  }
  if (lower.endsWith(".mp3")) {
    return "audio/mpeg";
  }
  return "application/octet-stream";
}

async function saveUsingFilePicker(url, filename) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Download failed (${response.status})`);
  }

  const blob = await response.blob();
  const handle = await window.showSaveFilePicker({
    suggestedName: filename,
    types: [
      {
        description: "File",
        accept: { [mimeTypeFromFilename(filename)]: [`.${filename.split(".").pop()}`] },
      },
    ],
  });
  const writable = await handle.createWritable();
  await writable.write(blob);
  await writable.close();
}

async function fallbackCopyText(text) {
  const hidden = document.createElement("textarea");
  hidden.value = text;
  hidden.style.position = "fixed";
  hidden.style.opacity = "0";
  hidden.style.pointerEvents = "none";
  document.body.appendChild(hidden);
  hidden.focus();
  hidden.select();
  document.execCommand("copy");
  document.body.removeChild(hidden);
}

async function copyOutput(text, label) {
  if (!text || !text.trim()) {
    setStatus(`${label} is empty.`, true);
    return;
  }
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      await fallbackCopyText(text);
    }
    setStatus(`${label} copied to clipboard.`);
  } catch (error) {
    setStatus(`Failed to copy ${label.toLowerCase()}: ${error.message}`, true);
  }
}

fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  setSelectedFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file) {
    return;
  }
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;
  setSelectedFile(file);
});

function activateTab(tabName) {
  tabs.forEach((tab) => {
    const isActive = tab.dataset.tab === tabName;
    tab.classList.toggle("active", isActive);
  });

  transcriptEl.classList.toggle("hidden", tabName !== "transcript");
  summaryEl.classList.toggle("hidden", tabName !== "summary");
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => activateTab(tab.dataset.tab));
});

copyTranscriptBtn.addEventListener("click", () => {
  copyOutput(transcriptEl.textContent || "", "Transcript");
});

copySummaryBtn.addEventListener("click", () => {
  copyOutput(summaryEl.textContent || "", "Summary");
});

async function handleDownloadClick(event) {
  const link = event.currentTarget;
  const isDisabled = link.classList.contains("disabled");
  if (isDisabled) {
    event.preventDefault();
    return;
  }

  const filename = link.dataset.filename || "download";
  const artifact = link.dataset.artifact || "";
  const href = link.href;

  if (window.pywebview?.api?.save_artifact && currentJobId) {
    event.preventDefault();
    link.classList.add("saving");
    try {
      const response = await window.pywebview.api.save_artifact(currentJobId, artifact);
      if (response?.ok) {
        setStatus(`Saved ${filename} to ${response.path}`);
      } else if (response?.cancelled) {
        setStatus("Download cancelled.");
      } else {
        setStatus(response?.error || "Download failed.", true);
      }
    } catch (error) {
      setStatus(`Download failed: ${error.message}`, true);
    } finally {
      link.classList.remove("saving");
    }
    return;
  }

  if (window.showSaveFilePicker) {
    event.preventDefault();
    link.classList.add("saving");
    try {
      await saveUsingFilePicker(href, filename);
      setStatus(`Saved ${filename}`);
    } catch (error) {
      if (error?.name === "AbortError") {
        setStatus("Download cancelled.");
      } else {
        setStatus(`Download failed: ${error.message}`, true);
      }
    } finally {
      link.classList.remove("saving");
    }
  }
}

downloadLinks.forEach((link) => {
  link.addEventListener("click", handleDownloadClick);
});

resetDownloads();
resetCostDisplay();

async function pollJob(jobId) {
  return new Promise((resolve, reject) => {
    let inFlight = false;
    const timer = setInterval(async () => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      try {
        const response = await fetch(`/api/jobs/${jobId}`);
        const payload = await response.json();

        if (!response.ok || !payload.ok) {
          clearInterval(timer);
          reject(new Error(payload.error || `Job polling failed (${response.status})`));
          return;
        }

        setProgress(payload.progress || 0, "Progress");
        setStatus(payload.message || "Processing...");

        if (payload.status === "completed") {
          clearInterval(timer);
          resolve(payload.result);
          return;
        }
        if (payload.status === "failed") {
          clearInterval(timer);
          reject(new Error(payload.error || "Transcription failed"));
          return;
        }
      } catch (error) {
        clearInterval(timer);
        reject(error);
        return;
      } finally {
        inFlight = false;
      }
    }, 1300);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const [file] = fileInput.files;
  if (!file) {
    setStatus("Select an MP3 file first.", true);
    return;
  }

  currentJobId = null;
  resultPanel.classList.add("hidden");
  resetDownloads();
  resetCostDisplay();
  setProgress(0, "Progress");
  setWorking(true);
  setStatus("Uploading file and creating job...");

  try {
    const formData = new FormData(form);
    const response = await fetch("/api/transcribe", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || `Request failed (${response.status})`);
    }
    if (!payload.job_id) {
      throw new Error("Server did not return a job id.");
    }

    currentJobId = payload.job_id;
    setStatus("Job accepted. Processing audio...");
    const result = await pollJob(payload.job_id);

    transcriptEl.textContent = result.transcript_text || "(No transcript returned)";
    summaryEl.textContent = result.summary_text || "Summary not generated.";
    metaEl.textContent = `${result.source_file} • ${result.chunk_count} chunk(s)`;
    outputDirEl.textContent = result.output_dir || "";
    transcriptPathEl.textContent = result.transcript_path || "";
    summaryPathEl.textContent = result.summary_path || "Not generated";
    costPathEl.textContent = result.cost_path || "";

    const formattedTotal = formatUsd(result.cost_estimate_usd_total);
    costTotalEl.textContent = formattedTotal;
    costTranscriptionEl.textContent = formatUsd(result.cost_estimate_usd_transcription);
    costSummaryEl.textContent = formatUsd(result.cost_estimate_usd_summary);
    costPartialNoteEl.classList.toggle("hidden", !Boolean(result.cost_estimate_is_partial));

    const downloadBase = `/api/jobs/${payload.job_id}/download`;
    setDownloadLink(
      downloadInputAudioEl,
      `${downloadBase}/input_audio`,
      true,
      filenameFromPath(result.copied_audio, "input.mp3"),
    );
    setDownloadLink(
      downloadTranscriptTxtEl,
      `${downloadBase}/transcript_md`,
      true,
      filenameFromPath(result.transcript_path, "transcript.md"),
    );
    setDownloadLink(
      downloadTranscriptJsonEl,
      `${downloadBase}/transcript_json`,
      true,
      filenameFromPath(result.transcript_json_path, "transcript.json"),
    );
    setDownloadLink(
      downloadSummaryTxtEl,
      `${downloadBase}/summary_md`,
      Boolean(result.summary_path),
      filenameFromPath(result.summary_path, "summary.md"),
    );
    setDownloadLink(
      downloadCostMdEl,
      `${downloadBase}/cost_md`,
      Boolean(result.cost_path),
      filenameFromPath(result.cost_path, "cost.md"),
    );
    downloadPanel.classList.remove("hidden");

    resultPanel.classList.remove("hidden");
    activateTab(result.summary_text ? "summary" : "transcript");
    setProgress(1, "Completed");
    setStatus(`Done. Estimated total cost: ${formattedTotal}`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setWorking(false);
  }
});
