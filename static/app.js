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
const downloadPanel = document.getElementById("download-panel");
const downloadInputAudioEl = document.getElementById("download-input-audio");
const downloadTranscriptTxtEl = document.getElementById("download-transcript-txt");
const downloadTranscriptJsonEl = document.getElementById("download-transcript-json");
const downloadSummaryTxtEl = document.getElementById("download-summary-txt");

const tabs = Array.from(document.querySelectorAll(".tab"));

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

function setDownloadLink(element, href, enabled) {
  element.href = href || "#";
  element.classList.toggle("disabled", !enabled);
}

function resetDownloads() {
  downloadPanel.classList.add("hidden");
  setDownloadLink(downloadInputAudioEl, "", false);
  setDownloadLink(downloadTranscriptTxtEl, "", false);
  setDownloadLink(downloadTranscriptJsonEl, "", false);
  setDownloadLink(downloadSummaryTxtEl, "", false);
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

resetDownloads();

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

  resultPanel.classList.add("hidden");
  resetDownloads();
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

    setStatus("Job accepted. Processing audio...");
    const result = await pollJob(payload.job_id);

    transcriptEl.textContent = result.transcript_text || "(No transcript returned)";
    summaryEl.textContent = result.summary_text || "Summary not generated.";
    metaEl.textContent = `${result.source_file} • ${result.chunk_count} chunk(s)`;
    outputDirEl.textContent = result.output_dir || "";
    transcriptPathEl.textContent = result.transcript_path || "";
    summaryPathEl.textContent = result.summary_path || "Not generated";

    const downloadBase = `/api/jobs/${payload.job_id}/download`;
    setDownloadLink(downloadInputAudioEl, `${downloadBase}/input_audio`, true);
    setDownloadLink(downloadTranscriptTxtEl, `${downloadBase}/transcript_md`, true);
    setDownloadLink(downloadTranscriptJsonEl, `${downloadBase}/transcript_json`, true);
    setDownloadLink(downloadSummaryTxtEl, `${downloadBase}/summary_md`, Boolean(result.summary_path));
    downloadPanel.classList.remove("hidden");

    resultPanel.classList.remove("hidden");
    activateTab(result.summary_text ? "summary" : "transcript");
    setProgress(1, "Completed");
    setStatus("Done. Output rendered below.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setWorking(false);
  }
});
