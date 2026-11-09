var currentController = null;

// MARKDOWN CONFIGURATION
marked.setOptions({
  breaks: true,
  gfm: true,
});
// ============================================

$(document).ready(function () {
  let room;
  // ============================================================
  // Assistant response state
  // ============================================================
  let currentAssistantMessage = null;
  let currentAssistantText = "";
  let currentAssistantGenerationId = null;
  const aiAudioElements = new Map();
  let latestGenerationId = 0;
  let isAudioStopped = false;
  let isMicMuted = false;

  // ============================================================
  // Audio state
  // ============================================================
  let speakingAudioTracks = 0;

  function scrollChatToBottom() {
    const chatBody = document.getElementById("chatBody");
    if (chatBody) {
      chatBody.scrollTop = chatBody.scrollHeight;
    }
  }

  function setConnecting() {
    const orb = document.getElementById("voiceOrb");
    if (orb) {
      orb.classList.add("active");
      orb.classList.remove("speaking");
    }
  }

  function setListening() {
    const orb = document.getElementById("voiceOrb");
    if (orb) {
      orb.classList.add("active");
      orb.classList.remove("speaking");
    }
  }

  function setSpeaking() {
    const orb = document.getElementById("voiceOrb");

    if (orb) {
      orb.classList.add("active");
      orb.classList.add("speaking");
    }
  }

  function setDisconnected() {
    const orb = document.getElementById("voiceOrb");
    if (orb) {
      orb.classList.remove("active");
      orb.classList.remove("speaking");
    }
  }

  // ============================================================
  // User messages
  // ============================================================

  function addUserMessage(text) {
    if (!text || !text.trim()) {
      return;
    }

    const chatBody = document.getElementById("chatBody");

    if (!chatBody) {
      console.warn("chatBody element not found.");
      return;
    }

    const row = $("<div>")
      .addClass("message-row user")
      .append(
        $("<div>")
          .addClass("message")
          .append(
            $("<div>").addClass("message-label").text("You"),
            text.trim(),
          ),
      );
    $(chatBody).append(row);
    scrollChatToBottom();
  }

  // ============================================================
  // Assistant DOM messages
  // ============================================================

  function createAssistantMessage() {
    const chatBody = document.getElementById("chatBody");

    if (!chatBody) {
      console.warn("chatBody element not found.");
      return;
    }

    const html = `
            <div class="message-row ai">
                <div class="message">
                    <div class="message-label">
                        <i class="bi bi-robot me-1"></i>
                      AI Assistant
                    </div>
                    <div class="ai-content"></div>
                </div>
            </div>
        `;
    const element = $(html);
    element.hide();
    $(chatBody).append(element);
    scrollChatToBottom();
    return element;
  }

  function renderMarkdown(element, markdownText) {
    if (!element) {
      return;
    }
    const content = element.find(".ai-content");
    element.show();

    if (typeof marked === "undefined") {
      console.warn("marked.js is not loaded. Falling back to plain text.");
      content.html(markdownText || "");
      return;
    }

    try {
      content.html(marked.parse(markdownText || ""));
    } catch (error) {
      console.error("Markdown rendering failed:", error);
      content.html(markdownText || "");
    }
  }

  // ============================================================
  // Generation helpers
  // ============================================================

  function parseGenerationId(message) {
    const generationId = Number(message?.generation_id);
    if (!Number.isFinite(generationId)) {
      return null;
    }
    return generationId;
  }

  function resetAssistantState() {
    currentAssistantMessage = null;
    currentAssistantText = "";
    currentAssistantGenerationId = null;
  }

  function startAssistantResponse(generationId) {
    currentAssistantText = "";
    currentAssistantGenerationId = generationId;
    currentAssistantMessage = createAssistantMessage();
    console.log(
      `[LLM UI] Started assistant response | generation=${generationId}`,
    );
    return currentAssistantMessage;
  }

  function appendAssistantDelta(text, generationId) {
    if (!text) {
      return;
    }

    if (generationId < latestGenerationId) {
      console.log(
        `[LLM UI] Ignoring stale delta | generation=${generationId} | latest=${latestGenerationId}`,
      );
      return;
    }

    if (
      currentAssistantGenerationId === null ||
      generationId !== currentAssistantGenerationId
    ) {
      latestGenerationId = Math.max(latestGenerationId, generationId);
      startAssistantResponse(generationId);
    }

    // ----------------------------------------------------------
    // Safety check
    // ----------------------------------------------------------

    if (!currentAssistantMessage) {
      console.warn(
        `[LLM UI] No assistant DOM message | generation=${generationId}`,
      );
      startAssistantResponse(generationId);
    }

    currentAssistantText += text;
    renderMarkdown(currentAssistantMessage, currentAssistantText);
    scrollChatToBottom();
  }

  // ============================================================
  // Generation-safe response completion
  // ============================================================

  function finishAssistantResponse(generationId, status) {
    console.log(
      `[LLM UI] Finish | status=${status} | generation=${generationId} | current=${currentAssistantGenerationId} | latest=${latestGenerationId}`,
    );

    if (generationId < latestGenerationId) {
      console.log(
        `[LLM UI] Ignoring stale ${status} | generation=${generationId} | latest=${latestGenerationId}`,
      );
      return;
    }

    // ----------------------------------------------------------
    // If there is no active assistant message, nothing to close.
    // ----------------------------------------------------------

    if (!currentAssistantMessage) {
      console.log(
        `[LLM UI] No active assistant message for ${status} | generation=${generationId}`,
      );
      return;
    }

    // ----------------------------------------------------------
    // Completion must belong to the currently displayed generation.
    // ----------------------------------------------------------

    if (generationId !== currentAssistantGenerationId) {
      console.log(
        `[LLM UI] Ignoring mismatched ${status} | generation=${generationId} | current=${currentAssistantGenerationId}`,
      );
      return;
    }

    renderMarkdown(currentAssistantMessage, currentAssistantText);
    console.log(
      `[LLM UI] Assistant response closed | status=${status} | generation=${generationId}`,
    );
    resetAssistantState();

    if (speakingAudioTracks === 0 && room) {
      setListening();
    }
  }

  // ============================================================
  // LLM error
  // ============================================================

  function showAssistantError(errorText, generationId) {
    if (!errorText) {
      return;
    }

    // Ignore stale errors.
    if (generationId < latestGenerationId) {
      console.log(
        `[LLM UI] Ignoring stale error | generation=${generationId} | latest=${latestGenerationId}`,
      );
      return;
    }

    // New generation error.
    if (
      currentAssistantGenerationId === null ||
      generationId !== currentAssistantGenerationId
    ) {
      latestGenerationId = Math.max(latestGenerationId, generationId);
      startAssistantResponse(generationId);
    }

    currentAssistantText += `\n\n**Error:** ${errorText}`;

    renderMarkdown(currentAssistantMessage, currentAssistantText);
    resetAssistantState();
    scrollChatToBottom();
  }

  // ============================================================
  // Clear chat
  // ============================================================

  function clearChat() {
    const chatBody = document.getElementById("chatBody");

    if (!chatBody) {
      return;
    }

    chatBody.innerHTML = `
      <div class="message-row ai">
        <div class="message">
            <div class="message-label">
                <i class="bi bi-robot me-1"></i>
               AI Assistant
            </div>
            <div class="ai-content">Hello! How can I help you today?</div>
        </div>
      </div>
    `;

    resetAssistantState();
  }

  // ============================================================
  // Live transcript
  // ============================================================

  function updateLiveTranscript(text) {
    const element = document.getElementById("thinkingText");

    if (element) {
      $("#thinkingContainer").show();
      element.textContent = text || "";
    }
  }

  function clearLiveTranscript() {
    updateLiveTranscript("");
    $("#thinkingContainer").hide();
  }

  // ============================================================
  // Transcript handling
  // ============================================================

  async function handleTranscriptMessage(message) {
    if (!message) {
      return;
    }

    const status = message.status;
    const text = message.text || "";

    console.log(`[TRANSCRIPT] ${status}: ${text}`);

    // --------------------------------------------------------
    // Partial/interim ASR
    // --------------------------------------------------------

    if (status === "partial") {
      await resumeCurrentAiAudio();
      updateLiveTranscript(text);
      return;
    }

    // --------------------------------------------------------
    // Final ASR
    // --------------------------------------------------------

    if (status === "final") {
      clearLiveTranscript();

      if (text.trim()) {
        addUserMessage(text.trim());
      }

      return;
    }
  }

  // ============================================================
  // LLM handling
  // ============================================================

  async function handleLlmMessage(message) {
    if (!message) {
      return;
    }

    const status = message.status;
    const text = message.text || "";
    const generationId = parseGenerationId(message);

    console.log(
      `[LLM] status=${status} generation=${message.generation_id}:`,
      text,
    );

    // ----------------------------------------------------------
    // Generation ID is mandatory.
    // ----------------------------------------------------------

    if (generationId === null) {
      console.warn(
        "[LLM] Ignoring message without valid generation_id:",
        message,
      );

      return;
    }

    // ----------------------------------------------------------
    // DELTA
    // ----------------------------------------------------------

    if (status === "delta") {
      appendAssistantDelta(text, generationId);
      return;
    }

    // ----------------------------------------------------------
    // DONE / INTERRUPTED
    // ----------------------------------------------------------

    if (status === "done" || status === "interrupted") {
      if (status === "interrupted") {
        stopCurrentAiAudio();
      }
      finishAssistantResponse(generationId, status);
      return;
    }

    // ----------------------------------------------------------
    // ERROR
    // ----------------------------------------------------------

    if (status === "error") {
      console.error(`[LLM] Error | generation=${generationId}:`, text);

      showAssistantError(
        text || "The AI failed to generate a response.",
        generationId,
      );
      return;
    }

    console.warn(`[LLM] Unknown status: ${status}`, message);
  }

  // ============================================================
  // LiveKit data handling
  // ============================================================

  function handleDataReceived(payload, participant, kind, topic) {
    try {
      const text = new TextDecoder().decode(payload);

      console.log(`[LIVEKIT-DATA] topic=${topic}`, text);

      // ----------------------------------------------------
      // Transcript
      // ----------------------------------------------------

      if (topic === "transcript") {
        const message = JSON.parse(text);

        if (message.type !== "transcript") {
          return;
        }

        handleTranscriptMessage(message);
        return;
      }

      // ----------------------------------------------------
      // LLM response
      // ----------------------------------------------------

      if (topic === "llm_response") {
        const message = JSON.parse(text);
        handleLlmMessage(message);
        return;
      }
    } catch (error) {
      console.error("[LIVEKIT-DATA] Failed to process message:", error);
    }
  }

  // ============================================================
  // Audio track handling
  // ============================================================

  function handleAudioTrackSubscribed(track) {
    if (!track || track.kind !== LivekitClient.Track.Kind.Audio) {
      return;
    }
    console.log("AI Audio track received:", track.sid);

    if (aiAudioElements.has(track.sid)) {
      return;
    }

    const element = track.attach();
    element.id = `track-${track.sid}`;
    element.autoplay = true;
    $("#audioContainer").append(element);
    aiAudioElements.set(track.sid, { track, element });

    speakingAudioTracks++;
    setSpeaking();
    console.log("Active AI audio tracks:", speakingAudioTracks);

    element.play().catch((err) => {
      console.warn("[AUDIO] Playback could not start:", err);
    });
  }

  async function resumeCurrentAiAudio() {
    console.log("Resume Current Ai Audio", isAudioStopped);
    if (isAudioStopped) {
      for (const { element } of aiAudioElements.values()) {
        if (element.paused) {
          try {
            await element.play();
            console.log("[AUDIO] Playback resumed");
          } catch (err) {
            console.error("[AUDIO] Resume failed:", err);
          }
        }
      }
      isAudioStopped = false;
    }
  }

  function stopCurrentAiAudio() {
    console.log("Stopping Current Ai Audio", isAudioStopped);
    if (isAudioStopped == false) {
      for (const { element } of aiAudioElements.values()) {
        element.pause();
        try {
          element.currentTime = 0;
        } catch (_) {}

        console.log("[AUDIO] AI playback stopped for barge-in");
      }
      isAudioStopped = true;
    }
  }

  function handleAudioTrackUnsubscribed(track, publication, participant) {
    if (!track || track.kind !== LivekitClient.Track.Kind.Audio) {
      return;
    }

    console.log("AI Audio track removed:", track.sid);
    const entry = aiAudioElements.get(track.sid);
    if (entry) {
      entry.element.pause();
      entry.track.detach(entry.element);
      entry.element.srcObject = null;
      entry.element.remove();
      aiAudioElements.delete(track.sid);
    }

    speakingAudioTracks = Math.max(0, speakingAudioTracks - 1);
    if (speakingAudioTracks === 0 && room) {
      setListening();
    }
  }

  // ============================================================
  // Room event registration
  // ============================================================

  function registerRoomEvents() {
    // --------------------------------------------------------
    // AI audio
    // --------------------------------------------------------

    room.on(
      LivekitClient.RoomEvent.TrackSubscribed,
      handleAudioTrackSubscribed,
    );

    room.on(
      LivekitClient.RoomEvent.TrackUnsubscribed,
      handleAudioTrackUnsubscribed,
    );

    // --------------------------------------------------------
    // Data messages
    // --------------------------------------------------------

    room.on(LivekitClient.RoomEvent.DataReceived, handleDataReceived);

    // --------------------------------------------------------
    // Disconnect
    // --------------------------------------------------------

    room.on(LivekitClient.RoomEvent.Disconnected, handleRoomDisconnected);
  }

  // ============================================================
  // Room disconnect
  // ============================================================

  function handleRoomDisconnected(reason) {
    console.log("LiveKit disconnected:", reason);
    $("#audioContainer").empty();
    speakingAudioTracks = 0;
    stopCurrentAiAudio();
    resetAssistantState();
    clearLiveTranscript();

    setDisconnected();
    showUiElements();
    room = null;
  }

  // ============================================================
  // Connect
  // ============================================================

  $("#connectBtn").on("click", async function () {
    if (room) {
      console.log("Already connected.");
      return;
    }

    try {
      setConnecting();
      const sessionId = window.sessionId;

      if (!sessionId) {
        throw new Error("Session has not been initialized");
      }
      $("#connectBtn").prop("disabled", true);
      $("#connectBtnSpinner").removeClass("d-none");
      $("#connectBtnTxt").addClass("d-none");
      const response = await fetch(
        `/api/sessions/${encodeURIComponent(sessionId)}/voice`,
        { method: "POST" },
      );
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Failed to activate voice");
      }

      const data = await response.json();
      const wsUrl = data?.url?.trim();
      const token = data?.token?.trim();

      if (!wsUrl || !token) {
        throw new Error("LiveKit URL or token is missing.");
      }

      room = new LivekitClient.Room({
        disconnectOnPageLeave: true,
      });

      registerRoomEvents();

      console.log("Connecting to LiveKit:", wsUrl);

      const allowed = await requestMicrophonePermission();

      room.on("connectionStateChanged", (state) => {
        console.log("[LiveKit] Connection state:", state);
      });

      room.on("disconnected", (reason) => {
        console.log("[LiveKit] Disconnected. Reason:", reason);
      });

      if (allowed) {
        console.log("Ready to use microphone");
        room.prepareConnection(wsUrl, token);
        await room.connect(wsUrl, token);
        console.log("Connected to room:", room.name);

        // ----------------------------------------------------
        // Enable microphone
        // ----------------------------------------------------

        await room.localParticipant.setMicrophoneEnabled(true, {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        });
        isMicMuted = false;
        console.log("Microphone enabled.");
      }

      hideUiElements();

      setListening();
    } catch (error) {
      console.error("Connection failed:", error);
      await deleteVoiceSession();
      if (room) {
        try {
          await room.disconnect();
        } catch (disconnectError) {
          console.error("Cleanup disconnect failed:", disconnectError);
        }
      }

      room = null;
      showUiElements();
      setDisconnected();
      alert("Failed to connect: " + error.message);
    }
  });

  async function requestMicrophonePermission() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      console.log("Microphone permission granted");
      // Stop the stream if you only needed permission
      stream.getTracks().forEach((track) => track.stop());
      return true;
    } catch (error) {
      console.error("Microphone permission denied:", error);
      if (error.name === "NotAllowedError") {
        alert(
          "Microphone permission was denied. Please allow microphone access in your browser settings.",
        );
      } else if (error.name === "NotFoundError") {
        alert("No microphone was found.");
      }
      return false;
    }
  }
  // ============================================================
  // Disconnect
  // ============================================================

  $("#disconnectBtn").on("click", async function () {
    showUiElements();
    console.log("Disconnecting from LiveKit...");

    try {
      stopCurrentAiAudio();
      latestGenerationId = 0;
      if (!room) {
        return;
      }
      await room.disconnect();
    } catch (error) {
      console.error("Disconnect failed:", error);

      // Make sure UI is reset even if LiveKit
      // throws an error.
      handleRoomDisconnected();
    }
  });

  $("#micMuteUnmuteBtn").on("click", async function () {
    const micPublication = await room?.localParticipant?.getTrackPublication(
      LivekitClient.Track.Source.Microphone,
    );
    console.log("micPublication", micPublication);
    console.log("isMicMuted", isMicMuted);
    if (micPublication && micPublication?.track) {
      if (!isMicMuted) {
        micPublication.track.mute();
        isMicMuted = true;
        $("#micMuteUnmuteBtn")
          .removeClass("btn-secondary")
          .addClass("btn-danger");
        console.log("Microphone track muted.");
      } else {
        micPublication.track.unmute();
        isMicMuted = false;
        $("#micMuteUnmuteBtn")
          .removeClass("btn-danger")
          .addClass("btn-secondary");
        console.log("Microphone track unmuted.");
      }
    } else {
      console.warn("No active microphone track found.");
    }
  });
  // ============================================================
  // Clear chat
  // ============================================================

  $("#clearChatBtn").on("click", function () {
    clearChat();
  });


  function hideUiElements() {
    $("#connectBtn").addClass("d-none");
    $("#connectBtnSpinner").addClass("d-none");
    $("#radioBtnsDiv").addClass("d-none");
    $("#connectBtnTxt").removeClass("d-none");
    $("#micMuteUnmuteBtn").removeClass("d-none");
    $("#btnSend").addClass("d-none");
    $("#voiceContainer").removeClass("d-none");
    $("#disconnectBtn").removeClass("d-none");
    $("#clearChatBtn").addClass("d-none");
    // $("#chatBody").css("visibility", "hidden");
    $("#chatBody").css("opacity", "0.2");
    $("#txtMessage").prop("disabled", true);
    setListening();
  }

  function showUiElements() {
    $("#connectBtn").prop("disabled", false);
    $("#connectBtnSpinner").addClass("d-none");
    $("#micMuteUnmuteBtn").addClass("d-none");
    $("#radioBtnsDiv").removeClass("d-none");
    $("#connectBtnTxt").removeClass("d-none");
    $("#connectBtn").removeClass("d-none");
    $("#btnSend").removeClass("d-none");
    $("#clearChatBtn").removeClass("d-none");
    $("#voiceContainer").addClass("d-none");
    $("#disconnectBtn").addClass("d-none");
    // $("#chatBody").css("visibility", "visible");
    $("#chatBody").css("opacity", "1");
    $("#txtMessage").prop("disabled", false);
  }

  $("#txtMessage").on("keydown", async function (event) {
    if (event.key === "Enter") {
      event.preventDefault();
      await sendMessage();
    }
  });

  function setSendingState(isSending) {
    if (isSending) {
      $("#btnSend").html('<i class="bi bi-stop-circle"></i>');
      $("#btnSend").attr("title", "Stop");
    } else {
      $("#btnSend").html('<i class="bi bi-send-fill"></i>');
      $("#btnSend").attr("title", "Send");
    }
    $("#txtMessage").prop("disabled", isSending);
    $("#connectBtn").prop("disabled", isSending);
  }

  $("#btnSend").on("click", async () => sendMessage());

  async function sendMessage() {
    if (currentController) {
      console.log("Cancelling request...");
      currentController.abort();
      currentController = null;
    }

    const sessionId = window.sessionId;

    if (!sessionId) {
      throw new Error("Session has not been initialized");
    }

    currentController = new AbortController();
    $("#errorTxt").text("");
    const input = $("#txtMessage");
    let message = input.val().trim();
    if (!message) {
      $("#errorTxt").text("Write something first...");
      input.focus();
      return;
    }
    setSendingState(true);
    addUserMessage(message);
    input.val("");

    showThinking();
    const aiMessage = createAssistantMessage();
    const content = aiMessage.find(".ai-content");
    const checkingText = "Checking that...";
    content.html(
      `<span class="text-muted"><i class="bi bi-hourglass-split me-1"></i> ${escapeHtml(checkingText)}</span>`,
    );
    aiMessage.show();
    scrollChatToBottom();
    let fullResponse = "";
    try {
      const API_URL = `/api/sessions/${encodeURIComponent(sessionId)}/chat`;
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
        }),
        signal: currentController.signal,
      });
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }
      if (!response.body) {
        throw new Error("Streaming response is not supported by this browser.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let buffer = "";
      let firstTokenReceived = false;
      let streamFinished = false;

      while (!streamFinished) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        // Decode incoming bytes
        buffer += decoder.decode(value, {
          stream: true,
        });

        // Split complete SSE lines
        const lines = buffer.split(/\r?\n/);
        // Keep incomplete line

        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          // Ignore empty lines

          if (!trimmed) {
            continue;
          }
          // Only process SSE data
          if (!trimmed.startsWith("data:")) {
            continue;
          }

          const rawData = trimmed.substring(6).trim();
          // ========================================
          // STREAM COMPLETE
          // ========================================
          console.log("rawdate", rawData);
          if (rawData === "[DONE]") {
            streamFinished = true;
            break;
          }

          if (!rawData) {
            continue;
          }
          // ========================================
          // PARSE JSON
          // ========================================

          let token = "";
          let data = null;
          try {
            data = JSON.parse(rawData);
          } catch {}

          if (data !== null) {
            if (data.error) {
              throw new Error(data.error);
            }                      

            token = data.token ?? "";
          } else {
            token = rawData;
          }

          if (!token) {
            continue;
          }
          // ========================================
          // FIRST TOKEN
          // ========================================

          if (!firstTokenReceived) {
            firstTokenReceived = true;
            hideThinking();
          }
          // ========================================
          // APPEND TOKEN
          // ========================================

          fullResponse += token;
          // ========================================
          // RENDER MARKDOWN
          // ========================================

          content.html(marked.parse(fullResponse));
          scrollChatToBottom();
        }
      }
      // If stream completed without receiving anything

      if (!firstTokenReceived) {
        hideThinking();
        content.html("<span class='text-muted'>No response received.</span>");
      }
    } catch (error) {
      if (error.name === "AbortError") {
        console.log("Request cancelled by user");
      } else {
        console.error("Streaming error:", error);
        aiMessage.show();
        content.html(`
                <div class="text-danger">
                    <i class="bi bi-exclamation-triangle me-1"></i>
                    Unable to receive AI response.
                </div>
            `);
        scrollChatToBottom();
      }
      hideThinking();
    } finally {
      if (fullResponse) {
        content.html(marked.parse(fullResponse));
      }
      setSendingState(false);
      input.focus();
      currentController = null;
    }
  }

  function escapeHtml(text) {
    return $("<div>").text(text).html();
  }
  // ============================================
  // THINKING
  // ============================================

  function showThinking() {
    const element = document.getElementById("thinkingText");
    if (element) {
      $("#thinkingContainer").show();
      element.textContent = "AI is thinking";
    }
    scrollChatToBottom();
  }

  function hideThinking() {
    $("#thinkingContainer").hide();
  }
  // ============================================================
  // Initial state
  // ============================================================

  setDisconnected();
  console.log("Voice AI Agent client initialized.");
});

async function initializeSession() {
  // Create a new session
  const response = await fetch("/api/session", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to create session: HTTP ${response.status}`);
  }

  const data = await response.json();

  if (!data.sessionId) {
    throw new Error("Session ID missing from server response");
  }

  // Save session ID for later chat and voice requests
  window.sessionId = data.sessionId;
  console.log("New session created:", data.sessionId);
  return data.sessionId;
}

async function deleteSession() {
  const sessionId = window.sessionId;
  if (sessionId) {
    await fetch(`/api/sessions/${sessionId}`, {
      method: "DELETE",
      keepalive: true,
      headers: {
        "Content-Type": "application/json",
      },
    });
    console.log(`Session - ${sessionId} has been removed`);
  }
}

async function deleteVoiceSession() {
  const sessionId = window.sessionId;
  if (sessionId) {
    await fetch(`/api/voice_sessions/${sessionId}`, {
      method: "DELETE",
      keepalive: true,
      headers: {
        "Content-Type": "application/json",
      },
    });
    console.log(`Voice Session - ${sessionId} has been removed`);
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  try {
    const session_Id = await initializeSession();
    console.log("Session initialized:", session_Id);

    // Enable chat and voice controls here if needed
  } catch (error) {
    console.error("Session initialization failed:", error);
  }
});

window.addEventListener("beforeunload", async (event) => {
  await deleteSession();
});
