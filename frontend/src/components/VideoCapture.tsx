// ClarifySign - VideoCapture Component
// Webcam access via getUserMedia, frame extraction via Canvas API every 100ms
import React, { useRef, useEffect, useCallback, useState } from "react";

interface VideoCaptureProps {
  onFrame: (frameB64: string) => void;
  isCapturing: boolean;
  onCameraReady?: (ready: boolean) => void;
}

const FRAME_INTERVAL_MS = 100; // 10 fps to backend

export const VideoCapture: React.FC<VideoCaptureProps> = ({
  onFrame,
  isCapturing,
  onCameraReady,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [cameraState, setCameraState] = useState<"idle" | "loading" | "active" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [fps, setFps] = useState(0);
  const fpsCountRef = useRef(0);

  // ── Start camera ────────────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setCameraState("loading");
    setErrorMsg("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraState("active");
      onCameraReady?.(true);
    } catch (err: unknown) {
      const msg =
        err instanceof DOMException
          ? err.name === "NotAllowedError"
            ? "Camera permission denied. Please allow camera access."
            : err.message
          : String(err);
      setErrorMsg(msg);
      setCameraState("error");
      onCameraReady?.(false);
    }
  }, [onCameraReady]);

  // ── Stop camera ─────────────────────────────────────────────────────────
  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraState("idle");
    onCameraReady?.(false);
  }, [onCameraReady]);

  // ── Frame extraction loop ────────────────────────────────────────────────
  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (!isCapturing || cameraState !== "active") return;

    fpsCountRef.current = 0;
    const fpsInterval = setInterval(() => {
      setFps(fpsCountRef.current);
      fpsCountRef.current = 0;
    }, 1000);

    intervalRef.current = setInterval(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      canvas.width = video.videoWidth || 320;
      canvas.height = video.videoHeight || 240;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const b64 = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
      onFrame(b64);
      fpsCountRef.current += 1;
    }, FRAME_INTERVAL_MS);

    return () => {
      clearInterval(fpsInterval);
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isCapturing, cameraState, onFrame]);

  // ── Cleanup on unmount ───────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <section className="video-card" aria-label="Camera feed">
      <div className="video-wrapper">
        <video
          ref={videoRef}
          id="camera-video"
          className="video-element"
          autoPlay
          playsInline
          muted
          aria-label="Live camera feed"
        />
        <canvas ref={canvasRef} style={{ display: "none" }} aria-hidden="true" />

        {/* Overlay when not active */}
        {cameraState !== "active" && (
          <div className="video-overlay" role="status">
            {cameraState === "loading" ? (
              <>
                <span className="spinner" aria-label="Loading camera" />
                <p className="text-secondary">Starting camera…</p>
              </>
            ) : cameraState === "error" ? (
              <>
                <span className="video-overlay-icon" aria-hidden="true">📷</span>
                <p className="text-danger" style={{ fontSize: "0.85rem", maxWidth: 260, textAlign: "center" }}>
                  {errorMsg}
                </p>
                <button id="retry-camera-btn" className="btn btn-secondary btn-sm" onClick={startCamera}>
                  Retry
                </button>
              </>
            ) : (
              <>
                <span className="video-overlay-icon" aria-hidden="true">📷</span>
                <p className="text-secondary">Camera not started</p>
              </>
            )}
          </div>
        )}

        {/* Status badges */}
        {cameraState === "active" && (
          <div className="video-badge-row" aria-hidden="true">
            <span className="badge badge-success">
              <span className="dot-live" /> LIVE
            </span>
          </div>
        )}
        {isCapturing && cameraState === "active" && (
          <span className="badge badge-info video-fps-badge" aria-hidden="true">
            {fps} fps
          </span>
        )}
      </div>

      <div className="video-controls">
        {cameraState === "idle" || cameraState === "error" ? (
          <button
            id="start-camera-btn"
            className="btn btn-primary w-full"
            onClick={startCamera}
            aria-label="Start camera"
          >
            📷 Start Camera
          </button>
        ) : (
          <button
            id="stop-camera-btn"
            className="btn btn-secondary"
            onClick={stopCamera}
            aria-label="Stop camera"
          >
            ⏹ Stop
          </button>
        )}
      </div>
    </section>
  );
};
