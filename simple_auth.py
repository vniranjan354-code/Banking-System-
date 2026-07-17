# simple_auth.py
#
# 2026-grade Face + Voice Authentication — CPU-optimised
#
# Face pipeline:   MediaPipe (detection + liveness) → InsightFace ArcFace (embedding)
# Voice pipeline:  Resemblyzer ECAPA-TDNN (d-vector embedding)
# Liveness:        EAR blink detection + head-pose jitter (defeats static photos)
# Anti-spoof audio: VAD pause analysis + pitch-variance check (defeats recordings)

import os
import cv2
import numpy as np
import json
import subprocess
import warnings
import logging
from pathlib import Path
from scipy.spatial.distance import cosine

warnings.filterwarnings("ignore")
logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# ── optional imports with graceful fallback ──────────────────────────────────

try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False
    print("⚠  insightface not installed  →  falling back to legacy face method")
    print("   pip install insightface onnxruntime")

try:
    import mediapipe as mp
    # MediaPipe 0.10+ dropped mp.solutions — detect which API is available
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        _MP_API = "legacy"          # 0.9.x
    else:
        # 0.10+ Tasks API
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            _MP_API = "tasks"
        except ImportError:
            _MP_API = "none"
    HAS_MEDIAPIPE = (_MP_API != "none")
except ImportError:
    HAS_MEDIAPIPE = False
    _MP_API = "none"
    print("⚠  mediapipe not installed  →  liveness detection disabled")
    print("   pip install mediapipe")

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    HAS_RESEMBLYZER = True
except ImportError:
    HAS_RESEMBLYZER = False
    print("⚠  resemblyzer not installed  →  falling back to legacy voice method")
    print("   pip install resemblyzer")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False


# ─────────────────────────────────────────────────────────────────────────────
#  Liveness Detector  (MediaPipe-based)
# ─────────────────────────────────────────────────────────────────────────────

class LivenessDetector:
    """
    Passive liveness using two complementary cues:
      1. EAR (Eye Aspect Ratio) blink  — a photo never blinks
      2. Head-pose jitter              — a photo has near-zero pose variance
    """

    EAR_THRESHOLD   = 0.22   # below this → eye closed
    EAR_CONSEC      = 2      # consecutive frames for a blink
    MIN_BLINKS      = 1      # minimum blinks required
    POSE_STD_MIN    = 0.8    # degrees — real head has more movement than this

    # MediaPipe landmark indices
    _LEFT_EYE  = [362, 385, 387, 263, 373, 380]
    _RIGHT_EYE = [33,  160, 158, 133, 153, 144]

    def __init__(self):
        self._api      = _MP_API
        self.face_mesh = None   # used by legacy API
        self._detector = None   # used by Tasks API

        if not HAS_MEDIAPIPE:
            print("  ⚠  MediaPipe unavailable — liveness disabled")
            return

        if self._api == "legacy":
            # MediaPipe 0.9.x
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print("  ✓ MediaPipe liveness (legacy solutions API)")

        elif self._api == "tasks":
            # MediaPipe 0.10+ Tasks API — needs the face_landmarker.task model
            try:
                import urllib.request, tempfile, os as _os
                model_path = _os.path.join(
                    tempfile.gettempdir(), "face_landmarker.task"
                )
                if not _os.path.exists(model_path):
                    print("  Downloading face_landmarker.task model (~5 MB)...")
                    urllib.request.urlretrieve(
                        "https://storage.googleapis.com/mediapipe-models/"
                        "face_landmarker/face_landmarker/float16/latest/"
                        "face_landmarker.task",
                        model_path,
                    )

                BaseOptions   = mp_tasks.BaseOptions
                FaceLandmarker      = mp_vision.FaceLandmarker
                FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
                VisionRunningMode   = mp_vision.RunningMode

                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=True,   # gives blink scores directly!
                )
                self._detector = FaceLandmarker.create_from_options(options)
                print("  ✓ MediaPipe liveness (Tasks API 0.10+, blendshapes)")
            except Exception as e:
                print(f"  ⚠  MediaPipe Tasks init failed: {e} — liveness disabled")
                self._api = "none"

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ear(self, landmarks, indices, w, h):
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C + 1e-6)

    def _process_frame_legacy(self, frame):
        """Returns (landmarks_list, w, h) or (None, w, h)"""
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None, w, h
        return res.multi_face_landmarks[0].landmark, w, h

    def _process_frame_tasks(self, frame):
        """
        Returns (landmarks_list, w, h, blink_score) or (None, w, h, 0).
        blink_score: average of eyeBlinkLeft + eyeBlinkRight blendshapes (0–1).
        """
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._detector.detect(mp_image)
        if not result.face_landmarks:
            return None, w, h, 0.0

        lm = result.face_landmarks[0]   # NormalizedLandmark list

        # Blendshapes give us direct blink scores — much more reliable than EAR
        blink = 0.0
        if result.face_blendshapes:
            bs = {b.category_name: b.score for b in result.face_blendshapes[0]}
            blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2.0

        return lm, w, h, blink

    def check_video(self, video_path, max_frames=150):
        """
        Returns (is_live: bool, reason: str)
        Samples up to max_frames from the video.
        """
        if self._api == "none" or (self.face_mesh is None and self._detector is None):
            return True, "liveness_skipped (mediapipe unavailable)"

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return False, "empty_video"

        blink_count    = 0
        ear_consec     = 0
        eye_closed     = False
        yaw_history    = []
        frames_checked = 0
        step = max(1, total // max_frames)

        for idx in range(0, min(total, max_frames * step), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if self._api == "tasks":
                lm, w, h, blink_score = self._process_frame_tasks(frame)
                if lm is None:
                    continue
                frames_checked += 1

                # Blink via blendshape score (> 0.4 = eye closed)
                if blink_score > 0.4:
                    ear_consec += 1
                    eye_closed  = True
                else:
                    if eye_closed and ear_consec >= self.EAR_CONSEC:
                        blink_count += 1
                    ear_consec = 0
                    eye_closed = False

                # Head-pose yaw
                nose_x    = lm[1].x
                eye_mid_x = (lm[33].x + lm[263].x) / 2
                yaw_history.append((nose_x - eye_mid_x) * 100)

            else:  # legacy
                lm, w, h = self._process_frame_legacy(frame)
                if lm is None:
                    continue
                frames_checked += 1

                left_ear  = self._ear(lm, self._LEFT_EYE,  w, h)
                right_ear = self._ear(lm, self._RIGHT_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0

                if ear < self.EAR_THRESHOLD:
                    ear_consec += 1
                    eye_closed  = True
                else:
                    if eye_closed and ear_consec >= self.EAR_CONSEC:
                        blink_count += 1
                    ear_consec = 0
                    eye_closed = False

                nose_x    = lm[1].x
                eye_mid_x = (lm[33].x + lm[263].x) / 2
                yaw_history.append((nose_x - eye_mid_x) * 100)

        cap.release()

        if frames_checked < 5:
            return False, "too_few_face_frames"

        pose_std = float(np.std(yaw_history)) if yaw_history else 0.0

        has_blink    = blink_count >= self.MIN_BLINKS
        has_movement = pose_std >= self.POSE_STD_MIN

        if has_blink and has_movement:
            return True, f"live (blinks={blink_count}, pose_std={pose_std:.2f})"
        elif has_blink:
            return True, f"live_blink_only (blinks={blink_count}, pose_std={pose_std:.2f})"
        elif has_movement:
            return True, f"live_movement_only (blinks={blink_count}, pose_std={pose_std:.2f})"
        else:
            return False, f"spoof_suspected (blinks={blink_count}, pose_std={pose_std:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
#  Audio Anti-Spoof
# ─────────────────────────────────────────────────────────────────────────────

class AudioAntiSpoof:
    """
    Heuristic checks that recorded/replayed audio tends to fail:
      1. Pause count  — real speech has natural pauses; replay is often continuous
      2. Pitch variance — recordings played back in a room have compressed F0 range
      3. WebRTC VAD    — verifies actual voice activity vs pure noise/silence
    """

    # Conservative defaults can be strict for short/quiet videos —
    # relax slightly to reduce false rejects in real-world mic conditions.
    MIN_PAUSES         = 2      # real speech has at least 2 silence intervals
    MIN_PITCH_STD      = 2.0    # Hz — lowered to tolerate quieter/monotone speech
    MIN_VOICE_RATIO    = 0.10   # at least 10 % of frames must be voiced

    def check(self, audio_path):
        """Returns (is_genuine: bool, reason: str)"""
        if not HAS_LIBROSA:
            return True, "audio_antispoof_skipped (librosa missing)"

        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000)

            if len(y) < sr * 0.5:
                return False, "audio_too_short"

            # ── Pause analysis ──
            intervals = librosa.effects.split(y, top_db=30)
            pause_count = max(0, len(intervals) - 1)

            # ── Pitch variance ──
            # Primary pitch estimator: pyin — good but can return many NaNs
            f0, voiced_flag, _ = librosa.pyin(
                y, fmin=50, fmax=500, sr=sr, frame_length=1024
            )
            # Safely compute voiced F0 values and basic stats
            if voiced_flag is None:
                voiced_flag = np.zeros_like(f0, dtype=bool)
            f0_voiced = f0[voiced_flag]
            pitch_std  = float(np.std(f0_voiced)) if len(f0_voiced) > 5 else 0.0
            voice_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0

            # If pyin fails to detect F0 (flat/zero), try a fallback (yin)
            if pitch_std < self.MIN_PITCH_STD and len(y) > 0:
                try:
                    f0_yin = librosa.yin(y, fmin=50, fmax=500, sr=sr)
                    # yin returns one f0 per frame — filter zeros/nans
                    f0y = f0_yin[~np.isnan(f0_yin) & (f0_yin > 0)]
                    if len(f0y) > 5:
                        pitch_std = float(np.std(f0y))
                except Exception:
                    pass

            # ── WebRTC VAD (optional refinement) ──
            vad_ok = True
            if HAS_VAD:
                try:
                    vad = webrtcvad.Vad(2)
                    import soundfile as sf
                    data, rate = sf.read(audio_path, dtype="int16")
                    frame_dur  = 20   # ms
                    frame_len  = int(rate * frame_dur / 1000)
                    voiced_frames = sum(
                        vad.is_speech(data[i:i+frame_len].tobytes(), rate)
                        for i in range(0, len(data) - frame_len, frame_len)
                    )
                    total_frames = len(data) // frame_len
                    vad_ratio = voiced_frames / max(1, total_frames)
                    vad_ok = vad_ratio >= self.MIN_VOICE_RATIO
                except Exception:
                    pass

            ok = (
                pause_count   >= self.MIN_PAUSES     and
                pitch_std     >= self.MIN_PITCH_STD  and
                voice_ratio   >= self.MIN_VOICE_RATIO and
                vad_ok
            )
            reason = (
                f"pauses={pause_count}, pitch_std={pitch_std:.1f}Hz, "
                f"voice_ratio={voice_ratio:.0%}, vad={'ok' if vad_ok else 'fail'}"
            )
            return ok, reason

        except Exception as e:
            return True, f"antispoof_error_skipped ({e})"


# ─────────────────────────────────────────────────────────────────────────────
#  Face Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class FacePipeline:
    """
    Primary:  InsightFace ArcFace (buffalo_sc) — 512-dim, ONNX, CPU ~150 ms/frame
    Fallback: OpenCV Haar + histogram/LBP (legacy)
    """

    def __init__(self):
        self.use_arcface = False
        self.app = None

        if HAS_INSIGHTFACE:
            try:
                self.app = FaceAnalysis(
                    name="buffalo_sc",           # small+fast model
                    providers=["CPUExecutionProvider"],
                )
                self.app.prepare(ctx_id=-1, det_size=(320, 320))
                self.use_arcface = True
                print("  ✓ ArcFace (buffalo_sc) loaded — 512-dim embeddings")
            except Exception as e:
                print(f"  ⚠  InsightFace load error: {e} — using legacy face")

        if not self.use_arcface:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            print("  ✓ Legacy face detector loaded (Haar + LBP)")

    # ── ArcFace path ──────────────────────────────────────────────────────────

    def _arcface_embedding(self, frame):
        """Return 512-dim L2-normalised ArcFace embedding, or None."""
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)
        if not faces:
            return None
        # Largest face by bbox area
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        emb  = face.embedding
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    # ── Legacy path ───────────────────────────────────────────────────────────

    def _legacy_embedding(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face  = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        hist  = cv2.calcHist([face], [0], None, [64], [0, 256]).flatten()
        hist /= hist.sum() + 1e-6
        lbp   = []
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                blk = face[i:i+10, j:j+10]
                lbp += [blk.mean() / 255.0, blk.std() / 255.0]
        small = cv2.resize(face, (20, 20)).flatten() / 255.0
        emb   = np.concatenate([hist, lbp, small])
        norm  = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    # ── Public API ────────────────────────────────────────────────────────────

    def embedding_from_frame(self, frame):
        if self.use_arcface:
            return self._arcface_embedding(frame)
        return self._legacy_embedding(frame)

    def best_embedding_from_video(self, video_path, num_samples=15):
        """
        Sample frames, return the embedding from the frame with the clearest face.
        Also returns the best frame for preview saving.
        """
        cap    = cv2.VideoCapture(video_path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return None, None

        indices = np.linspace(0, max(0, total - 1), num_samples, dtype=int)

        best_emb   = None
        best_frame = None
        best_conf  = -1.0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if self.use_arcface:
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.app.get(rgb)
                if not faces:
                    continue
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                area = (face.bbox[2]-face.bbox[0]) * (face.bbox[3]-face.bbox[1])
                det_score = float(getattr(face, "det_score", 0.5))
                conf  = area * det_score
                if conf > best_conf:
                    best_conf  = conf
                    norm       = np.linalg.norm(face.embedding)
                    best_emb   = face.embedding / norm if norm > 0 else face.embedding
                    best_frame = frame
            else:
                emb = self._legacy_embedding(frame)
                if emb is not None and best_emb is None:
                    best_emb   = emb
                    best_frame = frame

        cap.release()
        return best_emb, best_frame

    @property
    def threshold(self):
        # ArcFace cosine similarity is much tighter
        return 0.40 if self.use_arcface else 0.60

    @property
    def method_name(self):
        return "ArcFace (buffalo_sc)" if self.use_arcface else "Legacy (Haar+LBP)"


# ─────────────────────────────────────────────────────────────────────────────
#  Voice Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class VoicePipeline:
    """
    Primary:  Resemblyzer ECAPA-TDNN  — 256-dim d-vector, CPU-friendly
    Fallback: MFCC statistics         — legacy
    """

    def __init__(self):
        self.use_resemblyzer = False
        self.encoder = None

        if HAS_RESEMBLYZER:
            try:
                self.encoder = VoiceEncoder(device="cpu")
                self.use_resemblyzer = True
                print("  ✓ Resemblyzer (ECAPA-TDNN) loaded — 256-dim d-vectors")
            except Exception as e:
                print(f"  ⚠  Resemblyzer load error: {e} — using legacy voice")

        if not self.use_resemblyzer:
            print("  ✓ Legacy voice encoder loaded (MFCC stats)")

    def embedding_from_file(self, audio_path):
        if self.use_resemblyzer:
            return self._resemblyzer_embedding(audio_path)
        return self._mfcc_embedding(audio_path)

    def _resemblyzer_embedding(self, audio_path):
        try:
            wav = preprocess_wav(audio_path)
            if len(wav) < 16000 * 0.3:
                print("  Audio too short for voice embedding")
                return None
            emb  = self.encoder.embed_utterance(wav)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as e:
            print(f"  Voice embedding error: {e}")
            return None

    def _mfcc_embedding(self, audio_path):
        if not HAS_LIBROSA:
            return None
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000)
            if len(y) < sr * 0.3:
                print("  Audio too short")
                return None
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            emb   = np.concatenate([
                np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
                np.max(mfccs, axis=1),  np.min(mfccs, axis=1),
            ])
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as e:
            print(f"  Voice processing error: {e}")
            return None

    @property
    def threshold(self):
        return 0.50 if self.use_resemblyzer else 0.55

    @property
    def method_name(self):
        return "Resemblyzer (ECAPA-TDNN)" if self.use_resemblyzer else "Legacy (MFCC stats)"


# ─────────────────────────────────────────────────────────────────────────────
#  Main Authenticator
# ─────────────────────────────────────────────────────────────────────────────

class SimpleVideoAuthenticator:
    """
    Drop-in replacement with a modern biometric stack.
    Public API is identical to the original class.
    """

    def __init__(self, enrollment_dir="./enrolled_users"):
        self.enrollment_dir = Path(enrollment_dir)
        self.enrollment_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*55)
        print("  Initialising biometric pipelines...")
        print("="*55)

        self.face_pipeline   = FacePipeline()
        self.voice_pipeline  = VoicePipeline()
        self.liveness        = LivenessDetector()
        self.audio_antispoof = AudioAntiSpoof()

        print("\n  Face  method : " + self.face_pipeline.method_name)
        print("  Voice method : " + self.voice_pipeline.method_name)
        print("  Liveness     : " + ("MediaPipe EAR+pose" if HAS_MEDIAPIPE else "disabled"))
        print("  Audio spoof  : " + ("VAD+pitch analysis" if HAS_LIBROSA  else "disabled"))
        print("="*55 + "\n")

    # ── Audio extraction ──────────────────────────────────────────────────────

    def _extract_audio(self, video_path):
        output_audio = str(video_path).rsplit(".", 1)[0] + "_tmp_audio.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_audio,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if os.path.exists(output_audio):
                return output_audio
        except Exception:
            pass

        # moviepy fallback
        for moviepy_import in [
            lambda: __import__("moviepy.editor", fromlist=["VideoFileClip"]).VideoFileClip,
            lambda: __import__("moviepy",        fromlist=["VideoFileClip"]).VideoFileClip,
        ]:
            try:
                VideoFileClip = moviepy_import()
                v = VideoFileClip(str(video_path))
                v.audio.write_audiofile(output_audio, fps=16000, verbose=False, logger=None)
                v.close()
                return output_audio
            except Exception:
                pass

        print("  ❌ Could not extract audio. Install ffmpeg or moviepy.")
        return None

    # ── Enroll ────────────────────────────────────────────────────────────────

    def enroll(self, user_id, video_path):
        print(f"\n{'='*55}")
        print(f"  ENROLLING: {user_id}")
        print(f"{'='*55}")

        if not os.path.exists(video_path):
            print(f"  ❌ File not found: {video_path}")
            return {"success": False, "message": "Video file not found"}

        # ── Step 1: Liveness ──
        print("\n[1/4] Liveness check...")
        is_live, live_reason = self.liveness.check_video(video_path)
        print(f"  {'✓' if is_live else '⚠'} {live_reason}")
        if not is_live:
            print("  ❌ Liveness check failed — possible spoof")
            return {"success": False, "message": f"Liveness failed: {live_reason}"}

        # ── Step 2: Face ──
        print("\n[2/4] Extracting face embedding...")
        face_emb, face_frame = self.face_pipeline.best_embedding_from_video(video_path)
        if face_emb is None:
            print("  ❌ No face detected")
            return {"success": False, "message": "No face detected"}
        print(f"  ✓ Embedding: {len(face_emb)}-dim")

        # ── Step 3: Audio ──
        print("\n[3/4] Extracting audio...")
        audio_path = self._extract_audio(video_path)
        if audio_path is None or not os.path.exists(audio_path):
            print("  ❌ Audio extraction failed")
            return {"success": False, "message": "Could not extract audio"}
        print("  ✓ Audio extracted")

        # ── Step 3b: Audio anti-spoof ──
        print("\n[3b]  Audio anti-spoof check...")
        audio_ok, audio_reason = self.audio_antispoof.check(audio_path)
        print(f"  {'✓' if audio_ok else '⚠'} {audio_reason}")
        # (warn but don't block enrollment — mic conditions vary)

        # ── Step 4: Voice ──
        print("\n[4/4] Extracting voice embedding...")
        voice_emb = self.voice_pipeline.embedding_from_file(audio_path)

        if os.path.exists(audio_path):
            os.remove(audio_path)

        if voice_emb is None:
            print("  ❌ Voice embedding failed")
            return {"success": False, "message": "Could not process voice"}
        print(f"  ✓ Embedding: {len(voice_emb)}-dim")

        # ── Save ──
        print("\n  Saving enrollment data...")
        user_dir = self.enrollment_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        np.save(user_dir / "face_embedding.npy",  face_emb)
        np.save(user_dir / "voice_embedding.npy", voice_emb)

        if face_frame is not None:
            cv2.imwrite(str(user_dir / "enrolled_face.jpg"), face_frame)

        meta = {
            "user_id":       user_id,
            "face_method":   self.face_pipeline.method_name,
            "voice_method":  self.voice_pipeline.method_name,
            "face_dim":      int(len(face_emb)),
            "voice_dim":     int(len(voice_emb)),
            "face_threshold":  self.face_pipeline.threshold,
            "voice_threshold": self.voice_pipeline.threshold,
        }
        with open(user_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n{'='*55}")
        print(f"  ✅  USER '{user_id}' ENROLLED SUCCESSFULLY!")
        print(f"{'='*55}")
        return {"success": True, "message": f"User {user_id} enrolled"}

    # ── Authenticate ──────────────────────────────────────────────────────────

    def authenticate(self, user_id, video_path):
        print(f"\n{'='*55}")
        print(f"  AUTHENTICATING: {user_id}")
        print(f"{'='*55}")

        result = {
            "authenticated": False,
            "face_match":    False,
            "voice_match":   False,
            "face_score":    0.0,
            "voice_score":   0.0,
            "liveness":      False,
            "audio_genuine": False,
            "message":       "",
        }

        # ── Check enrolled ──
        user_dir = self.enrollment_dir / user_id
        if not user_dir.exists():
            result["message"] = f"User '{user_id}' not enrolled"
            print(f"\n  ❌ {result['message']}")
            return result

        enrolled_face  = np.load(user_dir / "face_embedding.npy")
        enrolled_voice = np.load(user_dir / "voice_embedding.npy")

        # Load thresholds saved at enroll time (handles method switch gracefully)
        try:
            with open(user_dir / "metadata.json") as f:
                meta = json.load(f)
            face_thr  = meta.get("face_threshold",  self.face_pipeline.threshold)
            voice_thr = meta.get("voice_threshold", self.voice_pipeline.threshold)
        except Exception:
            face_thr  = self.face_pipeline.threshold
            voice_thr = self.voice_pipeline.threshold

        # ── Step 1: Liveness ──
        print("\n[1/4] Liveness check...")
        is_live, live_reason = self.liveness.check_video(video_path)
        result["liveness"] = is_live
        print(f"  {'✓' if is_live else '✗'} {live_reason}")
        if not is_live:
            result["message"] = f"Liveness failed: {live_reason}"
            print(f"\n  ❌ ACCESS DENIED  (spoof suspected)")
            return result

        # ── Step 2: Face ──
        print("\n[2/4] Extracting face embedding...")
        auth_face, _ = self.face_pipeline.best_embedding_from_video(video_path)
        if auth_face is None:
            result["message"] = "No face detected in video"
            print(f"  ❌ {result['message']}")
            return result
        print(f"  ✓ Embedding extracted")

        # ── Step 3: Audio ──
        print("\n[3/4] Extracting audio...")
        audio_path = self._extract_audio(video_path)
        if audio_path is None:
            result["message"] = "Could not extract audio"
            print(f"  ❌ {result['message']}")
            return result
        print("  ✓ Audio extracted")

        # ── Step 3b: Audio anti-spoof ──
        print("\n[3b]  Audio anti-spoof check...")
        audio_ok, audio_reason = self.audio_antispoof.check(audio_path)
        result["audio_genuine"] = audio_ok
        print(f"  {'✓' if audio_ok else '✗'} {audio_reason}")
        if not audio_ok:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            result["message"] = f"Audio spoof suspected: {audio_reason}"
            print(f"\n  ❌ ACCESS DENIED  (audio anti-spoof)")
            return result

        # ── Step 4: Voice ──
        print("\n[4/4] Extracting voice embedding...")
        auth_voice = self.voice_pipeline.embedding_from_file(audio_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if auth_voice is None:
            result["message"] = "Could not process voice"
            print(f"  ❌ {result['message']}")
            return result
        print(f"  ✓ Embedding extracted")

        # ── Biometric comparison ──
        print("\n  Comparing biometrics...")

        def cosine_sim(a, b):
            try:
                s = 1 - cosine(a, b)
                return float(max(0.0, min(1.0, s)))
            except Exception:
                return 0.0

        face_score  = cosine_sim(enrolled_face,  auth_face)
        voice_score = cosine_sim(enrolled_voice, auth_voice)

        result["face_score"]  = round(face_score,  4)
        result["voice_score"] = round(voice_score, 4)
        result["face_match"]  = face_score  >= face_thr
        result["voice_match"] = voice_score >= voice_thr
        result["authenticated"] = result["face_match"] and result["voice_match"]

        # ── Print results ──
        print(f"\n{'='*55}")
        print("  BIOMETRIC RESULTS")
        print(f"{'='*55}")
        print(f"  Liveness  : {'✓ PASS' if is_live    else '✗ FAIL'}  ({live_reason})")
        print(f"  Audio spoof: {'✓ PASS' if audio_ok  else '✗ FAIL'}  ({audio_reason})")
        print(f"  Face score : {face_score:.1%}  (threshold {face_thr:.0%})  "
              f"{'✓ PASS' if result['face_match']  else '✗ FAIL'}")
        print(f"  Voice score: {voice_score:.1%}  (threshold {voice_thr:.0%})  "
              f"{'✓ PASS' if result['voice_match'] else '✗ FAIL'}")
        print(f"{'='*55}")

        if result["authenticated"]:
            result["message"] = "Authentication successful"
            print("  ✅  ACCESS GRANTED")
        else:
            result["message"] = "Authentication failed"
            print("  ❌  ACCESS DENIED")

        print(f"{'='*55}")
        return result

    # ── Utilities ─────────────────────────────────────────────────────────────

    def list_users(self):
        users = []
        if self.enrollment_dir.exists():
            for item in self.enrollment_dir.iterdir():
                if item.is_dir() and (item / "metadata.json").exists():
                    users.append(item.name)
        return sorted(users)

    def delete_user(self, user_id):
        user_dir = self.enrollment_dir / user_id
        if user_dir.exists():
            import shutil
            shutil.rmtree(user_dir)
            return True
        return False
