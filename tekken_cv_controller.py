"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         TEKKEN 3  —  Computer Vision Controller  (2 Players)                 ║
║                                                                              ║
║  Both players sit side-by-side in front of ONE webcam.                       ║
║  Left half = Player 1   |   Right half = Player 2                            ║
║                                                                              ║
║  SAME GESTURES FOR BOTH PLAYERS:                                             ║
║  ─────────────────────────────────────────────────────────────               ║
║  Head tilt LEFT        →  Walk Left                                          ║
║  Head tilt RIGHT       →  Walk Right                                         ║
║  Head look UP          →  Jump                                               ║
║  Head look DOWN        →  Crouch                                             ║
║  RIGHT hand fast punch →  Cross  (X button)                                  ║
║  LEFT  hand fast punch →  Square                                             ║
║  RIGHT hand swipe down →  Circle                                             ║
║  LEFT  hand swipe down →  Triangle                                           ║
║                                                                              ║
║  KEYBOARD SHORTCUTS (while CV window is focused):                            ║
║  Q = quit  |  1 = declare P1 winner  |  2 = declare P2 winner  |  R = reset  ║
╚══════════════════════════════════════════════════════════════════════════════╝

DuckStation Key Bindings Used:
  Player 1 — Left Analog: W/A/D/S  |  Face: Cross=G, Square=F, Circle=H, Triangle=T
  Player 2 — Left Analog: ↑←→↓     |  Face: Cross=K, Square=J, Circle=L, Triangle=I
"""

import cv2
import mediapipe as mp
mpf = mp.solutions.face_mesh
mph = mp.solutions.hands
mpd = mp.solutions.drawing_utils
import numpy as np
import time
import threading
from collections import deque
from pynput.keyboard import Key, Controller

# ─────────────────────────────────────────────────────────────────────────────
#  TUNING CONSTANTS  (adjust if gestures trigger too easily / not enough)
# ─────────────────────────────────────────────────────────────────────────────
YAW_THRESH   = 22.0   # degrees head must tilt sideways to walk
PITCH_UP     = -20.0  # head pitch threshold to jump   (negative = tilted back)
PITCH_DOWN   =  20.0  # head pitch threshold to crouch (positive = chin down)
VEL_THRESH   =  45    # wrist pixel-movement per frame to detect punch/swipe
COOLDOWN_SEC =  0.30  # seconds between same attack gesture
HOLD_SECS    =  0.13  # how long movement keys are held down per frame

# ─────────────────────────────────────────────────────────────────────────────
#  KEY BINDINGS  — matches your DuckStation screenshots exactly
#  Format: "ACTION": (Player1_key, Player2_key)
# ─────────────────────────────────────────────────────────────────────────────

KEY_MAP = {
    "MOVE_LEFT":  ('a',   Key.left),
    "MOVE_RIGHT": ('d',   Key.right),
    "MOVE_UP":    ('w',   Key.up),
    "MOVE_DOWN":  ('s',   Key.down),
    "CROSS":      ('g',   'k'),      # X / kick
    "SQUARE":     ('f',   'j'),      # square / punch
    "CIRCLE":     ('h',   'l'),      # circle
    "TRIANGLE":   ('t',   'i'),      # triangle
}

# What to display on the HUD above each player's head
LABEL_MAP = {
    "MOVE_LEFT":  "<-- LEFT",
    "MOVE_RIGHT": "--> RIGHT",
    "MOVE_UP":    "^ JUMP",
    "MOVE_DOWN":  "v CROUCH",
    "CROSS":      "X CROSS",
    "SQUARE":     "[] SQUARE",
    "CIRCLE":     "O CIRCLE",
    "TRIANGLE":   "/\\ TRI",
    "NEUTRAL":    "",
}

# Movement keys are held; attack keys are tapped
HOLD_SET = {"MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN"}

# ─────────────────────────────────────────────────────────────────────────────
#  KEY INJECTOR
# ─────────────────────────────────────────────────────────────────────────────
_kb = Controller()

def inject(action: str, player: int):
    """Fire the keyboard event for the given action + player in a daemon thread."""
    if action not in KEY_MAP:
        return
    key = KEY_MAP[action][player - 1]

    def _fire():
        _kb.press(key)
        time.sleep(HOLD_SECS if action in HOLD_SET else 0.05)
        try:
            _kb.release(key)
        except Exception:
            pass

    threading.Thread(target=_fire, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
#  HEAD ANGLE CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
def head_angles(face_lm):
    """
    Returns (yaw, pitch) in approximate degrees using MediaPipe face landmarks.
    yaw  > 0 → head tilted to its LEFT  (appears right in mirrored frame)
    pitch> 0 → chin down (crouch gesture)
    """
    nose  = face_lm.landmark[1]
    top   = face_lm.landmark[10]   # forehead top
    chin  = face_lm.landmark[152]
    l_ear = face_lm.landmark[234]
    r_ear = face_lm.landmark[454]

    # Yaw: nose offset from ear midpoint
    ear_mid_x = (l_ear.x + r_ear.x) / 2.0
    yaw = (nose.x - ear_mid_x) * 300.0

    # Pitch: nose vertical position relative to face centre
    face_h = max(abs(chin.y - top.y), 1e-5)
    mid_y  = top.y + face_h * 0.5
    pitch  = (nose.y - mid_y) / face_h * 180.0

    return yaw, pitch


def classify_head(yaw, pitch, neutral_yaw, neutral_pitch):
    """Map (yaw, pitch) relative to neutral baseline → action string."""
    dy = yaw   - neutral_yaw
    dp = pitch - neutral_pitch

    if   dy >  YAW_THRESH:  return "MOVE_LEFT"
    elif dy < -YAW_THRESH:  return "MOVE_RIGHT"
    elif dp <  PITCH_UP:    return "MOVE_UP"
    elif dp >  PITCH_DOWN:  return "MOVE_DOWN"
    return "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
#  ATTACK DETECTOR  (wrist velocity)
# ─────────────────────────────────────────────────────────────────────────────
class AttackDetector:
    def __init__(self):
        self.r_hist = deque(maxlen=4)   # right-hand wrist positions
        self.l_hist = deque(maxlen=4)   # left-hand wrist positions
        self.cd     = {}               # cooldown timestamps

    def _ok(self, act):
        return time.time() - self.cd.get(act, 0) >= COOLDOWN_SEC

    def _fire(self, act):
        self.cd[act] = time.time()
        return act

    def check(self, hand_results, half_w, half_h):
        """
        Returns the detected attack action or "NONE".
        hand_results : MediaPipe Hands result for one half-frame.
        half_w/h     : pixel dimensions of the half-frame.
        """
        if not hand_results or not hand_results.multi_hand_landmarks:
            return "NONE"

        for i, hlm in enumerate(hand_results.multi_hand_landmarks):
            # Determine handedness (Right / Left as seen in the frame)
            side = "Right"
            if hand_results.multi_handedness:
                side = hand_results.multi_handedness[i].classification[0].label

            wrist = hlm.landmark[0]
            pos   = (wrist.x * half_w, wrist.y * half_h)
            hist  = self.r_hist if side == "Right" else self.l_hist
            hist.append(pos)

            if len(hist) < 2:
                continue

            dx = abs(hist[-1][0] - hist[-2][0])
            dy = hist[-1][1] - hist[-2][1]   # positive = moving downward

            if side == "Right":
                # Fast horizontal swipe → Cross (kick)
                if dx > VEL_THRESH and self._ok("CROSS"):
                    return self._fire("CROSS")
                # Fast downward swipe → Circle
                if dy > VEL_THRESH and self._ok("CIRCLE"):
                    return self._fire("CIRCLE")
            else:
                # Fast horizontal swipe → Square (punch)
                if dx > VEL_THRESH and self._ok("SQUARE"):
                    return self._fire("SQUARE")
                # Fast downward swipe → Triangle
                if dy > VEL_THRESH and self._ok("TRIANGLE"):
                    return self._fire("TRIANGLE")

        return "NONE"


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
def calibrate(cap, fm_inst, duration=3):
    """
    Ask both players to hold neutral head position for `duration` seconds.
    Returns (neutral_yaw, neutral_pitch) averages from all detected faces.
    """
    yaws, pitches = [], []
    t0 = time.time()
    print(f"\n[CALIBRATE] Both players look straight ahead for {duration} seconds…\n")

    while time.time() - t0 < duration:
        ok, frame = cap.read()
        if not ok:
            continue
        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        mid    = fw // 2

        for xoff in (0, mid):
            crop = frame[:, xoff: xoff + mid]
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res  = fm_inst.process(rgb)
            if res.multi_face_landmarks:
                y, p = head_angles(res.multi_face_landmarks[0])
                yaws.append(y)
                pitches.append(p)

        rem = duration - int(time.time() - t0)
        cv2.putText(frame,
                    f"Look straight ahead — calibrating… {rem}s",
                    (fw // 8, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("Tekken 3 — CV Controller", frame)
        cv2.waitKey(1)

    ny  = float(np.mean(yaws))    if yaws    else 0.0
    np_ = float(np.mean(pitches)) if pitches else 0.0
    print(f"[CALIBRATE] Done  neutral_yaw={ny:.1f}  neutral_pitch={np_:.1f}\n")
    return ny, np_


# ─────────────────────────────────────────────────────────────────────────────
#  HUD DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
P1_COLOR = (180, 80, 255)    # purple  (BGR)
P2_COLOR = ( 60, 220, 100)   # green   (BGR)
FONT     = cv2.FONT_HERSHEY_SIMPLEX


def _text_size(text, scale=0.65, thickness=2):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


def draw_action_label(img, text, centre_x, top_y, color):
    """Draw a pill-shaped label with black background centred at centre_x."""
    if not text:
        return
    tw, th = _text_size(text, 0.62, 2)
    x  = max(2, centre_x - tw // 2)
    y  = max(th + 6, top_y)
    # Background
    cv2.rectangle(img,
                  (x - 6, y - th - 5),
                  (x + tw + 6, y + 6),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, 0.62, color, 2)


def draw_hud(frame, p1_info, p2_info, winner):
    """
    p1_info / p2_info = {
        "action"  : str,
        "face_box": (x1,y1,x2,y2) in half-frame coords, or None
    }
    """
    fh, fw = frame.shape[:2]
    mid    = fw // 2

    # Vertical dividing line
    cv2.line(frame, (mid, 0), (mid, fh), (180, 180, 180), 1)

    # Player name in top corner of each half
    cv2.putText(frame, "PLAYER 1", (8, 28),
                FONT, 0.75, P1_COLOR, 2)
    cv2.putText(frame, "PLAYER 2", (mid + 8, 28),
                FONT, 0.75, P2_COLOR, 2)

    for player, info, x_offset in ((1, p1_info, 0), (2, p2_info, mid)):
        color  = P1_COLOR if player == 1 else P2_COLOR
        action = info.get("action", "NEUTRAL")
        label  = LABEL_MAP.get(action, "")
        fb     = info.get("face_box")

        if fb:
            x1, y1, x2, y2 = fb
            # Draw face bounding box (shifted by half-frame offset)
            cv2.rectangle(frame,
                          (x_offset + x1, y1),
                          (x_offset + x2, y2),
                          color, 1)
            # Action label centred above the face box
            centre_x = x_offset + (x1 + x2) // 2
            draw_action_label(frame, label, centre_x, max(8, y1 - 16), color)
        else:
            # No face detected — show a hint
            draw_action_label(frame, "no face", x_offset + mid // 2, 55,
                              (80, 80, 80))

    # ── Winner Banner ─────────────────────────────────────────────────────────
    if winner:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, 70), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        color  = P1_COLOR if winner == 1 else P2_COLOR
        msg    = f"  PLAYER {winner} WINS !  "
        tw, th = _text_size(msg, 1.5, 3)
        cv2.putText(frame, msg,
                    (fw // 2 - tw // 2, 52),
                    FONT, 1.5, color, 3)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Use mediapipe standard API
    # Use new mediapipe API structure
    # Separate FaceMesh instances per half so tracking state is independent
    fm1 = mpf.FaceMesh(max_num_faces=1, refine_landmarks=True,
                            min_detection_confidence=0.60,
                            min_tracking_confidence=0.55)
    fm2 = mpf.FaceMesh(max_num_faces=1, refine_landmarks=True,
                            min_detection_confidence=0.60,
                            min_tracking_confidence=0.55)

    # Separate Hands instances per half
    hd1 = mph.Hands(max_num_hands=2,
                     min_detection_confidence=0.65,
                     min_tracking_confidence=0.55)
    hd2 = mph.Hands(max_num_hands=2,
                     min_detection_confidence=0.65,
                     min_tracking_confidence=0.55)

    # Open webcam (try index 0; change to 1 if wrong camera)
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  400)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  300)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera index.")
        return

    # Calibrate neutral head position (using fm1 temporarily for both halves)
    neutral_yaw, neutral_pitch = calibrate(cap, fm1, duration=3)

    atk1 = AttackDetector()
    atk2 = AttackDetector()

    winner    = None
    prev_p1   = "NEUTRAL"
    prev_p2   = "NEUTRAL"

    print("[RUNNING]  Q=quit | 1=P1 wins | 2=P2 wins | R=reset winner\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Empty frame — retrying…")
            continue

        frame     = cv2.flip(frame, 1)           # mirror for natural feel
        fh, fw    = frame.shape[:2]
        mid       = fw // 2
        half_w    = mid
        half_h    = fh

        p1_half  = frame[:, :mid].copy()
        p2_half  = frame[:, mid:].copy()

        # ──────────────────────────────────────────────────────────────────────
        #  Process Player 1  (left half)
        # ──────────────────────────────────────────────────────────────────────
        rgb1      = cv2.cvtColor(p1_half, cv2.COLOR_BGR2RGB)
        face_r1   = fm1.process(rgb1)
        hand_r1   = hd1.process(rgb1)

        p1_action = "NEUTRAL"
        p1_fb     = None

        if face_r1.multi_face_landmarks:
            lm = face_r1.multi_face_landmarks[0]
            yaw, pitch = head_angles(lm)
            p1_action  = classify_head(yaw, pitch, neutral_yaw, neutral_pitch)

            # Face bounding box (half-frame coords)
            xs = [l.x * half_w for l in lm.landmark]
            ys = [l.y * half_h for l in lm.landmark]
            p1_fb = (int(min(xs)), int(min(ys)),
                     int(max(xs)), int(max(ys)))

            # Face mesh contour drawing removed

        # Attack gestures override movement gestures
        atk = atk1.check(hand_r1, half_w, half_h)
        if atk != "NONE":
            p1_action = atk

        # Draw hand skeleton on half
        if hand_r1.multi_hand_landmarks:
            for hlm in hand_r1.multi_hand_landmarks:
                mpd.draw_landmarks(
                    p1_half, hlm, mph.HAND_CONNECTIONS,
                    mpd.DrawingSpec(color=P1_COLOR, thickness=2, circle_radius=4),
                    mpd.DrawingSpec(color=(160, 160, 160), thickness=1))

        # ──────────────────────────────────────────────────────────────────────
        #  Process Player 2  (right half)
        # ──────────────────────────────────────────────────────────────────────
        rgb2      = cv2.cvtColor(p2_half, cv2.COLOR_BGR2RGB)
        face_r2   = fm2.process(rgb2)
        hand_r2   = hd2.process(rgb2)

        p2_action = "NEUTRAL"
        p2_fb     = None

        if face_r2.multi_face_landmarks:
            lm = face_r2.multi_face_landmarks[0]
            yaw, pitch = head_angles(lm)
            p2_action  = classify_head(yaw, pitch, neutral_yaw, neutral_pitch)

            xs = [l.x * half_w for l in lm.landmark]
            ys = [l.y * half_h for l in lm.landmark]
            p2_fb = (int(min(xs)), int(min(ys)),
                     int(max(xs)), int(max(ys)))

            # Face mesh contour drawing removed

        atk = atk2.check(hand_r2, half_w, half_h)
        if atk != "NONE":
            p2_action = atk

        if hand_r2.multi_hand_landmarks:
            for hlm in hand_r2.multi_hand_landmarks:
                mpd.draw_landmarks(
                    p2_half, hlm, mph.HAND_CONNECTIONS,
                    mpd.DrawingSpec(color=P2_COLOR, thickness=2, circle_radius=4),
                    mpd.DrawingSpec(color=(160, 160, 160), thickness=1))

        # ──────────────────────────────────────────────────────────────────────
        #  Inject Keys into DuckStation  (only when game is running)
        # ──────────────────────────────────────────────────────────────────────
        if not winner:
            if p1_action != "NEUTRAL":
                inject(p1_action, 1)
            if p2_action != "NEUTRAL":
                inject(p2_action, 2)

        # ──────────────────────────────────────────────────────────────────────
        #  Merge processed halves back and draw HUD
        # ──────────────────────────────────────────────────────────────────────
        frame[:, :mid] = p1_half
        frame[:, mid:] = p2_half

        frame = draw_hud(
            frame,
            {"action": p1_action, "face_box": p1_fb},
            {"action": p2_action, "face_box": p2_fb},
            winner
        )

        # Use a constant window name to prevent opening multiple windows
        cv2.imshow("Tekken 3 — CV Controller", frame)

        # ──────────────────────────────────────────────────────────────────────
        #  Keyboard shortcuts for the CV window
        # ──────────────────────────────────────────────────────────────────────
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            print("[QUIT]")
            break
        elif k == ord('1'):
            winner = 1
            print("[INFO] Player 1 declared winner!")
        elif k == ord('2'):
            winner = 2
            print("[INFO] Player 2 declared winner!")
        elif k in (ord('r'), ord('R')):
            winner = None
            print("[INFO] Winner reset — game resumed.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    fm1.close(); fm2.close()
    hd1.close(); hd2.close()
    print("[DONE]")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()