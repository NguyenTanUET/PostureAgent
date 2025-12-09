import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import platform
import ctypes
import csv
import os
from datetime import datetime

# =========================
#      CONFIG THRESHOLDS
# =========================

# Cổ: baseline + NECK_OFFSET_DEG, nhưng không nhỏ hơn NECK_MIN_THRESH_DEG
NECK_OFFSET_DEG = 0.0
NECK_MIN_THRESH_DEG = 3.0

# Lưng
BACK_OFFSET_DEG = 7.0
BACK_MIN_THRESH_DEG = 10.0

# Vai
SHOULDER_OFFSET_DEG = 0.0
SHOULDER_MIN_THRESH_DEG = 3.0

# Lean-in (mặt to lên bao nhiêu lần so với baseline thì coi là ngồi sát)
LEAN_IN_RATIO_THRESH = 1.3

# Nhắc posture xấu & break
BAD_POSTURE_THRESHOLD_SEC = 60.0     # posture BAD liên tục > 60s
POPUP_COOLDOWN_SEC = 300.0          # ít nhất 5 phút mới nhắc lại posture xấu
BREAK_INTERVAL_SEC = 40.0         # 1 tiếng break 1 lần (test thì có thể set 60.0)

# File CSV lưu summary mỗi session
CSV_FILENAME = "posture_sessions.csv"

# MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# =========================
#      CONFIG VIDEO BREAK
# =========================

# Thư mục chứa video
VIDEO_DIR = "videos"

# Tên file video trong thư mục VIDEO_DIR (bạn tự tạo sẵn)
VIDEO_FILE_NECK      = "neck_exercise.mp4"        # cho bad_neck_time
VIDEO_FILE_BACK      = "back_exercise.mp4"        # cho bad_back_time
VIDEO_FILE_SHOULDER  = "shoulder_exercise.mp4"    # cho bad_shoulder_time
VIDEO_FILE_LEANIN    = "leanin_exercise.mp4"      # cho lean_in_time
VIDEO_FILE_FULL      = "full_break_exercise.mp4"  # dùng khi mọi thứ khá cân bằng / không có posture nổi trội

# =========================
#      HÀM HỖ TRỢ
# =========================

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    return np.degrees(theta_rad)


def get_point_norm(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=float)


def compute_face_scale(landmarks):
    """Khoảng cách giữa 2 tai (normalized) ~ kích thước mặt."""
    try:
        le = get_point_norm(landmarks, mp_pose.PoseLandmark.LEFT_EAR.value)
        re = get_point_norm(landmarks, mp_pose.PoseLandmark.RIGHT_EAR.value)
    except IndexError:
        return None
    return np.linalg.norm(le - re)


def check_camera_alignment(landmarks):
    """Check đơn giản xem người có ở giữa / xa / gần quá không."""
    try:
        le = get_point_norm(landmarks, mp_pose.PoseLandmark.LEFT_EAR.value)
        re = get_point_norm(landmarks, mp_pose.PoseLandmark.RIGHT_EAR.value)
    except IndexError:
        return "Khong thay mat ro", "warn"

    face_center = (le + re) / 2.0
    face_scale = np.linalg.norm(le - re)

    msg_list = []
    level = "ok"

    cx = face_center[0]
    if cx < 0.25:
        msg_list.append("Ban hoi lech TRAI khung hinh.")
        level = "warn"
    elif cx > 0.75:
        msg_list.append("Ban hoi lech PHAI khung hinh.")
        level = "warn"

    if face_scale is not None:
        if face_scale < 0.08:
            msg_list.append("Ban kha XA camera (nen ngoi gan hon).")
            level = "warn"
        elif face_scale > 0.25:
            msg_list.append("Ban RAT GAN camera (goc duoi len de sai so).")
            level = "warn"

    if not msg_list:
        return "Camera & vi tri OK. Giu tu the nay de calibration.", "ok"
    else:
        return " ".join(msg_list), level


def compute_neck_angle(landmarks):
    """Góc cổ: vector từ midpoint 2 vai -> mũi so với phương thẳng đứng."""
    try:
        ls = get_point_norm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        rs = get_point_norm(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        nose = get_point_norm(landmarks, mp_pose.PoseLandmark.NOSE.value)
    except IndexError:
        return None

    mid_shoulder = (ls + rs) / 2.0
    v_neck = nose - mid_shoulder
    vertical = np.array([0.0, -1.0])
    return angle_between(v_neck, vertical)


def compute_back_angle(landmarks):
    """Góc lưng: midpoint hông -> midpoint vai so với phương thẳng đứng."""
    try:
        ls = get_point_norm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        rs = get_point_norm(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        lh = get_point_norm(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
        rh = get_point_norm(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
    except IndexError:
        return None

    mid_shoulder = (ls + rs) / 2.0
    mid_hip = (lh + rh) / 2.0
    v_back = mid_shoulder - mid_hip
    vertical = np.array([0.0, -1.0])
    return angle_between(v_back, vertical)


def compute_shoulder_tilt(landmarks):
    """
    Lệch vai: độ nghiêng 0..90° so với phương ngang.
    0°  = vai ngang; càng lớn càng lệch.
    """
    try:
        ls = get_point_norm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        rs = get_point_norm(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    except IndexError:
        return None, None

    dx = rs[0] - ls[0]
    dy = rs[1] - ls[1]

    angle_raw = np.degrees(np.arctan2(dy, dx))
    a = abs(angle_raw)
    if a > 90:
        a = 180 - a  # 180° coi như 0° (ngang nhưng ngược chiều)

    direction = None
    if abs(dy) > 0.005:
        direction = "right_down" if dy > 0 else "left_down"

    return a, direction


def classify_posture(neck_angle, back_angle, neck_thresh, back_thresh):
    bad_neck = False
    bad_back = False

    if neck_angle is not None and neck_thresh is not None:
        bad_neck = neck_angle > neck_thresh
    if back_angle is not None and back_thresh is not None:
        bad_back = back_angle > back_thresh

    if bad_neck and bad_back:
        return "BAD: Co + Lung"
    elif bad_neck:
        return "BAD: Co (gap co)"
    elif bad_back:
        return "BAD: Lung (khom lung)"
    else:
        return "GOOD"


def show_posture_popup():
    """Popup posture xấu > 60s."""
    def _worker():
        if platform.system() == "Windows":
            ctypes.windll.user32.MessageBoxW(
                0,
                "Ban da ngoi sai tu the hon 60 giay.\n"
                "Hay dieu chinh lai co, lung, vai va tranh ngoi sat man hinh.",
                "PostureAgent - Nhac nho tu the",
                0x40 | 0x0
            )
        else:
            print("[PostureAgent] Nhac nho: posture xau > 60s.")
    threading.Thread(target=_worker, daemon=True).start()

def ask_break_confirmation():
    """
    Hỏi người dùng: đã đến giờ break, có muốn nghỉ và tập bài tập không?
    Trả về True nếu user chọn OK, False nếu Cancel/No.
    """
    if platform.system() == "Windows":
        MB_OKCANCEL = 0x00000001
        MB_ICONQUESTION = 0x00000020
        res = ctypes.windll.user32.MessageBoxW(
            0,
            "Ban da lam viec lien tuc ~1 gio.\n"
            "Ban co muon nghi va tap bai tap chinh tu the khong?\n\n"
            "OK = bat dau xem video bai tap\n"
            "Cancel = tiep tuc lam viec",
            "PostureAgent - Nhac nghi giai lao",
            MB_OKCANCEL | MB_ICONQUESTION
        )
        # IDOK = 1, IDCANCEL = 2
        return res == 1
    else:
        # fallback cho he dieu hanh khac
        ans = input("Da het 1 gio. Nghi ngoi va xem bai tap? (y/N): ")
        return ans.strip().lower().startswith("y")

def show_break_popup():
    """Popup nhắc nghỉ sau 1 tiếng."""
    def _worker():
        if platform.system() == "Windows":
            ctypes.windll.user32.MessageBoxW(
                0,
                "Ban da lam viec lien tuc ~1 gio.\n"
                "Hay dung len, di lai va nghi ngoi 3-5 phut.",
                "PostureAgent - Break reminder",
                0x40 | 0x0
            )
        else:
            print("[PostureAgent] Nhac nho: da 1 gio lam viec, hay nghi giai lao.")
    threading.Thread(target=_worker, daemon=True).start()


def append_session_csv(filename, row_dict):
    """Ghi 1 dòng summary session vào CSV (tự tạo file + header nếu chưa có)."""
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "session_id",
        "start_time",
        "end_time",
        "device_type",
        "total_runtime",
        "active_sitting_time",
        "good_posture_time",
        "bad_posture_time",
        "bad_neck_time",
        "bad_back_time",
        "bad_shoulder_time",
        "lean_in_time",
    ]
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def choose_break_video(bad_neck_time, bad_back_time, bad_shoulder_time, lean_in_time):
    """
    Chọn video break dựa trên posture xấu nào chiếm nhiều thời gian nhất.
    Trả về (posture_key, video_path).
    """
    # Map thời gian theo loại posture
    scores = {
        "neck": bad_neck_time,
        "back": bad_back_time,
        "shoulder": bad_shoulder_time,
        "leanin": lean_in_time,
    }

    # Xem loại nào lớn nhất
    max_key = max(scores, key=scores.get)
    max_val = scores[max_key]

    # Nếu tất cả ~0 (chưa log được gì mấy) thì dùng full break
    if max_val <= 0:
        video_file = VIDEO_FILE_FULL
        posture_key = "full"
    else:
        if max_key == "neck":
            video_file = VIDEO_FILE_NECK
        elif max_key == "back":
            video_file = VIDEO_FILE_BACK
        elif max_key == "shoulder":
            video_file = VIDEO_FILE_SHOULDER
        else:  # "leanin"
            video_file = VIDEO_FILE_LEANIN
        posture_key = max_key

    video_path = os.path.join(VIDEO_DIR, video_file)
    return posture_key, video_path

def play_exercise_video(video_path):
    """
    Play video bài tập trong một cửa sổ OpenCV riêng.
    Nhấn 'q', 'b' hoặc ESC để dừng video.
    Đóng cửa sổ (click X) cũng dừng video luôn.
    """
    if not os.path.exists(video_path):
        print("[Break] Video khong ton tai:", video_path)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[Break] Khong mo duoc video:", video_path)
        return

    window_name = "PostureAgent - Break Exercise"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    print("[Break] Dang play:", video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            # Het video
            break

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF

        # Neu user dong cua so (click X)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # User nhan 'q', 'b' hoac ESC de thoat som
        if key in (ord('q'), ord('b'), 27):
            break

    cap.release()
    cv2.destroyWindow(window_name)
    print("[Break] Ket thuc video break.")


# =========================
#           MAIN
# =========================

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Khong mo duoc webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "PostureAgent - Laptop/PC Calibration + Monitoring"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    MODE = "DEVICE_SELECT"   # DEVICE_SELECT -> CALIBRATION -> RUNNING
    DEVICE_TYPE = None

    baseline_ready = False
    collecting_baseline = False
    baseline_start_time = 0.0
    CALIB_DURATION = 3.0
    baseline_samples = []

    pitch0 = torso0 = face_scale0 = None
    neck_thresh = None
    back_thresh = None
    shoulder_tilt0 = None
    shoulder_tilt_thresh = None

    # Theo dõi posture xấu (để popup)
    bad_posture_start_time = None
    last_popup_time = None

    # ===== Thông tin session & các timer =====
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_start_time = time.time()
    start_time_str = datetime.fromtimestamp(session_start_time).isoformat(timespec="seconds")

    active_sitting_time = 0.0
    good_posture_time = 0.0
    bad_posture_time = 0.0
    bad_neck_time = 0.0
    bad_back_time = 0.0
    bad_shoulder_time = 0.0
    lean_in_time = 0.0

    last_frame_time = session_start_time

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Khong doc duoc frame")
                    break

                # dt giữa 2 frame
                now = time.time()
                dt = now - last_frame_time
                last_frame_time = now

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Pose estimation
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True

                has_pose = results.pose_landmarks is not None

                neck_angle = None
                back_angle = None
                shoulder_tilt_angle = None
                shoulder_tilt_dir = None
                face_scale = None
                camera_msg = ""
                camera_level = "warn"
                hips_visible = False
                lean_in_warning = False

                if has_pose:
                    landmarks = results.pose_landmarks.landmark

                    # Hông có thấy rõ không
                    try:
                        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                        if lh.visibility > 0.5 and rh.visibility > 0.5:
                            hips_visible = True
                    except IndexError:
                        hips_visible = False

                    neck_angle = compute_neck_angle(landmarks)
                    if hips_visible:
                        back_angle = compute_back_angle(landmarks)

                    shoulder_tilt_angle, shoulder_tilt_dir = compute_shoulder_tilt(landmarks)
                    face_scale = compute_face_scale(landmarks)
                    camera_msg, camera_level = check_camera_alignment(landmarks)

                    # Lean-in
                    if baseline_ready and face_scale is not None and face_scale0 is not None:
                        ratio = face_scale / face_scale0
                        if ratio > LEAN_IN_RATIO_THRESH:
                            lean_in_warning = True

                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                    # Tổng thời gian "ngồi trước máy"
                    active_sitting_time += dt
                else:
                    camera_msg = "Khong phat hien nguoi / pose."
                    camera_level = "warn"

                # ========= DEVICE SELECT =========
                if MODE == "DEVICE_SELECT":
                    cv2.rectangle(frame, (10, 10), (w - 10, 160), (0, 0, 0), -1)
                    cv2.putText(frame, "CHON THIET BI:", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, "1 - Laptop (camera tren man hinh laptop)", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, "2 - PC / Man hinh ngoai / Webcam roi", (20, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, "Nhan phim 1 hoac 2 de tiep tuc.", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # ========= CALIBRATION =========
                elif MODE == "CALIBRATION":
                    cv2.putText(frame, "CALIBRATION MODE", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    if DEVICE_TYPE == "laptop":
                        guide1 = "Dat laptop sao cho camera gan tam mat."
                        guide2 = "Ngoi tu the thoai mai, nhin thang man hinh."
                    else:
                        guide1 = "Dat webcam ngang tam mat, chinh dien voi man hinh."
                        guide2 = "Ngoi thang lung, nhin thang man hinh."

                    cv2.putText(frame, guide1, (20, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, guide2, (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, "Nhan 'C' de bat dau calibration (~3s).", (20, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

                    if camera_msg:
                        color = (0, 255, 0) if camera_level == "ok" else (0, 0, 255)
                        cv2.putText(frame, camera_msg, (20, 135),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if collecting_baseline:
                        elapsed = time.time() - baseline_start_time
                        remain = max(0.0, CALIB_DURATION - elapsed)
                        cv2.putText(frame, f"Thu baseline... {remain:.1f}s", (20, 165),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        if (neck_angle is not None and
                            face_scale is not None and
                            shoulder_tilt_angle is not None):

                            baseline_samples.append({
                                "neck": neck_angle,
                                "back": back_angle,
                                "face_scale": face_scale,
                                "shoulder_tilt": shoulder_tilt_angle
                            })

                        if elapsed >= CALIB_DURATION and len(baseline_samples) > 10:
                            neck_vals = [s["neck"] for s in baseline_samples]
                            face_vals = [s["face_scale"] for s in baseline_samples]
                            shoulder_vals = [s["shoulder_tilt"] for s in baseline_samples]
                            back_vals = [s["back"] for s in baseline_samples if s["back"] is not None]

                            pitch0 = float(np.mean(neck_vals)) if neck_vals else None
                            face_scale0 = float(np.mean(face_vals)) if face_vals else None
                            shoulder_tilt0 = float(np.mean(shoulder_vals)) if shoulder_vals else None

                            if pitch0 is not None:
                                neck_thresh = max(pitch0 + NECK_OFFSET_DEG, NECK_MIN_THRESH_DEG)
                            else:
                                neck_thresh = NECK_MIN_THRESH_DEG

                            if back_vals:
                                torso0 = float(np.mean(back_vals))
                                back_thresh = max(torso0 + BACK_OFFSET_DEG, BACK_MIN_THRESH_DEG)
                            else:
                                torso0 = None
                                back_thresh = None
                                print("Warning: Calibration khong thay ro hong, se danh gia theo co la chinh.")

                            if shoulder_tilt0 is not None:
                                shoulder_tilt_thresh = max(shoulder_tilt0 + SHOULDER_OFFSET_DEG, SHOULDER_MIN_THRESH_DEG)
                            else:
                                shoulder_tilt_thresh = SHOULDER_MIN_THRESH_DEG + 3.0

                            baseline_ready = True
                            collecting_baseline = False
                            MODE = "RUNNING"

                            print("=== Calibration xong ===")
                            print(f"Device: {DEVICE_TYPE}")
                            if pitch0 is not None:
                                print(f"pitch0 (neck): {pitch0:.2f} deg")
                                print(f"neck_thresh: {neck_thresh:.2f}")
                            if torso0 is not None:
                                print(f"torso0 (back): {torso0:.2f} deg")
                                print(f"back_thresh: {back_thresh:.2f}")
                            print(f"shoulder_tilt0: {shoulder_tilt0:.2f} deg")
                            print(f"shoulder_tilt_thresh: {shoulder_tilt_thresh:.2f}")
                            print(f"face_scale0: {face_scale0:.4f}")

                # ========= RUNNING =========
                elif MODE == "RUNNING":
                    cv2.putText(frame, f"RUNNING MODE ({DEVICE_TYPE.upper()})", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    if baseline_ready:
                        posture_text = classify_posture(
                            neck_angle, back_angle, neck_thresh, back_thresh
                        )
                    else:
                        posture_text = "No baseline"

                    shoulder_bad = False
                    if (baseline_ready and
                        shoulder_tilt_angle is not None and
                        shoulder_tilt_thresh is not None and
                        shoulder_tilt_angle > shoulder_tilt_thresh):
                        shoulder_bad = True

                    if shoulder_bad:
                        if posture_text.startswith("BAD"):
                            posture_text = posture_text + " + Vai lech"
                        else:
                            posture_text = "BAD: Vai lech"

                    # ---- Cập nhật các timer good/bad/neck/back/shoulder/lean_in ----
                    if has_pose and baseline_ready:
                        is_bad_posture = posture_text.startswith("BAD")
                        is_good_posture = not is_bad_posture

                        if is_good_posture:
                            good_posture_time += dt
                        else:
                            bad_posture_time += dt

                        if neck_angle is not None and neck_thresh is not None and neck_angle > neck_thresh:
                            bad_neck_time += dt
                        if back_angle is not None and back_thresh is not None and back_angle > back_thresh:
                            bad_back_time += dt
                        if shoulder_bad:
                            bad_shoulder_time += dt
                        if lean_in_warning:
                            lean_in_time += dt

                    # ---- Popup posture xấu > 60s ----
                    now2 = time.time()
                    is_bad_posture_for_popup = baseline_ready and posture_text.startswith("BAD")
                    if is_bad_posture_for_popup:
                        if bad_posture_start_time is None:
                            bad_posture_start_time = now2
                        else:
                            bad_duration = now2 - bad_posture_start_time
                            cooldown_ok = (last_popup_time is None) or (now2 - last_popup_time > POPUP_COOLDOWN_SEC)
                            if bad_duration >= BAD_POSTURE_THRESHOLD_SEC and cooldown_ok:
                                show_posture_popup()
                                last_popup_time = now2
                    else:
                        bad_posture_start_time = None

                    # ---- Break sau 1 giờ ngồi: play video bai tap ----
                    if baseline_ready and active_sitting_time >= BREAK_INTERVAL_SEC:
                        # Reset trước để tránh việc người dùng tắt video sớm rồi bị bật lại ngay
                        active_sitting_time = 0.0

                        # Hoi user co muon break + tap bai tap khong
                        want_break = ask_break_confirmation()

                        if want_break:
                            # Chon video tuong ung posture xau nhieu nhat
                            posture_key, video_path = choose_break_video(
                                bad_neck_time,
                                bad_back_time,
                                bad_shoulder_time,
                                lean_in_time
                            )
                            print(f"[Break] Chon video cho posture '{posture_key}': {video_path}")

                            # Play video break (user xem/hoac tat -> ham nay ket thuc)
                            play_exercise_video(video_path)

                    # Hộp info
                    cv2.rectangle(frame, (10, 50), (460, 230), (0, 0, 0), -1)
                    overlay = frame[50:230, 10:460]
                    cv2.addWeighted(overlay, 0.3, overlay, 0, 0, overlay)

                    color = (0, 255, 0) if not posture_text.startswith("BAD") else (0, 0, 255)
                    cv2.putText(frame, f"Posture: {posture_text}", (20, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Cổ
                    if neck_angle is not None and neck_thresh is not None:
                        co_status = "OK" if neck_angle <= neck_thresh else "GAP CO NHIEU"
                        cv2.putText(
                            frame,
                            f"Co: {neck_angle:.1f} deg (thr {neck_thresh:.1f}) [{co_status}]",
                            (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                        )

                    # Lưng
                    if back_angle is not None and back_thresh is not None:
                        lung_status = "OK" if back_angle <= back_thresh else "KHOM LUNG"
                        cv2.putText(
                            frame,
                            f"Lung: {back_angle:.1f} deg (thr {back_thresh:.1f}) [{lung_status}]",
                            (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                        )
                    elif back_angle is None:
                        cv2.putText(
                            frame,
                            "Lung: N/A (khong thay ro hong)",
                            (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
                        )

                    # Vai
                    if shoulder_tilt_angle is not None and shoulder_tilt_thresh is not None:
                        vai_status = "OK" if shoulder_tilt_angle <= shoulder_tilt_thresh else "VAI LECH"
                        dir_text = ""
                        if shoulder_tilt_angle > shoulder_tilt_thresh and shoulder_tilt_dir is not None:
                            if shoulder_tilt_dir == "right_down":
                                dir_text = " (vai TRAI thap hon)"
                            elif shoulder_tilt_dir == "left_down":
                                dir_text = " (vai PHAI thap hon)"
                        cv2.putText(
                            frame,
                            f"Vai: {shoulder_tilt_angle:.1f} deg (thr {shoulder_tilt_thresh:.1f}) [{vai_status}]{dir_text}",
                            (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                        )

                    # Lean-in
                    if lean_in_warning:
                        cv2.putText(
                            frame,
                            "Warning: Dang ngoi RAT GAN man hinh (lean-in).",
                            (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                        )

                # ===== HIỂN THỊ & BẮT PHÍM =====
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if MODE == "DEVICE_SELECT":
                    if key == ord("1"):
                        DEVICE_TYPE = "laptop"
                        MODE = "CALIBRATION"
                        print("Chon thiet bi: Laptop")
                    elif key == ord("2"):
                        DEVICE_TYPE = "pc"
                        MODE = "CALIBRATION"
                        print("Chon thiet bi: PC / Webcam roi")

                elif MODE == "CALIBRATION" and not collecting_baseline:
                    if key in (ord("c"), ord("C")):
                        baseline_samples = []
                        baseline_start_time = time.time()
                        collecting_baseline = True
                        print("Bat dau calibration... Giu dung tu the trong 3 giay.")

    except KeyboardInterrupt:
        print("Da dung chuong trinh.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        session_end_time = time.time()
        end_time_str = datetime.fromtimestamp(session_end_time).isoformat(timespec="seconds")
        total_runtime = session_end_time - session_start_time

        row = {
            "session_id": session_id,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "device_type": DEVICE_TYPE or "unknown",
            "total_runtime": round(total_runtime, 1),
            "active_sitting_time": round(active_sitting_time, 1),
            "good_posture_time": round(good_posture_time, 1),
            "bad_posture_time": round(bad_posture_time, 1),
            "bad_neck_time": round(bad_neck_time, 1),
            "bad_back_time": round(bad_back_time, 1),
            "bad_shoulder_time": round(bad_shoulder_time, 1),
            "lean_in_time": round(lean_in_time, 1),
        }

        try:
            append_session_csv(CSV_FILENAME, row)
            print("Da luu bao cao session vao", CSV_FILENAME)
            print(row)
        except Exception as e:
            print("Khong luu duoc CSV:", e)


if __name__ == "__main__":
    main()
