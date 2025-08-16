#之前效果最好的是maxlen = 50

from hopenet import hopenet
import time
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
from pykalman import KalmanFilter
import mediapipe as mp
import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.signal import kaiserord, lfilter, firwin, freqz, lfilter_zi
from typing import Optional, Union
from scipy.spatial.transform import Rotation as R

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
KEY_IDX = [33, 263, 1, 61, 291]

yawlist = []
pitchlist = []
rolllist = []

timing = []
class HeadPoseFilter:
    def __init__(self, max_speed_deg_s=15, fps=30, z_thresh=3):
        self.max_step = max_speed_deg_s / fps  # 每帧最大变化量
        self.z_thresh = z_thresh

    def remove_outliers(self, data):
        """用3σ原则检测跳变并用邻帧均值替代"""
        arr = np.array(data)
        diffs = np.diff(arr)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        filtered = arr.copy()
        for i in range(1, len(arr)-1):
            if abs(diffs[i-1] - mean_diff) > self.z_thresh * std_diff:
                filtered[i] = (filtered[i-1] + filtered[i+1]) / 2
        return filtered.tolist()

    def clamp_speed(self, data):
        """限制最大角速度"""
        filtered = [data[0]]
        for i in range(1, len(data)):
            delta = data[i] - filtered[-1]
            if abs(delta) > self.max_step:
                filtered.append(filtered[-1] + self.max_step * (1 if delta > 0 else -1))
            else:
                filtered.append(data[i])
        return filtered

    def kalman_smooth(self, data):
        """Kalman 滤波"""
        kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1)
        smoothed, _ = kf.smooth(np.expand_dims(data, axis=1))
        return smoothed.ravel().tolist()

    def process(self, yaw, pitch, roll):
        """对三轴姿态统一滤波"""
        result = {}
        for axis_name, axis_data in zip(["yaw", "pitch", "roll"], [yaw, pitch, roll]):
            step1 = self.remove_outliers(axis_data)
            step2 = self.clamp_speed(step1)
            step3 = self.kalman_smooth(step2)
            result[axis_name] = step3
        return result

class weifilter:
    def __init__(self, yawdeq, pitchdeq, rolldeq, dict, ystore, pstore, rstore):
        self.yawdeq = yawdeq
        self.pitchdeq = pitchdeq
        self.rolldeq = rolldeq
        self.dict = dict
        self.store = {
            'yaw':ystore,
            'pitch':pstore,
            'roll':rstore
        }

        self.thres = {
            'yaw': 3.4,
            'pitch':3.0,
            'roll':2.0
        }

    def get_weight(self, type, length):
        if type == 1:
            weights = np.array([1 for _ in range(length)])
            weights = weights/weights.sum()
        
        elif type == 2:
            base = 1.15
            weights = np.array([base ** i for i in range(length)])
            weights = weights / weights.sum()

        else:
            weights = None

        return weights
    
    def predict(self, id, deq, length):
        if self.dict[id][0] == 0:
            self.dict[id][1] = length

            pre = np.asarray(deq[:self.dict[id][1]], dtype=float)
            N = len(pre)
            x_pre = np.arange(N, dtype=float)

            p_pre = np.polyfit(x_pre, pre, 2)
            result = np.polyval(p_pre, N)
            self.dict[id][0] += 1
            return result, deq
        
        elif self.dict[id][0] <= 5:

            pre = np.asarray(deq[:(self.dict[id][1]+self.dict[id][0])], dtype=float)
            N = len(pre)
            x_pre = np.arange(N, dtype=float)

            p_pre = np.polyfit(x_pre, pre, 2)
            result = np.polyval(p_pre, N)
            self.dict[id][0] += 1
            return result, deq
            
        else:
            predict_std = np.std(np.asarray(deq[self.dict[id][1]: length]))
            actual_std = np.std(np.asarray(self.store[id]))

            if predict_std > actual_std and actual_std < self.thres[id]:
                deq[self.dict[id][1]:self.dict[id][1]+self.dict[id][0]] = self.store[id]

            #Initialization
            self.dict[id][0] = -1
            self.dict[id][1] = 0
            self.store[id] = []
            return deq[-1], deq
        
    def process(self, id, deq, thres1, thres2):
        length = len(deq)
        if length < 2:
            return deq[-1], deq
        
        delta = abs(deq[-1] - deq[-2])
        # if delta < thres2 and self.dict[id][0] == -1:
        #     alpha = min(1.0, max(0.0, (delta/(thres2-thres1))))
        #     w = (1-alpha) * self.get_weight(1, length) + alpha * self.get_weight(2, length)
        # else:
        #     w = self.get_weight(0, length)
        
        alpha = min(1.0, max(0.0, (delta/(thres2-thres1))))

        w = (1- alpha) * self.get_weight(1, length) + alpha * self.get_weight(2, length)

        if w is None:
            self.dict[id][0] += 1
            self.store[id].append(deq[-1])
            deq = np.array(deq)
            angle, deq = self.predict(id,deq,length)
            deq = deque(deq, maxlen=30)
        else:
            angle = np.array(deq) @ w
        
        return angle, deq

    def get_angle(self):
        thres1_yawpitch = 1.2
        thres1_roll = 1
        thres2_yawpitch = 4
        thres2_roll = 3

        yaw, self.yawdeq = self.process("yaw",self.yawdeq, thres1_yawpitch, thres2_yawpitch)
        pitch, self.pitchdeq = self.process("pitch",self.pitchdeq, thres1_yawpitch, thres2_yawpitch)
        roll, self.rolldeq = self.process("roll", self.rolldeq, thres1_roll, thres2_roll)

        return yaw, pitch, roll, self.yawdeq, self.pitchdeq, self.rolldeq

class AttitudeFilter:
    def __init__(self, fps, cutoff_hz=3.0, width=5.0, ripple_db=60.0):
        """
        优化的姿态角滤波器
        
        Args:
            fps: 采样频率
            cutoff_hz: 截止频率，默认3Hz
            width: 过渡带宽度，默认5Hz
            ripple_db: 阻带衰减，默认60dB
        """
        self.sample_rate = fps
        self.cutoff_hz = cutoff_hz
        self.width = width
        self.ripple_db = ripple_db
        
        # 计算滤波器系数
        nyq_rate = self.sample_rate / 2.0
        wid = self.width / nyq_rate
        N, beta = kaiserord(self.ripple_db, wid)
        
        if N % 2 == 0:
            N += 1
            
        self.N = N
        self.taps = firwin(N, self.cutoff_hz / nyq_rate, window=('kaiser', beta))
        
        pitch_cutoff = self.cutoff_hz * 0.5  # 缩小截止频率到1.5Hz
        self.pitch_taps = firwin(N, pitch_cutoff / nyq_rate, window=('kaiser', beta))
        
        roll_cutoff = self.cutoff_hz * 0.1
        self.roll_taps = firwin(N, roll_cutoff / nyq_rate, window=('kaiser', beta))
        self.reset()
    
    def reset(self):
        """重置滤波器状态"""
        # 为yaw, pitch, roll三个轴分别维护滤波器状态
        self.zi_yaw = lfilter_zi(self.taps, 1.0) * 0
        self.zi_pitch = lfilter_zi(self.pitch_taps, 1.0) * 0  
        self.zi_roll = lfilter_zi(self.roll_taps,1.0) * 0

        # 用于角度连续性处理
        self.last_yaw = None
        self.last_pitch = None
        self.last_roll = None
        
        # 数据缓冲区（用于批量处理）
        self.buffer_size = max(self.N * 2, 50)
        self.yaw_buffer = deque(maxlen=self.buffer_size)
        self.pitch_buffer = deque(maxlen=self.buffer_size)
        self.roll_buffer = deque(maxlen=self.buffer_size)

    def _wrap_angle(self, angle, last_angle):
        """处理角度的连续性问题，避免-180到180跳跃"""
        if last_angle is None:
            return angle
            
        diff = angle - last_angle
        if diff > 180:
            angle -= 360
        elif diff < -180:
            angle += 360
            
        return angle
    
    def update_single(self, yaw, pitch, roll):
        """
        单点更新模式（实时处理）
        
        Args:
            yaw, pitch, roll: 当前的姿态角（度）
            
        Returns:
            filtered_yaw, filtered_pitch, filtered_roll: 滤波后的姿态角
        """
        # 处理角度连续性
        yaw = self._wrap_angle(yaw, self.last_yaw)
        pitch = self._wrap_angle(pitch, self.last_pitch)
        roll = self._wrap_angle(roll, self.last_roll)

        # 使用lfilter的状态保持功能
        filtered_yaw, self.zi_yaw = lfilter(self.taps, 1.0, [yaw], zi=self.zi_yaw)
        filtered_pitch, self.zi_pitch = lfilter(self.taps, 1.0, [pitch], zi=self.zi_pitch)
        filtered_roll, self.zi_roll = lfilter(self.taps, 1.0, [roll], zi=self.zi_roll)

        self.last_yaw = yaw
        self.last_pitch = pitch
        self.last_roll = roll

        return filtered_yaw[0], filtered_pitch[0], filtered_roll[0]
    
    
    def _make_continuous(self, angles):
        """将角度数组转换为连续的（处理跳跃）"""
        if len(angles) == 0:
            return angles
            
        continuous = np.array(angles, dtype=float)
        for i in range(1, len(continuous)):
            diff = continuous[i] - continuous[i-1]
            if diff > 180:
                continuous[i:] -= 360
            elif diff < -180:
                continuous[i:] += 360
                
        return continuous
    
    def get_filter_info(self):
        """返回滤波器参数信息"""
        return {
            'sample_rate': self.sample_rate,
            'cutoff_hz': self.cutoff_hz,
            'filter_length': self.N,
            'group_delay_samples': (self.N - 1) // 2,
            'group_delay_ms': ((self.N - 1) // 2) * 1000 / self.sample_rate
        }
    

def get_face_bbox_and_center(pts, margin=30, frame_shape=None):
    """
    根据关键点计算人脸框和中心
    pts: np.array [[x1,y1], [x2,y2], ...] 像素坐标
    margin: 在四周扩展的像素
    frame_shape: (H, W)，用于防止越界
    """
    x_min = int(np.min(pts[:, 0])) - margin
    x_max = int(np.max(pts[:, 0])) + margin
    y_min = int(np.min(pts[:, 1])) - margin
    y_max = int(np.max(pts[:, 1])) + margin

    # 防止越界
    if frame_shape is not None:
        h, w = frame_shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w - 1, x_max)
        y_max = min(h - 1, y_max)

    # 人脸中心
    tdx = int((x_min + x_max) / 2)
    tdy = int((y_min + y_max) / 2)

    return tdx, tdy, x_min, y_min, x_max, y_max

def get_aligned_face(frame, desired_size=224, margin=30):
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None, None, None

    landmarks = results.multi_face_landmarks[0]
    pts = []
    for idx in KEY_IDX:
        lm = landmarks.landmark[idx]
        pts.append([lm.x * w, lm.y * h])
    pts = np.array(pts, dtype=np.float32)

    # 计算人脸框 & 中心
    tdx, tdy, x_min, y_min, x_max, y_max = get_face_bbox_and_center(
        pts, margin=margin, frame_shape=frame.shape
    )

    # 定义对齐模板
    target_pts = np.array([
        [0.3 * desired_size, 0.35 * desired_size],   # 左眼
        [0.7 * desired_size, 0.35 * desired_size],   # 右眼
        [0.5 * desired_size, 0.50 * desired_size],   # 鼻尖
        [0.35 * desired_size, 0.75 * desired_size],  # 嘴左
        [0.65 * desired_size, 0.75 * desired_size],  # 嘴右
    ], dtype=np.float32)

    # 仿射变换
    M, _ = cv2.estimateAffinePartial2D(pts, target_pts)
    aligned_face = cv2.warpAffine(frame, M, (desired_size, desired_size), flags=cv2.INTER_LINEAR)

    return aligned_face, (tdx, tdy, x_min, y_min, x_max, y_max), pts
  
def draw_axis(img, yaw, pitch, roll, tdx, tdy, size):
    pitch = pitch * np.pi / 180
    yaw = yaw * np.pi / -180
    roll = roll * np.pi / 180



    # 3D坐标轴方向（以头部为中心）
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    # 绘制X轴（红）
    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
    # 绘制Y轴（绿）
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
    # 绘制Z轴（蓝）
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img

def visualize(yawlist, pitchlist, rolllist):
    yaw_frames = list(range(len(yawlist)))
    pitch_frames = list(range(len(pitchlist)))
    roll_frames = list(range(len(rolllist)))

    # 设置中文字体（如果需要显示中文）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建包含3个子图的图表 - 垂直排列
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('三轴角度数据可视化', fontsize=16, fontweight='bold')

    # 第一个子图：Yaw
    axes[0].plot(yaw_frames, yawlist,  linewidth=2, markersize=6, color='#e74c3c')
    axes[0].set_title('Yaw 角度', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Yaw (度)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=10)

    # 第二个子图：Pitch
    axes[1].plot(pitch_frames, pitchlist,  linewidth=2, markersize=6, color='#2ecc71')
    axes[1].set_title('Pitch 角度', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Pitch (度)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=10)

    # 第三个子图：Roll
    axes[2].plot(roll_frames, rolllist, linewidth=2, markersize=6, color="#2e8fd0")
    axes[2].set_title('Roll 角度', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Roll (度)', fontsize=12)
    axes[2].set_xlabel('帧数 (Frame Number)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='both', which='major', labelsize=10)

    # 调整子图间距
    plt.tight_layout()

    # 显示图表
    #plt.show()
    plt.savefig('output/angle_processed.png')

def mediap(frame):
    global yawlist, pitchlist, rolllist
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False

    results = face_mesh.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])

                    face_3d.append([x, y, lm.z])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                      [0, focal_length, img_h / 2],
                                      [0, 0, 1]], dtype=np.float64)
            
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rotation_vector)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            yawlist.append(x)
            pitchlist.append(y)
            rolllist.append(z)

            nose_3d_projection , jacobian = cv2.projectPoints(nose_3d, rotation_vector, translation_vector, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0]+y*10), int(nose_2d[1]-x*10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)
    
    end = time.time()
    return image

def ddraw_virtual_screen(
    frame, yaw, pitch, roll, face_center, face_size,
    screen_image_path: Optional[str] = None,
    screen_image: Optional[np.ndarray] = None,
    screen_distance_mul=5,
    screen_w_mul=4.0,
    screen_h_mul=2.0,
    focal_length=None,
    alpha=0.8,
    border_color=(0, 200, 255),
    border_thickness=0,
    show_center_marker=False
):

    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    if focal_length is None:
        focal_length = max(w, h)

    tdx, tdy = face_center

    # 深度估算（可根据标定调整）
    Z_face = (focal_length * 300.0) / face_size  # Reversed for correct scaling; adjust 300.0 as needed

    x_cam = (tdx - cx) / focal_length * Z_face
    y_cam = (tdy - cy) / focal_length * Z_face
    face_center_cam = np.array([x_cam, y_cam, Z_face], dtype=np.float32)

    # -------------------- 把像素单位转换为相机坐标系的世界单位 --------------------
    screen_distance_pixels = screen_distance_mul * face_size

    screen_w_pixels = max(40.0, screen_w_mul * face_size)
    screen_h_pixels = max(40.0, screen_h_mul * face_size)

    # 将像素距离转换为相机坐标系长度： ΔX = (Δx_pixels / f) * Z
    # 使用 face 的深度作为基准来估算屏幕中心的世界距离
    screen_distance_world = (screen_distance_pixels / focal_length) * Z_face  # Removed *2 for standard conversion

    # 欧拉角转旋转矩阵（假设yaw绕y，pitch绕x，roll绕z） -- Moved earlier for unification
    yaw_rad = -np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)  # Keeping the sign flip for correct up/down perspective
    roll_rad = np.deg2rad(roll)

    R_yaw = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ], dtype=np.float32)

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ], dtype=np.float32)

    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    R = R_pitch @ R_yaw @ R_roll

    # Unified dir_vec from R (for both position and normal)
    dir_vec = R @ np.array([0, 0, 1], dtype=np.float32)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-9)  # Ensure unit vector

    # Now compute screen center using the unified dir_vec
    screen_center_cam = face_center_cam + dir_vec * screen_distance_world

    # 在屏幕中心的深度 Zs 处计算半宽/半高（世界单位）
    Zs = screen_center_cam[2]
    half_w = (screen_w_pixels / 2.0) / focal_length * Zs
    half_h = (screen_h_pixels / 2.0) / focal_length * Zs

    # u_axis（屏幕右方向） = 相机坐标系的 x 轴旋转后的位置
    u_axis = R @ np.array([1, 0, 0], dtype=np.float32)

    # v_axis（屏幕下方向） = 相机坐标系的 y 轴旋转后的位置
    v_axis = R @ np.array([0, 1, 0], dtype=np.float32)


    # 相机内参 & 畸变（若有真实标定，这里可替换）
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    # ---------- 关键检查：确保 u_axis 投影到图像后实际指向图像的右侧 ----------
    test_r_cam = screen_center_cam + u_axis * half_w
    test_l_cam = screen_center_cam - u_axis * half_w

    pt_r = cv2.projectPoints(test_r_cam.reshape(1, 3), (0,0,0), (0,0,0), camera_matrix, dist_coeffs)[0].ravel()
    pt_l = cv2.projectPoints(test_l_cam.reshape(1, 3), (0,0,0), (0,0,0), camera_matrix, dist_coeffs)[0].ravel()


    # 如果“右侧”的投影 x 小于“左侧”的投影 x -> 说明 u_axis 在图像上指向左边 -> 翻转它
    if pt_r[0] < pt_l[0]:
        u_axis = -u_axis
        v_axis = np.cross(dir_vec, u_axis)
        v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-9)
    # ---------------------------------------------------------------------------

    test_d_cam = screen_center_cam + v_axis * half_h
    test_u_cam = screen_center_cam - v_axis * half_h

    pt_d = cv2.projectPoints(test_d_cam.reshape(1, 3), (0,0,0), (0,0,0), camera_matrix, dist_coeffs)[0].ravel()
    pt_u = cv2.projectPoints(test_u_cam.reshape(1, 3), (0,0,0), (0,0,0), camera_matrix, dist_coeffs)[0].ravel()

    if pt_d[1] < pt_u[1]:  # 如果"下方"的投影y比"上方"还小 → 翻转v_axis
        v_axis = -v_axis

    # 四角（顺序：左上, 右上, 右下, 左下），注意使用上面计算出的 half_w/half_h（世界单位）
    corners_cam = np.array([
        screen_center_cam - u_axis * half_w - v_axis * half_h,  # 左上 (tl)
        screen_center_cam + u_axis * half_w - v_axis * half_h,  # 右上 (tr)
        screen_center_cam + u_axis * half_w + v_axis * half_h,  # 右下 (br)
        screen_center_cam - u_axis * half_w + v_axis * half_h   # 左下 (bl)
    ], dtype=np.float32)

    # 投影到图像平面（这次在最终 u_axis 确认后再投影）
    pts2d, _ = cv2.projectPoints(corners_cam, (0,0,0), (0,0,0), camera_matrix, dist_coeffs)
    pts2d = np.int32(pts2d).reshape(-1, 2)

    # debug 信息
    if show_center_marker:
        print(f"Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°")
        print(f"屏幕角点Z深度 - 左上:{corners_cam[0,2]:.1f}, 右上:{corners_cam[1,2]:.1f}, 右下:{corners_cam[2,2]:.1f}, 左下:{corners_cam[3,2]:.1f}")
        print(f"2D投影坐标 - 左上:{pts2d[0]}, 右上:{pts2d[1]}, 右下:{pts2d[2]}, 左下:{pts2d[3]}")
        print("---")

    # ------------------ 载入并绘制屏幕内容（与原逻辑一致） ------------------
    content_image = None
    if screen_image_path:
        try:
            content_image = cv2.imread(screen_image_path, cv2.IMREAD_UNCHANGED)
            content_image = cv2.flip(content_image,1)  
            if content_image is None:
                print(f"Warning: 无法加载图像文件 {screen_image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
    if content_image is None and screen_image is not None:
        content_image = screen_image.copy()

    overlay = frame.copy()
    if content_image is not None:
        if content_image.shape[2] == 4:
            bgr = content_image[:, :, :3]
            alpha_channel = content_image[:, :, 3] / 255.0
        else:
            bgr = content_image
            alpha_channel = np.ones((content_image.shape[0], content_image.shape[1]), dtype=np.float32)

        src_h, src_w = bgr.shape[:2]
        src_points = np.float32([
            [0, 0],           # 左上
            [src_w, 0],       # 右上
            [src_w, src_h],   # 右下
            [0, src_h]        # 左下
        ])

        try:
            M = cv2.getPerspectiveTransform(src_points, pts2d.astype(np.float32))
            warped = cv2.warpPerspective(bgr, M, (w, h), flags=cv2.INTER_LINEAR)
            warped_alpha = cv2.warpPerspective(alpha_channel, M, (w, h), flags=cv2.INTER_LINEAR)
            mask = warped_alpha > 0.01
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask,
                    overlay[:, :, c] * (1 - warped_alpha * alpha) + warped[:, :, c] * warped_alpha * alpha,
                    overlay[:, :, c]
                )
        except Exception as e:
            print(f"Error in perspective transformation: {e}")
            cv2.fillPoly(overlay, [pts2d], color=(100, 100, 100))
    else:
        cv2.fillPoly(overlay, [pts2d], color=(50, 50, 50))

    if border_thickness > 0:
        cv2.polylines(overlay, [pts2d], isClosed=True, color=border_color, thickness=border_thickness)

    frame[:] = overlay[:]

    if show_center_marker:
        center_px = tuple(np.int32(cv2.projectPoints(
            screen_center_cam.reshape(1,3), (0,0,0), (0,0,0), camera_matrix, dist_coeffs
        )[0].ravel()))
        cv2.drawMarker(frame, center_px, border_color, markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

    return frame

#调试之后感觉还不错的版本
def draw_virtual_screen(
    frame, yaw, pitch, roll, face_center, face_size,
    screen_image_path: Optional[str] = None,
    screen_image: Optional[np.ndarray] = None,
    screen_distance_mul=5,
    screen_w_mul=4.0,
    screen_h_mul=2.0,
    focal_length=None,
    alpha=0.8,
    border_color=(0, 200, 255),
    border_thickness=0,
    show_center_marker=False
):

    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    if focal_length is None:
        focal_length = max(w, h)

    tdx, tdy = face_center

    # 深度估算（可根据标定调整）
    Z_face = (focal_length * 6.0) / face_size

    
    x_cam = (tdx - cx) / focal_length * Z_face
    y_cam = (tdy - cy) / focal_length * Z_face
    face_center_cam = np.array([x_cam, y_cam, Z_face], dtype=np.float32)

    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)

    # 方向向量（保持你原有的角度约定）
    dir_vec = np.array([
        -np.sin(yaw_rad) * np.cos(pitch_rad),
        -np.sin(pitch_rad),
        np.cos(yaw_rad) * np.cos(pitch_rad)
    ], dtype=np.float32)
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-9)

    # -------------------- 把像素单位转换为相机坐标系的世界单位 --------------------
    screen_distance_pixels = screen_distance_mul * face_size

    screen_w_pixels = max(40.0, screen_w_mul * face_size)
    screen_h_pixels = max(40.0, screen_h_mul * face_size)
    
    # 将像素距离转换为相机坐标系长度： ΔX = (Δx_pixels / f) * Z
    # 使用 face 的深度作为基准来估算屏幕中心的世界距离
    screen_distance_world = (screen_distance_pixels / focal_length) * Z_face
    screen_center_cam = face_center_cam + dir_vec * screen_distance_world

    # 在屏幕中心的深度 Zs 处计算半宽/半高（世界单位）
    Zs = screen_center_cam[2]
    half_w = (screen_w_pixels / 2.0) / focal_length * Zs
    half_h = (screen_h_pixels / 2.0) / focal_length * Zs
    
    # 欧拉角转旋转矩阵（假设yaw绕y，pitch绕x，roll绕z）
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = -np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)

    R_yaw = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ], dtype=np.float32)

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ], dtype=np.float32)

    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    R =  R_pitch @ R_yaw @ R_roll 

    # dir_vec（屏幕法向） = 相机坐标系的 z 轴旋转后的位置
    dir_vec = R @ np.array([0, 0, 1], dtype=np.float32)

    # u_axis（屏幕右方向） = 相机坐标系的 x 轴旋转后的位置
    u_axis = R @ np.array([1, 0, 0], dtype=np.float32)

    # v_axis（屏幕下方向） = 相机坐标系的 y 轴旋转后的位置
    v_axis = R @ np.array([0, 1, 0], dtype=np.float32)

    # 相机内参 & 畸变（若有真实标定，这里可替换）
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    # ---------- 关键检查：确保 u_axis 投影到图像后实际指向图像的右侧 ----------
    test_r_cam = screen_center_cam + u_axis * half_w
    test_l_cam = screen_center_cam - u_axis * half_w

    pt_r = cv2.projectPoints(test_r_cam.reshape(1, 3), (0,0,0), (0,0,0), camera_matrix, dist_coeffs)[0].ravel()
    pt_l = cv2.projectPoints(test_l_cam.reshape(1, 3), (0,0,0), (0,0,0), camera_matrix, dist_coeffs)[0].ravel()


    # 如果“右侧”的投影 x 小于“左侧”的投影 x -> 说明 u_axis 在图像上指向左边 -> 翻转它
    if pt_r[0] < pt_l[0]:
        u_axis = -u_axis
        v_axis = np.cross(dir_vec, u_axis)
        v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-9)
    # ---------------------------------------------------------------------------

    test_d_cam = screen_center_cam + v_axis * half_h
    test_u_cam = screen_center_cam - v_axis * half_h

    pt_d = cv2.projectPoints(test_d_cam.reshape(1, 3), (0,0,0), (0,0,0), camera_matrix, dist_coeffs)[0].ravel()
    pt_u = cv2.projectPoints(test_u_cam.reshape(1, 3), (0,0,0), (0,0,0), camera_matrix, dist_coeffs)[0].ravel()

    if pt_d[1] < pt_u[1]:  # 如果"下方"的投影y比"上方"还小 → 翻转v_axis
        v_axis = -v_axis

    # 四角（顺序：左上, 右上, 右下, 左下），注意使用上面计算出的 half_w/half_h（世界单位）
    corners_cam = np.array([
        screen_center_cam - u_axis * half_w - v_axis * half_h,  # 左上 (tl)
        screen_center_cam + u_axis * half_w - v_axis * half_h,  # 右上 (tr)
        screen_center_cam + u_axis * half_w + v_axis * half_h,  # 右下 (br)
        screen_center_cam - u_axis * half_w + v_axis * half_h   # 左下 (bl)
    ], dtype=np.float32)

    # 投影到图像平面（这次在最终 u_axis 确认后再投影）
    pts2d, _ = cv2.projectPoints(corners_cam, (0,0,0), (0,0,0), camera_matrix, dist_coeffs)
    pts2d = np.int32(pts2d).reshape(-1, 2)

    # debug 信息
    if show_center_marker:
        print(f"Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°")
        print(f"屏幕角点Z深度 - 左上:{corners_cam[0,2]:.1f}, 右上:{corners_cam[1,2]:.1f}, 右下:{corners_cam[2,2]:.1f}, 左下:{corners_cam[3,2]:.1f}")
        print(f"2D投影坐标 - 左上:{pts2d[0]}, 右上:{pts2d[1]}, 右下:{pts2d[2]}, 左下:{pts2d[3]}")
        print("---")

    # ------------------ 载入并绘制屏幕内容（与原逻辑一致） ------------------
    content_image = None
    if screen_image_path:
        try:
            content_image = cv2.imread(screen_image_path, cv2.IMREAD_UNCHANGED)
            content_image = cv2.flip(content_image,1)  
            if content_image is None:
                print(f"Warning: 无法加载图像文件 {screen_image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
    if content_image is None and screen_image is not None:
        content_image = screen_image.copy()

    overlay = frame.copy()
    if content_image is not None:
        if content_image.shape[2] == 4:
            bgr = content_image[:, :, :3]
            alpha_channel = content_image[:, :, 3] / 255.0
        else:
            bgr = content_image
            alpha_channel = np.ones((content_image.shape[0], content_image.shape[1]), dtype=np.float32)

        src_h, src_w = bgr.shape[:2]
        src_points = np.float32([
            [0, 0],           # 左上
            [src_w, 0],       # 右上
            [src_w, src_h],   # 右下
            [0, src_h]        # 左下
        ])

        try:
            M = cv2.getPerspectiveTransform(src_points, pts2d.astype(np.float32))
            warped = cv2.warpPerspective(bgr, M, (w, h), flags=cv2.INTER_LINEAR)
            warped_alpha = cv2.warpPerspective(alpha_channel, M, (w, h), flags=cv2.INTER_LINEAR)
            mask = warped_alpha > 0.01
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask,
                    overlay[:, :, c] * (1 - warped_alpha * alpha) + warped[:, :, c] * warped_alpha * alpha,
                    overlay[:, :, c]
                )
        except Exception as e:
            print(f"Error in perspective transformation: {e}")
            cv2.fillPoly(overlay, [pts2d], color=(100, 100, 100))
    else:
        cv2.fillPoly(overlay, [pts2d], color=(50, 50, 50))

    if border_thickness > 0:
        cv2.polylines(overlay, [pts2d], isClosed=True, color=border_color, thickness=border_thickness)

    frame[:] = overlay[:]

    if show_center_marker:
        center_px = tuple(np.int32(cv2.projectPoints(
            screen_center_cam.reshape(1,3), (0,0,0), (0,0,0), camera_matrix, dist_coeffs
        )[0].ravel()))
        cv2.drawMarker(frame, center_px, border_color, markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

    return frame

def main():
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Path
    snapshot_path = 'hopenet_robust.pkl'
    video_path = 'video/talk2.mp4'

    print ('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path, map_location=device)
    model.load_state_dict(saved_state_dict)

    # Load data
    transformations = transforms.Compose([transforms.Resize(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.eval().to(device)

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
  
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    # New cv2
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    # Deque
    yawde = deque(maxlen=30)
    pitchde = deque(maxlen=30)
    rollde = deque(maxlen=30)

    #Store
    yawstore = []
    pitchstore = []
    rollstore = []
    #ID
    dict = {'yaw':[-1,0],
        'pitch':[-1,0],
        'roll':[-1,0]}

    #filter
    filt = AttitudeFilter(fps, cutoff_hz=2.5)


    # Define the codec and create VideoWriter object    
    out_writer = cv2.VideoWriter('output/processed_test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    while True:
        ret, frame = video.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aligned_face, facebox, keypoints = get_aligned_face(rgb_frame, desired_size=224)
        if aligned_face is None:
            print("No face detected.")
            continue

        tdx, tdy, x_min, y_min, x_max, y_max = facebox

        img = transformations(Image.fromarray(aligned_face))
        img = img.view(1, *img.size())  # 保持 (1, C, H, W)
        img = Variable(img).to(device)
        # img = frame[y_min:y_max,x_min:x_max]
        # img = transformations(Image.fromarray(img))
        # img_shape = img.size()
        # img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).to(device)

        start = time.time()
        yaw, pitch, roll = model(img)
        end = time.time()
        timing.append(end - start)

        yaw_predicted = F.softmax(yaw,dim=1)
        pitch_predicted = F.softmax(pitch,dim=1)
        roll_predicted = F.softmax(roll,dim=1)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

        yaw = yaw_predicted.item()
        pitch = pitch_predicted.item()
        roll = roll_predicted.item()

        tdx = int((x_min + x_max) / 2)
        tdy = int((y_min + y_max) / 2)
        size = int((y_max-y_min) / 2)

        #yawde.append(yaw)
        ##rollde.append(roll)

        #fliter instantiation
        # filterobj = weifilter(yawde, pitchde, rollde, dict, yawstore, pitchstore, rollstore)
        # yaw,pitch,roll,yawde,pitchde,rollde = filterobj.get_angle()

        # 每帧使用（标量）
        yaw, pitch, roll = filt.update_single(yaw,pitch,roll)

        yawlist.append(yaw)
        pitchlist.append(pitch)
        rolllist.append(roll)
        
        # Draw bounding box and angles
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        draw_axis(frame,yaw, pitch, roll, tdx, tdy, size)
        frame = draw_virtual_screen(
        frame,
        yaw,
        pitch,
        roll,
        (tdx, tdy),  # 人脸中心
        size,         # 人脸大小
        screen_image_path="texture.png"
    )


        out_writer.write(frame)
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
    return fps

if __name__ == "__main__":
    fps = main()
    print("Processing complete. Output saved to 'output/processed_{}.mp4'.")


    print("Average processing time per frame: {:.4f} seconds".format(np.mean(timing)))
    print(f'mean{np.median(timing)}')
    # 创建图表
    # filter_obj = HeadPoseFilter(max_speed_deg_s=15, fps=fps, z_thresh=3)
    # filtered_data = filter_obj.process(yawlist, pitchlist, rolllist)
    visualize(yawlist, pitchlist, rolllist)
    