from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np

# Konvensi landmark yang umum:
#  - Hip: LHIP, RHIP
#  - Knee: LKNE, RKNE
#  - Ankle: LANK, RANK
#  - Toe: LTOE, RTOE (atau foot_index)
#  - Shoulder: LSHO, RSHO
#  - Ear/Nose (opsional buat orientasi)
#
# Setiap landmark = (x,y,z,score). x meningkat ke kanan, y meningkat ke bawah (screen coords).
# Jika model kamu pakai urutan berbeda, sesuaikan indeksnya.

@dataclass
class SquatMetrics:
	trunk_lean_max_deg: float
	thorax_side_bend_max_deg: float
	pelvis_drop_deg_at_depth: float
	knee_flex_max_deg_L: float
	knee_flex_max_deg_R: float
	hip_flex_max_deg_L: float
	hip_flex_max_deg_R: float
	ankle_dorsi_deg_L_at_depth: float
	ankle_dorsi_deg_R_at_depth: float
	foot_ER_deg_L_at_depth: float
	foot_ER_deg_R_at_depth: float
	knee_valgus_deg_L_at_depth: float
	knee_valgus_deg_R_at_depth: float
	squat_depth_thigh_deg: float
	com_shift_ratio_right: float  # perkiraan proporsi bobot ke kanan (0..1)

@dataclass
class SquatFlags:
	knee_dominant: bool
	hip_dominant: bool
	trunk_leans_anterior: bool
	thorax_side_bend_right: bool
	thorax_side_bend_left: bool
	right_foot_ER: bool
	left_foot_ER: bool
	left_knee_valgus: bool
	right_knee_valgus: bool
	insufficient_right_ankle_dorsi: bool
	insufficient_left_ankle_dorsi: bool
	thoracolumbar_hyperextension: bool
	insufficient_depth_parallel: bool
	weight_bearing_right: bool
	weight_bearing_left: bool

def _angle_2d(a, b, c) -> float:
	"""Sudut ABC (derajat) pada bidang 2D (x,y)."""
	a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
	ba = a - b
	bc = c - b
	cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
	cosang = np.clip(cosang, -1.0, 1.0)
	return np.degrees(np.arccos(cosang))

def _vector(a, b):
	a, b = np.array(a[:2]), np.array(b[:2])
	v = b - a
	n = np.linalg.norm(v) + 1e-9
	return v / n

def _line_angle_deg(p, q, ref="vertical") -> float:
	"""Sudut garis pq terhadap vertikal/horizontal (derajat)."""
	v = _vector(p, q)
	if ref == "vertical":
		# sudut terhadap sumbu y (ke bawah)
		ref_v = np.array([0.0, 1.0])
	else:
		ref_v = np.array([1.0, 0.0])
	cosang = np.dot(v, ref_v)
	cosang = np.clip(cosang, -1.0, 1.0)
	return np.degrees(np.arccos(cosang))

def _signed_lateral_tilt_deg(left_pt, right_pt) -> float:
	"""Kemiringan bahu/pelvis (positif = kanan lebih tinggi)."""
	l, r = np.array(left_pt[:2]), np.array(right_pt[:2])
	dy = (r[1] - l[1])  # y turun ke bawah -> dy>0 artinya kanan lebih rendah
	dx = (r[0] - l[0])
	ang = np.degrees(np.arctan2(dy, dx))
	# Jika dy negatif, kanan lebih tinggi → sisi kanan naik → nilai negatif (kita balik)
	return -ang  # simplifikasi: negatif ~ right higher, positif ~ right lower

def _projected_com_x(landmarks: Dict[str, Tuple[float,float,float,float]]) -> float:
	"""Perkiraan COM lateral (x) sebagai rata-rata x dari bahu, panggul, lutut, pergelangan kaki."""
	keys = ["LSHO","RSHO","LHIP","RHIP","LKNE","RKNE","LANK","RANK"]
	xs = [landmarks[k][0] for k in keys if k in landmarks]
	if not xs: return 0.5
	return float(np.mean(xs))

def _thigh_angle_from_horizontal_deg(hip, knee) -> float:
	"""|sudut paha terhadap horizontal| (0 = sejajar lantai)."""
	return abs(_line_angle_deg(hip, knee, ref="horizontal"))

def _foot_progression_angle_deg(ank, toe) -> float:
	"""External rotation (+) bila garis ankle→toe mengarah lebih lateral dari sumbu tubuh."""
	# Bandingkan ke sumbu longitudinal tubuh: gunakan bahu tengah->pinggul tengah sebagai sumbu "maju".
	# Di call utama kita hitung relatif terhadap horizontal, cukup sebagai proxy:
	return _line_angle_deg(ank, toe, ref="vertical")  # makin besar ≈ makin ER di tampilan frontal

def _shank_foot_dorsiflex_deg(knee, ank, toe) -> float:
	"""Sudut dorsifleksi: antara shank (knee→ankle) dan foot (ankle→toe). Lebih besar = lebih dorsi."""
	shank = _angle_2d(knee, ank, toe)  # sudut di ankle
	# Pada posisi netral, sudut lebih kecil; saat dorsifleksi, sudut bertambah.
	return shank

def _valgus_angle_deg(hip, knee, ank) -> float:
	"""Deviasi frontal (valgus +): sudut H-K-A lebih kecil dari fisiologis → valgus.
	   Kita pakai 180 - sudut(H-K-A) sebagai skor valgus (derajat)."""
	theta = _angle_2d(hip, knee, ank)
	return max(0.0, 180.0 - theta)

def _flexion_angle_deg(prox, joint, dist) -> float:
	"""Sudut fleksi pada joint (mis. hip: shoulder/hip/knee atau pelvis/hip/knee)."""
	return 180.0 - _angle_2d(prox, joint, dist)

def _median_depth_frame(frames: List[Dict[str, Any]]) -> int:
	"""Ambil frame terdalam berdasar ketinggian rata-rata pinggul (y terendah)."""
	ys = []
	for i, lm in enumerate(frames):
		hips = [lm[k][1] for k in ["LHIP","RHIP"] if k in lm]
		ys.append(np.mean(hips) if hips else 1e9)
	return int(np.argmin(ys)) if ys else 0

def analyze_squat_from_sequence(frames: List[Dict[str, Tuple[float,float,float,float]]],
								score_thr: float = 0.5) -> Tuple[SquatMetrics, Any]:
	"""
	frames: list of dict {name: (x,y,z,score)} untuk setiap frame video.
	score_thr: abaikan landmark dengan confidence < thr.
	"""
	# Filtrasi score
	clean_frames = []
	for lm in frames:
		clean = {}
		for k, v in lm.items():
			if len(v) >= 4 and v[3] is not None and v[3] >= score_thr:
				clean[k] = v
		clean_frames.append(clean)

	# Frame terdalam
	d_idx = _median_depth_frame(clean_frames)
	f_depth = clean_frames[d_idx] if clean_frames else {}

	# Titik-titik yang dibutuhkan
	def G(lm, key, default=(np.nan, np.nan, 0.0, 0.0)): return lm.get(key, default)

	# Sudut trunk lean (bahu-mid ke panggul-mid vs vertikal), dan side-bend bahu
	trunk_lean_list = []
	thorax_side_bend_list = []
	weight_shift_list = []

	knee_flex_L, knee_flex_R = [], []
	hip_flex_L, hip_flex_R = [], []

	for lm in clean_frames:
		LSHO, RSHO = G(lm,"LSHO"), G(lm,"RSHO")
		LHIP, RHIP = G(lm,"LHIP"), G(lm,"RHIP")
		LKNE, RKNE = G(lm,"LKNE"), G(lm,"RKNE")
		LANK, RANK = G(lm,"LANK"), G(lm,"RANK")
		LTOE, RTOE = G(lm,"LTOE"), G(lm,"RTOE")

		# Mid points
		SHO = ((LSHO[0]+RSHO[0])/2, (LSHO[1]+RSHO[1])/2, 0, 1)
		HIP = ((LHIP[0]+RHIP[0])/2, (LHIP[1]+RHIP[1])/2, 0, 1)

		# Trunk lean vs vertikal
		trunk_lean = _line_angle_deg(SHO, HIP, ref="vertical")
		trunk_lean_list.append(trunk_lean)

		# Side-bend thorax (kemiringan bahu)
		thorax_tilt = _signed_lateral_tilt_deg(LSHO, RSHO)  # (+)=right lower, (-)=right higher
		thorax_side_bend_list.append(thorax_tilt)

		# Perkiraan bobot lateral berdasar COM x relatif terhadap titik tengah pergelangan kaki
		com_x = _projected_com_x(lm)
		if all(k in lm for k in ["LANK","RANK"]):
			mid_ank_x = (LANK[0] + RANK[0]) / 2
			# ratio ke kanan: 0 (penuh kiri) .. 1 (penuh kanan)
			width = max(1e-6, abs(RANK[0] - LANK[0]))
			ratio_right = 0.5 + (com_x - mid_ank_x) / width
			ratio_right = float(np.clip(ratio_right, 0.0, 1.0))
		else:
			ratio_right = 0.5
		weight_shift_list.append(ratio_right)

		# Fleksi lutut & panggul (pakai 2D)
		# Knee flexion: hip - knee - ankle
		kL = _flexion_angle_deg(LHIP, LKNE, LANK)
		kR = _flexion_angle_deg(RHIP, RKNE, RANK)
		# Hip flexion: shoulder - hip - knee (proxy)
		hL = _flexion_angle_deg(LSHO, LHIP, LKNE)
		hR = _flexion_angle_deg(RSHO, RHIP, RKNE)
		knee_flex_L.append(kL); knee_flex_R.append(kR)
		hip_flex_L.append(hL); hip_flex_R.append(hR)

	# Ambil nilai puncak
	trunk_lean_max = float(np.nanmax(trunk_lean_list) if trunk_lean_list else np.nan)
	thorax_side_bend_max = float(np.nanmax(np.abs(thorax_side_bend_list)) if thorax_side_bend_list else np.nan)

	# Di frame terdalam, hitung metrik sisi
	LSHO, RSHO = G(f_depth,"LSHO"), G(f_depth,"RSHO")
	LHIP, RHIP = G(f_depth,"LHIP"), G(f_depth,"RHIP")
	LKNE, RKNE = G(f_depth,"LKNE"), G(f_depth,"RKNE")
	LANK, RANK = G(f_depth,"LANK"), G(f_depth,"RANK")
	LTOE, RTOE = G(f_depth,"LTOE"), G(f_depth,"RTOE")

	# Pelvic drop (kemiringan pelvis L-R)
	pelvis_tilt_signed = _signed_lateral_tilt_deg(LHIP, RHIP)  # (+)=right lower
	# Depth: paha vs horizontal (semakin kecil semakin mendekati sejajar lantai)
	thigh_deg = min(_thigh_angle_from_horizontal_deg(LHIP, LKNE),
						_thigh_angle_from_horizontal_deg(RHIP, RKNE))

	# Foot external rotation (ER)
	foot_ER_L = _foot_progression_angle_deg(LANK, LTOE)
	foot_ER_R = _foot_progression_angle_deg(RANK, RTOE)

	# Dorsifleksi pergelangan kaki
	dorsi_L = _shank_foot_dorsiflex_deg(LKNE, LANK, LTOE)
	dorsi_R = _shank_foot_dorsiflex_deg(RKNE, RANK, RTOE)

	# Valgus (frontal misalignment)
	valg_L = _valgus_angle_deg(LHIP, LKNE, LANK)
	valg_R = _valgus_angle_deg(RHIP, RKNE, RANK)

	# Max flexion akumulatif dari deret
	knee_flex_max_L = float(np.nanmax(knee_flex_L) if knee_flex_L else np.nan)
	knee_flex_max_R = float(np.nanmax(knee_flex_R) if knee_flex_R else np.nan)
	hip_flex_max_L  = float(np.nanmax(hip_flex_L) if hip_flex_L else np.nan)
	hip_flex_max_R  = float(np.nanmax(hip_flex_R) if hip_flex_R else np.nan)

	# Thoracolumbar hyperextension (proxy): trunk vs pelvis divergen besar saat depth
	# pakai selisih sudut bahu-lumbal (bahu-mid→pinggul-mid) terhadap femur rata-rata;
	SHO = ((LSHO[0]+RSHO[0])/2, (LSHO[1]+RSHO[1])/2, 0, 1)
	HIPm = ((LHIP[0]+RHIP[0])/2, (LHIP[1]+RHIP[1])/2, 0, 1)
	trunk_vs_horizontal = _line_angle_deg(SHO, HIPm, ref="horizontal")
	femur_L = _line_angle_deg(LHIP, LKNE, ref="horizontal")
	femur_R = _line_angle_deg(RHIP, RKNE, ref="horizontal")
	tl_divergence = abs(trunk_vs_horizontal - 0.5*(femur_L+femur_R))

	# Bobot ke kanan
	com_right = float(np.nanmedian(weight_shift_list) if weight_shift_list else 0.5)

	metrics = SquatMetrics(
		trunk_lean_max_deg=trunk_lean_max,
		thorax_side_bend_max_deg=thorax_side_bend_max,
		pelvis_drop_deg_at_depth=pelvis_tilt_signed,
		knee_flex_max_deg_L=knee_flex_max_L,
		knee_flex_max_deg_R=knee_flex_max_R,
		hip_flex_max_deg_L=hip_flex_max_L,
		hip_flex_max_deg_R=hip_flex_max_R,
		ankle_dorsi_deg_L_at_depth=dorsi_L,
		ankle_dorsi_deg_R_at_depth=dorsi_R,
		foot_ER_deg_L_at_depth=foot_ER_L,
		foot_ER_deg_R_at_depth=foot_ER_R,
		knee_valgus_deg_L_at_depth=valg_L,
		knee_valgus_deg_R_at_depth=valg_R,
		squat_depth_thigh_deg=thigh_deg,
		com_shift_ratio_right=com_right
	)

	# ---- RULES (ambang sederhana; silakan sesuaikan dengan dataset Anda) ----
	# Knee- vs Hip-dominant 
	knee_dom = (max(knee_flex_max_L, knee_flex_max_R) - max(hip_flex_max_L, hip_flex_max_R) > 15) and (metrics.trunk_lean_max_deg < 25)
	hip_dom  = (max(hip_flex_max_L,  hip_flex_max_R)  - max(knee_flex_max_L, knee_flex_max_R) > 15) or (metrics.trunk_lean_max_deg > 35)

	# Trunk lean anterior 
	trunk_lean_flag = metrics.trunk_lean_max_deg > 25

	# Thorax side-bend kanan/kiri 
	thorax_sb_right = np.nanmedian([x for x in thorax_side_bend_list if not np.isnan(x)]) < -5  # right higher → side-bend right
	thorax_sb_left  = np.nanmedian([x for x in thorax_side_bend_list if not np.isnan(x)]) >  5

	# Foot ER (external rotation) 
	right_ER = metrics.foot_ER_deg_R_at_depth > 10
	left_ER  = metrics.foot_ER_deg_L_at_depth > 10

	# Knee valgus 
	left_valgus  = metrics.knee_valgus_deg_L_at_depth  > 5
	right_valgus = metrics.knee_valgus_deg_R_at_depth > 5

	# Insufficient ankle dorsiflexion 
	insuff_R_dorsi = metrics.ankle_dorsi_deg_R_at_depth < 15
	insuff_L_dorsi = metrics.ankle_dorsi_deg_L_at_depth < 15

	# Thoracolumbar hyperextension 
	tl_hyperext = tl_divergence > 25

	# Squat depth: paha ~ sejajar lantai 
	insufficient_depth = metrics.squat_depth_thigh_deg > 10  # >10° dari horizontal

	# Weight bearing side 
	wb_right = metrics.com_shift_ratio_right > 0.6
	wb_left  = metrics.com_shift_ratio_right < 0.4

	flags = SquatFlags(
		knee_dominant=knee_dom,
		hip_dominant=hip_dom,
		trunk_leans_anterior=trunk_lean_flag,
		thorax_side_bend_right=thorax_sb_right,
		thorax_side_bend_left=thorax_sb_left,
		right_foot_ER=right_ER,
		left_foot_ER=left_ER,
		left_knee_valgus=left_valgus,
		right_knee_valgus=right_valgus,
		insufficient_right_ankle_dorsi=insuff_R_dorsi,
		insufficient_left_ankle_dorsi=insuff_L_dorsi,
		thoracolumbar_hyperextension=tl_hyperext,
		insufficient_depth_parallel=insufficient_depth,
		weight_bearing_right=wb_right,
		weight_bearing_left=wb_left
	)

	return metrics, flags


