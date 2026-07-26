import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Segmentasi Label Gizi — YOLOv8-seg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Konfigurasi TETAP (disamakan dengan notebook yolov8-segmentasi.ipynb) ────
# Nilai ini BUKAN untuk diubah dari UI — diambil persis dari cell evaluasi
# notebook (model.val() dan model.predict()) agar hasil aplikasi konsisten
# dengan angka yang dilaporkan di skripsi.
IMG_SIZE = 640
CONF_THRESHOLD = 0.25   # sama seperti model.val(conf=0.25, ...) & model.predict(conf=0.25, ...)
IOU_THRESHOLD = 0.5     # sama seperti model.val(iou=0.5, ...)
MASK_ALPHA = 0.4        # transparansi overlay mask, murni visual (tidak ada di notebook)

# Satu aksen hangat di ruang gelap — ramp ember, bukan pelangi default YOLO. (BGR)
PALETTE = [(46, 157, 255), (24, 122, 255), (138, 210, 255), (77, 178, 255)]
INK = (10, 6, 5)  # teks label di atas chip ember

# ─── Kulit visual ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp {
    background:
      radial-gradient(120% 55% at 50% -10%, rgba(255,122,24,.16) 0%, rgba(255,122,24,0) 60%),
      #05060a;
  }
  [data-testid="stHeader"] { background: transparent; }
  .block-container { padding-top: 2.6rem; max-width: 1180px; }

  .eyebrow {
    font-size: .68rem; letter-spacing: .34em; text-transform: uppercase;
    color: #ff9d2e; margin-bottom: .55rem;
  }
  .masthead {
    font-size: clamp(1.9rem, 4.4vw, 3.1rem); line-height: 1.02;
    letter-spacing: -.035em; color: #f4f1ea; margin: 0 0 .5rem 0;
  }
  .sub { color: #7d8496; font-size: .78rem; letter-spacing: .02em; }
  .rule { height: 1px; background: linear-gradient(90deg,#ff7a18,rgba(255,122,24,.07) 55%,transparent);
          margin: 1.4rem 0 1.8rem; }
  .hair { height: 1px; background: rgba(244,241,234,.09); margin: 2.2rem 0 1.4rem; }

  .sect { font-size: .7rem; letter-spacing: .3em; text-transform: uppercase;
          color: #7d8496; margin: 0 0 .7rem 0; }
  .spec { color: #9aa2b4; font-size: .78rem; line-height: 1.85; }
  .spec b { color: #f4f1ea; font-weight: 400; }

  [data-testid="stMetricValue"] {
    font-size: 2rem; letter-spacing: -.04em; color: #f4f1ea; font-variant-numeric: tabular-nums;
  }
  [data-testid="stMetricLabel"] {
    font-size: .64rem; letter-spacing: .22em; text-transform: uppercase; color: #6e7688;
  }
  [data-testid="stImageContainer"] img { border: 1px solid rgba(244,241,234,.1); }

  .stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid rgba(244,241,234,.09); }
  .stTabs [data-baseweb="tab"] {
    background: transparent; padding: 0 0 .6rem 0;
    font-size: .7rem; letter-spacing: .22em; text-transform: uppercase; color: #6e7688;
  }
  .stTabs [aria-selected="true"] { color: #ff9d2e; }

  [data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"] > div {
    background: rgba(244,241,234,.02); border: 1px dashed rgba(255,157,46,.28);
  }
  details > summary { font-size: .8rem; letter-spacing: .04em; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="eyebrow">Instance segmentation · YOLOv8m-seg</div>
<h1 class="masthead">Label Informasi Gizi</h1>
<div class="sub">Muhamad Fazar Rizky Ardianto — deteksi panel gizi berbasis mask poligon</div>
<div class="rule"></div>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sect">Model</div>', unsafe_allow_html=True)
    yolo_model_path = st.text_input(
        "Path bobot", value="best.pt", label_visibility="collapsed",
        help="File .pt hasil training YOLOv8-seg (task=segment)",
    )
    show_conf_label = st.checkbox("Label confidence di gambar", value=True)

    st.markdown('<div class="hair"></div><div class="sect">Parameter inferensi</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="spec">'
        f'Confidence &nbsp;<b>{CONF_THRESHOLD}</b><br>'
        f'IoU (NMS) &nbsp;<b>{IOU_THRESHOLD}</b><br>'
        f'Image size &nbsp;<b>{IMG_SIZE}</b>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hair"></div><div class="spec">Nilai dikunci agar identik dengan '
        'cell evaluasi notebook — hasil aplikasi konsisten dengan angka di skripsi.</div>',
        unsafe_allow_html=True,
    )

# ─── Load model (cache) ───────────────────────────────────────────────────────
@st.cache_resource
def load_yolo_seg(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)

# ─── Segmentasi ───────────────────────────────────────────────────────────────
def segment_image(img_bgr, model):
    """Jalankan YOLOv8-seg (konfigurasi tetap) dan kumpulkan tiap instance (mask polygon + confidence)."""
    results = model.predict(
        img_bgr, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMG_SIZE,
        task="segment", verbose=False
    )
    r = results[0]
    instances = []
    if r.masks is None or r.boxes is None:
        return instances

    polygons = r.masks.xy                       # list poligon (koordinat piksel gambar asli)
    names    = r.names
    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls  = int(box.cls[0].item())
        conf_i = float(box.conf[0].item())
        poly = np.asarray(polygons[i], dtype=np.int32) if i < len(polygons) else np.empty((0, 2), np.int32)
        # luas mask dari poligon (piksel gambar asli), konsisten dengan bbox
        area_px = int(cv2.contourArea(poly)) if poly.shape[0] >= 3 else 0
        instances.append({
            "id":    i + 1,
            "class": names.get(cls, str(cls)),
            "conf":  conf_i,
            "box":   [int(x1), int(y1), int(x2), int(y2)],
            "polygon": np.asarray(poly, dtype=np.int32),
            "area_px": area_px,
        })
    return instances

def draw_segmentation(img_bgr, instances, alpha=MASK_ALPHA, show_label=True):
    """Overlay mask polygon berwarna + outline + label confidence per objek."""
    overlay = img_bgr.copy()
    outlined = img_bgr.copy()
    for inst in instances:
        color = PALETTE[(inst["id"] - 1) % len(PALETTE)]
        poly  = inst["polygon"]
        if poly.shape[0] >= 3:
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(outlined, [poly], isClosed=True, color=color, thickness=2)
    blended = cv2.addWeighted(overlay, alpha, outlined, 1 - alpha, 0)

    # Label teks digambar setelah blend agar tetap jelas
    for inst in instances:
        color = PALETTE[(inst["id"] - 1) % len(PALETTE)]
        x1, y1 = inst["box"][0], inst["box"][1]
        if show_label:
            label = f'{inst["id"]:02d}  {inst["class"]}  {inst["conf"]:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            yl = max(y1, th + 8)
            cv2.rectangle(blended, (x1, yl - th - 8), (x1 + tw + 10, yl), color, -1)
            cv2.putText(blended, label, (x1 + 5, yl - 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, INK, 1, cv2.LINE_AA)
    return blended

# ─── Input: berkas perangkat atau kamera ──────────────────────────────────────
tab_file, tab_cam = st.tabs(["Unggah berkas", "Ambil foto"])
with tab_file:
    uploaded_files = st.file_uploader(
        "Gambar kemasan produk (boleh lebih dari satu)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
with tab_cam:
    shot = st.camera_input("Arahkan kamera ke panel informasi gizi", label_visibility="collapsed")

# Satu daftar sumber, dua pintu masuk. getvalue() aman dipanggil ulang saat rerun.
sources = [(f.name, f.getvalue()) for f in (uploaded_files or [])]
if shot is not None:
    sources.append(("kamera.jpg", shot.getvalue()))

if sources:
    try:
        model = load_yolo_seg(yolo_model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    if getattr(model, "task", None) != "segment":
        st.warning(
            f"Model bertask '{getattr(model, 'task', '?')}', bukan 'segment'. "
            "Pastikan best.pt adalah model YOLOv8-seg."
        )

    total_objects = 0
    all_confs = []
    csv_rows = ["file,object_id,class,confidence,x1,y1,x2,y2,area_px"]

    st.markdown(
        f'<div class="hair"></div><div class="sect">{len(sources)} berkas diproses</div>',
        unsafe_allow_html=True,
    )

    for idx, (name, raw) in enumerate(sources):
        img_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error(f"Gagal membaca gambar: {name}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner(f"Segmentasi {name}…"):
            instances = segment_image(img_bgr, model)

        seg_img = draw_segmentation(img_bgr, instances, MASK_ALPHA, show_conf_label)
        seg_rgb = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

        total_objects += len(instances)
        all_confs.extend([inst["conf"] for inst in instances])

        with st.expander(f"{name}  —  {len(instances)} objek", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="sect">Sumber</div>', unsafe_allow_html=True)
                st.image(img_rgb, width="stretch")
                st.caption(f"{img_bgr.shape[1]} × {img_bgr.shape[0]} px")
            with col2:
                st.markdown('<div class="sect">Segmentasi</div>', unsafe_allow_html=True)
                st.image(seg_rgb, width="stretch")
                st.caption(f"conf ≥ {CONF_THRESHOLD} · IoU NMS {IOU_THRESHOLD}")

            if instances:
                confs = [inst["conf"] for inst in instances]
                m1, m2, m3 = st.columns(3)
                m1.metric("Objek", len(instances))
                m2.metric("Conf. tertinggi", f"{max(confs):.3f}")
                m3.metric("Conf. rata-rata", f"{sum(confs)/len(confs):.3f}")

                try:
                    import pandas as pd
                    df = pd.DataFrame([{
                        "Objek":       f'{inst["id"]:02d}',
                        "Kelas":       inst["class"],
                        "Confidence":  round(inst["conf"], 4),
                        "Bbox (x1,y1,x2,y2)": ", ".join(map(str, inst["box"])),
                        "Luas Mask (px)":     inst["area_px"],
                    } for inst in instances])
                    st.dataframe(df, width="stretch", hide_index=True)
                except Exception:
                    for inst in instances:
                        st.write(f'{inst["id"]:02d} {inst["class"]} — conf {inst["conf"]:.3f}')

                ok, buf = cv2.imencode(".png", seg_img)
                if ok:
                    st.download_button(
                        "Unduh gambar tersegmentasi",
                        data=buf.tobytes(),
                        file_name=f"seg_{name}.png",
                        mime="image/png",
                        key=f"dl_{idx}",
                    )

                for inst in instances:
                    csv_rows.append(
                        f'{name},{inst["id"]},{inst["class"]},{inst["conf"]:.4f},'
                        f'{inst["box"][0]},{inst["box"][1]},{inst["box"][2]},{inst["box"][3]},'
                        f'{inst["area_px"]}'
                    )
            else:
                st.warning(f"Tidak ada objek tersegmentasi (confidence < {CONF_THRESHOLD}).")

    # ─── Ringkasan keseluruhan ────────────────────────────────────────────────
    st.markdown('<div class="hair"></div><div class="sect">Ringkasan</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Berkas", len(sources))
    s2.metric("Objek", total_objects)
    s3.metric("Conf. tertinggi", f"{max(all_confs):.3f}" if all_confs else "—")
    s4.metric("Conf. rata-rata", f"{sum(all_confs)/len(all_confs):.3f}" if all_confs else "—")

    if len(csv_rows) > 1:
        st.download_button(
            "Unduh semua hasil (CSV)",
            data="\n".join(csv_rows),
            file_name="hasil_segmentasi.csv",
            mime="text/csv",
        )

    st.caption(
        "Confidence = tingkat keyakinan model pada tiap objek (0–1). Untuk evaluasi akurasi "
        "(Precision/Recall/mAP) gunakan model.val() pada dataset berlabel — bukan confidence sampel baru."
    )

else:
    st.markdown(
        '<div class="hair"></div>'
        '<div class="sect">Alur</div>'
        '<div class="spec">'
        'Unggah berkas <span style="color:#ff9d2e">atau</span> ambil foto '
        '&nbsp;→&nbsp; YOLOv8-seg &nbsp;→&nbsp; mask poligon per objek &nbsp;→&nbsp; confidence'
        '</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="hair"></div><div class="sect">Spesifikasi</div>'
            f'<div class="spec">'
            f'Model &nbsp;<b>YOLOv8m-seg (fine-tuned)</b><br>'
            f'Task &nbsp;<b>Instance segmentation — poligon</b><br>'
            f'Input &nbsp;<b>{IMG_SIZE} × {IMG_SIZE}</b><br>'
            f'Output &nbsp;<b>mask + bbox + confidence</b>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="hair"></div><div class="sect">Ambang (terkunci)</div>'
            f'<div class="spec">'
            f'Confidence &nbsp;<b>{CONF_THRESHOLD}</b><br>'
            f'IoU NMS &nbsp;<b>{IOU_THRESHOLD}</b><br>'
            f'Sumber &nbsp;<b>notebook yolov8-segmentasi.ipynb</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="hair"></div>'
    '<div class="sub">YOLOv8-seg · Muhamad Fazar Rizky Ardianto · 2025</div>',
    unsafe_allow_html=True,
)
