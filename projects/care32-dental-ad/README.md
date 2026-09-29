# Care32 dental ad (9:16, 30s)

1. Put the Magnific outputs in `assets/`: `footage.mp4` (Seedance 2.5 multi-shot, 28s),
   `vo.mp3` (Meera Iyer, Indian English), `music.mp3` (Indian-fusion bed). The logo is already there.
2. From the toolkit root: `pip install imageio-ffmpeg pillow numpy`
3. `python3 projects/care32-dental-ad/build.py` → `out/care32-ad.mp4`
   (`--preview 2 6 27.5` writes still frames for checking the layout)

Overlays: logo top-center + "BOOK NOW | 82 82 82 1409" bar the whole time, an animated
"20+ Years of Experience" badge, service chips for each shot, and a 4.5s end card with the address,
website, both numbers and "2 mins Away from Rainbow Vistas & Marina Skies".
