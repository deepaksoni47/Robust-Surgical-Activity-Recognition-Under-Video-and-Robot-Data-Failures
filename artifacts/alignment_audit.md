# JIGSAWS Multi-Modal Video-Kinematics Alignment Audit

- **Trials Audited:** 103
- **Mean Video vs Kinematics Difference:** 12.37 frames
- **Median Video vs Kinematics Difference:** 5.0 frames
- **Max Video vs Kinematics Difference:** 470 frames
- **Mean Kinematics vs Transcription Difference:** 67.39 frames

### Alignment Convention
Kinematics samples and video frames are 1-to-1 synchronized at nominal 30 Hz. Slight recording tail offsets (< 0.5s) are clipped to min(N_vid, N_kin) during windowing.

### Sample Trials Alignment Table

| Trial ID | Task | FPS | Video Frames | Kinematics Samples | Delta (Frames) | Vid Duration | Kin Duration |
|---|---|---|---|---|---|---|---|
| Suturing_B001 | Suturing | 30.0 | 5640 | 5635 | 5 | 188.0s | 187.83s |
| Suturing_B002 | Suturing | 30.0 | 3462 | 3445 | 17 | 115.4s | 114.83s |
| Suturing_B003 | Suturing | 30.0 | 3437 | 3432 | 5 | 114.57s | 114.4s |
| Suturing_B004 | Suturing | 30.0 | 3274 | 3257 | 17 | 109.13s | 108.57s |
| Suturing_B005 | Suturing | 30.0 | 2863 | 2848 | 15 | 95.43s | 94.93s |
| Suturing_C001 | Suturing | 30.0 | 3185 | 3180 | 5 | 106.17s | 106.0s |
| Suturing_C002 | Suturing | 30.0 | 3282 | 3267 | 15 | 109.4s | 108.9s |
| Suturing_C003 | Suturing | 30.0 | 3166 | 3161 | 5 | 105.53s | 105.37s |
| Suturing_C004 | Suturing | 30.0 | 2493 | 2479 | 14 | 83.1s | 82.63s |
| Suturing_C005 | Suturing | 30.0 | 2581 | 2577 | 4 | 86.03s | 85.9s |
| Suturing_D001 | Suturing | 30.0 | 2816 | 2800 | 16 | 93.87s | 93.33s |
| Suturing_D002 | Suturing | 30.0 | 2219 | 2214 | 5 | 73.97s | 73.8s |
| Suturing_D003 | Suturing | 30.0 | 2322 | 2317 | 5 | 77.4s | 77.23s |
| Suturing_D004 | Suturing | 30.0 | 2747 | 2742 | 5 | 91.57s | 91.4s |
| Suturing_D005 | Suturing | 30.0 | 1794 | 1779 | 15 | 59.8s | 59.3s |
