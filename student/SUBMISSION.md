# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

## Thông tin học viên

- Họ tên: Văn Thành Huy
- MSSV: 2A202602763
- Email: thanhhuvyan5@gmail.com
- Link repo (fork): https://github.com/thanhhuyvan/K4-L2L3-DAY23-VanThanhHuy-2A202602763-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): lấy HEAD của nhánh main sau commit CP6 để nộp trên LMS; commit chứa code và artifacts đã chạy là `c746dff8bc2495d533f8efef3a49407c95424d90`. Báo cáo không nhúng hash của chính commit chứa nó.

## Tóm tắt kết quả

- `fusion_mode`: `compare`; `frames`: `[0, 198]` (199 frame mỗi mode); `seed`: `0`.
- `segment`: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`.
- Detector FPN-ResNet-18 pretrained; chạy CPU, 4 luồng PyTorch, không training.
- Lệnh chính thức: `fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0` với `frame_start: 0`, `frame_end: 198`. Phiên chạy local đặt giới hạn luồng trước khi gọi CLI; tổng thời gian pipeline khoảng 393.8 giây.

### Detection

| Chỉ số | Giá trị |
|---|---:|
| `detection.tp` | 519 |
| `detection.fp` | 16 |
| `detection.fn` | 222 |
| `detection.precision` | 0.970093457944 |
| `detection.recall` | 0.700404858300 |

Detection giống nhau ở hai mode trên từng frame. Đây là kết quả của detector đã cung cấp, được ghi nhận riêng với chất lượng tracker.

### Tracking

| Chỉ số | LiDAR | LiDAR + camera |
|---|---:|---:|
| `rmse` (m) | 0.150322669212 | 0.135866762358 |
| `matches` | 502 | 502 |
| `sum_sq_err` | 11.343646249194 | 9.266808111039 |
| `ghost_track_frames` | 0 | 0 |
| `missed_gt_frames` | 239 | 239 |
| `mean_confirmed_tracks` | 2.522613065327 | 2.522613065327 |
| `precision_track` | 1.000000000000 | 1.000000000000 |
| `coverage` | 0.967244701349 | 0.967244701349 |

Giá trị đầy đủ nằm trong [metrics.json](artifacts/metrics.json); bảng trên làm tròn để đọc. RMSE tính theo `sqrt(sum_sq_err / matches)` trên vị trí 3D, với confirmed tracks ghép một-một với GT xe bằng gate XY 2 m.

Fused giảm RMSE khoảng 0.014456 m, tương đương 9.62% so với LiDAR. Hai mode giữ cùng 502 matches, 0 ghost và 239 miss, nên việc giảm RMSE đi kèm cùng số cặp đánh giá. `precision_track = 502 / (502 + 0) = 1`; `coverage = 502 / 519 ≈ 0.967245`. Cả hai đạt RMSE ≤ 0.45 m, precision_track ≥ 0.75 và coverage ≥ 0.70; `rmse_fused - rmse_lidar ≈ -0.014456 m`, đạt giới hạn ≤ 0.05 m.

239 miss cho thấy tracker chưa phủ tất cả GT xe trong từng frame dù precision_track bằng 1. Detection recall khoảng 70.04% là một giới hạn quan trọng; coverage của rubric chỉ dùng `det_tp` làm mẫu số. Kết quả trên toàn segment khác lượt thử 21 frame đầu, nơi fused có RMSE cao hơn LiDAR; không kết luận fusion luôn cải thiện ở mọi đoạn.

Camera dùng tâm hộp 2D ground-truth FRONT được thêm nhiễu theo seed. Kết quả này chỉ đánh giá fusion với phép đo camera mô phỏng từ nhãn, không đo chất lượng một camera detector hay perception system độc lập với GT.

Đã đối chiếu 398 record trong [grade_run.log](artifacts/grade_run.log), đúng 199 record cho mỗi mode, frame 0–198 không thiếu/trùng. Mỗi record thoả `matches + ghosts == confirmed` và `matches + misses == valid_gt`; tổng khớp metrics. Giữ nguyên bốn file per-mode để kiểm chứng; không sửa tay metrics/log.

## Giải thích ngắn (Parts E–H)

### 1. Khác biệt đo LiDAR 3D và camera 2D trong EKF

LiDAR có `z = [x, y, z]ᵀ` trong hệ cảm biến, đơn vị mét; ma trận R 3×3 là đường chéo của bình phương độ lệch chuẩn LiDAR (0.01 m² mỗi trục). Mô hình đo LiDAR tuyến tính theo vị trí, Jacobian H có 3 hàng và phần vận tốc bằng 0. Camera có `z = [u, v]ᵀ`, đơn vị pixel; R 2×2 là `diag(sigma_cam_i², sigma_cam_j²) = diag(25, 25)` pixel². Camera dùng phép chiếu phi tuyến sau biến đổi extrinsic: `u = c_i - f_i*y_s/x_s`, `v = c_j - f_j*z_s/x_s`; H là Jacobian 2×6 của phép chiếu. Xem [camera_fusion.py](workspace/camera_fusion.py), [kalman.py](workspace/kalman.py) và `Sensor.get_H` trong platform.

### 2. Vì sao gating Mahalanobis trước khi gán?

Khoảng cách bình phương `d² = γᵀ S⁻¹ γ`, với `γ = z - h(x)` và `S = HPHᵀ + R`, chuẩn hoá residual theo độ bất định của track và phép đo. Nó tránh dùng khoảng cách Euclid trực tiếp giữa mét và pixel, đồng thời xét cả tương quan covariance. Cổng χ² dùng xác suất 0.995 và số chiều phép đo: 3 cho LiDAR, 2 cho camera. Trong [association.py](workspace/association.py), kiểm tra FOV trước phép chiếu/Mahalanobis; các cặp ngoài FOV hoặc ngoài gate có chi phí vô hạn. Greedy chỉ chọn cặp hữu hạn nhỏ nhất và xoá hàng/cột đã dùng; đây là gán greedy theo đề, không bảo đảm tối ưu toàn cục như Hungarian.

### 3. Track-then-fuse hay fuse-then-track?

Pipeline là **track-then-fuse**, một tracker ở mỗi lượt chạy. Trong `platform/fusion_lab/scripts/run_lab.py`, mỗi frame predict các track một lần, associate/update LiDAR, rồi associate/update camera trong mode fused, sau đó mới ghi record đánh giá. Camera tinh chỉnh cùng trạng thái EKF mà LiDAR đang theo dõi, không sinh một bộ track camera riêng để ghép sau. Log cho thấy hai mode có cùng frame và detection; kết quả tracking có cùng số matches/ghost/miss nhưng `sum_sq_err` khác nhau. Thứ tự thao tác cần đối chiếu mã runner vì grade_run.log ghi kết quả mỗi frame, không ghi riêng từng bước predict/update.

### 4. Camera lệch calibration gây triệu chứng gì?

Extrinsic hoặc intrinsic sai làm `h(x)` và Jacobian H không còn khớp phép đo ảnh, khiến innovation có thể lệch có hệ thống, phụ thuộc độ sâu hoặc vị trí trong ảnh. Khi residual lớn so với S, d² vượt gate χ² và camera update bị từ chối; nếu vẫn lọt gate, update có thể kéo trạng thái sai và tăng RMSE fused. Code camera từ chối toạ độ không hữu hạn hoặc độ sâu ≤ 1e-6 để tránh phép chia không hợp lệ; điều này không phát hiện mọi sai lệch calibration. Đây là giải thích theo mô hình; chưa chạy thử nghiệm làm lệch calibration, nên không trình bày số liệu bonus về nội dung này.

### 5. Vì sao cần sensor tường minh ở frame rỗng?

Nếu `meas_list` rỗng thì không thể lấy sensor từ phần tử đầu để biết đây là lượt LiDAR hay camera. [association.py](workspace/association.py) vẫn gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)` khi không có đo hoặc mọi cặp bị gate. Với LiDAR, manager trừ score của track bị miss trong FOV, xét xoá và khởi tạo track từ đo chưa ghép. Với camera, lượt cập nhật chỉ tinh chỉnh EKF; không tăng/giảm score, không tạo hoặc xoá track. `handle_updated_track` chỉ được gọi khi sensor là LiDAR, còn `TrackManager.manage_tracks` của platform bỏ qua vòng đời ở lượt camera.

### 6. Xác nhận, giữ confirmed sau miss và xoá track

Trong [track_management.py](workspace/track_management.py), track mới đổi tâm đo sang hệ xe, đặt vận tốc ban đầu bằng 0, covariance vị trí là R sau xoay và covariance vận tốc là `diag(50², 50², 5²)`. Score ban đầu `1/6`, trạng thái `initialized`; mỗi hit LiDAR cộng `1/6` đến tối đa 1, miss trong FOV trừ `1/6`. Khi score **> 0.8** thì confirmed; track đã confirmed không bị hạ trạng thái chỉ vì miss, còn track chưa confirmed có hit chuyển tentative nếu chưa vượt ngưỡng. Track bị xoá khi **Pxx hoặc Pyy > 9 m²**, hoặc confirmed có **score < 0.6**, hoặc chưa confirmed có **score ≤ 0**. Do đó tại score 0.6 hay variance 9 m² chưa xoá theo điều kiện tương ứng; vòng đời chỉ được xét trong lượt LiDAR.

## Bonus

Không. Chưa thực hiện CVAT, trực quan hoá bonus hoặc thử nghiệm lệch calibration.

## Khai báo sử dụng AI

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): ChatGPT/Codex.
- Dùng cho phần nào (hàm, câu hỏi, debug): hỗ trợ triển khai toàn bộ 17 hàm Parts E–H; đọc tài liệu và công thức; cài môi trường CPU trên Windows, xử lý dependency/UTF-8; chạy tests và pipeline Waymo; tổng hợp số liệu và soạn báo cáo. Việc triển khai và kiểm tra được thực hiện trong phiên làm việc với Codex, không khai báo là tự viết hoàn toàn không có AI.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): giữ nguyên bộ test của đề, kết quả 128 passed, không failed/xfailed; kiểm tra F/Q, cập nhật EKF, Jacobian bằng sai phân hữu hạn, gating trước projection, frame rỗng và vòng đời chỉ theo LiDAR. Chạy segment mặc định ở compare/seed 0, đối chiếu 6 artifacts và 398 record, kiểm tra invariants và detection giống nhau giữa hai mode. Số liệu báo cáo lấy từ artifacts sinh bởi runtime, không tạo hoặc sửa tay log/metrics. Học viên chịu trách nhiệm đọc hiểu và giải thích code cũng như các kết quả khi vấn đáp.

## Checklist nộp

- [x] Parts E–H đã implement; 128 tests passed, không failed/xfailed.
- [x] Part A–D và platform giữ nguyên.
- [x] Lần chạy chính thức compare/seed 0, frame 0–198.
- [x] Sáu artifacts đã commit ở CP5, không sửa tay.
- [x] Báo cáo có thông tin học viên, đủ 6 câu giải thích và khai báo AI.
- [x] Dữ liệu, weights, paths.yaml và .env được gitignore; không đưa lên repo.
- [x] Nội dung, artifacts và các file cấm đã qua đối chiếu của check_submission.py; kiểm tra trạng thái commit cuối cùng trước khi push.
- [ ] Đã push và nộp link repo + commit hash trên LMS. Việc nộp LMS do học viên thực hiện sau khi nhận commit hash cuối.
