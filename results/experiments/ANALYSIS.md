# Phân tích kết quả experiment – Color Moments trên WANG/Corel-1K

Sinh lại toàn bộ số liệu: `python scripts/extract_features.py --color-space ALL` rồi `python scripts/run_experiments.py`.

## 1. Thiết lập

- **Dataset:** WANG/Corel-1K, 10 category × 100 ảnh.
- **Giao thức:** leave-one-out, mỗi ảnh làm truy vấn một lần (1000 truy vấn) và được xếp hạng với 999 ảnh còn lại. Ảnh liên quan là 99 ảnh cùng category.
- **Metric:** Precision@5/10/20, Recall@5/10/20, F1@5/10/20, Average Precision (tính trên toàn bộ 999 kết quả xếp hạng), mAP = trung bình AP.
  Lưu ý: Recall@20 tối đa chỉ đạt 20/99 ≈ 0.202, nên mAP là chỉ số tổng hợp chính.
- **Đặc trưng:** 9 chiều cho mỗi không gian màu (mean, std, skew × 3 kênh). Tổ hợp nhiều không gian màu thì ghép nối các vector (18 hoặc 27 chiều).
  - RGB: 0–255 (giống baseline). HSV: H 0–360, S/V 0–1. LAB: L 0–100, a/b ≈ −127..127 (OpenCV float).
- **Khoảng cách:** Euclidean, Manhattan, Cosine.
- **Chuẩn hoá:** `none` (giá trị gốc) và `zscore` (mỗi chiều về mean 0, std 1 trên toàn bộ database, không dùng nhãn).
- **Baseline:** hệ thống hiện tại gồm RGB, Euclidean, không chuẩn hoá. Kết quả baseline khớp với `results/evaluation` đã commit trước đó (P@10 = 0.5408).

Tổng cộng 7 phương pháp × 3 khoảng cách × 2 kiểu chuẩn hoá = 42 cấu hình.

## 2. Kết quả chính

Bảng dưới lấy cấu hình tốt nhất của mỗi phương pháp (đầy đủ trong `overall_results.csv`):

| Phương pháp | Dim | Khoảng cách | Chuẩn hoá | P@5 | P@10 | P@20 | F1@20 | mAP | Δ mAP so với baseline |
|---|---|---|---|---|---|---|---|---|---|
| **RGB+HSV+LAB** | 27 | cosine | zscore | 0.709 | **0.668** | 0.623 | 0.209 | **0.465** | **+30.2%** |
| HSV+LAB | 18 | cosine | zscore | 0.698 | 0.655 | 0.611 | 0.205 | 0.463 | +29.7% |
| RGB+LAB | 18 | euclidean | zscore | 0.681 | 0.641 | 0.591 | 0.199 | 0.438 | +22.8% |
| LAB | 9 | cosine | zscore | 0.657 | 0.616 | 0.573 | 0.193 | 0.438 | +22.7% |
| RGB+HSV | 18 | cosine | zscore | 0.670 | 0.625 | 0.581 | 0.195 | 0.430 | +20.5% |
| HSV | 9 | cosine | zscore | 0.605 | 0.572 | 0.528 | 0.178 | 0.391 | +9.5% |
| RGB | 9 | euclidean | zscore | 0.586 | 0.543 | 0.506 | 0.170 | 0.374 | +4.8% |
| *RGB baseline* | 9 | euclidean | none | 0.582 | 0.541 | 0.493 | 0.166 | 0.357 | — |

**Kết luận:** cấu hình tốt nhất là **RGB+HSV+LAB, z-score, Cosine**. So với baseline RGB, mAP tăng từ 0.357 lên 0.465 (+0.108, tức **+30.2%**) và P@10 tăng từ 0.541 lên 0.668 (+0.127, tức **+23.5%**).

- Cải thiện có ý nghĩa thống kê: AP tăng ở 705/1000 truy vấn và giảm ở 295 truy vấn. Kiểm định Wilcoxon signed-rank trên AP từng truy vấn cho p ≈ 3.6·10⁻⁶⁹ (`query_results_best.csv`).
- HSV+LAB (18 chiều) gần như ngang RGB+HSV+LAB (mAP 0.463 so với 0.465, P@5 và P@10 còn nhỉnh hơn khi dùng Manhattan). Nếu cần vector gọn hơn thì HSV+LAB là lựa chọn hợp lý. Trong nhóm top, chênh lệch giữa các cấu hình dưới 0.01 mAP, nên có thể coi là tương đương.
- Nếu chỉ được dùng một không gian màu thì **LAB** tốt nhất, với mAP +22.7% so với baseline.

### Ảnh hưởng của chuẩn hoá

| Phương pháp | mAP (none) | mAP (zscore) |
|---|---|---|
| RGB | 0.362 | 0.374 |
| HSV | 0.220 | 0.391 |
| LAB | 0.413 | 0.438 |
| HSV+LAB | 0.339 | 0.463 |
| RGB+HSV+LAB | 0.409 | 0.465 |

Chuẩn hoá z-score là **yếu tố quan trọng nhất** khi dùng HSV hoặc ghép nhiều không gian màu. Nếu không chuẩn hoá, kênh H (0–360) và các kênh RGB (0–255) sẽ lấn át S/V (0–1), khiến HSV gần như chỉ còn dựa vào Hue. Kết quả là HSV không chuẩn hoá còn kém hơn cả baseline.

### Ảnh hưởng của hàm khoảng cách

Khi đã chuẩn hoá z-score, mAP trung bình của ba hàm khoảng cách gần như bằng nhau: Cosine 0.428, Euclidean 0.425, Manhattan 0.422. Cosine nhỉnh hơn một chút ở các tổ hợp nhiều chiều, còn Manhattan thường cho P@5/P@10 cao hơn một chút. Hàm khoảng cách ảnh hưởng ít hơn nhiều so với việc chọn không gian màu và chuẩn hoá.

## 3. Phân tích theo category

mAP theo category (`category_map_pivot.csv`), so sánh baseline với cấu hình tốt nhất:

| Category | P@10 baseline | P@10 tốt nhất | mAP baseline | mAP tốt nhất | Δ mAP |
|---|---|---|---|---|---|
| dinosaurs | 0.989 | 0.975 | 0.985 | 0.947 | −0.038 |
| horses | 0.845 | 0.975 | 0.424 | 0.791 | **+0.367** |
| buses | 0.434 | 0.728 | 0.261 | 0.522 | **+0.262** |
| flowers | 0.670 | 0.773 | 0.432 | 0.486 | +0.054 |
| food | 0.528 | 0.652 | 0.241 | 0.378 | +0.137 |
| africa | 0.475 | 0.580 | 0.317 | 0.376 | +0.059 |
| elephants | 0.455 | 0.564 | 0.308 | 0.353 | +0.046 |
| beach | 0.426 | 0.488 | 0.250 | 0.290 | +0.040 |
| mountains | 0.308 | 0.441 | 0.202 | 0.277 | +0.074 |
| buildings | 0.278 | 0.505 | 0.151 | 0.229 | +0.078 |

Ma trận nhầm lẫn Top-10 nằm trong `confusion_top10_baseline.csv` và `confusion_top10_best.csv`. Mỗi ô là phần trăm kết quả Top-10 của category hàng rơi vào category cột.

### Category truy xuất tốt

- **Dinosaurs** (mAP 0.95–0.99): ảnh là mô hình khủng long trên nền trắng hoặc xám nhạt đồng nhất. Phân bố màu rất khác các category còn lại nên chỉ cần mean/std là tách được. Cấu hình tốt nhất giảm nhẹ ở lớp này vì z-score làm tăng trọng số của các chiều skew và Hue, vốn nhiễu khi ảnh gần như không có sắc độ (Hue không xác định khi S ≈ 0).
- **Horses** (0.42 → 0.79, cải thiện nhiều nhất): ảnh chủ yếu là ngựa nâu trên đồng cỏ xanh. Kênh a của LAB (trục xanh lá–đỏ) và Hue của HSV mô tả trực tiếp "màu xanh cỏ", trong khi RGB thô trộn màu này với độ sáng. Riêng LAB đã đạt mAP 0.86 ở lớp này.
- **Buses** (0.26 → 0.52): xe buýt thường có màu sơn bão hoà cao (đỏ, vàng). Kênh S và H của HSV bắt được độ bão hoà này, nên HSV đơn lẻ đạt 0.47 so với 0.26 của baseline.
- **Flowers**: màu hoa rực rỡ và nổi bật. Lớp này vẫn bị nhầm sang food khoảng 8% vì hai lớp có cùng tông đỏ/cam/vàng bão hoà.

### Category truy xuất kém

- **Buildings** (0.23), **mountains** (0.28), **beach** (0.29): ba lớp này bị nhầm chéo với nhau. Mountains có 18% kết quả rơi vào beach và 16% vào buildings, còn beach có 15% rơi vào buildings. Nguyên nhân:
  1. **Cùng bảng màu:** cả ba lớp đều có trời xanh, đất/đá/cát xám–be và mặt nước hoặc kính xanh, nên phân bố màu toàn cục rất giống nhau.
  2. **Mất thông tin không gian:** Color Moments toàn cục không phân biệt được "xanh ở nửa trên (trời) và xanh ở nửa dưới (biển)" với các bố cục khác.
  3. **Khác biệt nằm ở hình dạng và kết cấu** (đường thẳng của toà nhà, đường chân trời của núi) mà đặc trưng màu không mô tả được.
  4. **Biến thiên trong lớp lớn:** toà nhà có đủ loại màu, và ảnh chụp ở nhiều điều kiện ánh sáng khác nhau.
- **Africa, food, elephants** (0.35–0.38): africa bị nhầm với food khoảng 11% và ngược lại, do cùng tông nâu/cam của da người, đất và đồ ăn. Elephants bị nhầm sang buildings (11%) và dinosaurs (10%) do da voi xám giống tường đá và nền xám của ảnh khủng long.

## 4. Hạn chế và hướng cải thiện

- **Hue là đại lượng vòng tròn:** H = 1° và H = 359° đều là màu đỏ nhưng có khoảng cách lớn khi tính mean/std/skew. Nên dùng circular mean hoặc biểu diễn Hue bằng (cos H, sin H).
- **Color Moments theo vùng (block-based):** chia ảnh thành lưới 2×2 hoặc 3×3 để giữ bố cục không gian. Cách này có thể giúp phân biệt beach, mountains và buildings. Schema đã có sẵn cột `feature_type` (hiện là `GLOBAL`) để lưu thêm các loại đặc trưng như `GRID_3x3`.
- **Kết hợp đặc trưng kết cấu và hình dạng** (LBP, GLCM, HOG) cho các lớp khác nhau chủ yếu về cấu trúc như buildings và mountains.
- **Gán trọng số cho từng không gian màu hoặc từng moment** thay vì để các chiều ngang nhau sau z-score.

## 5. Các file kết quả

| File | Nội dung |
|---|---|
| `overall_results.csv` | 42 cấu hình, đủ P/R/F1@K và mAP, kèm mức tăng so với baseline (`map_gain`, `map_gain_pct`, `precision_at_10_gain_pct`) |
| `category_results.csv` | Kết quả theo cấu hình × category (420 dòng), kèm `map_gain` theo category |
| `category_map_pivot.csv` | mAP theo category × phương pháp (dùng để vẽ heatmap hoặc biểu đồ cột nhóm) |
| `query_results_best.csv` | P/R/F1/AP của từng truy vấn cho baseline và cấu hình tốt nhất |
| `confusion_top10_baseline.csv`, `confusion_top10_best.csv` | Ma trận nhầm lẫn Top-10 (%) |
