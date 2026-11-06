# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** 

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số "tin" của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về cả bức ảnh, không phải về một vật thể riêng. Vì tác vụ image classification chỉ đưa ra một nhãn duy nhất cho toàn bộ ảnh — trong JSON không có bounding box hay toạ độ nào chỉ ra vật thể cụ thể. Model nhìn tổng thể bức ảnh (ảnh giao thông có nhiều xe) rồi gán nhãn "cab" cho cả ảnh.

3. Danh sách tên lớp đi kèm taxonomy/checkpoint (ImageNet-1K, gồm 1000 lớp định sẵn), không phải model tự nghĩ ra. Model được huấn luyện trên bộ ImageNet-1K nên chỉ có thể dự đoán các lớp nằm trong danh sách 1000 lớp này.

4. Cần ghi cả ba vì:
   - `class_id` (468) là mã số duy nhất, tránh nhầm lẫn khi tên lớp viết khác nhau hoặc trùng tên.
   - `class_name` ("cab") giúp con người đọc hiểu nhanh nội dung nhãn.
   - `taxonomy_name` ("ImageNet-1K") cho biết bộ phân loại nào đang dùng — vì cùng `class_id = 468` ở taxonomy khác có thể mang nghĩa hoàn toàn khác. Ghi cả 3 đảm bảo truy vết chính xác, không mập mờ.

5. Guideline cần quy định rõ tiêu chí chọn lớp chính khi ảnh có nhiều vật/chủ đề, ví dụ: "Chọn lớp mô tả đối tượng/chủ đề chiếm diện tích lớn nhất hoặc nằm ở trung tâm ảnh. Nếu hai chủ đề cân bằng nhau không thể quyết định, đánh dấu ca đó là escalation để hỏi Lab Coach, không tự bịa nhãn."

6. Không được lấy `score` của model làm ground truth vì `score` (0.51) chỉ là độ tự tin (confidence) của model khi dự đoán, không phải nhãn đúng. Model hoàn toàn có thể sai — ví dụ ảnh `traffic` có nhiều xe bus, car nhưng model gán "cab" chỉ 51%. Ground truth phải do con người gán theo guideline dựa trên quan sát thực tế, độc lập với việc model đoán đúng hay sai.



## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn một record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Góc trên-trái của hộp ở toạ độ pixel (385, 69) — khoảng 60% chiều ngang ảnh tính từ trái, gần sát mép trên ảnh. Góc dưới-phải ở toạ độ pixel (499, 349) — khoảng 78% chiều ngang, hơn giữa chiều dọc ảnh. Hộp rộng khoảng 114 pixel, cao khoảng 280 pixel, bao trọn người đầu bếp từ đầu xuống gần đầu gối.

3.  Với dữ liệu ngưỡng 0.35 , ảnh `kitchen` có **11 vật thể**. Ước tính:
   - Ngưỡng **0.20**: số prediction sẽ **17 vật thể** (giữ thêm các hộp score thấp, bao gồm vật nhỏ, mờ, bị che).
   - Ngưỡng **0.60**: chỉ giữ lại **6 vật thể** (person 0.91, bowl 0.72, bowl 0.70, oven 0.69, oven 0.63, person 0.61) — chỉ các vật model rất tự tin.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), reviewer phải: kiểm tra thêm nhiều hộp hơn, xoá các false positive (hộp sai hoặc bao vật không tồn tại), loại hộp trùng (duplicate boxes cho cùng một vật), sửa hộp bao lệch, và xác nhận nhãn class cho các vật nhỏ/mờ khó nhận dạng → tốn thời gian và công sức đáng kể hơn.

5. Quy tắc box: "Hộp bao (bounding box) phải bao sát vật thể nhất có thể (tight fit): mỗi cạnh hộp chạm phần ngoài cùng nhìn thấy được của vật thể, không để khoảng trống thừa lớn hơn khoảng 5 pixel mỗi phía, và không cắt mất phần nào đang nhìn thấy được của vật thể."

6. Vật bị cắt mép: **person** ở góc dưới-trái ảnh (bbox = [0.12, 263.18, 61.55, 310.78]) — toạ độ x bắt đầu ≈ 0, tức người này bị **cắt mép trái ảnh**, chỉ nhìn thấy một phần nhỏ (tay/vai bên phải). Tương tự, **oven** bên trái (bbox x bắt đầu = 0.11) cũng bị cắt mép. Nên **viết thành quy tắc trong guideline**, ví dụ: "Vật thể bị cắt mép ảnh vẫn được gán nhãn nếu nhận dạng được, vẽ box đến sát mép ảnh. Vật bị che hoặc cắt >50% diện tích thì đánh dấu thuộc tính 'truncated/occluded'." Đây là tình huống thường gặp, cần quy tắc chung. Chỉ escalation khi không rõ vật đó là gì hoặc guideline chưa đề cập.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-002 ; `class_name`=bowl ; `score`=0.735744 ; số điểm polygon=`polygon_point_count`=67 ; 5 điểm đầu của `polygon_xy`= [53,344], [52,345], [50,345], [49,346], [48,346]

2. So với box (hình chữ nhật bao ngoài, bao cả phần nền xung quanh), polygon vẽ theo **đường viền (contour) bám sát hình dạng thực** của vật thể, cho biết: hình dạng chính xác của vật (tròn, bầu dục, méo…), diện tích thực tế (pixel-level), và phân biệt rõ phần nào là vật thể, phần nào là nền bên trong box.

3. `instance_id` ("kitchen-002") là **mã định danh duy nhất cho một cá thể vật thể cụ thể** trong ảnh. Nó **không phải class ID** vì nhiều cá thể khác nhau (vd. kitchen-002 và kitchen-003) có thể cùng class "bowl" nhưng là hai cái tô riêng biệt — class ID chỉ phân biệt loại, không phân biệt cá thể. Nó không phải tracking ID vì tracking ID dùng để theo dõi cùng một vật thể xuyên suốt nhiều frame trong video, còn instance_id chỉ phân biệt các vật thể trong cùng một ảnh tĩnh.

4. Quy tắc vẽ biên mask: "Vẽ polygon bám sát đường viền nhìn thấy được của vật thể, cách mép vật thể không quá 2 pixel. Không bao gồm bóng đổ, phản chiếu, hoặc phần bị vật khác che khuất. Nếu vật có viền mờ (lông, tóc, khói), vẽ theo đường viền rõ nhất có thể nhìn thấy."

5. Trên ảnh `kitchen`: **person** (kitchen-001) đứng trước **oven** — biên polygon của người đè lên lò nướng phía sau, tạo ra chỗ hai vật chạm/che nhau. Ngoài ra, khu vực bàn (**dining table**, kitchen-005) chồng lên nhiều **bowl** — biên bàn và biên tô gần như đè nhau. Nên **dùng guideline**, ví dụ: "Khi hai vật chạm/che nhau, mỗi vật có polygon riêng. Phần bị vật phía trước che khuất không được vẽ vào polygon của vật phía sau. Nếu biên quá mờ hoặc không phân biệt nổi hai vật riêng biệt, đánh dấu escalation." Đây là tình huống phổ biến nên cần quy tắc chung; chỉ escalation khi thực sự không quyết định được.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp duy nhất cho cả ảnh (vd. class_id + class_name + taxonomy) | Ảnh `traffic` có nhiều loại xe (cab, bus, car); model đoán "cab" chỉ score=0.51, top-5 đều là phương tiện → khó chọn 1 lớp đúng nhất | Nhìn toàn ảnh, chọn **một lớp phù hợp nhất** theo guideline; nếu không quyết định được → escalation | Kiểm tra nhãn có đúng chủ đề chính không; nhãn có tuân theo guideline không; trường hợp mập mờ có được escalation đúng không |
| Phát hiện vật thể | Mỗi vật thể nhìn thấy có 1 bounding box (4 toạ độ xyxy) + class_name | Ảnh `kitchen`: person ở góc trái bị cắt mép (x≈0); oven bị cắt mép trái; nhiều bowl chồng nhau trên bàn; cup score thấp (0.45, 0.38) dễ nhầm lẫn | Vẽ box bám sát từng vật thể nhìn thấy, gán đúng class name theo taxonomy COCO-80 | Kiểm tra box có sát không (tight fit), có thiếu vật hay thừa box sai không, nhãn class có đúng không, vật bị cắt/che có xử lý đúng quy tắc không |
| Instance segmentation | Mỗi cá thể vật thể có 1 polygon (danh sách toạ độ xy) + class_name + instance_id riêng biệt | Ảnh `kitchen`: person đè lên oven phía sau → biên chồng; dining table polygon rất phức tạp (598 điểm) chồng lên bowl; potted plant góc trên-trái có biên lá mờ | Vẽ polygon bám sát biên mỗi cá thể, gán instance_id riêng biệt cho từng cá thể, gán class_name | Kiểm tra polygon có bám sát biên thực không, các instance có bị thiếu/trùng không, instance_id có nhất quán không, biên chỗ vật chồng nhau có xử lý đúng không |

## 5. An toàn dữ liệu

1. Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo và output; không chia sẻ dữ liệu lab ra ngoài phạm vi bài nộp trên LMS/GitHub.

2. Dừng ngay việc xử lý, không mở/lưu/chia sẻ dữ liệu đó, và báo cho **giảng viên / Lab Coach** để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
