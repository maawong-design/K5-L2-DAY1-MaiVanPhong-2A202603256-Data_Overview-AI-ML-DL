# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không
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
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915; `taxonomy_name`=ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

Record này nói về cả bức ảnh (Image-level classification), không phải một vật thể riêng biệt.

Lý do:
- Model sử dụng task image_classification.
- Model phân tích toàn bộ hình ảnh đầu vào để đưa ra dự đoán lớp.
- Kết quả trả về một danh sách các lớp có khả năng mô tả nội dung ảnh, được sắp xếp theo thứ hạng.
- Không có bounding box (tọa độ khung giới hạn) xác định vị trí của từng phương tiện.
Trong ảnh có nhiều phương tiện như ô tô, xe buýt, xe tải và xe chuyên dụng. Tuy nhiên, model chỉ đưa ra dự đoán tổng thể dựa trên đặc trưng hình ảnh.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

Danh sách lớp được xác định bởi taxonomy và checkpoint của model, không phải do model tự nghĩ ra.

Cụ thể:
- ImageNet-1K là taxonomy (hệ thống phân loại) gồm 1.000 lớp.
- yolo11n-cls.pt là checkpoint chứa trọng số đã huấn luyện và cấu hình ánh xạ đầu ra với các lớp tương ứng.
- Model tính toán xác suất hoặc điểm tin cậy cho các lớp có sẵn trong hệ thống.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

Vì ba trường này có chức năng khác nhau và giúp dữ liệu annotation nhất quán, có thể kiểm tra và tái sử dụng.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

Guideline phải quy định rõ tiêu chí xác định chủ đề chính (Primary Subject) của toàn bộ bức ảnh.

Một guideline tốt cần quy định:
- Chủ thể chính: Chọn đối tượng hoặc chủ đề chiếm vai trò trung tâm trong ảnh.
- Mức độ nổi bật: Xem xét kích thước, vị trí, mức độ rõ nét và sự tập trung của nội dung.
- Ngữ cảnh tổng thể: Xác định bức ảnh đang mô tả hoạt động hoặc chủ đề gì.
- Quy tắc ưu tiên: Nếu có nhiều đối tượng, phải có quy định ưu tiên cụ thể để chọn một lớp duy nhất.
- Trường hợp không xác định được: Quy định khi nào sử dụng lớp other, ambiguous hoặc chuyển sang người kiểm duyệt.

6. Vì sao không được lấy `score` của model làm ground truth?

Vì score chỉ thể hiện mức độ tin cậy của model đối với dự đoán, không chứng minh dự đoán đó đúng.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[ 385.33 , 69.24 , 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

Góc trên-trái: nằm tại vị trí X = 385.33 pixel, Y = 69.24 pixel tính từ góc trên-trái của ảnh.
Góc dưới-phải: nằm tại vị trí X = 498.92 pixel, Y = 348.92 pixel tính từ góc trên-trái của ảnh.
Chiều rộng bounding box: 113.58 pixel.
Chiều cao bounding box: 279.68 pixel.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

Ngưỡng 0.20: Giữ lại 1 prediction.
Ngưỡng 0.60: Giữ lại 1 prediction.
Không có sự thay đổi về số lượng prediction trong record hiện có.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

Khi hạ Confidence Threshold, model giữ lại nhiều prediction hơn, bao gồm cả những Bounding Box có độ tin cậy thấp. Reviewer cần kiểm tra và loại bỏ False Positive, chỉnh sửa Bounding Box, xác minh Class Label, loại bỏ box trùng lặp và bổ sung những đối tượng bị bỏ sót. Điều này giúp tăng khả năng thu hồi đối tượng (Recall), nhưng đồng thời làm tăng khối lượng công việc và thời gian kiểm duyệt để đảm bảo chất lượng Ground Truth.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

Mỗi đối tượng được gán nhãn phải có một Bounding Box hình chữ nhật bao quanh toàn bộ phần đối tượng nhìn thấy được trong ảnh. Box phải sát với biên ngoài của đối tượng, không bao gồm quá nhiều vùng nền (background), không cắt mất các bộ phận nhìn thấy được và không bao gồm các đối tượng khác. Mỗi box chỉ được gán một class label tương ứng. Nếu đối tượng bị che khuất, chỉ vẽ box quanh phần đối tượng còn nhìn thấy được. Nếu không thể xác định chính xác đối tượng, đánh dấu để reviewer kiểm tra.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Vật được chọn: person 0.61 (góc dưới trái)

Chỉ có một bàn tay và một phần cẳng tay lọt vào khung hình, phần còn lại của người nằm ngoài mép ảnh. Hộp bao bị cắt sát cạnh trái, đồng thời bị chồng lấn bởi hộp oven 0.69 và các hộp bowl ngay bên cạnh. Mô hình vẫn gán nhãn person với độ tin cậy 0.61, thấp hơn nhiều so với person 0.91 của người đứng quay lưng ở giữa ảnh.

Quyết định: đưa lên hỏi Lab Coach (escalation), chưa viết thành quy tắc ngay

Lý do:

Đây là câu hỏi về ranh giới định nghĩa, không phải lỗi thao tác. Việc một bàn tay đơn lẻ có đáng gán nhãn person hay không là quyết định về phạm vi lớp đối tượng. Nếu tự quy ước, nhãn dễ thiếu nhất quán giữa các annotator.
Quy tắc sai sẽ lan rộng. Một dòng guideline áp dụng cho hàng nghìn ảnh. Nếu quy ước không khớp mục đích dự án (đếm người, theo dõi hành vi...), phải sửa lại nhãn hàng loạt.
Mới chỉ có một ví dụ. Chưa đủ để biết đây là trường hợp lặp lại hay ngoại lệ. Nên gom thêm vài ảnh tương tự rồi gửi kèm khi hỏi.
Đáp án phụ thuộc mục đích của mô hình, nên cần người nắm yêu cầu dự án xác nhận.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

Box chỉ cho 4 số bbox_xyxy = [385.45, 66.44, 498.02, 348.58], tức một hình chữ nhật khoảng 112.6 × 282.1 px bao quanh vật. Box cho biết vật nằm ở đâu và to cỡ nào, nhưng không cho biết vật có hình dạng ra sao bên trong hình chữ nhật đó.

Polygon (kitchen-001 có 348 điểm) đi theo đường viền thật của vật, nên cho thêm:

Hình dạng và đường biên chính xác: nhận ra đầu, vai, hai tay chống hông và hai chân của người đứng quay lưng, thay vì chỉ một khối chữ nhật.
Phân tách vật với nền: phần nền trong box (kệ bếp, nồi chảo, khoảng trống giữa tay và thân) không thuộc polygon.
Diện tích thực của vật: tính được từ polygon, chính xác hơn nhiều so với diện tích box vốn luôn thừa phần nền.
Xử lý chồng lấn tốt hơn: khi hai vật che nhau, polygon chỉ ra phần nào thuộc vật nào, còn hai box thì chồng lên nhau mà không phân biệt được.
Thông tin để kiểm tra nhãn: so polygon với ảnh gốc thấy ngay biên bị lệch hay thiếu, điều mà box thô khó lộ ra.

Đổi lại, polygon nặng hơn (hàng trăm cặp tọa độ so với 4 số) và tốn công gán nhãn hơn.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

instance_id là mã định danh của một đối tượng được gán nhãn trong một ảnh. Ví dụ kitchen-001 là đối tượng thứ nhất của mẫu kitchen (ghép từ sample_id và số thứ tự). Mã này dùng để chỉ đúng một bản ghi khi đối chiếu, kiểm tra, sửa nhãn hoặc báo lỗi, chẳng hạn: "polygon của kitchen-001 bị lệch ở vai trái".

class_id trả lời "là loại gì", tracking ID trả lời "là cùng vật đó qua thời gian", còn instance_id chỉ trả lời "là bản ghi nào trong ảnh này".

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

Vẽ biên mask bám sát mép ngoài của vật thể, nằm đúng trên ranh giới giữa vật và nền: không chừa nền thừa bên ngoài và không cắt vào phần thân vật. Ở chỗ mép mờ hoặc khó thấy, đặt đường biên chạm mép nhìn thấy được của vật rồi giữ nhất quán cách đặt đó cho cả bộ dữ liệu.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

các bowl ở góc dưới trái bàn (bowl 0.72 và bowl 0.38, cạnh nhau là bowl 0.50). Các hộp bao chồng sát nhau, các bát và đĩa nướng xếp gần như chạm vào nhau, và một phần bị bột, cán bột và mép khung ảnh che hoặc cắt. Quyết định: dùng guideline.

Bản chất vật không mơ hồ. Đều là bowl, chỉ là chúng chạm hoặc chồng lên nhau. Không có câu hỏi nào về việc "có gán nhãn hay không" hay về phạm vi của lớp đối tượng.

Guideline xử lý được bằng quy tắc cụ thể. Có thể viết thành quy tắc ngắn: mỗi điểm ảnh chỉ thuộc đúng một instance; ở chỗ hai vật chạm nhau, vẽ một đường biên chung dọc theo đường tiếp xúc, không để hai mask chồng lên nhau; chỉ vẽ phần nhìn thấy được. Quy tắc này khớp với quy tắc biên mask ở câu 4.

Tình huống phổ biến. Bát, chén, nồi xếp chồng gặp rất nhiều trong ảnh bếp, nên viết thành quy tắc giúp mọi annotator làm giống nhau và tiết kiệm công hỏi lại.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Mỗi ảnh có một nhãn đại diện cho chủ đề chính, thuộc taxonomy ImageNet-1K. | Ảnh có nhiều đối tượng, khó xác định chủ đề chính. | Quan sát toàn ảnh và chọn một class theo guideline, không phụ thuộc dự đoán của model. | Kiểm tra nhãn có phản ánh đúng chủ đề chính của ảnh và nhất quán với guideline không. |
| Phát hiện vật thể | Mỗi vật thể có một Bounding Box sát biên và Class Label chính xác. | Box bị lệch, vật thể bị che khuất hoặc model bỏ sót. | Vẽ hoặc chỉnh box theo phần vật thể nhìn thấy, bổ sung đối tượng bị bỏ sót. | Kiểm tra độ chính xác của Bounding Box, Class Label và các đối tượng bị bỏ sót. |
| Instance segmentation | Một instance = instance_id + class + bbox + polygon mask (từng vật một) | Biên mờ, vật chạm nhau hoặc bị che (vd person 0.61 cắt mép, các bowl chồng sát) | Vẽ biên bám mép vật, chỉ vẽ phần nhìn thấy; theo guideline, ca chưa rõ thì escalate | Mask khít biên, không chồng nhau, đúng class, ca khó đã xử lý đúng quy trình |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

Chỉ dùng ảnh COCO công khai (như coco_image_id 397133) cho bài lab, không đưa ảnh hay thông tin cá nhân, nhạy cảm lên công cụ bên ngoài, và chỉ chia sẻ file nhãn trong nhóm lab.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi (ví dụ ảnh có người thật nhận ra được, thông tin cá nhân, nguồn không rõ), tôi dừng xử lý ngay, không tải lên hay chia sẻ đi đâu, và báo cho Lab Coach kèm tên file/sample_id để họ quyết định.