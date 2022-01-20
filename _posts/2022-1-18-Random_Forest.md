---
layout: post
author: dinhhuyhoang
title: 14. Random Forest
---

## 1. Giới thiệu

Ở bài 12 và bài 13 mình đã giới thiệu về mô hình Decision Tree sử dụng các độ đo khác nhau để tìm ra được cây quyết định, ngoài ra lớp mô hình này có thể làm việc với cả 2 bài toán Classification và Regression. Ý tưởng chính là nó là xây dựng một chuỗi cây hỏi từ trên xuống để đưa ra dự đoán ở nhánh kết thúc. Mặc dù tính mạnh mẽ của nó đã được thể hiện nhưng những hạn chế còn lại khá nhiều, có thể kể đến là: dễ xảy ra hiện tượng _Overfiting_, không quá tốt trong các bộ dữ liệu lớn... và một điểm yếu khác đó là mô hình chỉ đưa ra dự báo dựa trên một kịch bản duy nhất (một cây được tạo ra), điều này sẽ làm cho sự phụ thuộc vào các thuộc tính được lựa chọn ở trên đỉnh rất cao, cho nên khi điểm một dữ liệu 'lạ' sẽ không được dự đoán chính xác.

Để cải thiện độ yếu kém của một cây duy nhất các nhà nghiên cứu đã đề xuất một phương pháp cải tiến đó là hợp nhất nhiều cây quyết định hơn để đưa ra kết quả. Và ý tưởng của sự kết hợp nhiều cây này sẽ tạo thành thuật toán _Random Forest (rừng ngẫu nhiên)_. 