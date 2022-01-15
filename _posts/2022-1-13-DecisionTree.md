---
layout: post
author: dinhhuyhoang
title: Bài 12 - Decision Tree
---

## 1. Giới thiệu

Trong thực tế, khi đưa ra một quyết định về một vấn đề nào đó ta thường dựa trên những yếu tố xung quanh để đặt ra những câu hỏi liên quan tới chủ đề mà ta đang quan tâm. Ví dụ trước khi bạn hỏi thằng bạn thân của bạn về cách tán một cô gái là có khả thi hay không, ta cần xem xét liệu có khả thi không dựa trên những gì chàng trai đang có (ở lứa tuổi học sinh) có thể biểu diễn dưới dạng sơ đồ quyết định như sau:

<img src="/assets/images/bai12/anh1.png" class="normalpic"/>

<p align="center"> <b>Hình 1</b>: Mức độ khả thi tán cô gái dựa trên câu hỏi</p>

Với hình 1 ta có thể dịch lại như sau: 

- Nếu cậu bạn đẹp trai thì khả thi, nếu không cậu ta có khả năng ăn nói trôi chảy thì cũng khả thi và các trường hợp ngược lại còn lại sẽ là không khả thi.

- Ô màu xanh dương là kí hiệu cho có khả thi (P) hoặc không khả thi (I).

- Ô màu vàng là câu hỏi đặt ra.

- Ô màu xanh lục là khả thi (P), màu đỏ là không khả thi (I).

Ta thấy rằng để đưa ra quyết định khả thi hay không khả thi, ta phải đi trả lời liên tiếp các câu hỏi liên quan tới vấn đề cần quyết định. Và đây cũng chính là ý tưởng của thuật toán hôm nay mình sẽ đề cập tới - Decision Tree. Thuật toán này sẽ mô tả những gì mà ta suy nghĩ thường ngày để đưa ra quyết định bằng cách đặt ra những câu hỏi.

## 2. Ý tưởng chính

Để hiểu rõ các bước mà mô hình Decision Tree làm việc, mình sẽ đưa ra một bộ dữ liệu 2D gồm 2 thuộc tính $x_1$ và $x_2$ như sau (ví dụ này được trích từ [Machine Learning cho dữ liệu dạng bảng](https://machinelearningcoban.com/tabml_book/ch_model/decision_tree.html)):

<img src="/assets/images/bai12/anh2.png" class="medianpic"/>

<p align="center"> <b>Hình 2</b>: Dataset</p>

Để phân chia tập dữ liệu này thành 2 lớp, ta có thể nghĩ ngay tới thuật toán [Logistic Regression](https://hnhoangdz.github.io/2021/11/12/LogisticRegression.html) để tìm ra 1 đường thẳng ngăn cách. Tuy nhiên, với Decision Tree có thể tìm ra đường nhiều đường phân cách hơn, cụ thể ta xét với ngưỡng và $x_1 > 5$ để chia tập dữ liệu như sau:

<img src="/assets/images/bai12/anh3.png" class="medianpic"/>

<p align="center"> <b>Hình 3</b>: $x_1 > 5$</p>

Từ đây ta có thể suy ra rằng, với bất kì điểm dữ liệu mới nào cần dự đoán chỉ cần có giá trị $x_1 > 5$ sẽ thuộc _nhãn 1_. Tuy nhiên với $x_1 < 5$ ta có thể thấy rằng vẫn còn có phần dữ liệu nhãn màu xanh lục, vì vậy khi có điểm dữ liệu có $x_1 < 5$ thì rất có thể dự đoán nhầm lẫn. Do đó, ta sẽ tiếp tục xét một ngưỡng $x_2 > 4$ như sau:

<img src="/assets/images/bai12/anh4.png" class="medianpic"/>

<p align="center"> <b>Hình 4</b>: $x_2 > 4$</p>

Lúc này ta có thể nói rằng với các điểm dữ liệu có $x_1 > 5$ thì sẽ thuộc _nhãn 1_, nếu $x_1 < 5$ mà có $x_2 > 4$ cũng sẽ thuộc _nhãn 1_ và các trường hợp còn lại thuộc _nhãn 0_. Và đây cũng chính là cách mà thuật toán Decision Tree sẽ làm:

<img src="/assets/images/bai12/anh5.png" class="large"/>

<p align="center"> <b>Hình 5</b>: Decision Tree</p>

Ở **hình 5** bên phải chính là cây quyết định mà ta đã tạo ra sau khi xét các ngưỡng. Một số kí hiệu về cây quyết định như sau:

- Trong decision tree, các ô hình chữ nhật màu xanh, đỏ trên được gọi là các **node**

- Các **node** thể hiện đầu ra màu đỏ (nhãn 1 và 0) được gọi là **node lá (leaf node hoặc terminal node)**

- Các **node** thể hiện câu hỏi màu xanh là các **non-leaf node**

- **Non-leaf node** trên cùng (câu hỏi đầu tiên) được gọi là **node gốc (root node)**

Vậy tiêu chí gì để mình tìm được điều kiện đầu tiên? tại sao lại là $x_1$ và tại sao lại là 5 mà không phải là một số khác? Nếu mọi người để ý ở trên thì mình sẽ tạo điều kiện để tách dữ liệu thành 2 phần mà dữ liệu mỗi phần có tính phân tách hơn dữ liệu ban đầu. Ví dụ: điều kiện $x_1 > 5$, tại nhánh đúng thì tất cả các phần tử đều thuộc lớp 1.

- Thế điều kiện $x_1 > 8$ cũng chia nhánh đúng thành toàn lớp 1 sao không chọn? vì nhánh đúng ở điều kiện x1>5 chứa nhiều phần tử lớp 1 hơn và tổng quát hơn nhán đúng của $x_1 > 8$.

- Còn điều kiện $x_1 > 2$ thì cả 2 nhánh con đều chứa dữ liệu của cả lớp 0 và lớp 1.

Từ ví dụ này ta có thể thấy rằng, ở mỗi bước chọn biến ($x_1$) và chọn ngưỡng ($t_1$) là các cách chọn tốt nhất có thể ở mỗi bước. Cách chọn này giống với ý tưởng của thuật toán _tham lam (greedy)_. Cách chọn này có thể không phải là tối ưu, nhưng trực giác cho chúng ta thấy rằng cách làm này sẽ gần với cách làm tối ưu. Ngoài ra, cách làm này khiến cho bài toán cần giải quyết trở nên đơn giản hơn. 

Tuy nhiên, tiêu chí chọn $x_1$ và $t_1$ để đưa ra điều kiện phân tách đều dựa trên trực giác mình có thể nhìn thấy được để đưa ra. Nhưng máy tính thì khác, nó cần một độ đo số liệu cụ thể để đưa ra các điều kiện phân tách. Vì vậy, trong bài hôm nay mình sẽ giới thiệu 2 thuật toán phổ biến nhất để làm với Decision Tree: ID3, CART.

## 3. ID3

