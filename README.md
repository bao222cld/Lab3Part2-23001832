Nội dung nộp

- notebook.ipynb — notebook gốc (23001832_NguyenLeNgocBao_Lab03Part02)

- src/representations/word_embedder.py — module chứa class WordEmbedder.

- test/lab4_test.py — script test demo (in vector length, similarity, top-10, document embedding shape).

- README.txt — file này.

Các bước đã thực hiện

Chuẩn bị môi trường: cài/gọi các thư viện chính (gensim, numpy, matplotlib, scikit-learn, umap-learn).

Phần 1: giảm chiều và trực quan hóa word vectors bằng PCA, t-SNE, UMAP (vẽ scatter plots và chú thích).

Phần 2:

Định nghĩa lớp WordEmbedder (load pretrained GloVe, tokenization, get_vector, get_similarity, get_most_similar, embed_document bằng trung bình vector từ).

Ghi file module src/representations/word_embedder.py.

Tạo file test/lab4_test.py có phần tự động thêm src/ vào sys.path để chạy thuận tiện.

Chạy test trong notebook: !python test/lab4_test.py — giữ toàn bộ output trong notebook trước khi xuất PDF.

Xuất PDF từ Colab (File → Print → Save as PDF) đảm bảo không ẩn outputs.

Lệnh chạy nhanh (trong Colab)
# cài thư viện nếu cần
!pip install --quiet gensim umap-learn scikit-learn matplotlib

# chạy test demo (sẽ tải model nếu lần đầu)
!python test/lab4_test.py

# (tuỳ chọn) nén submission
!zip -r submission.zip src test notebook.ipynb 23001832_NguyenLeNgocBao_Lab03.pdf README.txt

Khó khăn gặp phải

- Tải mô hình pretrained (GloVe) lần đầu khá nặng và mất thời gian; cần giữ thông báo tải trong PDF để chứng minh đã chạy.

- Huấn luyện Word2Vec trên tập rất nhỏ cho embedding kém chất lượng — dẫn tới kết quả so sánh không hoàn toàn công bằng với mô hình pretrained.

- Xử lý OOV: một vài từ không có vectơ trong GloVe; fastText cải thiện phần nào nhưng không giải quyết mọi trường hợp.

- Đồng bộ cấu trúc dự án (src/ vs representations/) cần xử lý sys.path khi chạy script test từ notebook.

Đánh giá tổng kết 

Bài lab giúp củng cố khái niệm về Word2Vec, GloVe và fastText và cách trực quan hóa không gian embedding. Kết quả cho thấy mô hình pretrained (GloVe) nắm bắt mối quan hệ ngữ nghĩa tốt hơn so với mô hình tự huấn luyện trên tập nhỏ; fastText có lợi thế với từ OOV. Phương pháp lấy trung bình vector là baseline đơn giản nhưng hạn chế về ngữ cảnh; có thể nâng cấp bằng mô hình ngữ cảnh hoặc trọng số TF-IDF.
